# app/services/model_loader.py
"""
模型加载服务，负责加载和管理模型。

模型采用懒加载：import 阶段不会触碰权重文件。否则 `import app.main` 就会
触发下载并加载 7B 模型（约 14GB 显存、可能数分钟），服务启动会被拖住。
"""

import os
import threading
from typing import Any

import torch

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("services.model_loader")


def parse_excluded_gpus(raw: str) -> set[int]:
    """解析 MODEL_GPU_EXCLUDE，容忍中英文逗号与空白"""
    text = raw.replace("，", ",")
    excluded: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            excluded.add(int(part))
    return excluded


def select_gpu_index() -> int:
    """决定把模型加载到哪张卡上（始终是单卡）。

    - `MODEL_GPU_INDEX` 显式指定时直接使用该卡；
    - 为 "auto" 时在候选卡里挑**空闲显存最多**的一张，
      跳过 `MODEL_GPU_EXCLUDE` 中的卡，以及空闲显存低于
      `MODEL_GPU_MIN_FREE_MB` 的卡（避免加载到一半 OOM）。
    """
    explicit = (settings.MODEL_GPU_INDEX or "").strip()
    if explicit and explicit.lower() != "auto":
        return int(explicit)

    excluded = parse_excluded_gpus(settings.MODEL_GPU_EXCLUDE)
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("未检测到可用 GPU，无法自动选择显卡")

    candidates: list[tuple[int, int]] = []  # (空闲 MB, 卡号)
    for index in range(count):
        if index in excluded:
            continue
        free_mb = torch.cuda.mem_get_info(index)[0] // (1024 * 1024)
        if free_mb < settings.MODEL_GPU_MIN_FREE_MB:
            logger.info("跳过 GPU %d：空闲 %d MB，低于阈值", index, free_mb)
            continue
        candidates.append((free_mb, index))

    if not candidates:
        raise RuntimeError(
            f"没有满足条件的 GPU：已排除 {sorted(excluded)}，"
            f"且要求空闲显存 ≥ {settings.MODEL_GPU_MIN_FREE_MB}MB"
        )

    # 空闲显存最多者优先；相同时取卡号较小的一张，保证选择可预期
    free_mb, index = max(candidates, key=lambda item: (item[0], -item[1]))
    logger.info("自动选择 GPU %d（空闲 %d MB）", index, free_mb)
    return index


class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._is_loaded = False
            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance._device = None
        return cls._instance

    def _ensure_initialized(self) -> None:
        """首次使用时才真正加载模型（双重检查加锁，避免并发重复加载）"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                self._load_model()
            finally:
                # 即使加载抛异常也置位：否则每个请求都会重试一次昂贵的加载，
                # 正确行为是失败一次后稳定回退到模拟模式。
                self._initialized = True

    def _load_model(self) -> None:
        self._device = settings.get_device()

        if self._device != "cuda":
            logger.info("未检测到 GPU，使用模拟推理模式（GPU 可用时自动切换）")
            return

        logger.info("检测到 GPU，准备加载真实模型...")
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_model_name = settings.BASE_MODEL
            lora_path = settings.LORA_PATH

            gpu_index = select_gpu_index()
            # Qwen3 以 bf16 训练，用 bf16 推理与之对齐且不易溢出；
            # 老架构（图灵及更早）不支持 bf16，退回 fp16。
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info("加载基座模型: %s → GPU %d (%s)", base_model_name, gpu_index, dtype)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            # 单卡加载。不要用 device_map="auto"：它会把模型拆到多张卡上，
            # 在与其他服务共存的机器上既会抢占显存，也可能因碎片化而 OOM。
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map={"": gpu_index},
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            if lora_path and os.path.exists(lora_path):
                logger.info("加载 LoRA 适配器: %s", lora_path)
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()

            self._model = model
            self._tokenizer = tokenizer
            self._is_loaded = True
            logger.info("真实模型加载完成（GPU %d）", gpu_index)
        except Exception as e:
            logger.error("模型加载失败，将回退至模拟模式: %s", e)
            self._is_loaded = False

    def is_ready(self) -> bool:
        self._ensure_initialized()
        return self._is_loaded and self._model is not None

    def get_model(self) -> tuple[Any, Any]:
        self._ensure_initialized()
        return self._model, self._tokenizer


model_loader = ModelLoader()
