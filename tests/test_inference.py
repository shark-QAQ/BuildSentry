# tests/test_inference.py
"""
推理服务测试。

重点关注：
1. 来源标记必须真实——模型已加载时输出解析失败只能报失败，
   绝不能回退到模拟（否则假违规项会被当成模型判断）；
2. 只解码新生成的 token，不能把输入 prompt 一起解码；
3. 审核结论与评分必须可复现；
4. 模型必须懒加载——import 阶段不得加载 7B 权重；
5. 显卡选择必须遵守排除列表，且始终单卡。
"""

import asyncio
import contextlib

import pytest

from app.services.audit_contract import build_prompt
from app.services.inference_service import (
    SOURCE_MODEL,
    SOURCE_MODEL_ERROR,
    SOURCE_SIMULATION,
    InferenceService,
)
from app.services.model_loader import ModelLoader

PROMPT_LEN = 120  # 模拟输入 prompt 的 token 数
MODEL_LEN = 25  # 模拟新生成部分的 token 数

VALID_OUTPUT = (
    '{"has_violation": true, "violations": [{"risk_level": "high",'
    ' "description": "堆载超限", "suggestion": "整改"}]}'
)

# 含审核关键词：模拟模式对这段文字必定产出违规项，
# 因此"返回空列表"可以证明代码没有偷偷回退到模拟。
PARAGRAPH_WITH_KEYWORD = "基坑周边1.2米范围内设置堆载区域，堆载高度约1.8米。"


def run(coro):
    return asyncio.run(coro)


class _FakeTensor:
    """最小张量替身：支持 .shape / .to() / 切片，并记录切片后的长度与去过的设备"""

    def __init__(self, length):
        self.shape = (1, length)
        self.length = length
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = self.length if item.stop is None else item.stop
            return _FakeTensor(max(0, stop - start))
        return _FakeTensor(1)


class _BaseTokenizer:
    """测试用 tokenizer，记录每次 decode 收到的 token 数"""

    chat_template = "qwen-chat-template"

    def __init__(self):
        self.decoded_lengths = []

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
        return _FakeTensor(PROMPT_LEN)

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_lengths.append(ids.length)
        return "我认为这段方案存在若干问题，但没有按要求输出 JSON。"


class _GoodTokenizer(_BaseTokenizer):
    """输出符合约定 schema 的 JSON"""

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_lengths.append(ids.length)
        return VALID_OUTPUT


class _PromptEchoTokenizer(_BaseTokenizer):
    """忠实模拟真实 tokenizer：喂进来的 token 越多，吐出的文本越多。

    用来暴露"解码了整个序列"这个错误——prompt 里含格式示例 JSON，
    一旦把 prompt 一起解码，find("{") 就会命中示例而非模型输出。
    """

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_lengths.append(ids.length)
        if ids.length <= MODEL_LEN:
            return VALID_OUTPUT
        return build_prompt("段落", []) + VALID_OUTPUT


class _FakeModel:
    def __init__(self, device="cpu"):
        self.device = device
        self.generate_calls = 0
        self.received_inputs = None

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.received_inputs = kwargs
        return [_FakeTensor(PROMPT_LEN + MODEL_LEN)]


def _use_model(monkeypatch, tokenizer, model=None):
    """让 model_loader 表现为已加载模型"""
    from app.services import model_loader as ml

    model = model or _FakeModel()
    monkeypatch.setattr(ml.model_loader, "is_ready", lambda: True)
    monkeypatch.setattr(ml.model_loader, "get_model", lambda: (model, tokenizer))
    return model


class TestGeneration:
    """生成路径测试"""

    def test_only_generated_tokens_are_decoded(self, monkeypatch):
        """必须只解码新生成的 token

        回归测试：此前 decode(outputs[0]) 把输入 prompt 一并解码，而 prompt
        里含有格式示例 JSON，导致 parse_model_output 的 find("{") 命中示例
        而非模型输出，真实推理必然解析失败。
        """
        tokenizer = _PromptEchoTokenizer()
        _use_model(monkeypatch, tokenizer)

        violations, source = run(InferenceService.audit_paragraph("段落", []))

        assert tokenizer.decoded_lengths == [MODEL_LEN]  # 不是 PROMPT_LEN + MODEL_LEN
        assert source == SOURCE_MODEL
        assert len(violations) == 1

    def test_output_is_parseable_after_slicing(self, monkeypatch):
        """切片后模型输出应能被正常解析"""
        _use_model(monkeypatch, _GoodTokenizer())
        violations, source = run(InferenceService.audit_paragraph("段落", []))
        assert source == SOURCE_MODEL
        assert violations[0]["risk_level"] == "high"

    def test_inputs_are_moved_to_model_device(self, monkeypatch):
        """输入张量必须搬到模型所在设备

        tokenizer 产出的是 CPU 张量，而模型可能在 GPU 上（单卡加载为
        device_map={"": 0}）。不搬会直接报设备不匹配。
        """
        model = _use_model(monkeypatch, _GoodTokenizer(), model=_FakeModel(device="cuda:0"))
        run(InferenceService.audit_paragraph("段落", []))

        moved = model.received_inputs["input_ids"]
        assert moved.moved_to == "cuda:0"


class TestProvenance:
    """来源标记测试"""

    def test_model_error_does_not_fall_back_to_simulation(self, monkeypatch):
        """模型已加载但输出无法解析时，必须报失败而不是返回模拟数据

        回归测试：此前异常被吞掉后继续走模拟分支，而 model_version 只看
        模型是否加载，于是模拟生成的假违规项被标记成 real-v1.0。
        """
        _use_model(monkeypatch, _BaseTokenizer())

        violations, source = run(InferenceService.audit_paragraph(PARAGRAPH_WITH_KEYWORD, []))

        assert source == SOURCE_MODEL_ERROR
        assert violations == []  # 关键：没有伪造任何违规项

    def test_simulation_only_when_model_absent(self, monkeypatch):
        """未加载模型时才走模拟，并明确标记来源"""
        from app.core.config import settings
        from app.services import model_loader as ml

        monkeypatch.setattr(settings, "SIMULATION_DELAY", 0)
        monkeypatch.setattr(ml.model_loader, "is_ready", lambda: False)

        violations, source = run(InferenceService.audit_paragraph(PARAGRAPH_WITH_KEYWORD, []))

        assert source == SOURCE_SIMULATION
        assert len(violations) == 1

    def test_model_is_retried_before_giving_up(self, monkeypatch):
        """解析失败应先重试，而不是一次就放弃"""
        model = _use_model(monkeypatch, _BaseTokenizer())
        run(InferenceService.audit_paragraph("段落", []))
        assert model.generate_calls == 2  # max_retries=1 → 共 2 次尝试


class TestAggregateResults:
    """结果聚合测试"""

    def test_identical_input_gives_identical_output(self):
        """同一批违规项重复聚合，结论与评分必须完全一致"""
        violations = [
            {"risk_level": "high", "description": "a"},
            {"risk_level": "medium", "description": "b"},
        ]
        results = [
            run(InferenceService.aggregate_results(violations, [SOURCE_MODEL] * 5))
            for _ in range(10)
        ]

        assert len({r["quality_score"] for r in results}) == 1
        assert len({r["conclusion"] for r in results}) == 1

    def test_conclusion_and_score_by_risk(self):
        """结论与评分应只由违规项构成决定"""
        cases = [
            ([], "passed", 90),
            ([{"risk_level": "low"}], "conditional_pass", 85),
            ([{"risk_level": "medium"}], "conditional_pass", 80),
            ([{"risk_level": "high"}], "rejected", 70),
            ([{"risk_level": "high"}, {"risk_level": "high"}], "rejected", 50),
        ]
        for violations, expected_conclusion, expected_score in cases:
            result = run(InferenceService.aggregate_results(violations, [SOURCE_MODEL]))
            assert result["conclusion"] == expected_conclusion, violations
            assert result["quality_score"] == expected_score, violations

    def test_score_is_clamped_to_zero(self):
        """高风险违规足够多时评分不应为负"""
        result = run(
            InferenceService.aggregate_results([{"risk_level": "high"}] * 10, [SOURCE_MODEL])
        )
        assert result["quality_score"] == 0

    def test_model_version_reflects_actual_sources(self):
        """model_version 必须反映真实来源，而不是模型是否加载"""
        cases = [
            ([SOURCE_MODEL], "real-v1.0"),
            ([SOURCE_SIMULATION], "simulation-v1.0"),
            ([SOURCE_MODEL, SOURCE_SIMULATION], "mixed"),
            ([SOURCE_MODEL, SOURCE_MODEL_ERROR], "mixed"),
            ([], "simulation-v1.0"),
        ]
        for sources, expected in cases:
            result = run(InferenceService.aggregate_results([], sources))
            assert result["model_version"] == expected, sources

    def test_failed_paragraphs_are_surfaced(self):
        """有段落解析失败时，结论里必须说明，不能装作审核完整"""
        result = run(
            InferenceService.aggregate_results(
                [], [SOURCE_MODEL, SOURCE_MODEL_ERROR, SOURCE_MODEL_ERROR]
            )
        )
        assert "2 个段落" in result["conclusion_detail"]

    def test_no_failure_note_when_all_succeeded(self):
        result = run(InferenceService.aggregate_results([], [SOURCE_MODEL]))
        assert "解析失败" not in result["conclusion_detail"]

    def test_no_longer_reports_fake_duration(self):
        """审核耗时由调用方实际计时写入，聚合结果里不应再有写死的常量"""
        result = run(InferenceService.aggregate_results([], [SOURCE_MODEL]))
        assert "audit_duration_ms" not in result

    def test_simulated_confidence_is_stable(self, monkeypatch):
        """模拟模式下同一段落的置信度应稳定"""
        from app.core.config import settings
        from app.services import model_loader as ml

        monkeypatch.setattr(settings, "SIMULATION_DELAY", 0)
        monkeypatch.setattr(ml.model_loader, "is_ready", lambda: False)

        clauses = [
            {
                "content": "基坑周边1.2米范围内严禁堆载。",
                "metadata": {"spec_code": "JGJ 59-2011", "article_id": "3.11.3"},
            }
        ]
        runs = [
            run(InferenceService.audit_paragraph(PARAGRAPH_WITH_KEYWORD, clauses)) for _ in range(3)
        ]
        confidences = {v["confidence"] for r in runs for v in r[0]}
        assert len(confidences) == 1


class TestModelLoader:
    """模型加载器测试"""

    def test_model_is_not_loaded_at_import(self, monkeypatch):
        """实例化不应触发加载，首次使用时才加载，且只加载一次"""
        loader = ModelLoader()
        monkeypatch.setattr(loader, "_initialized", False)

        calls = []
        monkeypatch.setattr(ModelLoader, "_load_model", lambda self: calls.append(1))

        assert calls == []  # 构造实例本身不加载

        loader.is_ready()
        assert len(calls) == 1  # 首次使用时加载

        loader.is_ready()
        loader.get_model()
        assert len(calls) == 1  # 之后不再重复加载

    def test_model_loads_on_selected_gpu(self, monkeypatch):
        """模型必须加载到选出的那张卡上，且不拆到多卡"""
        import sys
        import types

        from app.core.config import settings

        captured = {}

        class _FakeAutoModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                captured.update(kwargs)
                return object()

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return object()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = _FakeAutoModel
        fake_transformers.AutoTokenizer = _FakeAutoTokenizer

        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = object

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setattr(settings, "get_device", lambda: "cuda")
        monkeypatch.setattr(settings, "MODEL_GPU_INDEX", "5")

        loader = ModelLoader()
        monkeypatch.setattr(loader, "_initialized", False)
        loader._load_model()

        # device_map 必须是单卡形式，不能是 "auto"（会把模型拆到多张卡）
        assert captured["device_map"] == {"": 5}

    def test_device_map_is_single_card_not_auto(self, monkeypatch):
        """device_map 绝不能是 "auto"——那会把模型拆到多张卡上"""
        import sys
        import types

        from app.core.config import settings

        captured = {}

        class _Recorder:
            @staticmethod
            def from_pretrained(name, **kwargs):
                captured.update(kwargs)
                return object()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = _Recorder
        fake_transformers.AutoTokenizer = _Recorder
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = object

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setattr(settings, "get_device", lambda: "cuda")
        monkeypatch.setattr(settings, "MODEL_GPU_INDEX", "auto")
        monkeypatch.setattr(settings, "MODEL_GPU_EXCLUDE", "0,1")
        monkeypatch.setattr(settings, "MODEL_GPU_MIN_FREE_MB", 20000)
        # 只有 5 号卡够用（0/1 被排除，其余不足）
        _fake_gpu_memory(monkeypatch, dict.fromkeys(range(8), 18000) | {5: 21000})

        loader = ModelLoader()
        monkeypatch.setattr(loader, "_initialized", False)
        loader._load_model()

        device_map = captured["device_map"]
        assert isinstance(device_map, dict)
        assert list(device_map) == [""]
        assert device_map[""] == 5


def _fake_gpu_memory(monkeypatch, free_mb_by_index, total_mb=49140):
    """伪造各卡的空闲显存（MB）。

    卡数取 max(键) + 1，未列出的卡视作 0 空闲（必然低于阈值被跳过），
    这样测试里只需列出关心的那几张卡。
    """
    import torch

    count = max(free_mb_by_index) + 1 if free_mb_by_index else 0
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)

    def _mem_get_info(index):
        free = free_mb_by_index.get(index, 0) * 1024 * 1024
        return (free, total_mb * 1024 * 1024)

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)


class TestGpuSelection:
    """显卡选择测试"""

    def _settings(self, monkeypatch, index="auto", exclude="0,1", min_free=20000):
        from app.core.config import settings

        monkeypatch.setattr(settings, "MODEL_GPU_INDEX", index)
        monkeypatch.setattr(settings, "MODEL_GPU_EXCLUDE", exclude)
        monkeypatch.setattr(settings, "MODEL_GPU_MIN_FREE_MB", min_free)

    def test_auto_picks_freest_allowed_card(self, monkeypatch):
        """auto 应在允许的卡里挑空闲显存最多的一张"""
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch)
        _fake_gpu_memory(
            monkeypatch,
            {0: 40000, 1: 40000, 2: 21000, 3: 16000, 4: 18000, 5: 18000, 6: 22000, 7: 18000},
        )

        # 0/1 卡虽然最空但被排除；6 号卡在剩余里最空
        assert select_gpu_index() == 6

    def test_excluded_cards_are_never_chosen(self, monkeypatch):
        """被排除的卡即使空闲最多也不能选"""
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch, exclude="0,1")
        _fake_gpu_memory(monkeypatch, {0: 48000, 1: 48000, 2: 21000})

        assert select_gpu_index() == 2

    def test_skips_cards_below_min_free_memory(self, monkeypatch):
        """空闲显存不足的卡应跳过，避免加载到一半 OOM"""
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch, exclude="0,1", min_free=20000)
        _fake_gpu_memory(monkeypatch, {2: 21000, 3: 16000, 4: 18000, 5: 19500})

        assert select_gpu_index() == 2  # 3/4/5 都不够

    def test_raises_when_no_card_qualifies(self, monkeypatch):
        """所有卡都不满足条件时必须明确报错，而不是硬加载"""
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch, exclude="0,1", min_free=20000)
        _fake_gpu_memory(monkeypatch, {0: 48000, 1: 48000, 2: 15000, 3: 15000})

        with pytest.raises(RuntimeError, match="没有满足条件的 GPU"):
            select_gpu_index()

    def test_raises_when_no_gpu_present(self, monkeypatch):
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch)
        _fake_gpu_memory(monkeypatch, {})

        with pytest.raises(RuntimeError, match="未检测到可用 GPU"):
            select_gpu_index()

    def test_explicit_index_overrides_auto(self, monkeypatch):
        """显式指定卡号时不应走自动挑选"""
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch, index="5", exclude="0,1")
        _fake_gpu_memory(monkeypatch, {5: 100})  # 即使显存很少也照用

        assert select_gpu_index() == 5

    def test_ties_prefer_lower_index(self, monkeypatch):
        """空闲显存相同时取卡号较小的，保证选择可预期"""
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch, exclude="")
        _fake_gpu_memory(monkeypatch, {2: 20000, 3: 20000, 4: 20000})

        assert select_gpu_index() == 2

    def test_shared_machine_scenario(self, monkeypatch):
        """复现实机场景：8 张卡各被占用约 30GB，0/1 禁用，选最空的可用卡

        显存数据取自实际 nvidia-smi 输出（每卡 49140 MiB，已用 28092-32954 MiB）。
        """
        from app.services.model_loader import select_gpu_index

        self._settings(monkeypatch, exclude="0,1", min_free=18000)
        _fake_gpu_memory(
            monkeypatch,
            {0: 18670, 1: 18562, 2: 21048, 3: 16186, 4: 18670, 5: 18562, 6: 18670, 7: 18562},
        )

        # 0/1 被排除；3 号卡只剩 16186 MiB 不达标；其余里 2 号卡最空
        assert select_gpu_index() == 2

    def test_parse_excluded_tolerates_fullwidth_comma(self):
        from app.services.model_loader import parse_excluded_gpus

        assert parse_excluded_gpus("0,1") == {0, 1}
        assert parse_excluded_gpus("0，1") == {0, 1}
        assert parse_excluded_gpus(" 0 , 1 ") == {0, 1}
        assert parse_excluded_gpus("") == set()

    def test_marks_initialized_even_if_loading_fails(self, monkeypatch):
        """加载失败不应导致每次请求都重试一遍"""
        loader = ModelLoader()
        monkeypatch.setattr(loader, "_initialized", False)

        def boom(self):
            raise RuntimeError("模拟加载失败")

        monkeypatch.setattr(ModelLoader, "_load_model", boom)

        with contextlib.suppress(RuntimeError):
            loader.is_ready()

        assert loader._initialized is True
