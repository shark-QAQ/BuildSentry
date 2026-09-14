# app/services/inference_service.py
"""
推理服务模块。
提供对施工方案段落的违规审核功能，支持真实模型推理和模拟推理两种模式。

每次审核都会带回来源标记（见 SOURCE_* 常量），聚合时据此判定 model_version。
这一点很关键：此前模型输出解析失败会静默回退到模拟，而 model_version 只看
模型是否加载，导致模拟生成的假违规项被标记成 real-v1.0。
"""

import asyncio
import random
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.audit_contract import build_prompt, parse_model_output
from app.services.generation import generate_text
from app.services.model_loader import model_loader

logger = get_logger("services.inference")

# 违规项的来源标记
SOURCE_MODEL = "model"  # 真实模型给出
SOURCE_SIMULATION = "simulation"  # 未加载模型，走模拟模式
SOURCE_MODEL_ERROR = "model_error"  # 已加载模型但输出无法解析，重试后仍失败


class InferenceService:
    @staticmethod
    async def audit_paragraph(
        paragraph_text: str,
        context_clauses: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 1,
    ) -> tuple[list[dict[str, Any]], str]:
        """审核单个段落，返回 (违规项列表, 来源)。

        已加载模型时只可能返回 SOURCE_MODEL 或 SOURCE_MODEL_ERROR，
        绝不回退到模拟——合规审核里用生成的假违规项冒充模型输出是不可接受的。
        """
        if model_loader.is_ready():
            return await InferenceService._run_model(
                paragraph_text, context_clauses, temperature, max_tokens, max_retries
            )

        return (
            await InferenceService._run_simulation(paragraph_text, context_clauses),
            SOURCE_SIMULATION,
        )

    @staticmethod
    async def _run_model(
        paragraph_text: str,
        context_clauses: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ) -> tuple[list[dict[str, Any]], str]:
        """调用真实模型，失败时重试，仍失败则如实报告失败"""
        model, tokenizer = model_loader.get_model()
        prompt = build_prompt(paragraph_text, context_clauses)

        def _generate() -> str:
            return generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                # 生成是同步的 CPU/GPU 计算，直接 await 会阻塞整个事件循环，
                # 期间该 worker 无法处理任何其他请求。
                result_text = await asyncio.to_thread(_generate)
                return parse_model_output(result_text), SOURCE_MODEL
            except Exception as e:
                last_error = e
                logger.warning(
                    "段落审核失败（第 %d/%d 次尝试）: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )

        logger.error("段落审核重试 %d 次后仍失败: %s", max_retries, last_error)
        return [], SOURCE_MODEL_ERROR

    @staticmethod
    async def _run_simulation(
        paragraph_text: str, context_clauses: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """模拟模式：仅在未加载模型时使用"""
        await asyncio.sleep(settings.SIMULATION_DELAY)
        # 以段落内容为种子：同一段落重复审核得到相同置信度，便于结果复现
        rng = random.Random(paragraph_text)
        keywords = settings.AUDIT_KEYWORDS
        if any(kw in paragraph_text for kw in keywords):
            if context_clauses and len(context_clauses) > 0:
                first = context_clauses[0]
                return [
                    {
                        "clause_code": first["metadata"].get("spec_code", "JGJ 59-2011")
                        + " "
                        + first["metadata"].get("article_id", "3.11.x"),
                        "clause_content": first["content"][:100],
                        "risk_level": "high" if "严禁" in first["content"] else "medium",
                        "violation_type": "定量违规" if "严禁" in first["content"] else "定性缺失",
                        "description": f"段落内容与规范要求存在偏差，建议核查第{first['metadata'].get('article_id', '3.11.x')}条。",
                        "suggestion": "请严格按照规范要求整改。",
                        "confidence": round(0.80 + rng.random() * 0.15, 2),
                    }
                ]
            else:
                return [
                    {
                        "clause_code": "JGJ 59-2011 3.11.3",
                        "clause_content": "基坑周边1.2米范围内严禁堆载，堆载高度不得超过1.5米。",
                        "risk_level": "high",
                        "violation_type": "定量违规",
                        "description": f"段落中发现潜在违规：{paragraph_text[:30]}...",
                        "suggestion": "建议重新审查该部分内容。",
                        "confidence": round(0.80 + rng.random() * 0.15, 2),
                    }
                ]
        return []

    @staticmethod
    async def aggregate_results(
        all_violations: list[dict[str, Any]],
        sources: list[str],
    ) -> dict[str, Any]:
        """聚合成审核结论。

        sources 为各段落审核的来源标记，用于判定 model_version 与结论可信度。
        """
        high = [v for v in all_violations if v.get("risk_level") == "high"]
        medium = [v for v in all_violations if v.get("risk_level") == "medium"]
        low = [v for v in all_violations if v.get("risk_level") == "low"]

        if high:
            conclusion, detail = "rejected", f"存在 {len(high)} 项高风险违规，不予通过。"
        elif medium:
            conclusion, detail = (
                "conditional_pass",
                f"存在 {len(medium)} 项中风险违规，需整改后通过。",
            )
        elif low:
            conclusion, detail = "conditional_pass", f"存在 {len(low)} 项低风险违规，建议优化。"
        else:
            conclusion, detail = "passed", "未发现明显违规，予以通过。"

        # 评分必须只由违规项决定：此前叠加了 random.randint(-5, 5)，
        # 导致同一份文档重复审核分数会变。审核耗时由调用方实际计时后写入。
        base_score = 90 - len(high) * 20 - len(medium) * 10 - len(low) * 5
        quality_score = max(0, min(100, base_score))

        unique_sources = set(sources)
        if not unique_sources or unique_sources == {SOURCE_SIMULATION}:
            model_version = "simulation-v1.0"
        elif unique_sources == {SOURCE_MODEL}:
            model_version = "real-v1.0"
        else:
            # 同时存在模型输出、模拟输出或解析失败
            model_version = "mixed"

        failed = sum(1 for s in sources if s == SOURCE_MODEL_ERROR)
        if failed:
            # 结论建立在"部分段落未能分析"的基础上，必须让使用者看到
            detail += f"（另有 {failed} 个段落模型输出解析失败，未纳入结论，建议人工复核）"

        return {
            "conclusion": conclusion,
            "conclusion_detail": detail,
            "high_risk_count": len(high),
            "medium_risk_count": len(medium),
            "low_risk_count": len(low),
            "total_violations": len(all_violations),
            "quality_score": quality_score,
            "model_version": model_version,
        }
