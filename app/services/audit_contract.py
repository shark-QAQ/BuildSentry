# app/services/audit_contract.py
"""
审核任务的输入输出契约。

推理侧与训练侧必须共用此模块：
- 构造训练数据（JSONL）时，prompt 必须由 build_prompt() 生成，
  输出必须通过 parse_model_output() 校验；
- 微调后的适配器若与这里的模板不一致，接上去不会生效——
  这是 LoRA 微调最常见的失败原因。

因此本模块是唯一来源，任何一处改动都会同时影响推理与训练。
"""

import json
from typing import Any

# 风险等级取值。聚合逻辑（InferenceService.aggregate_results）按这三个字面量
# 统计，训练数据里出现其它取值会导致违规项被静默漏计。
RISK_LEVELS = ("high", "medium", "low")

# 输出中每条违规项必须包含的字段
REQUIRED_VIOLATION_FIELDS = ("risk_level", "description", "suggestion")

PROMPT_TEMPLATE = """你是一名资深建筑施工安全审核专家。请根据以下规范条文，审核输入的施工方案段落。

【规范条文】
{clauses_text}

【施工方案段落】
{paragraph_text}

请按以下JSON格式输出审核结果（只输出JSON，不要其他内容）：
{{
  "has_violation": true/false,
  "violations": [
    {{
      "clause_code": "规范编号+条文编号",
      "clause_content": "条文原文",
      "risk_level": "high/medium/low",
      "description": "违规描述",
      "suggestion": "整改建议",
      "confidence": 0.95
    }}
  ]
}}
"""


class OutputParseError(ValueError):
    """模型输出无法解析为约定的 JSON 结构"""


def to_clause_context(search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把向量检索结果转成 prompt 所需的条文结构。

    推理与数据生成共用，避免两边喂给模型的条文格式出现差异。
    """
    return [
        {"content": item.get("content", ""), "metadata": item.get("metadata", {})}
        for item in search_results
    ]


def format_clauses(context_clauses: list[dict[str, Any]]) -> str:
    """把检索到的规范条文渲染成 prompt 中的条文段落"""
    return "\n".join(
        f"- {c.get('metadata', {}).get('spec_code', '')} "
        f"{c.get('metadata', {}).get('article_id', '')}: {c.get('content', '')}"
        for c in context_clauses
    )


def build_prompt(paragraph_text: str, context_clauses: list[dict[str, Any]]) -> str:
    """构造审核 prompt（推理与训练共用的唯一实现）"""
    return PROMPT_TEMPLATE.format(
        clauses_text=format_clauses(context_clauses),
        paragraph_text=paragraph_text,
    )


def encode_for_model(
    tokenizer: Any, prompt: str, *, enable_thinking: bool = False
) -> dict[str, Any]:
    """把 prompt 编码成模型输入。

    Instruct 模型必须套用自身的对话模板（Qwen 系列需要 <|im_start|> 结构），
    直接喂裸文本会削弱指令遵循与输出格式的稳定性。

    训练侧由 LLaMA-Factory 套用同一个模板，两边必须一致：
    训练套、推理不套（或反之）会造成 train/serve skew，微调后的适配器接上去几乎无效。
    因此这里用 tokenizer 自带的 chat_template，保证与训练侧同源。

    `enable_thinking` 默认 False（推理与学生模型都用非思考模式）。
    数据生成时教师模型可以设为 True 以提升标注质量——其思考内容会被
    parse_model_output 剥离，不会进入训练目标。
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        # 基座模型没有对话模板，退回裸文本编码
        return {"input_ids": tokenizer(prompt, return_tensors="pt")["input_ids"]}

    template_kwargs: dict[str, Any] = {}
    if "enable_thinking" in chat_template:
        # Qwen3 默认开启思考模式，会先吐  thinking...<｜end▁of▁thinking｜> 再给答案。
        # 对推理侧：思考内容里的花括号会让 parse_model_output 的 find("{")
        # 命中思考文本而非最终 JSON，且思考 token 大幅拉长延迟，故默认关闭。
        template_kwargs["enable_thinking"] = enable_thinking

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        **template_kwargs,
    )
    return {"input_ids": input_ids}


def split_paragraphs(raw_text: str, min_length: int = 50) -> list[str]:
    """把文档正文切成待审核段落。

    审核推理与训练数据生成共用此实现——分段方式不一致会让训练数据
    呈现出与线上不同的段落粒度，模型学到的分布就对不上了。

    原实现还有一层「按单换行再切一次」的回退，但那层永远不会执行：
    单换行切出的片段必然是某个双换行片段的子串，所以「双换行片段全部
    不超过 min_length」已经蕴含「单换行片段同样如此」。已删除。
    """
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > min_length]
    if not paragraphs:
        paragraphs = [raw_text[:2000]]  # 保底：整篇过短时取前 2000 字
    return paragraphs


def _validate_violation(raw: Any) -> dict[str, Any]:
    """校验单条违规项，字段缺失或取值非法时抛 OutputParseError"""
    if not isinstance(raw, dict):
        raise OutputParseError(f"违规项不是对象: {raw!r}")

    missing = [f for f in REQUIRED_VIOLATION_FIELDS if not raw.get(f)]
    if missing:
        raise OutputParseError(f"违规项缺少必需字段: {missing}")

    risk_level = raw["risk_level"]
    if risk_level not in RISK_LEVELS:
        # 不能放行：聚合逻辑按字面量统计，非法取值会让这条违规项被静默丢弃
        raise OutputParseError(f"非法 risk_level: {risk_level!r}，应为 {RISK_LEVELS}")

    return raw


THINKING_END = "<｜end▁of▁thinking｜>"


def strip_thinking(raw_text: str) -> str:
    """剥离 Qwen3 思考模式产生的  thinking...<｜end▁of▁thinking｜> 前缀。

    这里做的是纵深防御：正常情况下推理侧已通过 enable_thinking=False 关闭思考，
    但教师模型生成数据时会主动开启；某些 tokenizer 版本也可能忽略该开关。
    思考内容里常出现花括号，不剥离就会让下面的 find("{") 取错位置。
    """
    if THINKING_END in raw_text:
        return raw_text.split(THINKING_END, 1)[1]
    return raw_text


def parse_model_output(raw_text: str) -> list[dict[str, Any]]:
    """从模型输出中解析并校验违规项。

    解析失败一律抛 OutputParseError，绝不返回空列表冒充"未发现违规"——
    空列表和"解析失败"是两回事，混为一谈会让审计结果失真。
    """
    raw_text = strip_thinking(raw_text)
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start == -1 or end <= start:
        raise OutputParseError("输出中未找到 JSON 对象")

    try:
        data = json.loads(raw_text[start:end])
    except json.JSONDecodeError as e:
        raise OutputParseError(f"JSON 解析失败: {e}") from e

    if not isinstance(data, dict):
        raise OutputParseError(f"顶层结构不是对象: {type(data).__name__}")

    if not data.get("has_violation"):
        return []

    violations = data.get("violations") or []
    if not isinstance(violations, list):
        raise OutputParseError(f"violations 不是数组: {type(violations).__name__}")

    return [_validate_violation(v) for v in violations]
