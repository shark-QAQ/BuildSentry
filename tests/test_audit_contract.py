# tests/test_audit_contract.py
"""
审核契约测试。

prompt 模板与输出 schema 是推理与训练共用的唯一来源，
这里的断言同时约束着训练数据的构造方式。
"""

import json

import pytest

from app.services.audit_contract import (
    PROMPT_TEMPLATE,
    RISK_LEVELS,
    OutputParseError,
    build_prompt,
    encode_for_model,
    format_clauses,
    parse_model_output,
    split_paragraphs,
    strip_thinking,
    to_clause_context,
)

CLAUSES = [
    {
        "content": "基坑周边1.2米范围内严禁堆载。",
        "metadata": {"spec_code": "JGJ 59-2011", "article_id": "3.11.3"},
    }
]


class TestBuildPrompt:
    """prompt 构造测试"""

    def test_prompt_contains_clauses_and_paragraph(self):
        prompt = build_prompt("基坑深4.5米", CLAUSES)
        assert "JGJ 59-2011 3.11.3: 基坑周边1.2米范围内严禁堆载。" in prompt
        assert "基坑深4.5米" in prompt

    def test_prompt_lists_every_risk_level(self):
        """模板必须列出全部合法取值，否则模型不知道该输出什么"""
        prompt = build_prompt("x", CLAUSES)
        for level in RISK_LEVELS:
            assert level in prompt

    def test_template_renders_without_leftover_placeholders(self):
        """渲染后不应残留 {clauses_text} 之类的占位符"""
        prompt = build_prompt("段落", CLAUSES)
        assert "{clauses_text}" not in prompt
        assert "{paragraph_text}" not in prompt

    def test_format_clauses_handles_missing_metadata(self):
        assert format_clauses([{"content": "条文"}]) == "-  : 条文"

    def test_format_clauses_empty(self):
        assert format_clauses([]) == ""

    def test_template_is_valid_format_string(self):
        """模板必须是可 format 的，且只含两个具名占位符"""
        assert PROMPT_TEMPLATE.format(clauses_text="a", paragraph_text="b")


class _FakeTensor:
    def __init__(self, length):
        self.shape = (1, length)
        self.length = length


class _TokenizerWithChatTemplate:
    chat_template = "qwen"

    def __init__(self):
        self.template_calls = []
        self.raw_calls = []

    def apply_chat_template(
        self, messages, add_generation_prompt=True, return_tensors=None, **kwargs
    ):
        self.template_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "return_tensors": return_tensors,
                **kwargs,
            }
        )
        return _FakeTensor(7)

    def __call__(self, text, return_tensors=None):
        self.raw_calls.append(text)
        return {"input_ids": _FakeTensor(3)}


class _Qwen3Tokenizer(_TokenizerWithChatTemplate):
    """Qwen3 的模板里含 enable_thinking 开关"""

    chat_template = "{% if enable_thinking %}...{% endif %}<|im_start|>assistant"


class _BaseTokenizer:
    chat_template = None

    def __init__(self):
        self.raw_calls = []

    def __call__(self, text, return_tensors=None):
        self.raw_calls.append(text)
        return {"input_ids": _FakeTensor(3)}


class TestEncodeForModel:
    """输入编码测试——训练与推理必须套用同一个对话模板"""

    def test_uses_chat_template_when_available(self):
        """Instruct 模型必须走对话模板，否则指令遵循会明显变差"""
        tokenizer = _TokenizerWithChatTemplate()
        result = encode_for_model(tokenizer, "提示词")

        assert len(tokenizer.template_calls) == 1
        call = tokenizer.template_calls[0]
        assert call["messages"] == [{"role": "user", "content": "提示词"}]
        assert call["add_generation_prompt"] is True
        assert call["return_tensors"] == "pt"
        assert tokenizer.raw_calls == []  # 不应退回裸文本编码
        assert result["input_ids"].length == 7

    def test_disables_thinking_for_qwen3(self):
        """Qwen3 必须关闭思考模式

        思考内容里几乎必然出现花括号，会让 parse_model_output 的 find("{")
        命中思考文本而非最终 JSON，解析必然失败。
        """
        tokenizer = _Qwen3Tokenizer()
        encode_for_model(tokenizer, "提示词")

        assert tokenizer.template_calls[0]["enable_thinking"] is False

    def test_does_not_pass_thinking_kwarg_to_templates_without_it(self):
        """模板本身不支持 enable_thinking 时不应传该参数（Qwen2.5 等）"""
        tokenizer = _TokenizerWithChatTemplate()  # chat_template 里没有 enable_thinking
        encode_for_model(tokenizer, "提示词")

        assert "enable_thinking" not in tokenizer.template_calls[0]

    def test_falls_back_to_raw_when_no_chat_template(self):
        """基座模型没有对话模板时退回裸文本编码"""
        tokenizer = _BaseTokenizer()
        result = encode_for_model(tokenizer, "提示词")

        assert tokenizer.raw_calls == ["提示词"]
        assert result["input_ids"].length == 3


class TestSplitParagraphs:
    """分段测试——线上推理与训练数据生成共用"""

    def test_splits_on_blank_lines(self):
        text = "第一段" + "甲" * 60 + "\n\n" + "第二段" + "乙" * 60
        assert len(split_paragraphs(text)) == 2

    def test_drops_short_fragments(self):
        """只保留超过 min_length 的片段"""
        text = "短\n\n" + "甲" * 60 + "\n\n" + "也短"
        assert split_paragraphs(text) == ["甲" * 60]

    def test_long_text_without_blank_lines_stays_one_paragraph(self):
        """没有空行分隔时不拆分，整段保留（不截断）"""
        text = "甲" * 60 + "\n" + "乙" * 60
        assert split_paragraphs(text) == [text]

    def test_all_short_falls_back_to_truncated_text(self):
        """整篇都很短时保底返回前 2000 字"""
        assert split_paragraphs("短文本") == ["短文本"]

    def test_fallback_truncates_very_long_short_paragraph(self):
        """全部片段都不够长时，保底取前 2000 字"""
        text = ("甲" * 40 + "\n\n") * 200  # 每片 40 字，都不过线
        result = split_paragraphs(text)
        assert len(result) == 1
        assert len(result[0]) == 2000


class TestToClauseContext:
    """检索结果 → prompt 条文结构"""

    def test_keeps_content_and_metadata(self):
        items = [{"id": "1", "content": "条文", "metadata": {"spec_code": "JGJ"}, "score": 0.9}]
        assert to_clause_context(items) == [{"content": "条文", "metadata": {"spec_code": "JGJ"}}]

    def test_tolerates_missing_fields(self):
        assert to_clause_context([{}]) == [{"content": "", "metadata": {}}]


class TestStripThinking:
    """思考内容剥离"""

    def test_strips_thinking_prefix(self):
        raw = '思考中……出现了 { 花括号 }\n<｜end▁of▁thinking｜>{"has_violation": false}'
        assert strip_thinking(raw).startswith('{"has_violation"')

    def test_passthrough_without_thinking(self):
        raw = '{"has_violation": false}'
        assert strip_thinking(raw) == raw

    def test_only_strips_first_block(self):
        raw = "a<｜end▁of▁thinking｜>b<｜end▁of▁thinking｜>c"
        assert strip_thinking(raw) == "b<｜end▁of▁thinking｜>c"


class TestParseModelOutputWithThinking:
    """带思考内容时仍能正确解析"""

    def test_parses_json_after_thinking(self):
        """思考里的花括号不能干扰 JSON 定位"""
        raw = (
            "让我分析一下这段方案。\n"
            "规范条文提到 {堆载高度} 的限值……\n"
            "<｜end▁of▁thinking｜>\n"
            '{"has_violation": true, "violations": '
            '[{"risk_level": "high", "description": "d", "suggestion": "s"}]}'
        )
        result = parse_model_output(raw)
        assert len(result) == 1
        assert result[0]["risk_level"] == "high"

    def test_thinking_without_json_still_raises(self):
        with pytest.raises(OutputParseError):
            parse_model_output("想了半天{但没有给答案}")


class TestParseModelOutput:
    """输出解析与校验测试"""

    def _wrap(self, payload: dict) -> str:
        return "一些前置说明\n" + json.dumps(payload, ensure_ascii=False) + "\n后置说明"

    def test_parses_valid_violation(self):
        raw = self._wrap(
            {
                "has_violation": True,
                "violations": [
                    {
                        "clause_code": "JGJ 59-2011 3.11.3",
                        "clause_content": "基坑周边1.2米范围内严禁堆载。",
                        "risk_level": "high",
                        "description": "堆载距离仅0.8米",
                        "suggestion": "调整至1.2米以外",
                        "confidence": 0.95,
                    }
                ],
            }
        )
        result = parse_model_output(raw)
        assert len(result) == 1
        assert result[0]["risk_level"] == "high"

    def test_no_violation_returns_empty_list(self):
        assert parse_model_output(self._wrap({"has_violation": False, "violations": []})) == []

    def test_missing_json_raises(self):
        with pytest.raises(OutputParseError):
            parse_model_output("模型说了一大段话但没有 JSON")

    def test_malformed_json_raises(self):
        """解析失败必须抛异常，不能返回空列表冒充『未发现违规』"""
        with pytest.raises(OutputParseError):
            parse_model_output('{"has_violation": true, "violations": [')

    def test_invalid_risk_level_raises(self):
        """非法 risk_level 会被聚合逻辑静默漏计，必须拦下"""
        raw = self._wrap(
            {
                "has_violation": True,
                "violations": [{"risk_level": "critical", "description": "d", "suggestion": "s"}],
            }
        )
        with pytest.raises(OutputParseError, match="非法 risk_level"):
            parse_model_output(raw)

    def test_missing_required_field_raises(self):
        raw = self._wrap({"has_violation": True, "violations": [{"risk_level": "high"}]})
        with pytest.raises(OutputParseError, match="缺少必需字段"):
            parse_model_output(raw)

    def test_violations_not_a_list_raises(self):
        raw = self._wrap({"has_violation": True, "violations": {"a": 1}})
        with pytest.raises(OutputParseError, match="不是数组"):
            parse_model_output(raw)

    def test_takes_first_json_object_when_model_chatters(self):
        """模型在 JSON 前后多说几句时应仍能解析"""
        raw = self._wrap({"has_violation": False, "violations": []})
        assert parse_model_output(raw) == []
