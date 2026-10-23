# tests/test_training_dataset.py
"""
训练数据集工具测试。

最关键的一条：转换产出的 prompt 必须与推理侧 build_prompt() 逐字节一致。
这是"训完接上去不生效"的唯一防线。
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(TRAINING_DIR))

from generate_dataset import (  # noqa: E402
    build_labeled_record,
    find_documents,
    label_paragraph,
)
from prepare_dataset import build_target, to_sharegpt, validate_sharegpt  # noqa: E402

from app.services.audit_contract import build_prompt  # noqa: E402

CLAUSES = [
    {
        "content": "基坑周边1.2米范围内严禁堆载。",
        "metadata": {"spec_code": "JGJ 59-2011", "article_id": "3.11.3"},
    }
]

VIOLATION = {
    "clause_code": "JGJ 59-2011 3.11.3",
    "clause_content": "基坑周边1.2米范围内严禁堆载。",
    "risk_level": "high",
    "description": "堆载距离仅0.8米",
    "suggestion": "调整至1.2米以外",
    "confidence": 0.95,
}


def _record(paragraph="基坑深4.5米", clauses=None, violations=None):
    return {
        "paragraph": paragraph,
        "clauses": CLAUSES if clauses is None else clauses,
        "violations": [VIOLATION] if violations is None else violations,
    }


class TestToSharegpt:
    """标注数据 → 训练集转换"""

    def test_prompt_is_byte_identical_to_inference(self):
        """核心保证：数据集里的 prompt 必须与推理侧完全一致"""
        record = _record()
        converted = to_sharegpt(record)
        human = converted["conversations"][0]["value"]

        assert human == build_prompt(record["paragraph"], record["clauses"])

    def test_target_is_parseable_and_self_consistent(self):
        converted = to_sharegpt(_record())
        target = converted["conversations"][1]["value"]

        from app.services.audit_contract import parse_model_output

        assert parse_model_output(target)[0]["risk_level"] == "high"

    def test_empty_violations_means_no_violation(self):
        converted = to_sharegpt(_record(violations=[]))
        assert '"has_violation": false' in converted["conversations"][1]["value"]

    def test_roles_are_llamafactory_sharegpt(self):
        converted = to_sharegpt(_record())
        roles = [c["from"] for c in converted["conversations"]]
        assert roles == ["human", "gpt"]

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="缺少字段"):
            to_sharegpt({"paragraph": "x"})

    def test_pretty_flag_produces_indented_target(self):
        compact = to_sharegpt(_record())["conversations"][1]["value"]
        pretty = to_sharegpt(_record(), pretty=True)["conversations"][1]["value"]
        assert "\n" not in compact
        assert "\n" in pretty

    def test_build_target_reflects_violations(self):
        assert '"has_violation": true' in build_target([VIOLATION])
        assert '"has_violation": false' in build_target([])


class TestTrainingPythonCompatibility:
    """训练脚本及其依赖的 app 模块的 Python 版本兼容性

    服务侧跑在 Python 3.14，但训练环境通常是 3.11/3.12 ——
    LLaMA-Factory、torch、bitsandbytes、flash-attn 都还没跟上 3.14。
    而训练脚本要 import app 的模块，所以这些模块也必须能在 3.12 下解析。

    这里用 ast 的 feature_version 卡住语法，避免引入 3.14 才支持的写法
    （例如 PEP 758 的 `except A, B:`，不带括号，3.12 解析不了）。

    app/ 一并纳入检查：此前只扫 training/*.py，于是
    app/auth/dependencies.py 里的 `except TypeError, ValueError:` 长期没被发现 ——
    而它一旦被训练脚本间接 import，整个训练环境就起不来。
    """

    def _assert_parses_on_312(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path), feature_version=(3, 12))
        except SyntaxError as e:
            pytest.fail(
                f"{path.name} 无法在 Python 3.12 下解析: 第 {e.lineno} 行 {e.msg}\n"
                "训练环境用不了 3.14，请改用 3.12 也支持的写法。"
            )

    def test_training_scripts_parse_on_python312(self):
        scripts = sorted(TRAINING_DIR.glob("*.py"))
        assert scripts, "没有找到训练脚本"

        for script in scripts:
            self._assert_parses_on_312(script)

    def test_app_modules_parse_on_python312(self):
        modules = sorted(APP_DIR.rglob("*.py"))
        assert modules, "没有找到 app 模块"

        for module in modules:
            self._assert_parses_on_312(module)


class _StubTensor:
    """最小张量替身：支持 .shape / .to() / 切片（generate_text 会切掉输入部分）"""

    def __init__(self, length):
        self.shape = (1, length)
        self.length = length

    def to(self, device):
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = self.length if item.stop is None else item.stop
            return _StubTensor(max(0, stop - start))
        return _StubTensor(1)


class _StubTokenizer:
    chat_template = "qwen3 {% if enable_thinking %}{% endif %}"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None, **kw):
        return _StubTensor(50)

    def decode(self, ids, skip_special_tokens=True):
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return text


class _StubModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [_StubTensor(70)]


VALID = '{"has_violation": true, "violations": [{"risk_level": "high", "description": "d", "suggestion": "s"}]}'


class TestLabelParagraph:
    """教师标注核心逻辑"""

    def test_returns_parsed_violations(self):
        tokenizer = _StubTokenizer([VALID])
        result = label_paragraph(
            _StubModel(),
            tokenizer,
            "段落",
            [],
            thinking=False,
            max_new_tokens=64,
            max_retries=2,
        )
        assert len(result) == 1
        assert result[0]["risk_level"] == "high"

    def test_retries_then_gives_up_returning_none(self):
        """教师反复输出不合规内容时应返回 None，而不是把坏样本混进数据集"""
        tokenizer = _StubTokenizer(["我不会输出 JSON"] * 3)
        result = label_paragraph(
            _StubModel(),
            tokenizer,
            "段落",
            [],
            thinking=False,
            max_new_tokens=64,
            max_retries=2,
        )
        assert result is None
        assert tokenizer.calls == 3  # 首次 + 2 次重试

    def test_recovers_on_retry(self):
        """第一次失败、重试成功时应正常返回"""
        tokenizer = _StubTokenizer(["格式错了", VALID])
        result = label_paragraph(
            _StubModel(),
            tokenizer,
            "段落",
            [],
            thinking=False,
            max_new_tokens=64,
            max_retries=2,
        )
        assert result is not None
        assert tokenizer.calls == 2

    def test_thinking_output_is_stripped(self):
        """教师开启思考时，思考内容不应污染结果"""
        tokenizer = _StubTokenizer([f"分析中 {{花括号}}\n<｜end▁of▁thinking｜>{VALID}"])
        result = label_paragraph(
            _StubModel(),
            tokenizer,
            "段落",
            [],
            thinking=True,
            max_new_tokens=64,
            max_retries=0,
        )
        assert len(result) == 1


class TestFindDocuments:
    """文档发现"""

    def test_rejects_unsupported_file(self, tmp_path):
        bad = tmp_path / "a.txt"
        bad.write_text("x", encoding="utf-8")
        with pytest.raises(SystemExit, match="不支持的文件类型"):
            find_documents(bad)

    def test_rejects_missing_path(self, tmp_path):
        with pytest.raises(SystemExit, match="路径不存在"):
            find_documents(tmp_path / "nope")

    def test_finds_supported_files_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "sub" / "b.docx").write_bytes(b"PK")
        (tmp_path / "c.txt").write_text("x", encoding="utf-8")

        found = {p.name for p in find_documents(tmp_path)}
        assert found == {"a.pdf", "b.docx"}


class TestGenerateToConvertPipeline:
    """造数据脚本与转换脚本必须能串起来"""

    def test_labeled_record_converts_and_validates(self):
        """generate_dataset 产出的记录，应能被 prepare_dataset 直接转换并通过校验"""
        record = build_labeled_record("基坑深4.5米", CLAUSES, [VIOLATION])

        converted = to_sharegpt(record)
        assert validate_sharegpt(converted, 0) == []

    def test_records_without_violations_also_convert(self):
        record = build_labeled_record("基坑深4.5米", CLAUSES, [])
        converted = to_sharegpt(record)
        assert validate_sharegpt(converted, 0) == []

    def test_labeled_record_shape_matches_converter_expectations(self):
        """字段名必须是转换器要求的 paragraph / clauses / violations"""
        record = build_labeled_record("段落", CLAUSES, [VIOLATION])
        assert set(record) == {"paragraph", "clauses", "violations"}


class TestValidateSharegpt:
    """数据集校验"""

    def test_valid_record_passes(self):
        assert validate_sharegpt(to_sharegpt(_record()), 0) == []

    def test_detects_foreign_prompt(self):
        """prompt 不是本项目模板生成的，必须拦下"""
        record = {
            "conversations": [
                {"from": "human", "value": "随手写的提示词"},
                {"from": "gpt", "value": '{"has_violation": false, "violations": []}'},
            ]
        }
        problems = validate_sharegpt(record, 0)
        assert any("不是由本项目模板生成" in p for p in problems)

    def test_detects_invalid_risk_level(self):
        """非法 risk_level 会被推理侧聚合逻辑静默漏计"""
        record = to_sharegpt(_record())
        record["conversations"][1]["value"] = (
            '{"has_violation": true, "violations":'
            ' [{"risk_level": "critical", "description": "d", "suggestion": "s"}]}'
        )
        problems = validate_sharegpt(record, 0)
        assert any("非法 risk_level" in p for p in problems)

    def test_detects_has_violation_inconsistency(self):
        """标记为无违规、却带着违规项——这种矛盾会让模型学到混乱信号"""
        record = to_sharegpt(_record())
        record["conversations"][1]["value"] = (
            '{"has_violation": false, "violations":'
            ' [{"risk_level": "high", "description": "d", "suggestion": "s"}]}'
        )
        problems = validate_sharegpt(record, 0)
        assert any("has_violation 与 violations 不一致" in p for p in problems)

    def test_detects_malformed_json(self):
        record = to_sharegpt(_record())
        record["conversations"][1]["value"] = "模型没有输出 JSON"
        problems = validate_sharegpt(record, 0)
        assert any("不是合法 JSON" in p for p in problems)

    def test_detects_wrong_message_count(self):
        record = {"conversations": [{"from": "human", "value": "x"}]}
        assert validate_sharegpt(record, 0) != []

    def test_detects_wrong_roles(self):
        record = to_sharegpt(_record())
        record["conversations"][0]["from"] = "user"
        problems = validate_sharegpt(record, 0)
        assert any("from 必须是 human/gpt" in p for p in problems)
