#!/usr/bin/env python
# training/prepare_dataset.py
"""
训练数据准备与校验。

核心保证：prompt 一律由 app.services.audit_contract.build_prompt() 生成，
训练目标一律由 parse_model_output() 校验。两者与推理侧共用同一份实现，
因此"训练用的 prompt 和推理不一致"这类问题在构造上就不可能发生。

用法：
    # 把标注数据转成 LLaMA-Factory 数据集
    python training/prepare_dataset.py convert \\
        --input labeled.jsonl --output training/data/train.jsonl

    # 校验已有数据集（不重新生成）
    python training/prepare_dataset.py check --input training/data/train.jsonl

输入（标注）格式，每行一条：
    {
      "paragraph": "施工方案段落原文",
      "clauses": [{"content": "条文原文", "metadata": {"spec_code": "...", "article_id": "..."}}],
      "violations": [
        {"clause_code": "...", "clause_content": "...", "risk_level": "high",
         "description": "...", "suggestion": "...", "confidence": 0.95}
      ]
    }
violations 为空表示该段落无违规。

输出为 LLaMA-Factory 的 sharegpt 格式：
    {"conversations": [{"from": "human", "value": "<prompt>"}, {"from": "gpt", "value": "<JSON>"}]}
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.audit_contract import (  # noqa: E402
    RISK_LEVELS,
    OutputParseError,
    build_prompt,
    parse_model_output,
)

# prompt 中必须出现的固定片段，用于确认它确实由本项目的模板生成
PROMPT_MARKERS = ("你是一名资深建筑施工安全审核专家", "【规范条文】", "【施工方案段落】")

TARGET_JSON_HINT = "只输出JSON"


def build_target(violations: list[dict], pretty: bool = False) -> str:
    """构造训练目标（模型应答）"""
    payload = {"has_violation": bool(violations), "violations": violations}
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False)


def to_sharegpt(record: dict, pretty: bool = False) -> dict:
    """把一条标注记录转成 LLaMA-Factory 的 sharegpt 格式"""
    missing = [k for k in ("paragraph", "clauses", "violations") if k not in record]
    if missing:
        raise ValueError(f"记录缺少字段: {missing}")

    prompt = build_prompt(record["paragraph"], record["clauses"])
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": build_target(record["violations"], pretty)},
        ]
    }


def validate_sharegpt(record: dict, index: int) -> list[str]:
    """校验一条数据集记录，返回问题列表（空列表表示通过）"""
    problems: list[str] = []

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        return [f"#{index}: conversations 必须是长度为 2 的数组"]

    human, gpt = conversations
    if human.get("from") != "human" or gpt.get("from") != "gpt":
        return [f"#{index}: conversations 的 from 必须是 human/gpt"]

    prompt = human.get("value", "")
    for marker in PROMPT_MARKERS:
        if marker not in prompt:
            problems.append(f"#{index}: prompt 缺少标记 {marker!r}，可能不是由本项目模板生成")
    if TARGET_JSON_HINT not in prompt:
        problems.append(f"#{index}: prompt 缺少输出格式说明，模型无从知道该输出什么")

    target = gpt.get("value", "")

    # has_violation 与 violations 必须自洽，否则模型会学到矛盾信号。
    # 注意要直接比对原始数据：parse_model_output 在 has_violation 为 false 时
    # 会提前返回空列表，拿它的返回值比对会漏掉"标记为无违规、却带着违规项"这种矛盾。
    try:
        payload = json.loads(target[target.find("{") : target.rfind("}") + 1])
    except ValueError:  # json.JSONDecodeError 是其子类
        problems.append(f"#{index}: 训练目标不是合法 JSON")
        return problems

    if bool(payload.get("has_violation")) != bool(payload.get("violations")):
        problems.append(f"#{index}: has_violation 与 violations 不一致")

    try:
        parse_model_output(target)
    except OutputParseError as e:
        problems.append(f"#{index}: 训练目标无法通过推理侧解析器: {e}")

    return problems


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"找不到文件: {path}\n\n"
            "若还没有数据集，先用标注数据生成：\n"
            f"    python {Path(__file__).name} convert "
            f"--input <标注数据.jsonl> --output {path}\n\n"
            "标注数据的格式见 training/README.md。"
        )

    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno} JSON 解析失败: {e}") from e
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def report(records: list[dict], problems: list[str]) -> None:
    """打印分布统计——风险等级失衡是训练效果差的常见原因"""
    risk_counter: Counter[str] = Counter()
    with_violation = 0
    for record in records:
        for conv in record.get("conversations", []):
            if conv.get("from") != "gpt":
                continue
            value = conv.get("value")
            if not isinstance(value, str):
                continue
            try:
                payload = json.loads(value[value.find("{") : value.rfind("}") + 1])
            except ValueError:  # json.JSONDecodeError 是其子类
                continue
            violations = payload.get("violations") or []
            if violations:
                with_violation += 1
            for v in violations:
                risk_counter[v.get("risk_level", "<缺失>")] += 1

    total = len(records)
    print(f"\n样本总数: {total}")
    if total:
        print(f"  含违规: {with_violation} ({with_violation / total:.1%})")
        print(f"  无违规: {total - with_violation} ({(total - with_violation) / total:.1%})")
    print("  风险等级分布:", dict(risk_counter) or "（无）")

    if problems:
        print(f"\n发现 {len(problems)} 个问题:")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... 另有 {len(problems) - 20} 个")
    else:
        print("\n校验通过")

    # 数据质量提示
    if total == 0:
        print("\n提示: 数据集为空")
    elif with_violation == 0:
        print("\n警告: 没有一条样本包含违规项，模型将学不到如何识别违规")
    elif with_violation == total:
        print("\n警告: 所有样本都含违规，模型会倾向于无条件报违规")
    missing_levels = set(RISK_LEVELS) - set(risk_counter)
    if missing_levels and risk_counter:
        print(f"\n警告: 缺少这些风险等级的样本: {sorted(missing_levels)}")


def cmd_convert(args: argparse.Namespace) -> int:
    labeled = read_jsonl(Path(args.input))
    records = []
    for i, record in enumerate(labeled):
        try:
            records.append(to_sharegpt(record, pretty=args.pretty))
        except ValueError as e:
            raise SystemExit(f"第 {i + 1} 条记录无法转换: {e}") from e

    problems = [p for i, r in enumerate(records) for p in validate_sharegpt(r, i)]
    if problems:
        report(records, problems)
        print("\n转换中止：请先修正上述问题")
        return 1

    write_jsonl(Path(args.output), records)
    print(f"已写入 {args.output}")
    report(records, problems)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    records = read_jsonl(Path(args.input))
    problems = [p for i, r in enumerate(records) for p in validate_sharegpt(r, i)]
    report(records, problems)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="标注数据 → LLaMA-Factory 数据集")
    p_convert.add_argument("--input", required=True, help="标注数据 JSONL")
    p_convert.add_argument("--output", required=True, help="输出 JSONL")
    p_convert.add_argument(
        "--pretty", action="store_true", help="训练目标使用缩进格式（更贴近 prompt 中的示例排版）"
    )
    p_convert.set_defaults(func=cmd_convert)

    p_check = sub.add_parser("check", help="校验已有数据集")
    p_check.add_argument("--input", required=True, help="数据集 JSONL")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
