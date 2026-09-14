#!/usr/bin/env python
# training/generate_dataset.py
"""
用教师模型为真实施工方案生成审核标注，产出训练数据。

这是「没有标注数据」时的起步路径：让大模型先产出候选标注，再由人工抽检修正。
**未经复核的教师输出不要直接用于训练** —— 那会把教师的错误固化进学生模型。

流水线与线上推理完全一致（共用 app.services.audit_contract）：
    文档 → 分段 → RAG 检索条文 → 教师审核 → 校验 → 标注记录
因此生成的标注与线上看到的输入分布一致。最后一步的校验用的是推理侧同一个
解析器，教师输出不合规的样本会被丢弃而不是混进训练集。

用法：
    # 单个文件
    python training/generate_dataset.py \\
        --documents /path/to/plan.pdf --output training/data/labeled.jsonl

    # 整个目录（递归找 .pdf / .docx）
    python training/generate_dataset.py \\
        --documents /path/to/plans/ --output training/data/labeled.jsonl

    # 指定教师模型与显卡
    python training/generate_dataset.py --documents ... --output ... \\
        --model Qwen/Qwen3-32B --gpu 3

生成后接：
    python training/prepare_dataset.py convert \\
        --input training/data/labeled.jsonl --output training/data/train.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.audit_contract import (  # noqa: E402
    OutputParseError,
    build_prompt,
    parse_model_output,
    split_paragraphs,
    to_clause_context,
)
from app.services.document_service import DocumentService  # noqa: E402
from app.services.generation import generate_text  # noqa: E402
from app.services.model_loader import select_gpu_index  # noqa: E402
from app.services.vector_service import vector_service  # noqa: E402

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def find_documents(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise SystemExit(f"不支持的文件类型: {path}（仅支持 {sorted(SUPPORTED_SUFFIXES)}）")
        return [path]
    if not path.exists():
        raise SystemExit(f"路径不存在: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)


def load_teacher(model_name: str, gpu_index: int):
    """独立加载教师模型。

    故意不复用 app.services.model_loader —— 那是线上服务模型的单例，
    教师模型是另一个（通常更大的）模型，两者不能混用。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"加载教师模型 {model_name} → GPU {gpu_index} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": gpu_index},
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def label_paragraph(
    model,
    tokenizer,
    paragraph: str,
    context_clauses: list[dict],
    *,
    thinking: bool,
    max_new_tokens: int,
    max_retries: int,
):
    """让教师审核一个段落，返回违规项列表；失败返回 None"""
    prompt = build_prompt(paragraph, context_clauses)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            text = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                # 重试时提高温度，避免每次都在同一处卡住
                temperature=0.1 if attempt == 0 else 0.6,
                enable_thinking=thinking,
            )
            return parse_model_output(text)
        except (OutputParseError, ValueError) as e:
            last_error = e

    print(f"    [跳过] 教师输出无法解析: {last_error}")
    return None


def build_labeled_record(
    paragraph: str, context_clauses: list[dict], violations: list[dict]
) -> dict:
    """构造标注记录（prepare_dataset.py convert 的输入格式）"""
    return {"paragraph": paragraph, "clauses": context_clauses, "violations": violations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--documents", required=True, help="施工方案文件或目录")
    parser.add_argument("--output", required=True, help="输出的标注 JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="教师模型")
    parser.add_argument("--gpu", default="auto", help='显卡选择："auto" 或具体卡号')
    parser.add_argument("--top-k", type=int, default=3, help="每段检索的条文数")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--limit-per-doc", type=int, default=0, help="每个文档最多处理多少段（0 表示不限）"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="关闭教师思考模式。默认开启，思考能提升标注质量，其内容会被解析器剥离",
    )
    args = parser.parse_args()

    documents = find_documents(Path(args.documents))
    if not documents:
        raise SystemExit(f"在 {args.documents} 下没有找到 {sorted(SUPPORTED_SUFFIXES)} 文件")
    print(f"待处理文档 {len(documents)} 个")

    # auto 时复用线上同一套显存排除策略（MODEL_GPU_EXCLUDE / MIN_FREE），
    # 避免抢到被禁用或放不下的卡；显式指定卡号时以用户为准。
    gpu_index = int(args.gpu) if args.gpu != "auto" else select_gpu_index()
    model, tokenizer = load_teacher(args.model, gpu_index)

    records: list[dict] = []
    failed = 0
    risk_counter: Counter[str] = Counter()

    for doc_path in documents:
        print(f"\n处理 {doc_path.name}")
        try:
            raw_text, _ = DocumentService.parse_document(
                doc_path.read_bytes(), doc_path.suffix.lower()
            )
        except Exception as e:
            print(f"  [跳过] 解析失败: {e}")
            continue

        paragraphs = split_paragraphs(raw_text)
        if args.limit_per_doc > 0:
            paragraphs = paragraphs[: args.limit_per_doc]
        print(f"  切分出 {len(paragraphs)} 段")

        for idx, paragraph in enumerate(paragraphs, 1):
            context_clauses = to_clause_context(vector_service.search(paragraph, top_k=args.top_k))
            violations = label_paragraph(
                model,
                tokenizer,
                paragraph,
                context_clauses,
                thinking=not args.no_thinking,
                max_new_tokens=args.max_new_tokens,
                max_retries=args.max_retries,
            )
            print(
                f"  [{idx}/{len(paragraphs)}] "
                + ("失败" if violations is None else f"{len(violations)} 条违规")
            )

            if violations is None:
                failed += 1
                continue

            for v in violations:
                risk_counter[v.get("risk_level", "<缺失>")] += 1
            records.append(build_labeled_record(paragraph, context_clauses, violations))

    if not records:
        print("\n没有生成任何标注记录，请检查文档与教师输出")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with_violation = sum(1 for r in records if r["violations"])
    total = len(records)
    print(f"\n已写入 {output_path}")
    print(f"  标注段落: {total}")
    print(f"  含违规:   {with_violation} ({with_violation / total:.1%})")
    print(f"  无违规:   {total - with_violation} ({(total - with_violation) / total:.1%})")
    print(f"  解析失败: {failed} 段（已丢弃，未写入）")
    print(f"  风险分布: {dict(risk_counter)}")
    print(
        "\n下一步：\n"
        f"  python training/prepare_dataset.py convert "
        f"--input {output_path} --output training/data/train.jsonl\n"
        "注意：教师输出未经复核，务必人工抽检后再用于训练。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
