# app/api/kb_admin.py
"""
知识库管理员接口。
提供对知识库的增删改查操作，以及知识库的训练和推理功能。
"""

import re
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth.dependencies import require_role
from app.models.db_models import User
from app.services.vector_service import vector_service

router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base Admin"])

# 规范条文的起始标记。数字后可以跟「条/款」标记、分隔符或空白，也可以直接
# 跟汉字——PDF 抽取经常把 "3.1" 与正文之间的空白吃掉（尤其全角版式）。
_ARTICLE_NUM = r"\d+(?:\.\d+){1,3}"
# 条文号之后的边界。除了条/款标记、分隔符、空白、直接跟汉字之外，还要接受
# 「本行到此结束」——把拆行的编号并回去之后，编号会独占一行（"4.1.1\n正文"）。
# 独占一行这一支要求编号里带小数点，否则整行的页码 "12" 也会被当成条文号，
# 而它会截断上一条条文的正文。
_ARTICLE_SEP = r"(?:[条款]|[:：、]|\s|(?=[一-鿿])|(?<=\.\d)$)"
_ARTICLE_LINE = re.compile(rf"^[ \t]*第?[ \t]*({_ARTICLE_NUM}){_ARTICLE_SEP}")

# 目录行：条文标题与页码之间由点线连接。点线有 "." / "…" 两种画法（实测
# 有的规范整本用省略号 U+2026 连排）；有的版式还把页码排在点线之前
# （"型号2………………"），所以只要出现 4 个以上连续点号就判定为目录行——
# 正文里不会出现这种写法。
_TOC_PATTERN = re.compile(r"[.．·…]{4,}")

# 表格行常以对勾/叉号打头（"√控制响应特性"），不是条文正文
_TABLE_MARK = re.compile(r"^[√×✓✗\-—–]+")

# 正文里出现的规范编号，前面可能紧跟着页码（页眉 "4GB/T46261—2025"）
_RUNNING_HEADER = re.compile(
    r"\d{0,3}\s*([A-Z]{2,4}\s*/?\s*T?\s*\d+(?:\.\d+)?\s*[—\-–]\s*\d{4})", re.IGNORECASE
)


def canonical_code(value: str) -> str:
    """归一化规范编号，便于比较。

    连接号在不同来源里是 — / – / -，空格也时有时无
    （"GB/T 19964-2024" / "GB/T19964—2024"），直接比字符串会全部判成不一致。
    """
    return re.sub(r"[—–\-]", "-", re.sub(r"\s+", "", value)).upper()


# 过短的片段大概率是页眉页脚或残句，不是条文正文
_MIN_CLAUSE_LEN = 5
# 条文正文至少要有的汉字数，用来滤掉表格行、页码行这类噪声
_MIN_CJK = 4

_CJK_RE = re.compile(r"[一-鿿]")
_CJK_CHARS = "".join(chr(c) for c in range(0x4E00, 0xA000))

# 句子结束标点。用于判断上一行是否已经把话说完。
_SENTENCE_END = "。；;：:！？!?.)）】》」』\"'"
# 句读停顿标点。行内出现这些说明该行还停在句子中间。
_CLAUSE_BREAK = "，,、；;"

# 全角数字 -> 半角。"３．１" 这类写法里的 \d 匹配不上，必须先拉平。
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_pdf_text(text: str) -> str:
    """把 PDF 抽取出的文本拉回可比对的形态。

    抽取结果与「干净文本」差得很远，实测到的三类版式问题都在这里处理：

    1. 全角数字与全角句点（"３．１"）—— 正则里的 \\d 与 \\. 都匹配不上；
    2. 表意空格 U+3000 夹在条文号与正文之间；
    3. 条文号被排版拆到两行（"3." 换行 "1"），正则要求编号在同一行内完整出现，
       于是整本规范一条都切不出来（实测 49 份里有 18 份属于这种）。

    这里刻意只动数字与条文号里的点号，不做整篇 NFKC：NFKC 会把全角逗号
    "，" 降级成半角 ","、全角括号变成半角括号，把整库中文标点带成英文排版，
    而入库条文是要给人看、也要进训练数据的。
    """
    text = text.translate(_FULLWIDTH_DIGITS)
    text = text.replace("．", ".").replace("　", " ")
    # 把 "3.\n1" 这类被拆开的条文号并回一行。三级编号会被拆成三段
    # （"4." / "1." / "1"），一次替换只能并回前两段，所以反复替换到稳定。
    for _ in range(4):
        merged = re.sub(r"(\d)\.[ \t]*\n[ \t]*(\d)", r"\1.\2", text)
        if merged == text:
            break
        text = merged
    return text


def _is_wrapped_continuation(prev_line: str, tail: str) -> bool:
    """行首条文号是否只是上一句被硬换行切断的产物。

    只有「数字后直接跟汉字」这一种写法有歧义：正文里 "…堆载高度不得超过
    \\n1.5米。当采用…" 换行后同样以 "1.5" 开头，长得和真实条文
    "3.1光伏发电站" 一模一样。

    区分点放在上一行：被切断的上一行还停在句子中间，必然含有读号
    （、，）而没有句末标点；而条文标题、术语条目的上一行是完整的，
    不含读号。
    """
    if not tail or tail[0] not in _CJK_CHARS:
        return False
    stripped = prev_line.strip()
    if not stripped:
        return False
    if stripped[-1] in _SENTENCE_END:
        return False
    return any(ch in stripped for ch in _CLAUSE_BREAK)


def _previous_line(lines: list[str], index: int) -> str:
    for i in range(index - 1, -1, -1):
        if lines[i].strip():
            return lines[i]
    return ""


def _strip_running_header(content: str, spec_code: str) -> str:
    """去掉串进正文里的页眉——页码 + 本规范编号（"…不应4GB/T46261—2025低于…"）。

    只删本文档自己的编号：正文里引用别的规范（"应符合GB/T4208—2017"）是有效
    内容。做法是先把正文里所有「编号样」的串找出来，归一化后与本规范编号比对，
    而不是拿编号去拼正则——PDF 里 "GB/T 44803—2024" 可能写成 "GB/T44803—2024"
    或 "GB/T 44803-2024"，按字面拼出来的模式匹配不上。
    """
    target = canonical_code(spec_code)
    if not target:
        return content

    def drop_if_own(match: re.Match) -> str:
        return "" if canonical_code(match.group(1)) == target else match.group(0)

    return _RUNNING_HEADER.sub(drop_if_own, content)


def _clean_content(raw: str, spec_code: str) -> str:
    # PDF 在句中硬换行，先并回一行
    content = re.sub(r"\s*\n\s*", "", raw).strip()
    content = _TABLE_MARK.sub("", content).strip()
    return _strip_running_header(content, spec_code).strip()


def _is_noise(content: str) -> bool:
    if len(content) < _MIN_CLAUSE_LEN:
        return True
    if _TOC_PATTERN.search(content):
        return True
    return len(_CJK_RE.findall(content)) < _MIN_CJK


def extract_clauses(
    text: str,
    spec_code: str,
    spec_name: str,
    is_mandatory: bool = False,
    category: str | None = None,
):
    """从规范全文中切出条文。

    spec_code / spec_name 必须由上传方给出。此前这里把 spec_code 写死成
    "JGJ 59-2011"，上传任何规范都会被挂到该编号下——模型随后引用出错误的
    规范编号，而这种错误还会经由训练数据固化进微调模型。

    切分按行首条文号进行，兼容 "3.11.3 正文"、"第3.11.5条 正文" 以及
    被排版拆行的 "3.\\n1 正文" 三种写法，并剔除目录页条目与表格行。
    规范 PDF 版式差异很大，正式入库前建议先 dry_run 预览。
    """
    normalized = normalize_pdf_text(text)
    lines = normalized.split("\n")

    # 先定位所有真实的条文起始行。被判定为「硬换行产物」的行不算起点，
    # 它会自然并入上一条条的正文里。
    starts: list[tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        match = _ARTICLE_LINE.match(line)
        if not match:
            continue
        if _is_wrapped_continuation(_previous_line(lines, i), line[match.end() :]):
            continue
        starts.append((i, match.group(1), match.end()))

    clauses = []
    for idx, (line_no, article_id, tail_start) in enumerate(starts):
        end_line = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body = "\n".join([lines[line_no][tail_start:], *lines[line_no + 1 : end_line]])
        content = _clean_content(body, spec_code)
        if _is_noise(content):
            continue

        metadata = {
            "spec_code": spec_code,
            "spec_name": spec_name,
            "article_id": article_id,
            "is_mandatory": is_mandatory,
        }
        if category:
            metadata["category"] = category
        clauses.append(
            {
                "article_id": article_id,
                "content": content[:500],
                "metadata": metadata,
            }
        )
    return clauses


@router.post("/specs")
async def upload_spec(
    file: UploadFile = File(...),
    spec_code: str = Form(..., description='规范编号，如 "JGJ 59-2011"'),
    spec_name: str = Form(..., description="规范全称"),
    is_mandatory: bool = Form(False, description="是否强制性标准"),
    category: str | None = Form(None, description="分类：房建/市政/公路/水利"),
    dry_run: bool = Form(False, description="仅预览提取结果，不写入向量库"),
    current_user: User = Depends(require_role(["admin"])),
):
    # 入库的知识库会直接参与 RAG 检索、并成为训练数据的条款来源，
    # 因此写入口必须限管理员（此前该路由没有任何鉴权）。
    # 规范编号必须由上传方明确给出，不能替它猜：猜错会让条文挂到别的规范名下，
    # 且会经由训练数据固化进模型。空值直接拒绝，不做静默兜底。
    spec_code = spec_code.strip()
    spec_name = spec_name.strip()
    if not spec_code or not spec_name:
        raise HTTPException(
            status_code=400,
            detail={"code": 40002, "message": "spec_code 与 spec_name 均不能为空"},
        )

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"code": 40001, "message": "仅支持PDF"})

    pdf_bytes = await file.read()
    try:
        import pymupdf

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF解析失败: {e!s}") from e

    clauses = extract_clauses(full_text, spec_code, spec_name, is_mandatory, category)
    if not clauses:
        raise HTTPException(status_code=400, detail="未提取到任何条文")

    if dry_run:
        # 规范版式差异很大，入库前先让人看一眼提取质量
        return {
            "code": 200,
            "message": "预览模式，未写入向量库",
            "data": {
                "spec_code": spec_code,
                "spec_name": spec_name,
                "total_extracted": len(clauses),
                "preview": [
                    {"article_id": c["article_id"], "content": c["content"]} for c in clauses[:20]
                ],
            },
            "timestamp": datetime.now().isoformat() + "Z",
        }

    count = 0
    for clause in clauses:
        # 简单去重（基于内容前50字）
        existing = vector_service.search(clause["content"][:50], top_k=1)
        if not existing or existing[0]["similarity_score"] < 0.95:
            vector_service.add_clause(clause["content"], clause["metadata"])
            count += 1

    return {
        "code": 200,
        "message": "规范上传成功",
        "data": {
            "spec_code": spec_code,
            "spec_name": spec_name,
            "total_extracted": len(clauses),
            "new_stored": count,
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }
