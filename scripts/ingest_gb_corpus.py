# scripts/ingest_gb_corpus.py
"""
把 data/GB 下的国家标准 PDF 批量灌入向量知识库。

用法：
    # 预览（默认，不写入任何数据）
    python scripts/ingest_gb_corpus.py

    # 换过嵌入模型后重建集合，再正式入库
    python scripts/ingest_gb_corpus.py --reset --write

为什么要单独有这么一个脚本：`POST /api/v1/kb/specs` 一次只吃一个文件、且要
管理员登录，而这里是一次几十本规范的离线灌库，用同一套 extract_clauses
逻辑直接写库更实际。

两件事必须由人来确认，所以默认只预览不写入：
  1. 规范编号取错会把条文挂到别的规范名下，并经由训练数据固化进模型；
  2. 规范 PDF 版式差异极大，条文切分质量要看过才算数。
"""

import argparse
import contextlib
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf  # noqa: E402

from app.api.kb_admin import canonical_code, extract_clauses, normalize_pdf_text  # noqa: E402

DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "GB"
# 封面损坏、自动取不到编号/全称时，由人工在这里补齐（见 --names）。
DEFAULT_NAMES_FILE = DEFAULT_CORPUS_DIR / "names_override.json"

# 可读性门槛：正文里汉字的占比。实测可正常抽取的规范都在 0.46 以上，
# 而字体编码损坏（方正书版 FzBookMaker 之类，没有可用的 ToUnicode 表）
# 的文件抽出来是一堆乱码，占比只有 0.02~0.08。取 0.15 能把两者干净分开。
MIN_CJK_RATIO = 0.15

_CJK_RE = re.compile(r"[一-鿿]")

# 封面/页眉上的套话，不是规范名称
_COVER_NOISE = re.compile(
    r"^(中华人民共和国|国家标准|国家标准化管理委员会|国家质量监督检验检疫总局|"
    r"国家市场监督管理总局|发布|实施|代替|前言|目\s*[次录]|ICS|CCS|"
    r"[\d\s.．—\-–/]*$)"
)

# 从正文里找规范编号。全角写法（"ＧＢ／Ｔ３４９３１—２０１７"）已被
# normalize_pdf_text 拉平，连接号可能是 — / – / -。
_CODE_IN_TEXT = re.compile(
    r"(GB|JGJ|JTG|SL|DL|CJJ|CECS)\s*/?\s*(T)?\s*(\d+(?:\.\d+)?)\s*[—\-–]\s*(\d{4})"
)

# 封面通常在最前面，扫这么多字符足够覆盖；扫太多会扫到前言里引用的别的规范
COVER_SCAN_CHARS = 2000
# 每本规范的「前言」都会引用《标准化工作导则》GB/T 1.1，不能当成本规范的编号
_DRAFTING_RULE = "GB/T1.1"

# 从文件名找规范编号，兼容两种命名：
#   GBT+19964-2024.pdf / GB+47739-2026.pdf      —— 编号在前
#   30153-2013-gbt-e-300.pdf                    —— 编号在前、gbt 在后
_CODE_IN_NAME = re.compile(r"([A-Za-z]{2,4})[\s+_-]*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d{4})", re.I)


def read_pdf_text(path: Path) -> str:
    doc = pymupdf.open(path)
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def cjk_ratio(text: str) -> float:
    return len(_CJK_RE.findall(text)) / len(text) if text else 0.0


def split_prefix(raw: str) -> tuple[str, bool]:
    """把 "GBT" / "GB/T" / "GB" 归一成 (前缀, 是否推荐性标准)。"""
    prefix = raw.upper().replace("/", "")
    if prefix.endswith("T"):
        return prefix[:-1], True
    return prefix, False


def format_code(prefix: str, is_recommended: bool, number: str, year: str) -> str:
    # 推荐性标准写作 GB/T；强制性标准只有 GB。JGJ、JTG 等先按原样保留。
    if prefix == "GB" and is_recommended:
        return f"GB/T {number}-{year}"
    return f"{prefix} {number}-{year}"


def code_from_text(text: str) -> str | None:
    """从 PDF 封面上取规范编号，用于和文件名互相印证。

    这里额外做一次 NFKC，把全角字母与全角斜杠也拉平——成书较早的规范封面
    写作 "ＧＢ／Ｔ３４９３１—２０１７"，不拉平就找不到编号，会被误判成
    「封面损坏」。NFKC 只用于这一步查找，入库正文仍走 normalize_pdf_text，
    中文标点不受影响。

    只扫封面范围，并且跳过 GB/T 1.1：每本规范的「前言」都写着
    "本标准按照GB/T 1.1—2009给出的规则起草"，那不是这本规范的编号。
    封面本身取不出文字时直接返回 None，以免把前言里引用的别的规范
    （或这条套话）当成自己的编号。
    """
    flattened = unicodedata.normalize("NFKC", normalize_pdf_text(text[:COVER_SCAN_CHARS]))
    for match in _CODE_IN_TEXT.finditer(flattened):
        prefix, t, number, year = match.groups()
        code = format_code(prefix.upper(), bool(t), number, year)
        if canonical_code(code) == _DRAFTING_RULE:
            continue
        return code
    return None


def code_from_filename(path: Path) -> str | None:
    """从文件名兜底推编号。"""
    match = _CODE_IN_NAME.search(path.stem)
    if not match:
        return None
    raw_prefix, number, year = match.groups()
    prefix, is_recommended = split_prefix(raw_prefix)
    return format_code(prefix, is_recommended, number, year)


def _is_title_continuation(line: str) -> bool:
    """判断某行是不是封面标题被排到下一行的后半截。"""
    if not line or line[0].isdigit():
        return False
    if _COVER_NOISE.match(line):
        return False
    return len(_CJK_RE.findall(line)) >= 3


def _pick_title(lines: list[str], indices) -> str | None:
    """在一串候选行里挑第一行像规范名称的。"""
    for i in indices:
        if not 0 <= i < len(lines):
            continue
        line = lines[i].strip()
        if len(_CJK_RE.findall(line)) < 4:
            continue
        if _COVER_NOISE.match(line):
            continue
        # 去掉夹在标题里的英文名（"光伏发电站 photovoltaic(PV)powerstation"）
        line = re.split(r"\s{2,}", line)[0].strip()
        if len(_CJK_RE.findall(line)) < 4:
            continue
        # 封面标题常被排成两行（"…火灾监测预警" / "系统通用技术要求"），
        # 紧跟的下一行若仍是中文就并回来。章节标题以数字开头，不会误并。
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _is_title_continuation(nxt):
            line += nxt
        return line[:120]
    return None


def title_from_text(text: str, code: str) -> str | None:
    """取规范中文全称。

    封面有两种版式，都要照顾到：
      · 老版式：中华人民共和国国家标准 / GB/T xxxxx—yyyy / 规范名称（编号在前）
      · 新版式：规范名称排在最上面，编号在下面（2024 年后的标准多为这种）
    所以先往编号后面找，找不到再往前找。

    找不到编号所在行就直接放弃。成书较早的规范封面用替换字体排版，整片
    封面（含中文标题）都取不出文字，硬凑只会把替换字体的英文乱码
    （"犜犲狊狋犻狀犵…"）当成规范名称入库——这种文件名交给 --names 人工补。
    """
    lines = [ln.strip() for ln in normalize_pdf_text(text).split("\n")]
    code_key = canonical_code(code)

    start = None
    for i, line in enumerate(lines):
        if canonical_code(line).startswith(code_key):
            start = i
            break
    if start is None:
        return None

    forward = range(start + 1, start + 11)
    backward = range(start - 1, max(-1, start - 9), -1)
    return _pick_title(lines, list(forward) + list(backward))


def resolve_identity(path: Path, text: str, override: dict) -> tuple[str, str, list[str]]:
    """确定 (spec_code, spec_name)，并返回需要人工确认的告警。

    override 是 names_override.json 里针对该文件的人工指定值，优先级最高。
    """
    warnings: list[str] = []
    text_code = code_from_text(text)
    name_code = code_from_filename(path)

    if text_code and name_code and canonical_code(text_code) != canonical_code(name_code):
        warnings.append(f"文件名编号 {name_code} 与封面编号 {text_code} 不一致，采用文件名编号")
    # 以文件名为主：本批语料里文件名都带规范编号，而封面在大半文件上取不出
    # 文字；封面编号只用来印证。真正防错的是两者不一致时的告警。
    code = override.get("code") or name_code or text_code
    if not code:
        warnings.append("无法确定规范编号，需人工指定")
        code = ""

    name = override.get("name") or (title_from_text(text, code) if code else None)
    if not name:
        warnings.append("无法从封面取到规范全称，需人工指定")
        name = ""

    return code, name, warnings


def collect(corpus_dir: Path, overrides: dict):
    """扫描目录，返回 (可入库, 跳过, 待人工命名) 三组结果。"""
    ingestable, skipped, needs_name = [], [], []

    for path in sorted(corpus_dir.glob("*.pdf")):
        text = read_pdf_text(path)
        ratio = cjk_ratio(text)

        if ratio < MIN_CJK_RATIO:
            skipped.append(
                {
                    "file": path.name,
                    "reason": f"文本层损坏（汉字占比 {ratio:.1%}），抽取结果是乱码",
                }
            )
            continue

        code, name, warnings = resolve_identity(path, text, overrides.get(path.name, {}))
        if not code or not name:
            reason = "；".join(warnings) or "缺少编号或名称"
            skipped.append({"file": path.name, "reason": reason})
            # 正文可读、只是封面取不到名称的，值得人工补一个名字救回来
            if code and not name:
                needs_name.append({"file": path.name, "code": code})
            continue

        clauses = extract_clauses(text, code, name, is_mandatory="/T" not in code)
        if not clauses:
            skipped.append({"file": path.name, "reason": "未切出任何条文"})
            continue

        ingestable.append(
            {
                "file": path.name,
                "code": code,
                "name": name,
                "clauses": clauses,
                "warnings": warnings,
            }
        )

    return ingestable, skipped, needs_name


def write_names_template(needs_name: list[dict], path: Path) -> None:
    """给「正文可读、封面取不到名称」的文件生成待填模板。

    规范全称按项目约定必须由上传方给出，不能猜，所以这里只列文件名和已经
    确定的编号，名称留空等人补。
    """
    if not needs_name:
        return
    payload = {item["file"]: {"code": item["code"], "name": ""} for item in needs_name}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            existing = {}
        # 保留已经填好的条目
        for key, value in payload.items():
            if existing.get(key, {}).get("name"):
                value["name"] = existing[key]["name"]
        payload = {**existing, **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"待人工命名：{len(needs_name)} 个文件已写入 {path}")
    for item in needs_name:
        print(f"  {item['file']:32} {item['code']}")
    print()


def print_report(ingestable, skipped, corpus_dir: Path) -> None:
    total = sum(len(item["clauses"]) for item in ingestable)

    print("=" * 78)
    print(f"语料目录：{corpus_dir}")
    print(f"可入库：{len(ingestable)} 份，共 {total} 条条文")
    print(f"跳过：  {len(skipped)} 份")
    print("=" * 78)
    print()

    for item in ingestable:
        print(f"[{item['code']}] {item['name']}")
        print(f"    {item['file']}  ->  {len(item['clauses'])} 条")
        for warning in item["warnings"]:
            print(f"    ! {warning}")
        for clause in item["clauses"][:2]:
            print(f"      · {clause['article_id']}  {clause['content'][:60]}")
        print()

    if skipped:
        print("-" * 78)
        print("跳过的文件：")
        for item in skipped:
            print(f"  {item['file']:32} {item['reason']}")
        print()


def write_to_kb(ingestable) -> int:
    from app.services.vector_service import vector_service

    written = 0
    for item in ingestable:
        code = item["code"]
        # 先删掉该规范已有条文，让脚本可以重复执行而不产生重复条目
        with contextlib.suppress(Exception):
            vector_service.collection.delete(where={"spec_code": code})

        written += vector_service.add_clauses(item["clauses"])
        print(f"  已写入 {code}：{len(item['clauses'])} 条")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="把 data/GB 下的规范 PDF 灌入向量知识库")
    parser.add_argument("--dir", type=Path, default=DEFAULT_CORPUS_DIR, help="语料目录")
    parser.add_argument("--write", action="store_true", help="正式写入（默认只预览）")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先删除并重建向量集合。换嵌入模型后必须执行（向量维度会变）",
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=DEFAULT_NAMES_FILE,
        help="人工指定规范编号/全称的 JSON（封面损坏时用）",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"目录不存在：{args.dir}")
        return 1

    if args.reset:
        from app.services.vector_service import vector_service

        vector_service.reset()

    overrides = {}
    if args.names.exists():
        try:
            overrides = json.loads(args.names.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"读取 {args.names} 失败：{exc}")
            return 1

    ingestable, skipped, needs_name = collect(args.dir, overrides)
    print_report(ingestable, skipped, args.dir)
    if needs_name:
        write_names_template(needs_name, args.names)

    if not args.write:
        print("预览模式，未写入向量库。确认无误后加 --write 正式入库。")
        return 0

    written = write_to_kb(ingestable)
    print()
    print(f"完成：{len(ingestable)} 份规范，{written} 条条文已写入知识库。")
    print(f"当前集合条目数：{_collection_count()}")
    return 0


def _collection_count() -> int:
    from app.services.vector_service import vector_service

    return vector_service.collection.count()


if __name__ == "__main__":
    raise SystemExit(main())
