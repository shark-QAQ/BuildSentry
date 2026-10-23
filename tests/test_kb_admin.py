# tests/test_kb_admin.py
"""
规范条文提取测试。

这里守着两个此前都坏掉的行为：
1. 规范编号必须来自上传方 —— 曾经写死成 "JGJ 59-2011"，上传任何规范都被
   挂到该编号下，模型随后引用出错误的规范编号；
2. 必须真能把一本规范切成多条 —— 旧正则的前瞻是
   `(?=\\d+\\.\\d+\\s*[条款]|$)`，要求数字后紧跟「条」或「款」，而最常见的
   真实写法是 "3.11.5 深度超过…"（无「条」字），于是只剩 `$` 分支成立，
   整本规范被截成 1 条。
"""

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

from app.api.kb_admin import extract_clauses
from tests.conftest import get_auth_headers

CJK_FONT = "STSong-Light"
if CJK_FONT not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT))


def make_pdf_bytes(lines: list[str]) -> bytes:
    """生成一个可被解析的中文 PDF，每个元素独占一行"""
    buffer = BytesIO()
    style = ParagraphStyle("s", fontName=CJK_FONT, fontSize=11, leading=18)
    SimpleDocTemplate(buffer, pagesize=A4).build([Paragraph(line, style) for line in lines])
    return buffer.getvalue()


SPEC_CODE = "JGJ 59-2011"
SPEC_NAME = "建筑施工安全检查标准"

# 接口测试用的 PDF 正文，每个元素独占一行
PDF_LINES = [
    "3.11.3 基坑周边1.2米范围内严禁堆载，堆载高度不得超过1.5米。",
    "3.11.5 深度超过2米的基坑必须设置专用人员上下通道，宽度不小于1米。",
]

# 最常见的真实写法：条文号后直接跟正文，不带「条」字。
# 旧正则正是被这种写法击穿的——它只在下面 FULL_TEXT 那种带「条」字的
# 写法下才碰巧能切开，所以切分回归测试必须用这个样例，否则测不出问题。
PLAIN_TEXT = """3.11.3 基坑周边1.2米范围内严禁堆载，堆载高度不得超过1.5米。
3.11.5 深度超过2米的基坑必须设置专用人员上下通道，通道宽度不小于1米。
"""

# 贴近 PDF 抽取的真实版式：目录页、句中硬换行、「第X条」写法
FULL_TEXT = """建筑施工安全检查标准

目 录
3.11.3 基坑工程 ....................... 12
3.11.5 基坑工程 ....................... 13

3.11.3 基坑周边1.2米范围内严禁堆载，堆载高度不得超过
1.5米。当采用机械开挖时，应保证机械与边坡的安全距离。
第3.11.5条 深度超过2米的基坑必须设置专用人员上下通道，
通道宽度不小于1米。
"""


class TestExtractClauses:
    def test_spec_code_comes_from_caller(self):
        """规范编号必须原样采用上传方给的值，不能写死"""
        clauses = extract_clauses(PLAIN_TEXT, "JGJ 130-2011", "脚手架安全技术规范")

        assert clauses
        assert {c["metadata"]["spec_code"] for c in clauses} == {"JGJ 130-2011"}

    def test_plain_form_splits_into_multiple_clauses(self):
        """不带「条」字的常见写法也必须能切开

        回归测试：旧前瞻要求数字后紧跟「条/款」，本样例不含该标记，
        整段文字会被吞成 1 条（内容是全文截断到 500 字）。
        """
        clauses = extract_clauses(PLAIN_TEXT, SPEC_CODE, SPEC_NAME)

        assert [c["article_id"] for c in clauses] == ["3.11.3", "3.11.5"]

    def test_di_tiao_form_is_also_supported(self):
        """「第3.11.5条 正文」这种写法同样要能识别，且编号不带「第」「条」"""
        clauses = extract_clauses(FULL_TEXT, SPEC_CODE, SPEC_NAME)

        assert "3.11.5" in {c["article_id"] for c in clauses}

    def test_toc_entries_are_dropped(self):
        """目录页的点线条目不是正文，不能入库"""
        clauses = extract_clauses(FULL_TEXT, SPEC_CODE, SPEC_NAME)

        assert not any("...." in c["content"] for c in clauses)

    def test_hard_wrap_does_not_split_a_clause(self):
        """句中硬换行后的数字不能被当成新条文号

        "…堆载高度不得超过\\n1.5米。" 换行后以 "1.5" 开头，若只按数字切分
        会被误判成条文起点。
        """
        clauses = extract_clauses(FULL_TEXT, SPEC_CODE, SPEC_NAME)

        assert "1.5" not in {c["article_id"] for c in clauses}
        # 被换行切断的两半应重新拼成完整一句
        assert "堆载高度不得超过1.5米" in clauses[0]["content"]

    def test_metadata_carries_spec_name_and_flags(self):
        clauses = extract_clauses(PLAIN_TEXT, SPEC_CODE, SPEC_NAME, True, "房建")

        meta = clauses[0]["metadata"]
        assert meta["spec_name"] == SPEC_NAME
        assert meta["is_mandatory"] is True
        assert meta["category"] == "房建"

    def test_category_omitted_when_not_given(self):
        """没传分类时不应写入空值，否则检索结果里多出无意义字段"""
        clauses = extract_clauses(PLAIN_TEXT, SPEC_CODE, SPEC_NAME)

        assert "category" not in clauses[0]["metadata"]

    def test_empty_text_yields_nothing(self):
        """没有条文时应返回空列表，交由调用方报错，而不是编一条出来"""
        assert extract_clauses("", SPEC_CODE, SPEC_NAME) == []


class TestRealPdfLayouts:
    """实测 49 份国标 PDF 时踩到的版式问题。

    在真实语料上跑之前，这里只覆盖了「条文号是半角、且独占行首」这一种
    理想写法。实测 49 份里只有 7 份长这样，其余 42 份的条文一条都切不出来。
    """

    def test_number_split_across_lines(self):
        """条文号被排版拆到多行：三级编号会拆成三段

        PDF 里 "4.1.1" 实际抽成 "4." / "1." / "1" 三行，且正则要求编号在
        同一行内完整出现。只并一次只能拼回 "4.1"，三级编号仍然丢失，
        所以必须反复并到稳定。
        """
        text = "4.\n1.\n1\n光伏发电站应配置有功功率控制系统。\n"

        clauses = extract_clauses(text, SPEC_CODE, SPEC_NAME)

        assert [c["article_id"] for c in clauses] == ["4.1.1"]

    def test_fullwidth_digits_and_period(self):
        """全角数字与全角句点："３．１" 里的 ３ 和 ．都匹配不上 \\d \\."""
        text = "３．１ 光伏发电站应配置有功功率控制系统。\n"

        clauses = extract_clauses(text, SPEC_CODE, SPEC_NAME)

        assert [c["article_id"] for c in clauses] == ["3.1"]

    def test_number_glued_to_cjk(self):
        """全角版式常把条文号与正文之间的空白吃掉："3.1光伏发电站"

        数字后直接跟汉字，没有分隔符。上一行是完整句子（以。收尾），
        所以这是真条文，不是硬换行的产物。
        """
        text = (
            "下列术语和定义适用于本文件。\n"
            "3.1光伏发电站 利用光伏电池的光生伏特效应，将太阳辐射能直接转换成电能的发电系统。\n"
            "3.2并网点 对于有升压站的光伏发电站，指升压站高压侧母线或节点。\n"
        )

        clauses = extract_clauses(text, SPEC_CODE, SPEC_NAME)

        assert [c["article_id"] for c in clauses] == ["3.1", "3.2"]

    def test_cjk_punctuation_is_preserved(self):
        """入库正文必须保留中文标点

        NFKC 会把全角逗号 "，" 降级成半角 ","、全角括号变成半角括号，
        一旦用它做整篇归一化，整库中文标点都会被带成英文排版。
        """
        text = "3.1 基坑周边（含栈桥）严禁堆载，堆载高度不得超过1.5米。\n"

        clauses = extract_clauses(text, SPEC_CODE, SPEC_NAME)

        assert "，" in clauses[0]["content"]
        assert "（含栈桥）" in clauses[0]["content"]

    def test_ellipsis_toc_lines_are_dropped(self):
        """目录点线用省略号 U+2026 连排时也要能识别

        有的规范整本用 "…" 画点线，且把页码排在点线之前，只看 "末尾是数字"
        的旧写法漏掉了整整 300 多条目录行，全部进了库。
        """
        text = (
            "3.2 控制装置分类2…………………………………………………………………………\n"
            "3.3 监测报警装置分类2……………………………………………………………………\n"
            "4.1 系统要求\n"
            "控制装置应符合本章要求。\n"
        )

        clauses = extract_clauses(text, SPEC_CODE, SPEC_NAME)

        assert "…" not in {c["article_id"] for c in clauses}
        assert all("…" not in c["content"] for c in clauses)
        assert [c["article_id"] for c in clauses] == ["4.1"]

    def test_cross_page_running_header_is_stripped(self):
        """页眉会被并进正文："…不应4GB/T46261—2025低于…"

        但正文里引用别的规范（"应符合GB/T4208—2017"）是有效内容，
        只删本文档自己的编号。
        """
        text = (
            "4.1 外壳防护等级不应低于GB/T4208—2017中IP54的要求，\n"
            "5GB/T46261—2025\n"
            "并应满足下列要求。\n"
        )

        clauses = extract_clauses(text, "GB/T 46261-2025", "可燃气体报警控制器")

        content = clauses[0]["content"]
        assert "GB/T46261—2025" not in content
        assert "GB/T4208—2017" in content

    def test_running_header_spacing_variant_is_stripped(self):
        """PDF 里的编号写法与调用方给的不一致时也要能删掉

        "GB/T 44803—2024"（带空格）与调用方给的 "GB/T 44803-2024" 逐字比较
        并不相等，按字面拼正则匹配不上，页眉就会留在条文里。
        """
        text = (
            "8.2.2 电站应采取措施防止应急物资的损坏，并有防火、防盗、\n"
            "防4GB/T 44803—2024\n"
            "潮、防污染等措施。\n"
        )

        clauses = extract_clauses(text, "GB/T 44803-2024", "电化学储能电站应急物资技术导则")

        content = clauses[0]["content"]
        assert "GB/T 44803—2024" not in content
        assert "防火、防盗、防潮、防污染" in content

    def test_table_rows_are_not_clauses(self):
        """表格里的对勾行不是条文正文"""
        text = "3.1 检测项目\n5.2.1 √控制响应特性\n5.2.2 √动态响应特性\n"

        clauses = extract_clauses(text, SPEC_CODE, SPEC_NAME)

        assert all(not c["content"].startswith("√") for c in clauses)


class TestUploadSpecEndpoint:
    """接口层：鉴权与表单参数接线

    一律用 dry_run=true —— 正式入库会写进真实的 data/chroma_data，
    测试不该污染开发机上的知识库。
    """

    def _files(self):
        return {"file": ("spec.pdf", BytesIO(make_pdf_bytes(PDF_LINES)), "application/pdf")}

    def test_dry_run_previews_extraction(self, client, admin_headers):
        response = client.post(
            "/api/v1/kb/specs",
            files=self._files(),
            data={
                "spec_code": "JGJ 130-2011",
                "spec_name": "建筑施工扣件式钢管脚手架安全技术规范",
                "dry_run": "true",
            },
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["spec_code"] == "JGJ 130-2011"
        assert data["total_extracted"] == 2
        # 预览里必须带上真实的规范编号，而不是某个写死的值
        assert {item["article_id"] for item in data["preview"]} == {"3.11.3", "3.11.5"}

    def test_spec_code_is_required(self, client, admin_headers):
        """缺少 spec_code 时必须直接报错，不能退回某个默认编号"""
        response = client.post(
            "/api/v1/kb/specs",
            files=self._files(),
            data={"spec_name": SPEC_NAME},
            headers=admin_headers,
        )

        assert response.status_code == 422

    def test_requires_authentication(self, client):
        """知识库写入口必须登录后才能访问

        回归测试：该路由此前没有任何鉴权，任何人都能往知识库灌数据，
        而这些数据会直接参与 RAG 检索、并成为训练数据的条款来源。
        """
        response = client.post(
            "/api/v1/kb/specs",
            files=self._files(),
            data={"spec_code": SPEC_CODE, "spec_name": SPEC_NAME, "dry_run": "true"},
        )

        assert response.status_code == 401

    def test_requires_admin_role(self, client, test_user_data):
        """普通用户不得写入知识库"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.post(
            "/api/v1/kb/specs",
            files=self._files(),
            data={"spec_code": SPEC_CODE, "spec_name": SPEC_NAME, "dry_run": "true"},
            headers=headers,
        )

        assert response.status_code == 403


class TestDemoSeedGating:
    """示例条文的插入门控

    示例条文会与真实规范一起参与检索、带偏审核依据，并经由数据生成流进
    训练集。此前它在 __init__ 里无条件执行，每次进程启动都会塞回去。
    """

    def _build(self, monkeypatch, tmp_path, *, environment, seed_flag):
        from app.core.config import settings
        from app.services.vector_service import VectorService

        monkeypatch.setattr(settings, "CHROMA_DATA_DIR", tmp_path)
        monkeypatch.setattr(settings, "ENVIRONMENT", environment)
        monkeypatch.setattr(settings, "SEED_DEMO_DATA", seed_flag)
        return VectorService()

    def test_production_never_seeds(self, monkeypatch, tmp_path):
        """生产环境绝不插入示例条文"""
        service = self._build(monkeypatch, tmp_path, environment="production", seed_flag=True)

        assert service.collection.count() == 0

    def test_development_seeds_on_empty_store(self, monkeypatch, tmp_path):
        """开发环境且库为空时插入，便于开箱试用"""
        service = self._build(monkeypatch, tmp_path, environment="development", seed_flag=True)

        assert service.collection.count() == 2

    def test_disabled_by_flag(self, monkeypatch, tmp_path):
        """显式关闭后不插入，即使处于开发环境"""
        service = self._build(monkeypatch, tmp_path, environment="development", seed_flag=False)

        assert service.collection.count() == 0

    def test_existing_clauses_are_not_duplicated(self, monkeypatch, tmp_path):
        """库里已有真实条文时不得再插入示例，避免与真实规范混在一起"""
        first = self._build(monkeypatch, tmp_path, environment="development", seed_flag=True)
        first.collection.delete(ids=["demo1", "demo2"])
        first.add_clause("真实规范条文内容，用于占位。", {"spec_code": "JGJ 130-2011"})

        from app.core.config import settings
        from app.services.vector_service import VectorService

        monkeypatch.setattr(settings, "SEED_DEMO_DATA", True)
        second = VectorService()

        assert second.collection.count() == 1
