# tests/test_documents.py
"""
文档模块测试。
"""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

from app.core.config import settings
from tests.conftest import get_auth_headers

CJK_FONT = "STSong-Light"
if CJK_FONT not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT))


def make_pdf_bytes(text: str = "基坑开挖专项施工方案，基坑深度4.5米。") -> bytes:
    """生成一个可被解析的中文 PDF"""
    buffer = BytesIO()
    style = ParagraphStyle("s", fontName=CJK_FONT, fontSize=11, leading=18)
    SimpleDocTemplate(buffer, pagesize=A4).build([Paragraph(text, style)])
    return buffer.getvalue()


@pytest.fixture
def auth_headers(client: TestClient, test_user_data):
    client.post("/api/v1/auth/register", json=test_user_data)
    return get_auth_headers(client, test_user_data["username"], test_user_data["password"])


class TestDocuments:
    """文档相关测试"""

    def test_upload_pdf_success(self, client: TestClient, auth_headers):
        """上传合法 PDF 应成功入库并完成文本提取"""
        files = {"file": ("plan.pdf", BytesIO(make_pdf_bytes()), "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["id"] is not None
        assert data["file_name"] == "plan.pdf"
        assert data["status"] == "parsed"
        assert data["word_count"] > 0

        # 提取到的正文应可读回
        detail = client.get(f"/api/v1/documents/{data['id']}", headers=auth_headers)
        assert detail.status_code == 200
        assert detail.json()["data"]["id"] == data["id"]

    def test_upload_unsupported_format(self, client: TestClient, auth_headers):
        """测试上传不支持的格式"""
        files = {"file": ("test.txt", BytesIO(b"content"), "text/plain")}
        response = client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
        assert response.status_code == 400

    def test_upload_legacy_doc_rejected(self, client: TestClient, auth_headers):
        """.doc 不受支持：解析走 python-docx，无法处理旧版二进制格式"""
        files = {"file": ("legacy.doc", BytesIO(b"content"), "application/msword")}
        response = client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
        assert response.status_code == 400

    def test_upload_requires_auth(self, client: TestClient):
        """未认证不得上传"""
        files = {"file": ("plan.pdf", BytesIO(make_pdf_bytes()), "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 401

    def test_upload_exceeding_size_limit(self, client: TestClient, auth_headers, monkeypatch):
        """超过大小上限应返回 413，且不得继续解析"""
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)
        oversized = b"x" * (1024 * 1024 + 1024)  # 略大于 1MB

        files = {"file": ("huge.pdf", BytesIO(oversized), "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files, headers=auth_headers)

        assert response.status_code == 413

    def test_get_document_not_found(self, client: TestClient, auth_headers):
        """测试获取不存在的文档"""
        response = client.get("/api/v1/documents/99999", headers=auth_headers)
        assert response.status_code == 404
