# app/services/document_service.py
"""
文档服务模块。
包含文档解析、文本提取等业务逻辑。
"""

import io

from fastapi import HTTPException

# 导入文档解析器
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import docx
except ImportError:
    docx = None
try:
    import pymupdf
except ImportError:
    pymupdf = None


class DocumentService:
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        """
        从 PDF 文件中提取文本。
        """
        if pymupdf:
            # 必须显式关闭：pymupdf 的 Document 持有底层资源
            with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            return text
        elif pypdf:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        raise HTTPException(status_code=500, detail="PDF解析器未安装")

    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        """
        从 DOCX 文件中提取文本。
        """
        if docx:
            document = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in document.paragraphs if p.text)
        raise HTTPException(status_code=500, detail="DOCX解析器未安装")

    @staticmethod
    def parse_document(file_bytes: bytes, ext: str):
        """
        解析文档，提取文本并返回文本和文本长度。
        """
        if ext == ".pdf":
            raw_text = DocumentService.extract_text_from_pdf(file_bytes)
        else:
            raw_text = DocumentService.extract_text_from_docx(file_bytes)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail={"code": 40002, "message": "文档内容为空"})
        return raw_text, len(raw_text)
