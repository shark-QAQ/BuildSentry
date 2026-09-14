# app/schemas/document.py
"""
文档相关请求和响应模型。
"""

from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    file_name: str
    file_size: int
    status: str
    word_count: int
    title: str | None = None
    project_name: str | None = None
    company: str | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True  # 允许从 ORM 模型自动转换
