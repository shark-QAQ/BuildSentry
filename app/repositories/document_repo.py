# app/repositories/document_repo.py
"""
文档仓库模块。
包含文档相关的数据库操作。
"""

from sqlalchemy.orm import Session

from app.models.db_models import Document


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Document:
        doc = Document(**kwargs)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_by_id(self, doc_id: int) -> Document | None:
        return self.db.query(Document).filter(Document.id == doc_id).first()
