# app/models/db_models.py
"""
数据库模型模块。
包含数据库表的 ORM 模型。
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.sql import func

from app.core.database import Base

# SQLite 只把 "INTEGER PRIMARY KEY" 视为 rowid 别名并自动赋值，
# BIGINT 主键不会自增，插入时会报 NOT NULL constraint failed。
# 因此 SQLite 下降级为 INTEGER，其他数据库（如 PostgreSQL）保持 BIGINT。
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    full_name = Column(String(128))
    role = Column(String(32), default="viewer")  # viewer, submitter, auditor, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))


class Document(Base):
    __tablename__ = "documents"
    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64))
    file_size = Column(BigInteger)
    raw_text = Column(Text)
    word_count = Column(Integer)
    title = Column(String(255))
    project_name = Column(String(255))
    company = Column(String(255))
    status = Column(String(32), default="parsed")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditTask(Base):
    __tablename__ = "audit_tasks"
    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    task_no = Column(String(32), unique=True, nullable=False)
    document_id = Column(BigIntPK, ForeignKey("documents.id"), nullable=False)
    status = Column(String(32), default="queued")
    progress = Column(Integer, default=0)
    current_step = Column(String(128))
    result_id = Column(BigIntPK)
    queued_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))


class AuditResult(Base):
    __tablename__ = "audit_results"
    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    task_id = Column(BigIntPK, ForeignKey("audit_tasks.id"), nullable=False)
    conclusion = Column(String(32), nullable=False)
    conclusion_detail = Column(Text)
    high_risk_count = Column(Integer, default=0)
    medium_risk_count = Column(Integer, default=0)
    low_risk_count = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    quality_score = Column(Float)
    model_version = Column(String(64))
    audit_duration_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Violation(Base):
    __tablename__ = "violations"
    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    task_id = Column(BigIntPK, ForeignKey("audit_tasks.id"), nullable=False)
    result_id = Column(BigIntPK, ForeignKey("audit_results.id"))
    clause_code = Column(String(64))
    clause_content = Column(Text)
    location_ref = Column(String(255))
    violation_text = Column(Text)
    risk_level = Column(String(16), nullable=False)
    violation_type = Column(String(64))
    description = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=False)
    confidence = Column(Float)
    human_review_status = Column(String(32), default="pending")
    # 人工复核留痕：谁在什么时候改成了什么、理由是什么
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    review_comment = Column(Text)
    reviewed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
