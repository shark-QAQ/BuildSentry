# app/api/documents.py
"""
文档接口模块。
包含上传、解析和提取文档文本的接口。
"""

import hashlib
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.db_models import User
from app.repositories.document_repo import DocumentRepository
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService
from app.utils.file_utils import get_file_extension, is_valid_document

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# 分块读取的块大小，避免把整个上传一次性读进内存
CHUNK_SIZE = 1024 * 1024


async def _read_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """分块读取上传内容，超过上限立即中止"""
    buffer = bytearray()
    while chunk := await file.read(CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": 41301,
                    "message": f"文件超过大小上限 {settings.MAX_UPLOAD_SIZE_MB}MB",
                },
            )
    return bytes(buffer)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = None,
    project_name: str | None = None,
    company: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. 校验格式（白名单以 file_utils 为唯一来源）
    filename = file.filename or ""
    if not is_valid_document(filename):
        raise HTTPException(status_code=400, detail={"code": 40001, "message": "不支持的格式"})
    ext = get_file_extension(filename)

    # 2. 读取字节（带大小上限）
    content = await _read_within_limit(file, settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    file_hash = hashlib.sha256(content).hexdigest()

    # 3. 调用业务层解析
    raw_text, word_count = DocumentService.parse_document(content, ext)

    # 4. 调用数据层存储
    repo = DocumentRepository(db)
    new_doc = repo.create(
        file_name=file.filename,
        file_hash=file_hash,
        file_size=len(content),
        raw_text=raw_text,
        word_count=word_count,
        title=title or file.filename,
        project_name=project_name,
        company=company,
        status="parsed",
    )

    # 5. 用 Pydantic Schema 返回
    return {
        "code": 200,
        "message": "上传成功",
        "data": DocumentResponse.model_validate(new_doc).model_dump(),
        "timestamp": datetime.now().isoformat() + "Z",
    }


@router.get("/{doc_id}")
async def get_document(
    doc_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    repo = DocumentRepository(db)
    doc = repo.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "文档不存在"})

    return {
        "code": 200,
        "message": "success",
        "data": DocumentResponse.model_validate(doc).model_dump(),
        "timestamp": datetime.now().isoformat() + "Z",
    }
