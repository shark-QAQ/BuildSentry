# app/api/kb_admin.py
"""
知识库管理员接口。
提供对知识库的增删改查操作，以及知识库的训练和推理功能。
"""

import re
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.vector_service import vector_service

router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base Admin"])


def extract_clauses(text: str):
    pattern = r"(\d+\.\d+(?:\.\d+)?)\s*[条款]?\s*(.*?)(?=\d+\.\d+(?:\.\d+)?\s*[条款]|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    clauses = []
    for article_id, content in matches:
        content = content.strip()
        if len(content) > 10:
            clauses.append(
                {
                    "article_id": article_id,
                    "content": content[:500],
                    "metadata": {"spec_code": "JGJ 59-2011", "article_id": article_id},
                }
            )
    return clauses


@router.post("/specs")
async def upload_spec(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"code": 40001, "message": "仅支持PDF"})

    content = await file.read()
    try:
        import pymupdf

        doc = pymupdf.open(stream=content, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF解析失败: {e!s}") from e

    clauses = extract_clauses(full_text)
    if not clauses:
        raise HTTPException(status_code=400, detail="未提取到任何条文")

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
            "total_extracted": len(clauses),
            "new_stored": count,
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }
