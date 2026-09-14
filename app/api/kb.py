# app/api/kb.py
"""
知识库接口模块。
包含知识库相关的 API, 如查询、检索等。
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import get_current_user
from app.models.db_models import User
from app.services.vector_service import vector_service

router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base"])


@router.get("/clauses")
async def search_clauses(
    q: str = Query(..., description="查询关键词或自然语言"),
    mode: str = Query("vector", description="检索模式，目前仅支持 vector"),
    top_k: int = Query(5, ge=1, le=20, description="返回条数"),
    current_user: User = Depends(get_current_user),
):
    if mode != "vector":
        # 暂不支持 keyword 模式，返回提示
        raise HTTPException(
            status_code=400, detail={"code": 40004, "message": "仅支持 vector 模式"}
        )

    if not q.strip():
        raise HTTPException(status_code=400, detail={"code": 40004, "message": "查询词不能为空"})

    results = vector_service.search(q, top_k=top_k)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "query": q,
            "mode": mode,
            "total": len(results),
            "items": [
                {
                    "id": item["id"],
                    "spec_code": item["metadata"].get("spec_code"),
                    "article_id": item["metadata"].get("article_id"),
                    "content": item["content"],
                    "similarity_score": round(item["similarity_score"], 4),
                }
                for item in results
            ],
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }
