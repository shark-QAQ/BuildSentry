# app/services/vector_service.py
"""
向量服务模块。
包含向量数据库的交互逻辑。
"""

# === 关键：重定向 Chroma 缓存到项目 data 目录 ===
import contextlib
import os
import uuid
from typing import Any

from app.core.config import settings

os.environ["CHROMA_HOME"] = str(settings.CHROMA_CACHE_DIR)

import chromadb


class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_DATA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._seed_demo_data()

    def _seed_demo_data(self):
        # 清空演示数据，防止重复插入（不存在时忽略）
        with contextlib.suppress(Exception):
            self.collection.delete(ids=["demo1", "demo2"])

        self.collection.add(
            documents=[
                "JGJ 59-2011 第3.11.3条：基坑周边1.2米范围内严禁堆载，堆载高度不得超过1.5米。",
                "JGJ 59-2011 第3.11.5条：深度超过2米的基坑必须设置专用人员上下通道。",
            ],
            metadatas=[
                {"spec_code": "JGJ 59-2011", "article_id": "3.11.3", "is_mandatory": True},
                {"spec_code": "JGJ 59-2011", "article_id": "3.11.5", "is_mandatory": True},
            ],
            ids=["demo1", "demo2"],
        )
        print("[INFO] 向量服务初始化完成，已插入两条示例条文")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"]
        )
        items = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i],
                    }
                )
        return items

    def add_clause(self, content: str, metadata: dict) -> str:
        doc_id = str(uuid.uuid4())
        self.collection.add(documents=[content], metadatas=[metadata], ids=[doc_id])
        return doc_id


# 单例
vector_service = VectorService()
