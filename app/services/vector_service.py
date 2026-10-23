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
from app.core.logging import get_logger

os.environ["CHROMA_HOME"] = str(settings.CHROMA_CACHE_DIR)

# HuggingFace 官方域名在部分网络环境下不可达（连接被重置），未显式配置时
# 回退到镜像站。用 setdefault：已设置 HF_ENDPOINT 的环境变量优先生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = get_logger("services.vector")

# 开箱试用用的示例条文，仅非生产环境插入
_DEMO_CLAUSES = [
    "JGJ 59-2011 第3.11.3条：基坑周边1.2米范围内严禁堆载，堆载高度不得超过1.5米。",
    "JGJ 59-2011 第3.11.5条：深度超过2米的基坑必须设置专用人员上下通道。",
]
_DEMO_METADATA = [
    {"spec_code": "JGJ 59-2011", "article_id": "3.11.3", "is_mandatory": True},
    {"spec_code": "JGJ 59-2011", "article_id": "3.11.5", "is_mandatory": True},
]
_DEMO_IDS = ["demo1", "demo2"]

# bge 系列中文模型官方推荐的检索指令前缀，只加在查询侧、不加在文档侧。
# 这是模型训练时的非对称用法，漏掉会明显拉低召回。
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def build_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """构造中文嵌入模型。

    此前用的是 ChromaDB 默认的 all-MiniLM-L6-v2 —— 英文模型，中文只能按
    子词匹配，而本库条文全部是中文，检索质量明显偏弱。
    """
    return SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        normalize_embeddings=True,
    )


class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_DATA_DIR))
        self.embedding_function = build_embedding_function()
        self.collection = self._open_collection()
        self._seed_demo_data()

    def _create_kwargs(self) -> dict[str, Any]:
        return {
            "name": settings.CHROMA_COLLECTION_NAME,
            "embedding_function": self.embedding_function,
            "metadata": {"hnsw:space": "cosine"},
        }

    def _open_collection(self):
        try:
            return self.client.get_or_create_collection(**self._create_kwargs())
        except Exception as e:
            # 换嵌入模型会改变向量维度，旧集合无法直接复用。这里不静默删库
            # （那会丢数据），而是把恢复方式直接告诉使用者。
            raise RuntimeError(
                f"打开向量集合失败：{e}\n"
                f"若刚更换过嵌入模型（当前为 {settings.EMBEDDING_MODEL}），"
                f"向量维度已变化，需要重建集合后重新灌库：\n"
                f"  python scripts/ingest_gb_corpus.py --reset"
            ) from e

    def reset(self) -> None:
        """删除并重建集合。换成不同维度的嵌入模型后必须执行。"""
        # 集合本就不存在时 delete_collection 会抛错，这是预期情况，忽略即可
        with contextlib.suppress(Exception):
            self.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(**self._create_kwargs())
        logger.info("向量集合已重建：%s", settings.CHROMA_COLLECTION_NAME)

    def _seed_demo_data(self):
        """插入开箱试用用的示例条文（仅非生产环境、且库为空时）。

        此前这里无条件执行，每次进程启动都会把这两条塞回知识库。它们会和
        真实规范一起参与检索，把审核依据带偏，还会经由数据生成流进训练集 ——
        因此在生产环境绝不插入。
        """
        if settings.is_production() or not settings.SEED_DEMO_DATA:
            logger.info("跳过示例条文插入（生产环境或已显式关闭）")
            return
        if self.collection.count() > 0:
            return

        self.collection.add(
            documents=_DEMO_CLAUSES,
            metadatas=_DEMO_METADATA,
            ids=_DEMO_IDS,
        )
        logger.info("向量服务初始化完成，已插入 %d 条示例条文", len(_DEMO_CLAUSES))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[_QUERY_INSTRUCTION + query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
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

    def add_clauses(self, clauses: list[dict]) -> int:
        """批量写入条文。

        逐条 add_clause 会让嵌入模型按单条推理，灌一本规范要几百次前向；
        批量编码快得多（CPU 上差别尤其明显）。
        """
        if not clauses:
            return 0

        documents = [c["content"] for c in clauses]
        metadatas = [c["metadata"] for c in clauses]
        ids = [c.get("id") or str(uuid.uuid4()) for c in clauses]

        # 分批，避免一次性把整个语料塞进模型导致内存峰值过高
        batch_size = 64
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self.collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        return len(documents)


# 单例
vector_service = VectorService()
