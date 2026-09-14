# app/services/audit_service.py
"""
审核服务模块。
包含审核任务的模拟逻辑。
"""

import random
import time
from datetime import datetime

from sqlalchemy.orm import Session

from app.repositories.audit_repo import (
    AuditResultRepository,
    AuditTaskRepository,
    ViolationRepository,
)
from app.repositories.document_repo import DocumentRepository
from app.services.audit_contract import split_paragraphs, to_clause_context
from app.services.inference_service import InferenceService
from app.services.vector_service import vector_service


class AuditService:
    @staticmethod
    def generate_task_no() -> str:
        return f"AT-{datetime.now().strftime('%Y%m%d')}-{random.randint(100, 999)}"

    @staticmethod
    async def run_simulation(task_id: int, db: Session):
        """真实的审核流程（使用推理服务）"""
        started_at = time.perf_counter()
        task_repo = AuditTaskRepository(db)
        result_repo = AuditResultRepository(db)
        violation_repo = ViolationRepository(db)

        # 1. 获取文档
        task = task_repo.get_by_id(task_id)
        doc_repo = DocumentRepository(db)
        doc = doc_repo.get_by_id(task.document_id)
        if not doc or not doc.raw_text:
            task_repo.update_status(task_id, "failed", progress=0, current_step="文档内容为空")
            return

        # 2. 更新状态：开始解析
        task_repo.update_status(task_id, "auditing", progress=10, current_step="正在分割段落...")

        # 3. 分段（与训练数据生成共用同一实现，避免线上/训练段落粒度不一致）
        paragraphs = split_paragraphs(doc.raw_text)

        total_paragraphs = len(paragraphs)
        all_violations = []
        sources = []  # 各段落的审核来源，用于判定 model_version

        # 4. 逐段审核（并行处理）
        task_repo.update_status(
            task_id, "auditing", progress=20, current_step=f"正在审核 1/{total_paragraphs} 段..."
        )

        for idx, para in enumerate(paragraphs):
            # 更新进度
            progress = 20 + int((idx + 1) / total_paragraphs * 60)
            task_repo.update_status(
                task_id,
                "auditing",
                progress=progress,
                current_step=f"正在审核 {idx + 1}/{total_paragraphs} 段...",
            )

            # RAG 检索：查找相关条文
            clause_context = to_clause_context(vector_service.search(para, top_k=3))

            # 调用推理服务
            violations, source = await InferenceService.audit_paragraph(
                paragraph_text=para,
                context_clauses=clause_context,
            )
            sources.append(source)

            # 添加位置信息
            for v in violations:
                v["task_id"] = task_id
                v["location_ref"] = f"第 {idx + 1} 段"
                v["violation_text"] = para[:200] + ("..." if len(para) > 200 else "")
                all_violations.append(v)

        # 5. 保存所有违规项
        violation_repo.create_batch(all_violations)

        # 6. 聚合结果
        task_repo.update_status(
            task_id, "auditing", progress=90, current_step="正在生成综合结论..."
        )
        aggregated = await InferenceService.aggregate_results(all_violations, sources)

        # 7. 保存审核结果
        result = result_repo.create(
            task_id=task_id,
            conclusion=aggregated["conclusion"],
            conclusion_detail=aggregated["conclusion_detail"],
            high_risk_count=aggregated["high_risk_count"],
            medium_risk_count=aggregated["medium_risk_count"],
            low_risk_count=aggregated["low_risk_count"],
            total_violations=aggregated["total_violations"],
            quality_score=aggregated["quality_score"],
            model_version=aggregated["model_version"],
            # 实际耗时，而非写死的常量
            audit_duration_ms=int((time.perf_counter() - started_at) * 1000),
        )

        # 8. 关联结果到任务
        task = task_repo.get_by_id(task_id)
        task.result_id = result.id
        db.commit()

        # 9. 更新任务状态为完成
        task_repo.update_status(task_id, "completed", progress=100, current_step="审核完成")
