# app/api/audit.py
"""
审计路由模块。
包含审计相关的 API 路由。
"""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, require_role
from app.core.database import get_db
from app.models.db_models import User
from app.repositories.audit_repo import (
    AuditResultRepository,
    AuditTaskRepository,
    ViolationRepository,
)
from app.repositories.document_repo import DocumentRepository
from app.services.audit_service import AuditService
from app.services.report_service import ReportService

# ===== 关键：定义 router 对象 =====
router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])


@router.post("/tasks")
async def create_audit_task(
    doc_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 先确认文档存在，否则会留下指向不存在文档的孤儿任务
    doc_repo = DocumentRepository(db)
    if not doc_repo.get_by_id(doc_id):
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "文档不存在"})

    task_no = AuditService.generate_task_no()
    task_repo = AuditTaskRepository(db)
    task = task_repo.create(task_no=task_no, document_id=doc_id, status="queued")
    background_tasks.add_task(AuditService.run_simulation, task.id, db)
    return {
        "code": 200,
        "message": "审核任务已创建",
        "data": {
            "task_id": task.id,
            "task_no": task.task_no,
            "status": task.status,
            "queued_at": task.queued_at.isoformat(),
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    task_repo = AuditTaskRepository(db)
    task = task_repo.get_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "任务不存在"})
    return {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": task.id,
            "task_no": task.task_no,
            "status": task.status,
            "progress": task.progress,
            "current_step": task.current_step,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }


@router.get("/tasks/{task_id}/result")
async def get_audit_result(
    task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    task_repo = AuditTaskRepository(db)
    task = task_repo.get_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "任务不存在"})
    if task.status != "completed":
        raise HTTPException(status_code=400, detail={"code": 40004, "message": "任务尚未完成"})

    result_repo = AuditResultRepository(db)
    result = result_repo.get_by_task_id(task_id)
    if not result:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "结果不存在"})

    violation_repo = ViolationRepository(db)
    violations = violation_repo.get_by_task_id(task_id)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": task.id,
            "task_no": task.task_no,
            "conclusion": result.conclusion,
            "conclusion_detail": result.conclusion_detail,
            "statistics": {
                "high_risk_count": result.high_risk_count,
                "medium_risk_count": result.medium_risk_count,
                "low_risk_count": result.low_risk_count,
                "total_violations": result.total_violations,
            },
            "quality_score": result.quality_score,
            "model_version": result.model_version,
            "audit_duration_ms": result.audit_duration_ms,
            "violations": [
                {
                    "id": v.id,
                    "clause_code": v.clause_code,
                    "clause_content": v.clause_content,
                    "risk_level": v.risk_level,
                    "violation_type": v.violation_type,
                    "location_ref": v.location_ref,
                    "violation_text": v.violation_text,
                    "description": v.description,
                    "suggestion": v.suggestion,
                    "confidence": v.confidence,
                    "human_review_status": v.human_review_status,
                }
                for v in violations
            ],
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }


@router.get("/tasks/{task_id}/report")
async def download_report(
    task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """下载审核报告 PDF"""
    task_repo = AuditTaskRepository(db)
    task = task_repo.get_by_id(task_id)

    if not task:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "任务不存在"})

    if task.status != "completed":
        raise HTTPException(status_code=400, detail={"code": 40004, "message": "任务尚未完成"})

    result_repo = AuditResultRepository(db)
    result = result_repo.get_by_task_id(task_id)

    if not result:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "结果不存在"})

    violation_repo = ViolationRepository(db)
    violations = violation_repo.get_by_task_id(task_id)

    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_by_id(task.document_id)

    # 生成报告
    result_data = {
        "conclusion": result.conclusion,
        "conclusion_detail": result.conclusion_detail,
        "statistics": {
            "high_risk_count": result.high_risk_count,
            "medium_risk_count": result.medium_risk_count,
            "low_risk_count": result.low_risk_count,
            "total_violations": result.total_violations,
        },
        "quality_score": result.quality_score,
        "model_version": result.model_version,
        "audit_duration_ms": result.audit_duration_ms,
    }

    violations_data = [
        {
            "clause_code": v.clause_code,
            "risk_level": v.risk_level,
            "violation_type": v.violation_type,
            "location_ref": v.location_ref,
            "description": v.description,
            "suggestion": v.suggestion,
            "confidence": v.confidence,
        }
        for v in violations
    ]

    report_path = ReportService.generate_report(
        task_no=task.task_no,
        document_title=doc.title if doc else "未知文档",
        result=result_data,
        violations=violations_data,
    )

    return FileResponse(
        path=report_path, filename=f"审核报告_{task.task_no}.pdf", media_type="application/pdf"
    )


@router.put("/violations/{violation_id}/review")
async def review_violation(
    violation_id: int,
    review_status: str,
    review_comment: str | None = None,
    current_user: User = Depends(require_role(["auditor", "admin"])),
    db: Session = Depends(get_db),
):
    """
    人工复核违规项

    仅审核员与管理员可操作：复核会改变审核结论，viewer 不应有权修改。

    review_status 可选值:
    - confirmed: 确认违规
    - rejected: 驳回（非违规）
    - pending: 待复核
    """
    # 验证复核状态
    valid_statuses = ["confirmed", "rejected", "pending"]
    if review_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail={"code": 40007, "message": f"无效的复核状态，可选值: {valid_statuses}"},
        )

    violation_repo = ViolationRepository(db)
    violation = violation_repo.get_by_id(violation_id)

    if not violation:
        raise HTTPException(status_code=404, detail={"code": 40404, "message": "违规项不存在"})

    # 更新复核状态
    updated = violation_repo.update_review_status(
        violation_id=violation_id,
        status=review_status,
        reviewer_id=current_user.id,
        review_comment=review_comment,
    )

    return {
        "code": 200,
        "message": "复核状态更新成功",
        "data": {
            "violation_id": updated.id,
            "human_review_status": updated.human_review_status,
            "reviewed_by": current_user.username,
            "reviewed_by_id": updated.reviewed_by,
            "review_comment": updated.review_comment,
            "reviewed_at": updated.reviewed_at.isoformat() if updated.reviewed_at else None,
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }


@router.get("/tasks/{task_id}/violations")
async def get_task_violations(
    task_id: int,
    risk_level: str | None = None,
    review_status: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取任务的违规项列表（支持筛选）"""
    task_repo = AuditTaskRepository(db)
    task = task_repo.get_by_id(task_id)

    if not task:
        raise HTTPException(status_code=404, detail={"code": 40402, "message": "任务不存在"})

    violation_repo = ViolationRepository(db)
    violations = violation_repo.get_by_task_id(task_id)

    # 应用筛选条件
    if risk_level:
        violations = [v for v in violations if v.risk_level == risk_level]

    if review_status:
        violations = [v for v in violations if v.human_review_status == review_status]

    return {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": task_id,
            "task_no": task.task_no,
            "total_count": len(violations),
            "violations": [
                {
                    "id": v.id,
                    "clause_code": v.clause_code,
                    "clause_content": v.clause_content,
                    "risk_level": v.risk_level,
                    "violation_type": v.violation_type,
                    "location_ref": v.location_ref,
                    "description": v.description,
                    "suggestion": v.suggestion,
                    "confidence": v.confidence,
                    "human_review_status": v.human_review_status,
                    "reviewed_by": v.reviewed_by,
                    "review_comment": v.review_comment,
                    "reviewed_at": v.reviewed_at.isoformat() if v.reviewed_at else None,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
                for v in violations
            ],
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }
