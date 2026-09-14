# app/repositories/audit_repo.py
"""
审计仓库模块。
包含审计记录的数据库操作。
"""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models.db_models import AuditResult, AuditTask, Violation


class AuditTaskRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> AuditTask:
        task = AuditTask(**kwargs)
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get_by_id(self, task_id: int):
        return self.db.query(AuditTask).filter(AuditTask.id == task_id).first()

    def update_status(
        self,
        task_id: int,
        status: str,
        progress: int | None = None,
        current_step: str | None = None,
    ):
        task = self.get_by_id(task_id)
        if task:
            task.status = status
            if progress is not None:
                task.progress = progress
            if current_step is not None:
                task.current_step = current_step
            self.db.commit()
            self.db.refresh(task)
        return task


class AuditResultRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> AuditResult:
        result = AuditResult(**kwargs)
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def get_by_task_id(self, task_id: int):
        return self.db.query(AuditResult).filter(AuditResult.task_id == task_id).first()


class ViolationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_batch(self, violations_data: list) -> list:
        violations = [Violation(**data) for data in violations_data]
        self.db.add_all(violations)
        self.db.commit()
        for v in violations:
            self.db.refresh(v)
        return violations

    def get_by_task_id(self, task_id: int):
        return self.db.query(Violation).filter(Violation.task_id == task_id).all()

    def get_by_id(self, violation_id: int):
        return self.db.query(Violation).filter(Violation.id == violation_id).first()

    def update_review_status(
        self,
        violation_id: int,
        status: str,
        reviewer_id: int | None = None,
        review_comment: str | None = None,
    ):
        """更新违规项的人工复核状态，并记录复核人与复核意见"""
        violation = self.get_by_id(violation_id)
        if violation:
            violation.human_review_status = status
            violation.reviewed_by = reviewer_id
            violation.review_comment = review_comment
            violation.reviewed_at = datetime.now(UTC)
            self.db.commit()
            self.db.refresh(violation)
        return violation
