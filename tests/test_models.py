# tests/test_models.py
"""
数据库模型测试。
"""

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.db_models import AuditResult, AuditTask, Document, User, Violation


class TestUserModel:
    """用户模型测试"""

    def test_create_user(self, db_session):
        """测试创建用户"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123",
            full_name="测试用户",
            role="viewer",
        )
        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == "viewer"
        assert user.is_active is True

    def test_user_unique_username(self, db_session):
        """测试用户名唯一性"""
        user1 = User(username="test", email="a@test.com", hashed_password="pass1")
        user2 = User(username="test", email="b@test.com", hashed_password="pass2")

        db_session.add(user1)
        db_session.commit()

        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestDocumentModel:
    """文档模型测试"""

    def test_create_document(self, db_session):
        """测试创建文档"""
        doc = Document(
            file_name="test.pdf",
            file_hash="abc123",
            file_size=1024,
            raw_text="测试内容",
            word_count=100,
            title="测试文档",
            project_name="测试项目",
            company="测试公司",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.id is not None
        assert doc.file_name == "test.pdf"
        assert doc.status == "parsed"


class TestAuditTaskModel:
    """审核任务模型测试"""

    def test_create_audit_task(self, db_session):
        """测试创建审核任务"""
        # 先创建文档
        doc = Document(file_name="test.pdf")
        db_session.add(doc)
        db_session.commit()

        task = AuditTask(task_no="AT-20260908-001", document_id=doc.id, status="queued")
        db_session.add(task)
        db_session.commit()

        assert task.id is not None
        assert task.task_no == "AT-20260908-001"
        assert task.status == "queued"
        assert task.progress == 0


class TestAuditResultModel:
    """审核结果模型测试"""

    def test_create_audit_result(self, db_session):
        """测试创建审核结果"""
        doc = Document(file_name="test.pdf")
        db_session.add(doc)
        db_session.commit()

        task = AuditTask(task_no="AT-001", document_id=doc.id)
        db_session.add(task)
        db_session.commit()

        result = AuditResult(
            task_id=task.id,
            conclusion="passed",
            conclusion_detail="未发现违规",
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            total_violations=0,
            quality_score=95.0,
            model_version="simulation-v1.0",
            audit_duration_ms=1500,
        )
        db_session.add(result)
        db_session.commit()

        assert result.id is not None
        assert result.conclusion == "passed"
        assert result.quality_score == 95.0


class TestViolationModel:
    """违规项模型测试"""

    def test_create_violation(self, db_session):
        """测试创建违规项"""
        doc = Document(file_name="test.pdf")
        db_session.add(doc)
        db_session.commit()

        task = AuditTask(task_no="AT-001", document_id=doc.id)
        db_session.add(task)
        db_session.commit()

        violation = Violation(
            task_id=task.id,
            clause_code="JGJ 59-2011 3.11.3",
            clause_content="基坑周边1.2米范围内严禁堆载",
            risk_level="high",
            violation_type="定量违规",
            description="段落中发现潜在违规",
            suggestion="请严格按照规范要求整改",
            confidence=0.95,
            human_review_status="pending",
        )
        db_session.add(violation)
        db_session.commit()

        assert violation.id is not None
        assert violation.risk_level == "high"
        assert violation.human_review_status == "pending"
