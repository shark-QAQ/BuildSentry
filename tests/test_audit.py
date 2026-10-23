# tests/test_audit.py
"""
审核模块测试。
"""

from fastapi.testclient import TestClient

from app.models.db_models import AuditTask, Document, Violation
from tests.conftest import TestingSessionLocal, get_auth_headers


class TestAudit:
    """审核相关测试"""

    def test_create_audit_task_not_found(self, client: TestClient, test_user_data):
        """文档不存在时应返回 404，而不是创建一条孤儿任务"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.post("/api/v1/audit/tasks?doc_id=99999", headers=headers)
        assert response.status_code == 404

    def test_get_task_status_not_found(self, client: TestClient, test_user_data):
        """测试获取不存在的任务状态"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.get("/api/v1/audit/tasks/99999", headers=headers)
        assert response.status_code == 404

    def test_get_result_before_completion(self, client: TestClient, test_user_data):
        """测试任务未完成时获取结果"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.get("/api/v1/audit/tasks/1/result", headers=headers)
        assert response.status_code in [400, 404]

    def test_review_violation_invalid_status(self, client: TestClient, auditor_headers):
        """测试无效的复核状态"""
        headers = auditor_headers

        response = client.put(
            "/api/v1/audit/violations/1/review?review_status=invalid", headers=headers
        )
        assert response.status_code == 400

    def test_review_violation_not_found(self, client: TestClient, auditor_headers):
        """测试复核不存在的违规项"""
        headers = auditor_headers

        response = client.put(
            "/api/v1/audit/violations/99999/review?review_status=confirmed", headers=headers
        )
        assert response.status_code == 404

    def test_review_violation_requires_auditor_role(self, client: TestClient, test_user_data):
        """viewer 不应有权复核违规项：复核会改变审核结论"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.put(
            "/api/v1/audit/violations/99999/review?review_status=confirmed", headers=headers
        )
        assert response.status_code == 403

    def test_review_violation_records_trail(self, client: TestClient, db_session, auditor_headers):
        """复核必须留下可追溯的记录，而不只是改一个状态字段"""
        headers = auditor_headers

        doc = Document(file_name="trail.pdf")
        db_session.add(doc)
        db_session.commit()
        task = AuditTask(task_no="AT-TRAIL-001", document_id=doc.id)
        db_session.add(task)
        db_session.commit()
        violation = Violation(
            task_id=task.id, risk_level="high", description="描述", suggestion="建议"
        )
        db_session.add(violation)
        db_session.commit()
        violation_id = violation.id

        response = client.put(
            f"/api/v1/audit/violations/{violation_id}/review"
            "?review_status=confirmed&review_comment=已现场核实",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["human_review_status"] == "confirmed"
        assert data["review_comment"] == "已现场核实"
        assert data["reviewed_at"] is not None

        # 复查数据库，确认确实落库而非只出现在响应里
        session = TestingSessionLocal()
        try:
            fresh = session.query(Violation).filter(Violation.id == violation_id).first()
            assert fresh.human_review_status == "confirmed"
            assert fresh.review_comment == "已现场核实"
            assert fresh.reviewed_by is not None
            assert fresh.reviewed_at is not None
        finally:
            session.close()

    def test_get_violations_task_not_found(self, client: TestClient, test_user_data):
        """测试获取不存在任务的违规项"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.get("/api/v1/audit/tasks/99999/violations", headers=headers)
        assert response.status_code == 404
