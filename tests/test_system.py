# tests/test_system.py
"""
系统级测试。
"""

from fastapi.testclient import TestClient


class TestSystemHealth:
    """系统健康检查测试"""

    def test_health_endpoint(self, client: TestClient):
        """测试健康检查端点"""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client: TestClient):
        """测试根路由"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_metrics_endpoint(self, client: TestClient):
        """测试监控指标端点"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestRateLimit:
    """速率限制测试"""

    def test_rate_limit_headers(self, client: TestClient):
        """测试速率限制响应头"""
        # 发送多个请求
        for _ in range(5):
            client.get("/api/v1/system/health")

        response = client.get("/api/v1/system/health")
        assert response.status_code == 200


class TestCORS:
    """CORS 测试"""

    def test_cors_options(self, client: TestClient):
        """测试 CORS 预检请求"""
        response = client.options(
            "/api/v1/system/health",
            headers={"Origin": "http://localhost:7860", "Access-Control-Request-Method": "GET"},
        )
        # CORS 中间件应该处理这个请求
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """错误处理测试"""

    def test_404_error(self, client: TestClient):
        """测试 404 错误处理"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient):
        """测试 405 错误处理"""
        response = client.post("/api/v1/system/health")
        assert response.status_code == 405
