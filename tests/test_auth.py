# tests/test_auth.py
"""
认证模块测试。
"""

from fastapi.testclient import TestClient


class TestAuth:
    """认证相关测试"""

    def test_register_user(self, client: TestClient, test_user_data):
        """测试用户注册"""
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user_data["username"]
        assert data["email"] == test_user_data["email"]
        assert data["role"] == test_user_data["role"]
        assert "id" in data

    def test_register_duplicate_username(self, client: TestClient, test_user_data):
        """测试重复用户名注册"""
        client.post("/api/v1/auth/register", json=test_user_data)
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 400

    def test_login_success(self, client: TestClient, test_user_data):
        """测试登录成功"""
        client.post("/api/v1/auth/register", json=test_user_data)
        response = client.post(
            "/api/v1/auth/login",
            json={"username": test_user_data["username"], "password": test_user_data["password"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["username"] == test_user_data["username"]

    def test_login_wrong_password(self, client: TestClient, test_user_data):
        """测试错误密码登录"""
        client.post("/api/v1/auth/register", json=test_user_data)
        response = client.post(
            "/api/v1/auth/login",
            json={"username": test_user_data["username"], "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_get_current_user(self, client: TestClient, test_user_data):
        """测试获取当前用户信息"""
        client.post("/api/v1/auth/register", json=test_user_data)
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": test_user_data["username"], "password": test_user_data["password"]},
        )
        token = login_response.json()["access_token"]

        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["username"] == test_user_data["username"]

    def test_change_password(self, client: TestClient, test_user_data):
        """测试修改密码"""
        client.post("/api/v1/auth/register", json=test_user_data)
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": test_user_data["username"], "password": test_user_data["password"]},
        )
        token = login_response.json()["access_token"]

        response = client.put(
            "/api/v1/auth/me/password",
            json={"old_password": test_user_data["password"], "new_password": "newpassword123"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_admin_list_users(self, client: TestClient, admin_user_data, test_user_data):
        """测试管理员获取用户列表"""
        # 创建管理员
        client.post("/api/v1/auth/register", json=admin_user_data)
        # 创建普通用户
        client.post("/api/v1/auth/register", json=test_user_data)

        # 管理员登录
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": admin_user_data["username"], "password": admin_user_data["password"]},
        )
        token = login_response.json()["access_token"]

        response = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert len(response.json()) >= 2

    def test_non_admin_cannot_list_users(self, client: TestClient, test_user_data):
        """测试非管理员无法获取用户列表"""
        client.post("/api/v1/auth/register", json=test_user_data)
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": test_user_data["username"], "password": test_user_data["password"]},
        )
        token = login_response.json()["access_token"]

        response = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403
