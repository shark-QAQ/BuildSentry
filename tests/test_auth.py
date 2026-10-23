# tests/test_auth.py
"""
认证模块测试。
"""

from fastapi.testclient import TestClient

from tests.conftest import get_auth_headers


class TestAuth:
    """认证相关测试"""

    def test_register_user(self, client: TestClient, test_user_data):
        """测试用户注册"""
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user_data["username"]
        assert data["email"] == test_user_data["email"]
        assert data["role"] == "viewer"
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

    def test_admin_list_users(self, client: TestClient, admin_headers, test_user_data):
        """测试管理员获取用户列表"""
        # 再建一个普通用户
        client.post("/api/v1/auth/register", json=test_user_data)

        response = client.get("/api/v1/auth/users", headers=admin_headers)
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


class TestPrivilegeEscalation:
    """注册接口不得成为提权通道

    回归测试：UserCreate.role 此前由客户端指定并被直接透传给
    UserRepository.create，任何人都能 POST {"role": "admin"} 注册成管理员，
    使全部 require_role(["admin"]) 形同虚设。
    """

    def test_register_rejects_role_field(self, client: TestClient, test_user_data):
        """注册时带 role 必须被明确拒绝，而不是静默忽略

        静默忽略会让调用方以为提权成功了，属于更隐蔽的失败方式。
        """
        response = client.post("/api/v1/auth/register", json={**test_user_data, "role": "admin"})

        assert response.status_code == 422

    def test_register_creates_viewer_even_without_role_field(
        self, client: TestClient, test_user_data
    ):
        response = client.post("/api/v1/auth/register", json=test_user_data)

        assert response.status_code == 200
        assert response.json()["role"] == "viewer"

    def test_admin_can_create_auditor(self, client: TestClient, admin_headers):
        """特权账号只能由管理员创建，且建出来的号真的能用"""
        response = client.post(
            "/api/v1/auth/users",
            json={
                "username": "newauditor",
                "email": "newauditor@example.com",
                "password": "auditorpass123",
                "full_name": "新审核员",
                "role": "auditor",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        assert response.json()["role"] == "auditor"

        headers = get_auth_headers(client, "newauditor", "auditorpass123")
        assert headers  # 拿到 token 说明账号可用

    def test_non_admin_cannot_create_user(self, client: TestClient, test_user_data):
        """普通用户不得建号，否则等于绕开注册接口自行提权"""
        client.post("/api/v1/auth/register", json=test_user_data)
        headers = get_auth_headers(client, test_user_data["username"], test_user_data["password"])

        response = client.post(
            "/api/v1/auth/users",
            json={
                "username": "sneaky",
                "email": "sneaky@example.com",
                "password": "sneakypass123",
                "role": "admin",
            },
            headers=headers,
        )

        assert response.status_code == 403

    def test_unknown_role_is_rejected(self, client: TestClient, admin_headers):
        """角色取值受契约约束，避免写入聚合逻辑认不出的角色"""
        response = client.post(
            "/api/v1/auth/users",
            json={
                "username": "weird",
                "email": "weird@example.com",
                "password": "weirdpass123",
                "role": "superuser",
            },
            headers=admin_headers,
        )

        assert response.status_code == 422
