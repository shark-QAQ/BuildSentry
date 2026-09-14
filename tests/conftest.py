# tests/conftest.py
"""
测试配置和固定装置。
"""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# 设置测试环境变量（必须在导入 app 之前）
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["ENVIRONMENT"] = "testing"

# 关键：把 DATABASE_URL 指向临时文件。
# 依赖覆盖只作用于 FastAPI 的 Depends，而 lifespan 会直接用真实的 engine/SessionLocal
# 建表并创建初始管理员。若不重定向，`DATABASE_URL=<生产库> pytest` 会在生产库中
# 写入 admin 账户；即使本地运行也会污染 data/buildsentry.db。
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp()) / 'test.db'}"

from app.core.database import Base, get_db
from app.main import app

# 使用内存数据库进行测试。
# 必须使用 StaticPool：默认连接池会让每个线程拿到各自独立的空内存库，
# 而 TestClient 的请求运行在 portal 线程中，会导致 "no such table"。
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def fresh_middleware_stack():
    """每个测试重建中间件栈。

    限流中间件把计数保存在实例属性里，而 app 是模块级单例，
    因此计数会跨测试累积（登录限流 5 次/分钟 + 锁定 5 分钟），
    导致后续测试拿到 429。重建栈即可让每个测试从空计数开始。
    """
    app.middleware_stack = app.build_middleware_stack()
    yield


@pytest.fixture(scope="function")
def setup_database():
    """创建测试数据库表"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(setup_database):
    """创建测试客户端"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
def db_session(setup_database):
    """创建测试数据库会话"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_user_data():
    """测试用户数据"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "测试用户",
        "role": "viewer",
    }


@pytest.fixture
def admin_user_data():
    """管理员用户数据"""
    return {
        "username": "testadmin",
        "email": "admin@example.com",
        "password": "adminpass123",
        "full_name": "测试管理员",
        "role": "admin",
    }


@pytest.fixture
def auditor_user_data():
    """审核员用户数据"""
    return {
        "username": "testauditor",
        "email": "auditor@example.com",
        "password": "auditorpass123",
        "full_name": "测试审核员",
        "role": "auditor",
    }


def get_auth_headers(client: TestClient, username: str, password: str) -> dict:
    """获取认证头"""
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}
