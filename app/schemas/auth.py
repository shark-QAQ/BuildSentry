# app/schemas/auth.py
"""
认证相关的 Pydantic 模型。
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

# 与 User.role 的取值保持一致（见 app/models/db_models.py）
Role = Literal["viewer", "submitter", "auditor", "admin"]


class UserCreate(BaseModel):
    """自助注册。

    刻意不提供 role：注册一律创建 viewer。此前该字段客户端可控，任何人都能
    直接注册成 admin，使所有 require_role(["admin"]) 形同虚设。
    extra="forbid" 让试图带上 role 的请求直接 422，而不是被静默忽略——
    静默忽略会让调用方以为提权成功了。

    特权角色请走 UserAdminCreate（管理员接口）。
    """

    model_config = {"extra": "forbid"}

    username: str
    email: str
    password: str
    full_name: str | None = None


class UserAdminCreate(UserCreate):
    """管理员创建用户，可指定角色"""

    role: Role = "viewer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime
    last_login: datetime | None

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class PasswordChange(BaseModel):
    old_password: str
    new_password: str
