# app/schemas/auth.py
"""
认证相关的 Pydantic 模型。
"""

from datetime import datetime

from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str | None = None
    role: str | None = "viewer"


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
