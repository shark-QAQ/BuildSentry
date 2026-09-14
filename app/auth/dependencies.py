# app/auth/dependencies.py
"""
认证依赖项模块。
提供当前用户获取和权限验证功能。
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.core.database import get_db
from app.repositories.user_repo import UserRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """获取当前认证用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": 40101, "message": "无效的认证凭证"},
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    sub = payload.get("sub")
    if sub is None:
        raise credentials_exception
    try:
        user_id = int(sub)
    except TypeError, ValueError:
        # 不链式抛出，避免把解析细节暴露给调用方
        raise credentials_exception from None

    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def require_role(roles: list):
    """角色权限验证装饰器"""

    async def role_checker(current_user=Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail={"code": 40301, "message": "权限不足"}
            )
        return current_user

    return role_checker
