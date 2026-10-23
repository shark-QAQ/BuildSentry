# app/api/auth.py
"""
认证路由模块。
包含用户注册、登录、用户管理等 API。
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, require_role
from app.auth.security import create_access_token, get_password_hash, verify_password
from app.core.database import get_db
from app.models.db_models import User
from app.repositories.user_repo import UserRepository
from app.schemas.auth import (
    LoginRequest,
    PasswordChange,
    TokenResponse,
    UserAdminCreate,
    UserCreate,
    UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """用户注册"""
    user_repo = UserRepository(db)

    # 检查用户名是否已存在
    if user_repo.get_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": 40002, "message": "用户名已存在"},
        )

    # 检查邮箱是否已存在
    if user_repo.get_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": 40003, "message": "邮箱已被注册"},
        )

    # 角色固定为 viewer，绝不取自请求体：此前 role 由客户端指定，
    # 任何人都能直接注册成 admin，使全部管理员鉴权形同虚设。
    # 特权角色只能由管理员经 POST /users 创建。
    hashed_password = get_password_hash(user_data.password)
    user = user_repo.create(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role="viewer",
    )

    return user


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    user_repo = UserRepository(db)
    user = user_repo.get_by_username(request.username)

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": 40102, "message": "用户名或密码错误"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail={"code": 40302, "message": "账户已被禁用"}
        )

    # 更新最后登录时间
    user_repo.update_last_login(user.id)

    # 创建访问令牌（JWT 规范要求 sub 为字符串，python-jose 在解码时会强制校验）
    access_token = create_access_token(data={"sub": str(user.id)})

    return TokenResponse(access_token=access_token, user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user


@router.put("/me/password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """修改密码"""
    # 验证旧密码
    if not verify_password(password_data.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail={"code": 40005, "message": "旧密码错误"}
        )

    # 更新密码
    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()

    return {"message": "密码修改成功"}


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """获取用户列表（仅管理员）"""
    user_repo = UserRepository(db)
    users = user_repo.get_all(skip=skip, limit=limit)
    return users


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserAdminCreate,
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """创建用户并指定角色（仅管理员）

    特权角色（auditor / admin）只能从这里产生：注册接口一律创建 viewer，
    否则任何人都能自助提权。
    """
    user_repo = UserRepository(db)

    if user_repo.get_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": 40002, "message": "用户名已存在"},
        )

    if user_repo.get_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": 40003, "message": "邮箱已被注册"},
        )

    return user_repo.create(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        role=user_data.role,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """删除用户（仅管理员）"""
    user_repo = UserRepository(db)

    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": 40006, "message": "不能删除自己"},
        )

    if not user_repo.delete(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"code": 40403, "message": "用户不存在"}
        )

    return {"message": "用户删除成功"}
