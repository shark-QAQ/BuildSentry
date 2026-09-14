# app/main.py
"""
主应用文件，包含 FastAPI 实例和路由。
"""

import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api import audit, auth, documents, kb, kb_admin
from app.auth.security import get_password_hash
from app.core.config import settings
from app.core.database import engine
from app.core.logging import get_logger, setup_logging
from app.middleware.error_handler import ErrorHandlerMiddleware, RequestLoggingMiddleware
from app.middleware.metrics import MetricsMiddleware
from app.middleware.rate_limit import LoginRateLimitMiddleware, RateLimitMiddleware
from app.models.db_models import Base

# 初始化日志
setup_logging()
logger = get_logger("main")

# 全局指标实例
metrics_middleware = None


def _initial_admin_password() -> tuple[str, bool]:
    """确定初始管理员密码。

    返回 (密码, 是否为随机生成)。生产环境绝不使用已知的固定密码：
    未显式配置 DEFAULT_ADMIN_PASSWORD 时随机生成，避免部署即带默认凭据。
    """
    if settings.DEFAULT_ADMIN_PASSWORD:
        return settings.DEFAULT_ADMIN_PASSWORD, False
    if settings.is_production():
        return secrets.token_urlsafe(16), True
    return "admin123", False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global metrics_middleware

    # 启动时执行
    logger.info("BuildSentry API 启动中...")

    # 创建数据库表
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表已同步")

    # 创建默认管理员账户
    from app.core.database import SessionLocal
    from app.repositories.user_repo import UserRepository

    db = SessionLocal()
    try:
        user_repo = UserRepository(db)
        admin = user_repo.get_by_username("admin")
        if admin:
            logger.info("管理员账户已存在")
        else:
            password, generated = _initial_admin_password()
            user_repo.create(
                username="admin",
                email="admin@buildsentry.com",
                hashed_password=get_password_hash(password),
                full_name="系统管理员",
                role="admin",
            )
            if generated:
                logger.warning(
                    "已创建管理员账户 admin，密码为本次随机生成：%s —— 请立即登录修改，"
                    "或通过 DEFAULT_ADMIN_PASSWORD 环境变量预设。",
                    password,
                )
            else:
                logger.info("默认管理员账户已创建 (admin)")
    finally:
        db.close()

    # 初始化指标中间件
    metrics_middleware = MetricsMiddleware(app)

    logger.info("BuildSentry API 启动完成")

    yield

    # 关闭时执行
    logger.info("BuildSentry API 正在关闭...")
    engine.dispose()
    logger.info("BuildSentry API 已关闭")


# 创建 FastAPI 实例
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs" if settings.LOG_LEVEL == "DEBUG" else None,
    redoc_url="/redoc" if settings.LOG_LEVEL == "DEBUG" else None,
)

# 配置中间件（顺序很重要，后添加的先执行）
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(LoginRateLimitMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-Duration"],
)

# 包含路由
app.include_router(auth.router)  # 认证相关接口
app.include_router(documents.router)  # 文档相关接口
app.include_router(audit.router)  # 审核相关接口
app.include_router(kb.router)  # 知识库相关接口
app.include_router(kb_admin.router)  # 知识库管理员接口


# 系统健康检查接口
@app.get("/api/v1/system/health")
async def health():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
    }


# Prometheus 指标端点
@app.get("/metrics")
async def metrics():
    """Prometheus 格式的监控指标"""
    if metrics_middleware:
        return Response(content=metrics_middleware.get_metrics(), media_type="text/plain")
    return Response(content="Metrics not available", status_code=503)


# 根路由，返回系统状态
@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "BuildSentry API running",
        "docs": "/docs" if settings.LOG_LEVEL == "DEBUG" else "disabled",
        "metrics": "/metrics",
    }
