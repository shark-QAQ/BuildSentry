# app/middleware/rate_limit.py
"""
速率限制中间件。
防止 API 滥用。
"""

import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("middleware.rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""

    def __init__(
        self, app, rate_limit_per_minute: int | None = None, rate_limit_per_hour: int | None = None
    ):
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self.rate_limit_per_hour = rate_limit_per_hour or settings.RATE_LIMIT_PER_HOUR
        self.requests: dict[str, list] = defaultdict(list)
        self._cleanup_interval = 60  # 每60秒清理一次
        self._last_cleanup = time.time()

    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用认证用户ID，否则使用IP
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return f"user:{auth_header[7:20]}"  # 使用token前20字符作为标识

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self):
        """清理过期的请求记录"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        cutoff_hour = now - 3600

        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [t for t in self.requests[client_id] if t > cutoff_hour]
            if not self.requests[client_id]:
                del self.requests[client_id]

    def _check_rate_limit(self, client_id: str) -> tuple[bool, str | None]:
        """检查速率限制"""
        now = time.time()
        self._cleanup_old_requests()

        # 获取当前请求时间戳
        timestamps = self.requests.get(client_id, [])

        # 检查每分钟限制
        minute_ago = now - 60
        recent_requests = [t for t in timestamps if t > minute_ago]
        if len(recent_requests) >= self.rate_limit_per_minute:
            return False, f"速率限制: 每分钟最多 {self.rate_limit_per_minute} 次请求"

        # 检查每小时限制
        hour_ago = now - 3600
        hourly_requests = [t for t in timestamps if t > hour_ago]
        if len(hourly_requests) >= self.rate_limit_per_hour:
            return False, f"速率限制: 每小时最多 {self.rate_limit_per_hour} 次请求"

        # 记录当前请求
        self.requests[client_id].append(now)
        return True, None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        # 跳过健康检查
        if request.url.path in ["/api/v1/system/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_id = self._get_client_id(request)
        allowed, message = self._check_rate_limit(client_id)

        if not allowed:
            logger.warning(f"速率限制触发: {client_id} - {message}")
            return JSONResponse(
                status_code=429, content={"code": 42901, "message": message, "retry_after": 60}
            )

        return await call_next(request)


class LoginRateLimitMiddleware(BaseHTTPMiddleware):
    """登录接口专用速率限制"""

    def __init__(self, app):
        super().__init__(app)
        self.login_attempts: dict[str, list] = defaultdict(list)
        self.max_attempts = settings.LOGIN_RATE_LIMIT_PER_MINUTE
        self.lockout_duration = 300  # 锁定5分钟

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        # 只拦截登录接口
        if request.url.path != "/api/v1/auth/login" or request.method != "POST":
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()

        # 清理过期记录
        self.login_attempts[client_ip] = [t for t in self.login_attempts[client_ip] if t > now - 60]

        # 检查是否被锁定
        if len(self.login_attempts[client_ip]) >= self.max_attempts:
            # 检查是否还在锁定期内
            last_attempt = max(self.login_attempts[client_ip])
            if now - last_attempt < self.lockout_duration:
                remaining = int(self.lockout_duration - (now - last_attempt))
                logger.warning(f"登录锁定: {client_ip} - 剩余 {remaining} 秒")
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": 42902,
                        "message": f"登录尝试次数过多，请 {remaining} 秒后重试",
                        "retry_after": remaining,
                    },
                )
            else:
                # 锁定期过，重置计数
                self.login_attempts[client_ip] = []

        # 记录尝试
        self.login_attempts[client_ip].append(now)

        return await call_next(request)
