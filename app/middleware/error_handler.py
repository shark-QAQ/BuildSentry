# app/middleware/error_handler.py
"""
错误处理中间件。
统一处理异常和错误响应。
"""

from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger("middleware.error_handler")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """全局错误处理中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # 记录错误
            logger.error(f"未处理的异常: {e!s} | path={request.url.path} method={request.method}")

            # 返回统一错误响应
            return JSONResponse(
                status_code=500,
                content={"code": 50000, "message": "服务器内部错误", "detail": "请联系管理员"},
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import time

        start_time = time.time()

        # 记录请求
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"请求开始: {request.method} {request.url.path} | client={client_ip}")

        response = await call_next(request)

        # 记录响应
        process_time = time.time() - start_time
        logger.info(
            f"请求完成: {request.method} {request.url.path} | status={response.status_code} | time={round(process_time, 3)}s"
        )

        # 添加处理时间头
        response.headers["X-Process-Time"] = str(round(process_time, 3))

        return response
