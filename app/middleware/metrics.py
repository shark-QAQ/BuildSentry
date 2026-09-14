# app/middleware/metrics.py
"""
监控指标中间件。
提供 Prometheus 格式的监控指标。
"""

import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus 指标中间件"""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
        self.error_count = defaultdict(int)
        self._start_time = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过指标端点
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        method = request.method
        path = self._normalize_path(request.url.path)

        try:
            response = await call_next(request)
            status_code = response.status_code

            # 记录请求
            self.request_count[f"{method}:{path}:{status_code}"] += 1

            # 记录错误
            if status_code >= 400:
                self.error_count[f"{method}:{path}:{status_code}"] += 1

            # 记录延迟
            duration = time.time() - start_time
            self.request_duration[f"{method}:{path}"].append(duration)

            # 添加指标头
            response.headers["X-Request-Duration"] = str(round(duration, 3))

            return response

        except Exception:
            # 记录异常
            self.error_count[f"{method}:{path}:exception"] += 1
            raise

    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _normalize_path(self, path: str) -> str:
        """标准化路径（移除动态参数）"""
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/".join(normalized)

    def get_metrics(self) -> str:
        """生成 Prometheus 格式的指标"""
        uptime = time.time() - self._start_time

        lines = [
            "# HELP buildsentry_uptime_seconds Application uptime in seconds",
            "# TYPE buildsentry_uptime_seconds gauge",
            f"buildsentry_uptime_seconds {uptime:.2f}",
            "",
            "# HELP buildsentry_requests_total Total number of requests",
            "# TYPE buildsentry_requests_total counter",
        ]

        for key, count in self.request_count.items():
            method, path, status = key.split(":")
            lines.append(
                f'buildsentry_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        lines.extend(
            [
                "",
                "# HELP buildsentry_errors_total Total number of errors",
                "# TYPE buildsentry_errors_total counter",
            ]
        )

        for key, count in self.error_count.items():
            method, path, status = key.split(":")
            lines.append(
                f'buildsentry_errors_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        lines.extend(
            [
                "",
                "# HELP buildsentry_request_duration_seconds Request duration in seconds",
                "# TYPE buildsentry_request_duration_seconds summary",
            ]
        )

        for key, durations in self.request_duration.items():
            if durations:
                method, path = key.split(":")
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                lines.append(
                    f'buildsentry_request_duration_seconds{{method="{method}",path="{path}",quantile="0.5"}} {avg_duration:.3f}'
                )
                lines.append(
                    f'buildsentry_request_duration_seconds{{method="{method}",path="{path}",quantile="0.9"}} {max_duration:.3f}'
                )
                lines.append(
                    f'buildsentry_request_duration_seconds{{method="{method}",path="{path}",quantile="0.99"}} {max_duration:.3f}'
                )

        return "\n".join(lines)


# 全局指标实例
metrics = None


def get_metrics_middleware() -> MetricsMiddleware:
    """获取指标中间件实例"""
    global metrics
    if metrics is None:
        metrics = MetricsMiddleware(None)
    return metrics
