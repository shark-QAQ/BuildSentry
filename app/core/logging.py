# app/core/logging.py
"""
日志配置模块。
提供结构化日志记录功能。
"""

import json
import logging
import sys
from datetime import datetime

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        return f"{datetime.utcnow().isoformat()} | {record.levelname:8} | {record.name} | {record.getMessage()}"


def setup_logging() -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("buildsentry")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_FORMAT == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if settings.LOG_FILE:
        file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


# 全局日志实例
logger = setup_logging()


def get_logger(name: str) -> logging.Logger:
    """获取子日志器"""
    return logging.getLogger(f"buildsentry.{name}")
