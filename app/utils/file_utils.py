# app/utils/file_utils.py
"""
文件工具模块。
提供文件操作相关的工具函数。
"""

import os

# 支持的文档格式白名单，作为唯一来源：
# 上传接口与文档解析器都以此为准，避免出现「接口拒绝但解析器支持」这类不一致。
# 注意 .doc（旧版二进制格式）不在其中——解析走的是 python-docx，无法处理。
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        文件扩展名（小写，包含点号）
    """
    return os.path.splitext(filename)[1].lower()


def is_valid_document(filename: str) -> bool:
    """
    检查文件是否为支持的文档格式

    Args:
        filename: 文件名

    Returns:
        是否为有效文档
    """
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS
