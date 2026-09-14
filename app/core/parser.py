# app/core/parser.py
"""
文档解析模块。
提供文档内容解析和结构化功能。
"""

from typing import Any


class DocumentParser:
    """文档解析器"""

    @staticmethod
    def extract_sections(text: str) -> list[dict[str, Any]]:
        """
        从文本中提取章节结构

        Args:
            text: 原始文本

        Returns:
            章节列表，每个章节包含标题和内容
        """
        sections = []
        current_section = {"title": "", "content": ""}

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                current_section["content"] += "\n"
                continue

            # 简单的章节识别（以数字开头的行）
            if len(line) < 100 and (line[0].isdigit() or line.startswith("第")):
                if current_section["title"] or current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"title": line, "content": ""}
            else:
                current_section["content"] += line + "\n"

        if current_section["title"] or current_section["content"].strip():
            sections.append(current_section)

        return sections

    @staticmethod
    def split_paragraphs(text: str, min_length: int = 50) -> list[str]:
        """
        将文本分割为段落

        Args:
            text: 原始文本
            min_length: 最小段落长度

        Returns:
            段落列表
        """
        # 先按双换行分割
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_length]

        # 如果段落太少，尝试按单换行分割
        if len(paragraphs) < 3:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_length]

        # 如果还是太少，使用原始文本
        if not paragraphs and text.strip():
            paragraphs = [text.strip()[:2000]]

        return paragraphs
