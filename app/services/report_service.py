# app/services/report_service.py
"""
报告生成服务模块。
提供审核报告 PDF 导出功能。
"""

from datetime import datetime
from typing import Any, ClassVar

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import settings

# 报告中文字体：reportlab 内置的 Adobe-GB1 CID 字体。
# 无需字体文件，也不增加仓库与镜像体积；字形由阅读器提供
# （Acrobat / Chrome / Edge / Firefox / macOS 预览均内置支持）。
# 注意：CID 字体不内嵌，若归档或送审要求字形绝对固定，需改用内嵌 TTF。
CJK_FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT))
# CID 字体没有独立粗体，注册字体族让 <b> 等标签回退到同一字体，而不是抛缺少字体错误
pdfmetrics.registerFontFamily(
    CJK_FONT, normal=CJK_FONT, bold=CJK_FONT, italic=CJK_FONT, boldItalic=CJK_FONT
)


class ReportService:
    """审核报告生成服务"""

    # 风险等级对应的颜色
    RISK_COLORS: ClassVar[dict[str, Any]] = {
        "high": colors.HexColor("#DC2626"),  # 红色
        "medium": colors.HexColor("#F59E0B"),  # 黄色
        "low": colors.HexColor("#10B981"),  # 绿色
    }

    # 结论对应的中文
    CONCLUSION_MAP: ClassVar[dict[str, str]] = {
        "passed": "通过",
        "conditional_pass": "有条件通过",
        "rejected": "不通过",
    }

    @classmethod
    def generate_report(
        cls,
        task_no: str,
        document_title: str,
        result: dict[str, Any],
        violations: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> str:
        """
        生成审核报告 PDF

        Args:
            task_no: 任务编号
            document_title: 文档标题
            result: 审核结果
            violations: 违规项列表
            output_path: 输出路径（可选）

        Returns:
            生成的 PDF 文件路径
        """
        if output_path is None:
            report_dir = settings.DATA_DIR / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(report_dir / f"report_{task_no}.pdf")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        elements = []

        # 自定义样式（全部指定中文字体，reportlab 默认字体不含中文字形）
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontName=CJK_FONT,
            fontSize=18,
            spaceAfter=20,
            alignment=1,  # 居中
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontName=CJK_FONT,
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
        )

        subheading_style = ParagraphStyle(
            "CustomSubHeading", parent=styles["Heading3"], fontName=CJK_FONT
        )

        footer_style = ParagraphStyle("CustomNormal", parent=styles["Normal"], fontName=CJK_FONT)

        # 标题
        elements.append(Paragraph("施工安全合规审核报告", title_style))
        elements.append(Spacer(1, 10 * mm))

        # 基本信息表格
        info_data = [
            ["任务编号", task_no],
            ["文档标题", document_title],
            ["审核时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            [
                "审核结论",
                cls.CONCLUSION_MAP.get(result.get("conclusion", ""), result.get("conclusion", "")),
            ],
            ["质量评分", f"{result.get('quality_score', 0)} 分"],
        ]

        info_table = Table(info_data, colWidths=[4 * cm, 12 * cm])
        info_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
                    ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, -1), CJK_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(info_table)
        elements.append(Spacer(1, 10 * mm))

        # 统计信息
        elements.append(Paragraph("审核统计", heading_style))
        stats = result.get("statistics", {})
        stats_data = [
            ["高风险违规", "中风险违规", "低风险违规", "总违规数"],
            [
                str(stats.get("high_risk_count", 0)),
                str(stats.get("medium_risk_count", 0)),
                str(stats.get("low_risk_count", 0)),
                str(stats.get("total_violations", 0)),
            ],
        ]

        stats_table = Table(stats_data, colWidths=[4 * cm] * 4)
        stats_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B82F6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, -1), CJK_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(stats_table)
        elements.append(Spacer(1, 10 * mm))

        # 违规项详情
        if violations:
            elements.append(Paragraph("违规项详情", heading_style))

            for idx, v in enumerate(violations, 1):
                risk_level = v.get("risk_level", "medium")
                risk_color = cls.RISK_COLORS.get(risk_level, colors.gray)

                # 违规项标题
                elements.append(
                    Paragraph(
                        f"<font color='{risk_color.hexval()}'>[{risk_level.upper()}]</font> 违规项 {idx}",
                        subheading_style,
                    )
                )

                # 违规项详情表格
                detail_data = [
                    ["规范条款", v.get("clause_code", "N/A")],
                    ["风险等级", risk_level.upper()],
                    ["违规类型", v.get("violation_type", "N/A")],
                    ["位置引用", v.get("location_ref", "N/A")],
                    ["违规描述", v.get("description", "N/A")],
                    ["整改建议", v.get("suggestion", "N/A")],
                    ["置信度", f"{v.get('confidence', 0):.2%}"],
                ]

                detail_table = Table(detail_data, colWidths=[3 * cm, 13 * cm])
                detail_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F9FAFB")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                            ("FONTNAME", (0, 0), (-1, -1), CJK_FONT),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("PADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                elements.append(detail_table)
                elements.append(Spacer(1, 5 * mm))
        else:
            elements.append(Paragraph("未发现违规项", heading_style))

        # 页脚信息
        elements.append(Spacer(1, 15 * mm))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D1D5DB")))
        elements.append(Spacer(1, 5 * mm))
        elements.append(
            Paragraph(
                f"<font size='8' color='#6B7280'>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"模型版本: {result.get('model_version', 'N/A')} | "
                f"审核耗时: {result.get('audit_duration_ms', 0)}ms</font>",
                footer_style,
            )
        )

        # 构建 PDF
        doc.build(elements)

        return output_path
