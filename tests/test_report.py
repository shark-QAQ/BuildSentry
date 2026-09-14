# tests/test_report.py
"""
审核报告生成测试。

重点覆盖中文字形：reportlab 默认字体不含中文字形，且不会报错，
只是把中文静默丢弃，因此必须断言文本能被提取回来。
"""

import pytest
from pypdf import PdfReader

from app.services.report_service import CJK_FONT, ReportService


@pytest.fixture
def sample_result():
    return {
        "conclusion": "conditional_pass",
        "quality_score": 82.5,
        "model_version": "simulation-v1.0",
        "audit_duration_ms": 1520,
        "statistics": {
            "high_risk_count": 1,
            "medium_risk_count": 1,
            "low_risk_count": 0,
            "total_violations": 2,
        },
    }


@pytest.fixture
def sample_violations():
    return [
        {
            "risk_level": "high",
            "clause_code": "JGJ 59-2011 3.11.3",
            "violation_type": "定量违规",
            "location_ref": "第 1 段",
            "description": "基坑周边1.2米范围内严禁堆载，方案中堆载距离为0.8米",
            "suggestion": "调整堆载位置至基坑边1.2米以外",
            "confidence": 0.95,
        },
        {
            "risk_level": "medium",
            "clause_code": "JGJ 59-2011 3.11.5",
            "violation_type": "缺失项",
            "location_ref": "第 3 段",
            "description": "深度超过2米的基坑未设置专用人员上下通道",
            "suggestion": "补充人员上下通道设计",
            "confidence": 0.88,
        },
    ]


def _extract_text(pdf_path) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(str(pdf_path)).pages)


class TestReportService:
    """报告生成测试"""

    def test_generate_report_returns_path(self, tmp_path, sample_result, sample_violations):
        """生成的报告文件应存在且非空"""
        output = tmp_path / "report.pdf"
        returned = ReportService.generate_report(
            task_no="AT-20260914-001",
            document_title="XX 项目基坑开挖专项施工方案",
            result=sample_result,
            violations=sample_violations,
            output_path=str(output),
        )

        assert returned == str(output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_chinese_text_is_rendered(self, tmp_path, sample_result, sample_violations):
        """中文必须真正写入 PDF，而不是被静默丢弃"""
        output = tmp_path / "report.pdf"
        ReportService.generate_report(
            task_no="AT-20260914-001",
            document_title="XX 项目基坑开挖专项施工方案",
            result=sample_result,
            violations=sample_violations,
            output_path=str(output),
        )

        text = _extract_text(output)
        for probe in [
            "施工安全合规审核报告",
            "任务编号",
            "XX 项目基坑开挖专项施工方案",
            "审核结论",
            "有条件通过",
            "审核统计",
            "高风险违规",
            "违规项详情",
            "规范条款",
            "整改建议",
            "基坑周边1.2米范围内严禁堆载",
            "补充人员上下通道设计",
        ]:
            assert probe in text, f"报告中缺少中文内容: {probe}"

    def test_generate_report_without_violations(self, tmp_path, sample_result):
        """无违规项时应生成报告并给出对应提示"""
        output = tmp_path / "empty.pdf"
        ReportService.generate_report(
            task_no="AT-20260914-002",
            document_title="空报告测试",
            result=sample_result,
            violations=[],
            output_path=str(output),
        )

        text = _extract_text(output)
        assert "未发现违规项" in text

    def test_cjk_font_is_registered(self):
        """中文字体应在模块导入时完成注册"""
        from reportlab.pdfbase import pdfmetrics

        assert CJK_FONT == "STSong-Light"
        assert pdfmetrics.getFont(CJK_FONT) is not None
