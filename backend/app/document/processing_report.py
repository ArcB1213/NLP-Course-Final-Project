import json
from pathlib import Path

from app.core.schemas import RawPage


WARNING_LABELS = {
    "page_title_uncertain": "页面主标题无法可靠确定，已保留标题候选供复核",
    "title_resolution_failed": "标题判定失败，已保留正文与候选，未强行指定页面主标题",
    "recognized_math_table_requires_review": "表格／公式已识别，字符、上下标与单元格关系仍需核对原页",
    "image_semantics_not_extracted": "已保留图片，图中节点、连线及关系尚未转为文字证据",
    "visual_content_requires_recognition": "图片内容尚未识别，需视觉模型或 OCR 处理",
    "formula_layout_requires_recognition": "公式排列可能损坏，需视觉识别或核对原页",
    "no_native_text": "未提取到可用文字",
    "table_detection_failed": "表格检测失败，已保留原生文字，请核对表格结构",
}


def processing_report_path(file_path: str | Path) -> Path:
    return Path(file_path).with_suffix(".processing.json")


def build_processing_report(pages: list[RawPage]) -> dict:
    warnings = [
        f"第 {page.page} 页：{WARNING_LABELS[warning]}"
        for page in pages
        for warning in page.metadata.get("warnings", [])
    ]
    return {
        "version": "pdf-processing-v2",
        "page_count": len(pages),
        "warnings": warnings,
        "pages": [page.model_dump() for page in pages],
    }


def save_processing_report(file_path: str | Path, pages: list[RawPage]) -> dict:
    report = build_processing_report(pages)
    path = processing_report_path(file_path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return report


def read_processing_report(file_path: str | Path) -> dict | None:
    path = processing_report_path(file_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
