import math
import re
from collections import Counter

import fitz

from app.core.schemas import RawPage
from app.document.normalizer import normalize_text


def load_pdf(file_path: str) -> list[RawPage]:
    """Extract native PDF content and keep diagnostics for visual-only content."""
    with fitz.open(file_path) as document:
        page_blocks = [
            page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES, sort=True)["blocks"]
            for page in document
        ]
        repeated = _repeated_margins(document, page_blocks)
        images = [page.get_image_info(hashes=True) for page in document]
        margin_images: Counter[bytes] = Counter()
        for page, items in zip(document, images):
            margin_images.update({item["digest"] for item in items if _in_margin(item["bbox"], page.rect.height)})
        threshold = max(3, math.ceil(len(document) * 0.6))
        logos = {digest for digest, count in margin_images.items() if count >= threshold}
        return [_extract_page(page, page_blocks[index], repeated, images[index], logos)
                for index, page in enumerate(document)]


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _margin_key(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _in_margin(bbox, height: float) -> bool:
    return bbox[3] <= height * 0.12 or bbox[1] >= height * 0.88


def _repeated_margins(document, page_blocks: list[list[dict]]) -> set[str]:
    counts: Counter[str] = Counter()
    for page, blocks in zip(document, page_blocks):
        keys = {
            _margin_key(_line_text(line))
            for block in blocks
            for line in block.get("lines", [])
            if _in_margin(line["bbox"], page.rect.height) and _line_text(line)
        }
        counts.update(keys)
    threshold = max(3, math.ceil(len(document) * 0.6))
    return {key for key, count in counts.items() if count >= threshold}


def _extract_page(page, blocks: list[dict], repeated: set[str], page_images: list[dict], logos: set[bytes]) -> RawPage:
    elements: list[tuple[tuple, str, str]] = []
    warnings: list[str] = []
    try:
        tables = _extract_tables(page)
    except (RuntimeError, ValueError):
        tables = []
        warnings.append("table_detection_failed")
    body_lines: list[dict] = []
    for block in blocks:
        lines = []
        for line in block.get("lines", []):
            text = _line_text(line)
            if not text:
                continue
            if _in_margin(line["bbox"], page.rect.height) and _margin_key(text) in repeated:
                continue
            if any(_inside(line["bbox"], bbox) for bbox, _ in tables):
                continue
            lines.append(line)
        if lines:
            text = "\n".join(_line_text(line) for line in lines)
            elements.append((block["bbox"], "text", normalize_text(text)))
            body_lines.extend(lines)

    for bbox, text in tables:
        elements.append((bbox, "table", text))

    images = [
        image["bbox"] for image in page_images
        if not _in_margin(image["bbox"], page.rect.height)
        and image["digest"] not in logos
        and fitz.Rect(image["bbox"]).get_area() >= page.rect.get_area() * 0.005
    ]
    if images:
        warnings.append("visual_content_requires_recognition")
    if _fragmented_math(body_lines):
        warnings.append("formula_layout_requires_recognition")
    if not body_lines and not tables and (images or page.get_drawings()):
        warnings.append("no_native_text")

    elements.sort(key=lambda item: (item[0][1], item[0][0]))
    heading = _page_heading(body_lines, page.rect.height)
    return RawPage(
        text="\n\n".join(text for _, _, text in elements),
        page=page.number + 1,
        metadata={
            "heading": heading,
            "extraction_method": "native_layout",
            "warnings": warnings,
            "image_regions": [list(bbox) for bbox in images],
            "blocks": [{"bbox": list(bbox), "type": kind, "text": text} for bbox, kind, text in elements],
        },
    )


def _inside(inner, outer) -> bool:
    rect = fitz.Rect(inner)
    return rect.get_area() > 0 and (rect & fitz.Rect(outer)).get_area() / rect.get_area() >= 0.5


def _extract_tables(page) -> list[tuple[tuple, str]]:
    result = []
    for table in page.find_tables(strategy="lines_strict").tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        rows = table.extract()
        # Arrows and enclosing frames can be mistaken for a cell covering the actual grid.
        enclosing = {
            tuple(cell) for cell in table.cells
            if sum(other != cell and _inside(other, cell) for other in table.cells) >= 2
        }
        cell_rows = [row.cells for row in table.rows]
        columns = [index for index in range(table.col_count)
                   if any(row[index] is not None and tuple(row[index]) not in enclosing for row in cell_rows)]
        if len(columns) < 2:
            continue
        rows = [[values[index] if cells[index] is not None and tuple(cells[index]) not in enclosing else None
                 for index in columns] for values, cells in zip(rows, cell_rows)]
        if sum(bool(cell and str(cell).strip()) for row in rows for cell in row) < 4:
            continue
        escaped = [
            [normalize_text(cell or "").replace("|", "\\|").replace("\n", "<br>") for cell in row]
            for row in rows
        ]
        lines = ["| " + " | ".join(row) + " |" for row in escaped]
        lines.insert(1, "| " + " | ".join("---" for _ in escaped[0]) + " |")
        cells = [cell for row in cell_rows for cell in row if cell is not None and tuple(cell) not in enclosing]
        bbox = (min(cell[0] for cell in cells), min(cell[1] for cell in cells),
                max(cell[2] for cell in cells), max(cell[3] for cell in cells))
        result.append((bbox, "\n".join(lines)))
    return result


def _fragmented_math(lines: list[dict]) -> bool:
    texts = [_line_text(line) for line in lines]
    short_lines = sum(len(text) <= 3 for text in texts)
    math_symbols = sum(char in "=∑∏∫≤≥∈∞√" for text in texts for char in text)
    return short_lines >= 8 and short_lines / max(len(texts), 1) >= 0.25 and math_symbols >= 2


def _page_heading(lines: list[dict], height: float) -> str | None:
    for line in lines:
        text = _line_text(line)
        if line["bbox"][1] > height * 0.35:
            break
        if len(text) <= 100 and not text.startswith(("•", "", "▪", "- ")):
            return text
    return None
