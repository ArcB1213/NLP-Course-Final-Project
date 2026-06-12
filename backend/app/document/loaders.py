from pathlib import Path

import fitz

from app.core.schemas import RawPage
from app.document.normalizer import normalize_text


class BaseLoader:
    def load(self, file_path: str) -> list[RawPage]:
        raise NotImplementedError


class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> list[RawPage]:
        pages: list[RawPage] = []
        with fitz.open(file_path) as document:
            for index, page in enumerate(document, start=1):
                text = normalize_text(page.get_text("text"))
                if text:
                    pages.append(RawPage(text=text, page=index, metadata={}))
        return pages


class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> list[RawPage]:
        text = _read_text_file(file_path)
        pages: list[RawPage] = []
        current_heading: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                pages.append(
                    RawPage(
                        text=normalize_text("\n".join(buffer)),
                        page=None,
                        metadata={"heading": current_heading},
                    )
                )
                buffer.clear()

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                flush()
                current_heading = stripped.lstrip("#").strip() or None
            buffer.append(line)
        flush()
        return [page for page in pages if page.text]


class TxtLoader(BaseLoader):
    def load(self, file_path: str) -> list[RawPage]:
        text = _read_text_file(file_path)
        paragraphs = [normalize_text(part) for part in text.split("\n\n")]
        joined = "\n\n".join(part for part in paragraphs if part)
        return [RawPage(text=joined, page=None, metadata={})] if joined else []


def _read_text_file(file_path: str) -> str:
    path = Path(file_path)
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def get_loader(file_type: str) -> BaseLoader:
    if file_type == "pdf":
        return PDFLoader()
    if file_type in {"md", "markdown"}:
        return MarkdownLoader()
    if file_type == "txt":
        return TxtLoader()
    raise ValueError(f"Unsupported file type: {file_type}")

