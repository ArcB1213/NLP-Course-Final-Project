from pathlib import Path

from app.core.schemas import RawPage
from app.config import Settings
from app.document.normalizer import normalize_text
from app.document.pdf_parser import load_pdf


class BaseLoader:
    def load(self, file_path: str) -> list[RawPage]:
        raise NotImplementedError


class PDFLoader(BaseLoader):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings

    def load(self, file_path: str) -> list[RawPage]:
        if self.settings is not None and self.settings.pdf_parser == "mineru":
            from app.document.mineru_parser import load_mineru_pdf
            return load_mineru_pdf(file_path, self.settings)
        return load_pdf(file_path)


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


def get_loader(file_type: str, settings: Settings | None = None) -> BaseLoader:
    if file_type == "pdf":
        return PDFLoader(settings)
    if file_type in {"md", "markdown"}:
        return MarkdownLoader()
    if file_type == "txt":
        return TxtLoader()
    raise ValueError(f"Unsupported file type: {file_type}")

