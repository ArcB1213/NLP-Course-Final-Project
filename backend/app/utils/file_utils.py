from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


SUPPORTED_FILE_TYPES = {"pdf", "md", "markdown", "txt"}


def get_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {suffix}")
    return suffix


def new_document_id() -> str:
    return f"doc_{uuid4().hex[:16]}"


async def save_upload_file(upload_file: UploadFile, upload_dir: Path, document_id: str) -> Path:
    file_type = get_file_type(upload_file.filename or "")
    safe_name = Path(upload_file.filename or f"{document_id}.{file_type}").name
    target = upload_dir / f"{document_id}_{safe_name}"
    content = await upload_file.read()
    target.write_bytes(content)
    return target

