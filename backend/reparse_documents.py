"""Reparse registered PDFs in the current database, then rebuild retrieval indexes."""
import json
from datetime import datetime
import shutil
import sqlite3

from app.config import get_settings
from app.db import crud
from app.db.database import SessionLocal
from app.dependencies import get_ingestion_service


def main():
    settings = get_settings()
    backup = settings.sqlite_path.parent / ("before-mineru-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    backup.mkdir(parents=True)
    with sqlite3.connect(settings.sqlite_path) as source, sqlite3.connect(backup / "app.db") as target:
        source.backup(target)
    if settings.index_dir.exists():
        shutil.copytree(settings.index_dir, backup / "indexes")
    service = get_ingestion_service()
    results = []
    try:
        with SessionLocal() as db:
            documents = [document for document in crud.list_documents(db) if document.file_type == "pdf"]
            for document in documents:
                print(json.dumps({"starting": document.id, "filename": document.filename}, ensure_ascii=False), flush=True)
                try:
                    service.reparse_document(db, document_id=document.id, rebuild=False)
                    result = {"id": document.id, "filename": document.filename, "status": document.status, "chunks": document.chunk_count}
                except Exception as exc:
                    db.rollback()
                    result = {"id": document.id, "error": str(exc)}
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            service.retriever.rebuild(db)
    finally:
        (backup / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"backup": str(backup.resolve()), "pdf_count": len(results)}, ensure_ascii=False), flush=True)
    if any("error" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
