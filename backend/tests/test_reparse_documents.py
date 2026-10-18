import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import reparse_documents


def test_interruption_rebuilds_committed_documents_and_saves_progress(tmp_path, monkeypatch):
    settings = SimpleNamespace(sqlite_path=tmp_path / "app.db", index_dir=tmp_path / "indexes")
    documents = [SimpleNamespace(id=str(n), filename=f"{n}.pdf", file_type="pdf",
                                 status="indexed", chunk_count=1) for n in range(2)]
    events = []
    db = SimpleNamespace(rollback=lambda: events.append("rollback"))

    def reparse(db, document_id, rebuild):
        events.append(document_id)
        if document_id == "1":
            raise KeyboardInterrupt

    service = SimpleNamespace(reparse_document=reparse,
                              retriever=SimpleNamespace(rebuild=lambda db: events.append("rebuild")))
    monkeypatch.setattr(reparse_documents, "get_settings", lambda: settings)
    monkeypatch.setattr(reparse_documents, "get_ingestion_service", lambda: service)
    monkeypatch.setattr(reparse_documents, "SessionLocal", lambda: nullcontext(db))
    monkeypatch.setattr(reparse_documents.crud, "list_documents", lambda db: documents)
    with pytest.raises(KeyboardInterrupt):
        reparse_documents.main()
    assert events == ["0", "1", "rollback", "rebuild"]
    report = json.loads(next(tmp_path.glob("before-mineru-*/results.json")).read_text(encoding="utf-8"))
    assert [r["id"] for r in report] == ["0"]
