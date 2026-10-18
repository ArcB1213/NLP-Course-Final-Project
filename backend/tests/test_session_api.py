from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.memory import router
from app.db import crud
from app.db.database import Base, get_db
from app.db.models import ChatMessage, ConversationSummary


@pytest.fixture
def session_api():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: db
        with TestClient(app) as client:
            yield db, client
    engine.dispose()


def test_session_list_uses_first_question_and_recent_activity(session_api):
    db, client = session_api
    older = crud.get_or_create_session(db, "older")
    newer = crud.get_or_create_session(db, "newer")
    crud.create_chat_message(db, session_id=older.id, role="user", content="First question")
    crud.create_chat_message(db, session_id=older.id, role="assistant", content="Answer")
    crud.create_chat_message(db, session_id=older.id, role="user", content="Follow-up")
    newer.title = "Custom title"
    older.updated_at = datetime(2026, 1, 1)
    newer.updated_at = datetime(2026, 1, 2)
    db.commit()
    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert [row["id"] for row in sessions] == ["newer", "older"]
    assert sessions[0]["title"] == "Custom title" and sessions[0]["message_count"] == 0
    assert sessions[1]["title"] == "First question" and sessions[1]["message_count"] == 3


def test_session_history_is_complete_ordered_and_isolated(session_api):
    db, client = session_api
    crud.get_or_create_session(db, "s")
    crud.get_or_create_session(db, "other")
    crud.create_chat_message(db, session_id="other", role="user", content="Not included")
    for i in range(30):
        message = crud.create_chat_message(db, session_id="s", role="user" if i % 2 == 0 else "assistant",
                                           content=f"Message {i}", task_type="qa")
        message.created_at = datetime(2026, 1, 1)
        db.commit()
    response = client.get("/api/sessions/s/messages")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "s"
    assert [row["content"] for row in data["messages"]] == [f"Message {i}" for i in range(30)]
    assert data["messages"][1]["role"] == "assistant"
    assert data["messages"][1]["task_type"] == "qa"
    assert "created_at" in data["messages"][0]


def test_empty_and_missing_sessions(session_api):
    db, client = session_api
    assert client.get("/api/sessions").json() == {"sessions": []}
    crud.get_or_create_session(db, "empty")
    assert client.get("/api/sessions").json()["sessions"][0]["title"] == "未命名会话"
    assert client.get("/api/sessions/empty/messages").json()["messages"] == []
    assert client.get("/api/sessions/missing/messages").status_code == 404


def test_delete_session_removes_history_but_keeps_long_term_memory(session_api):
    db, client = session_api
    crud.get_or_create_session(db, "s")
    user = crud.create_chat_message(db, session_id="s", role="user", content="Remember preference")
    memory = crud.create_memory_record(db, kind="preference", content="Persistent preference", importance=3,
                                       source_session_id="s", source_message_id=user.id)
    db.add(ConversationSummary(session_id="s", content="summary"))
    db.commit()
    response = client.delete("/api/sessions/s")
    assert response.status_code == 200 and response.json() == {"deleted": 1}
    assert client.get("/api/sessions/s/messages").status_code == 404
    assert client.get("/api/sessions").json() == {"sessions": []}
    assert not list(db.scalars(select(ChatMessage)))
    assert db.get(ConversationSummary, "s") is None
    assert crud.list_memory_records(db)[0].id == memory.id
    assert client.delete("/api/sessions/s").json() == {"deleted": 0}
