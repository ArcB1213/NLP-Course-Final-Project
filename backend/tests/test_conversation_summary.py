import asyncio
import json
import re
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.conversation_summary import read_summary, update_summary
from app.core.llm_client import IncompleteResponseError
from app.core.memory_service import MemoryResolvedRequest, MemoryService
from app.db import crud
from app.db.database import Base
from app.db.models import ChatMessage, ConversationSession


class SummaryLLM:
    api_key = "test"

    def __init__(self):
        self.calls = []
        self.failure = None

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        ids = re.findall(r"message_id=([^ ]+) role=", kwargs["messages"][1]["content"])
        return json.dumps({
            "current_task": "Review the course with source evidence.",
            "constraints": ["Use original course formulas."],
            "results": [{"description": "Previous exercise; read original for exact content.", "message_id": ids[-1]}],
            "unresolved": ["Verify the formula against the source document."],
        })


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(ConversationSession(id="session"))
        session.commit()
        yield session
    engine.dispose()


def add_turns(db, start, stop):
    for turn in range(start, stop):
        for index, role in enumerate(["user", "assistant"]):
            db.add(ChatMessage(
                id=f"{turn}-{role}", session_id="session", role=role,
                content=f"turn {turn} {role}", task_type="qa",
                created_at=datetime(2026, 1, 1) + timedelta(seconds=turn * 2 + index),
            ))
    db.commit()


def update(db, llm):
    asyncio.run(update_summary(db, "session", llm, first_turn=12, interval=6, keep_turns=8))


def service(llm):
    return MemoryService(Settings(_env_file=None), llm, embedder=None, vector_store=object())


def test_first_at_12_then_every_6_completed_turns(db):
    llm = SummaryLLM()
    add_turns(db, 0, 11)
    update(db, llm)
    assert not llm.calls
    add_turns(db, 11, 12)
    update(db, llm)
    record = crud.get_conversation_summary(db, "session")
    assert record.last_message_id == "3-assistant"
    assert read_summary(record.content).updated_at_turn == 12
    first_calls = len(llm.calls)
    for turn in range(12, 17):
        add_turns(db, turn, turn + 1)
        update(db, llm)
    assert len(llm.calls) == first_calls
    context = service(llm).load_context(db, "session", "follow-up")
    assert context.recent_messages[0].id == "4-user"
    assert len(context.recent_messages) == 26  # All uncovered turns, not just the latest eight.
    add_turns(db, 17, 18)
    update(db, llm)
    assert record.last_message_id == "9-assistant"
    assert read_summary(record.content).updated_at_turn == 18
    add_turns(db, 18, 24)
    update(db, llm)
    assert read_summary(record.content).updated_at_turn == 24
    assert record.last_message_id == "15-assistant"


def test_long_message_tail_survives_summarization_and_recent_context(db):
    llm = SummaryLLM()
    add_turns(db, 0, 12)
    original = "first question\n" + "explanation\n" * 1800 + "THIRD_QUESTION_AT_END"
    db.get(ChatMessage, "0-assistant").content = original
    db.get(ChatMessage, "11-assistant").content = "x" * 1500 + "RECENT_THIRD_QUESTION"
    db.commit()
    update(db, llm)
    inputs = "\n".join(call["messages"][1]["content"] for call in llm.calls)
    assert "THIRD_QUESTION_AT_END" in inputs
    assert sum(call["messages"][1]["content"].count("explanation") for call in llm.calls) == 1800
    assert all(call["require_complete"] for call in llm.calls)
    context = service(llm).load_context(db, "session", "explain the third question")
    text = service(llm).build_context_text(context)
    assert "RECENT_THIRD_QUESTION" in text
    assert "[message_id=0-assistant role=assistant]" not in text


@pytest.mark.parametrize("failure", [RuntimeError("offline"), IncompleteResponseError("length"), ValueError("invalid JSON")])
def test_failure_keeps_old_summary_and_cursor(db, failure):
    llm = SummaryLLM()
    add_turns(db, 0, 12)
    update(db, llm)
    record = crud.get_conversation_summary(db, "session")
    previous = (record.content, record.last_message_id)
    add_turns(db, 12, 18)
    llm.failure = failure
    update(db, llm)
    assert (record.content, record.last_message_id) == previous


def test_legacy_summary_rebuilds_from_originals(db):
    llm = SummaryLLM()
    add_turns(db, 0, 18)
    crud.upsert_conversation_summary(db, session_id="session", content="OUTDATED_TRANSCRIPT", last_message_id="17-assistant")
    update(db, llm)
    record = crud.get_conversation_summary(db, "session")
    assert read_summary(record.content).updated_at_turn == 18
    assert record.last_message_id == "9-assistant"
    inputs = "\n".join(call["messages"][1]["content"] for call in llm.calls)
    assert "OUTDATED_TRANSCRIPT" not in inputs
    assert "turn 0 user" in inputs


@pytest.mark.parametrize("turn,expected_calls", [(0, 2), (11, 1)])
def test_reference_reads_full_original_without_redundant_resolution(db, turn, expected_calls):
    llm = SummaryLLM()
    add_turns(db, 0, 12)
    original = "x" * 1500 + "ARCHIVED_THIRD_QUESTION"
    db.get(ChatMessage, f"{turn}-assistant").content = original
    db.commit()
    update(db, llm)
    memory = service(llm)
    context = memory.load_context(db, "session", "explain the earlier exercise")

    class Resolver:
        calls = []

        async def resolve(self, text, query):
            self.calls.append(text)
            return MemoryResolvedRequest(
                answer_query="Explain the archived exercise", retrieval_query="CRF",
                mode="rag_with_memory", referenced_turns=[f"{turn}-assistant"],
            )

    memory.resolver = Resolver()
    result = asyncio.run(memory.resolve_request(context, "explain the earlier exercise"))
    assert len(memory.resolver.calls) == expected_calls
    if turn == 0:
        assert "ARCHIVED_THIRD_QUESTION" not in memory.resolver.calls[0]
    assert original in memory.resolver.calls[-1]
    assert original in result.answer_query
    assert result.retrieval_query == "CRF"


def test_incomplete_turn_does_not_trigger_summary(db):
    llm = SummaryLLM()
    add_turns(db, 0, 11)
    db.add(ChatMessage(id="pending", session_id="session", role="user", content="pending"))
    db.commit()
    update(db, llm)
    assert not llm.calls


def test_timestamp_ties_keep_turns_in_insertion_order(db):
    add_turns(db, 0, 12)
    for message in crud.list_session_messages(db, "session"):
        message.created_at = datetime(2026, 1, 1)
    db.commit()
    llm = SummaryLLM()
    update(db, llm)
    assert crud.get_conversation_summary(db, "session").last_message_id == "3-assistant"


def test_truncation_retries_smaller_parts_before_committing(db):
    class RetryLLM(SummaryLLM):
        failed_once = False

        async def chat(self, **kwargs):
            if not self.failed_once:
                self.failed_once = True
                self.original_size = len(kwargs["messages"][1]["content"])
                raise IncompleteResponseError("length")
            return await super().chat(**kwargs)

    add_turns(db, 0, 12)
    llm = RetryLLM()
    update(db, llm)
    assert len(llm.calls) == 2
    assert len(llm.calls[0]["messages"][1]["content"]) < llm.original_size
    assert crud.get_conversation_summary(db, "session").last_message_id == "3-assistant"


@pytest.mark.parametrize("text", ["not json", '{"current_task":"missing fields"}', json.dumps({
    "current_task": "task", "constraints": [], "unresolved": [],
    "results": [{"description": "invented reference", "message_id": "OTHER_SESSION"}],
})])
def test_invalid_summary_never_advances_coverage(db, text):
    class InvalidLLM(SummaryLLM):
        async def chat(self, **kwargs):
            return text

    add_turns(db, 0, 12)
    update(db, InvalidLLM())
    assert crud.get_conversation_summary(db, "session") is None
