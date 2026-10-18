import asyncio
import json

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core.memory_writer import MemoryWriter
from app.db import crud
from app.db.database import Base
from app.db.models import ChatMessage, MemoryEvidence, MemoryProcessingState, MemoryRecord


class WriterLLM:
    def __init__(self, extract=None, decide=None):
        self.extract = extract or (lambda payload: [])
        self.decide = decide or self.default_decision
        self.calls = []

    async def chat(self, **kwargs):
        assert kwargs["require_complete"]
        payload = json.loads(kwargs["messages"][1]["content"])
        self.calls.append(payload)
        if "mode" in payload:
            return json.dumps({"candidates": self.extract(payload)}, ensure_ascii=False)
        return json.dumps(self.decide(payload), ensure_ascii=False)

    @staticmethod
    def default_decision(payload):
        return {"action": "add", "content": payload["candidate"]["content"], "target_ids": [],
                "supporting_evidence_ids": list(payload["supporting_errors"]),
                "independent_questions": True, "same_knowledge_point": True,
                "relation": "unrelated", "reason": "supported by original evidence"}


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        crud.get_or_create_session(session, "s")
        yield session
    engine.dispose()


def add_turn(db, writer, text="解释一下 HMM", answer="Course explanation", *, task="qa", confidence="high", message=None, sources=None):
    user = crud.create_chat_message(db, session_id="s", role="user", content=text, task_type=task)
    assistant = crud.create_chat_message(db, session_id="s", role="assistant", content=answer, task_type=task)
    report = asyncio.run(writer.process_turn(db, user, assistant, {
        "task_type": task, "confidence": confidence, "message": message, "sources": sources or [],
    }))
    return report, user, assistant


def candidate(db, uid, *, kind="preference", basis="explicit_statement", content="解释公式时举例", correction="none"):
    user = db.get(ChatMessage, uid)
    return {"kind": kind, "content": content, "topic": "explanation preference", "evidence_type": basis,
            "user_message_id": uid, "quotes": [{"message_id": uid, "quote": user.content}],
            "durability": "durable", "confidence": "high", "correction": correction}


def test_six_ordinary_turns_can_write_zero_memories(db):
    llm = WriterLLM()
    writer = MemoryWriter(llm)
    for i in range(6):
        report, _, _ = add_turn(db, writer)
        assert report.batch_processed == (i == 5)
    assert not crud.list_memory_records(db)
    assert db.get(MemoryProcessingState, "s").batch_turn_count == 6
    assert len([p for p in llm.calls if p.get("mode") == "batch"]) == 1
    assert db.scalar(select(func.count()).select_from(ChatMessage)) == 12


def test_explicit_preference_immediate_and_paraphrase_merges(db):
    def extract(payload):
        return [candidate(db, uid) for uid in payload["new_user_ids"]] if payload["mode"] == "immediate" else []

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        if payload["existing"]:
            result.update(action="noop", target_ids=[payload["existing"][0]["id"]], relation="equivalent")
        return result

    writer = MemoryWriter(WriterLLM(extract, decide))
    first, _, _ = add_turn(db, writer, "以后解释公式时都举例")
    second, _, _ = add_turn(db, writer, "我希望公式讲解配上例子")
    assert first.added == 1
    assert second.ignored == 1 and not second.changed
    assert len(crud.list_memory_records(db)) == 1
    assert len(list(db.scalars(select(MemoryEvidence).where(MemoryEvidence.evidence_type == "explicit_statement")))) == 2


def test_user_correction_replaces_and_forgetting_does_not_resurrect(db):
    texts = ["以后讲解时给我举例", "我改主意了，之后只要简短定义", "请忘掉我的讲解偏好"]

    def extract(payload):
        if payload["mode"] != "immediate":
            return []
        uid = payload["new_user_ids"][0]
        text = db.get(ChatMessage, uid).content
        index = texts.index(text)
        return [candidate(db, uid, basis="explicit_statement" if index == 0 else "correction",
                          content="简短定义" if index == 1 else "讲解偏好", correction=["none", "replace", "forget"][index])]

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        if payload["existing"]:
            result.update(action="remove" if payload["candidate"]["correction"] == "forget" else "update",
                          target_ids=[payload["existing"][0]["id"]], relation="contradicts")
        return result

    writer = MemoryWriter(WriterLLM(extract, decide))
    add_turn(db, writer, texts[0])
    report, _, _ = add_turn(db, writer, texts[1])
    assert report.merged == 1
    assert crud.list_memory_records(db)[0].content == "简短定义"
    report, _, _ = add_turn(db, writer, texts[2])
    assert report.removed == 1 and not crud.list_memory_records(db)
    assert list(db.scalars(select(MemoryEvidence).where(MemoryEvidence.evidence_type == "retired")))


def graded_candidate(db, uid, payload):
    user = db.get(ChatMessage, uid)
    receipt = payload["response_metadata"][uid]
    assistant = db.get(ChatMessage, receipt["assistant_id"])
    question, answer = user.content.split("|", 1)
    result = candidate(db, uid, kind="weakness", basis="graded_error", content="需要练习 CRF 的归一化机制")
    result.update(topic="CRF normalization", question={"message_id": uid, "quote": question},
                  answer={"message_id": uid, "quote": answer}, grading={"message_id": assistant.id, "quote": assistant.content},
                  course_evidence=[{"chunk_id": "course-1", "quote": "CRF uses global normalization"}])
    return result


@pytest.mark.parametrize("questions,independent,expected", [
    (["CRF 如何归一化？", "CRF 的配分函数是什么？"], True, 1),
    (["CRF 如何归一化？", "CRF 如何归一化？"], True, 0),
    (["CRF 如何归一化？", "请说明CRF怎样做归一化"], False, 0),
])
def test_weakness_requires_independent_source_backed_errors(db, questions, independent, expected):
    def extract(payload):
        if payload["mode"] != "batch":
            return []
        return [graded_candidate(db, uid, payload) for uid in payload["new_user_ids"]
                if db.get(ChatMessage, uid).task_type == "grade"]

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        result["independent_questions"] = independent
        return result

    writer = MemoryWriter(WriterLLM(extract, decide))
    for question in questions:
        add_turn(db, writer, question + "|局部归一化", "该答案错误，CRF使用全局归一化", task="grade",
                 sources=[{"chunk_id": "course-1", "text": "CRF uses global normalization"}])
    for _ in range(4):
        report, user, assistant = add_turn(db, writer)
    assert len(crud.list_memory_records(db)) == expected
    if expected:
        assert crud.list_memory_records(db)[0].content.startswith("练习观察：")
    assert len(list(db.scalars(select(MemoryEvidence).where(MemoryEvidence.evidence_type == "graded_error")))) == 2
    again = asyncio.run(writer.process_turn(db, user, assistant, {}))
    assert not again.changed
    assert len(crud.list_memory_records(db)) == expected


@pytest.mark.parametrize("confidence,message,sources", [
    ("low", None, [{"chunk_id": "course-1", "text": "CRF uses global normalization"}]),
    ("high", "evidence_refusal", []), ("high", "llm_error", []), ("high", None, []),
])
def test_unreliable_grading_is_excluded_without_stalling_batch(db, confidence, message, sources):
    def extract(payload):
        if payload["mode"] != "batch":
            return []
        return [graded_candidate(db, uid, payload) for uid in payload["new_user_ids"]
                if db.get(ChatMessage, uid).task_type == "grade"]

    writer = MemoryWriter(WriterLLM(extract))
    add_turn(db, writer, "CRF如何归一化？|局部", "未找到足够依据", task="grade", confidence=confidence, message=message, sources=sources)
    for _ in range(5):
        report, _, _ = add_turn(db, writer)
    assert not crud.list_memory_records(db)
    assert "unreliable_grading_excluded" in report.reasons
    assert db.get(MemoryProcessingState, "s").batch_turn_count == 6


def test_self_report_is_attributed_to_user(db):
    llm = WriterLLM(lambda payload: [candidate(db, uid, kind="weakness", basis="self_report", content="我仍分不清HMM与CRF")
                                    for uid in payload["new_user_ids"]])
    report, _, _ = add_turn(db, MemoryWriter(llm), "我仍分不清HMM与CRF，这一直困扰我")
    assert report.added == 1
    assert crud.list_memory_records(db)[0].content.startswith("用户自述：")


def test_invalid_second_operation_rolls_back_whole_batch_and_retry_is_idempotent(db):
    fail = [True]

    def extract(payload):
        return [candidate(db, uid, content=f"goal {i}") for i, uid in enumerate(payload["new_user_ids"][:2])] if payload["mode"] == "batch" else []

    def decide(payload):
        if payload["existing"] and fail[0]:
            raise ValueError("invalid second operation")
        return WriterLLM.default_decision(payload)

    writer = MemoryWriter(WriterLLM(extract, decide))
    for _ in range(6):
        report, user, assistant = add_turn(db, writer)
    assert report.reasons == ["memory_write_failed:ValueError"]
    assert not crud.list_memory_records(db)
    assert db.get(MemoryProcessingState, "s").batch_turn_count == 0
    assert db.scalar(select(func.count()).select_from(ChatMessage)) == 12
    fail[0] = False
    report = asyncio.run(writer.process_turn(db, user, assistant, {}))
    assert report.added == 2 and report.batch_processed
    again = asyncio.run(writer.process_turn(db, user, assistant, {}))
    assert not again.changed and len(crud.list_memory_records(db)) == 2


def test_invalid_source_quotes_do_not_write(db):
    def extract(payload):
        item = candidate(db, payload["new_user_ids"][0])
        item["quotes"][0]["quote"] = "not in original"
        return [item]

    report, _, _ = add_turn(db, MemoryWriter(WriterLLM(extract)), "以后举例解释")
    assert report.reasons == ["memory_write_failed:ValueError"]
    assert not crud.list_memory_records(db)


def test_long_message_tail_is_processed(db):
    def extract(payload):
        if "TAIL_PREFERENCE" not in payload["messages"]:
            return []
        item = candidate(db, payload["new_user_ids"][0])
        item["quotes"][0]["quote"] = "TAIL_PREFERENCE"
        return [item]

    llm = WriterLLM(extract)
    report, _, _ = add_turn(db, MemoryWriter(llm), "课程资料内容。" * 1800 + "TAIL_PREFERENCE")
    assert report.added == 1
    assert "TAIL_PREFERENCE" in "".join(p.get("messages", "") for p in llm.calls)


def test_unresolved_conflict_cannot_overwrite_preference(db):
    def decide(payload):
        result = WriterLLM.default_decision(payload)
        if payload["existing"]:
            result.update(action="update", target_ids=[payload["existing"][0]["id"]], relation="contradicts")
        return result

    llm = WriterLLM(lambda p: [candidate(db, uid, content=db.get(ChatMessage, uid).content) for uid in p["new_user_ids"]], decide)
    writer = MemoryWriter(llm)
    add_turn(db, writer, "我长期偏好举例讲解")
    report, _, _ = add_turn(db, writer, "这一次只要定义")
    assert not report.changed
    assert crud.list_memory_records(db)[0].content == "我长期偏好举例讲解"


def test_legacy_record_is_not_automatically_rewritten(db):
    legacy = crud.create_memory_record(db, kind="learning_goal", content="legacy inferred goal", importance=3,
                                       source_session_id="s", source_message_id=None)

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        result.update(action="update", target_ids=[legacy.id], relation="compatible")
        return result

    writer = MemoryWriter(WriterLLM(lambda p: [candidate(db, p["new_user_ids"][0])], decide))
    report, _, _ = add_turn(db, writer, "我喜欢示例")
    assert not report.changed and "legacy_record_requires_manual_review" in report.reasons
    assert legacy.content == "legacy inferred goal"


def test_unrelated_errors_do_not_form_a_weakness(db):
    def extract(payload):
        return [graded_candidate(db, uid, payload) for uid in payload["new_user_ids"]] if payload["mode"] == "batch" else []

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        result["same_knowledge_point"] = False
        return result

    writer = MemoryWriter(WriterLLM(extract, decide), interval=2)
    for question in ["HMM如何解码", "QLoRA如何压缩权重"]:
        report, _, _ = add_turn(db, writer, question + "|错误答案", "答案错误", task="grade",
                               sources=[{"chunk_id": "course-1", "text": "CRF uses global normalization"}])
    assert not report.changed and not crud.list_memory_records(db)


def test_memory_service_rebuilds_only_for_content_changes_and_filters_stale_ids(db):
    import numpy as np
    from app.config import Settings
    from app.core.memory_service import MemoryService

    class Embedder:
        def encode_texts(self, texts):
            return np.ones((len(texts), 2), dtype="float32")

        def encode_query(self, query):
            return np.ones(2, dtype="float32")

    class Store:
        builds = 0
        ids = []

        def build(self, ids, embeddings):
            self.builds += 1
            self.ids = ids

        def search(self, query, k):
            return [(mid, 1.0) for mid in [*self.ids, "deleted-id"]]

    def extract(payload):
        return [candidate(db, uid) for uid in payload["new_user_ids"]]

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        if payload["existing"]:
            result.update(action="noop", target_ids=[payload["existing"][0]["id"]], relation="equivalent")
        return result

    store = Store()
    memory = MemoryService(Settings(_env_file=None), WriterLLM(extract, decide), Embedder(), store)
    for text in ["以后讲解时举例", "以后继续举例"]:
        asyncio.run(memory.remember_turn(db, session_id="s", user_query=text, assistant_answer="好", task_type="qa"))
    assert store.builds == 1
    memory._recall_memories(db, "preference", 5)
    memory._recall_memories(db, "preference", 5)
    assert store.builds == 1  # last_used timestamps do not invalidate content embeddings.
    record = crud.list_memory_records(db)[0]
    crud.delete_memory_record(db, record.id)
    assert memory._recall_memories(db, "preference", 5) == []


@pytest.mark.parametrize("failure", ["invalid_json", "length", "offline"])
def test_failed_extraction_keeps_receipt_retryable(db, failure):
    from app.core.llm_client import IncompleteResponseError

    class FailingLLM:
        async def chat(self, **kwargs):
            if failure == "invalid_json":
                return '{"candidates": ['
            raise IncompleteResponseError("length") if failure == "length" else RuntimeError("offline")

    report, _, _ = add_turn(db, MemoryWriter(FailingLLM()), "以后请举例")
    assert report.reasons[0].startswith("memory_write_failed:")
    assert not crud.list_memory_records(db)
    receipt = db.scalar(select(MemoryEvidence).where(MemoryEvidence.evidence_type == "turn"))
    assert not json.loads(receipt.payload)["immediate_done"]
    assert db.get(MemoryProcessingState, "s").batch_turn_count == 0


def test_two_explicit_facts_in_one_message_keep_separate_evidence(db):
    def extract(payload):
        uid = payload["new_user_ids"][0]
        return [candidate(db, uid, content="解释时举例"),
                candidate(db, uid, kind="learning_goal", content="备考NLP")]

    report, _, _ = add_turn(db, MemoryWriter(WriterLLM(extract)), "以后讲解时举例，我的长期目标是通过NLP考试")
    assert report.added == 2
    assert len(crud.list_memory_records(db)) == 2


def test_one_error_is_only_evidence(db):
    def extract(payload):
        if payload["mode"] != "batch":
            return []
        return [graded_candidate(db, uid, payload) for uid in payload["new_user_ids"]
                if db.get(ChatMessage, uid).task_type == "grade"]

    writer = MemoryWriter(WriterLLM(extract))
    add_turn(db, writer, "CRF怎么归一化|局部", "答案错误", task="grade",
             sources=[{"chunk_id": "course-1", "text": "CRF uses global normalization"}])
    for _ in range(5):
        report, _, _ = add_turn(db, writer)
    assert not crud.list_memory_records(db)
    assert "awaiting_two_independent_questions" in report.reasons
    assert len(list(db.scalars(select(MemoryEvidence).where(MemoryEvidence.evidence_type == "graded_error")))) == 1


def test_forgotten_source_cannot_be_repromoted_by_batch_paraphrase(db):
    def extract(payload):
        items = []
        for uid in payload["new_user_ids"]:
            text = db.get(ChatMessage, uid).content
            if text == "以后举例":
                items.append(candidate(db, uid, content="举例偏好" if payload["mode"] == "immediate" else "喜欢配合例子讲解"))
            elif text == "忘掉这个偏好":
                items.append(candidate(db, uid, basis="correction", correction="forget"))
        return items

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        if payload["candidate"]["correction"] == "forget":
            result.update(action="remove", target_ids=[payload["existing"][0]["id"]], relation="contradicts")
        return result

    writer = MemoryWriter(WriterLLM(extract, decide))
    add_turn(db, writer, "以后举例")
    add_turn(db, writer, "忘掉这个偏好")
    for _ in range(4):
        report, _, _ = add_turn(db, writer)
    assert report.batch_processed
    assert not crud.list_memory_records(db)
    assert "retired_source_not_repromoted" in report.reasons


def test_explicit_correction_retires_old_evidence(db):
    def extract(payload):
        uid = payload["new_user_ids"][0]
        correction = db.get(ChatMessage, uid).content == "之后改成简短定义"
        return [candidate(db, uid, basis="correction" if correction else "explicit_statement",
                          correction="replace" if correction else "none", content="简短定义" if correction else "举例")]

    def decide(payload):
        result = WriterLLM.default_decision(payload)
        if payload["existing"]:
            result.update(action="update", target_ids=[payload["existing"][0]["id"]], relation="contradicts")
        return result

    writer = MemoryWriter(WriterLLM(extract, decide))
    add_turn(db, writer, "以后举例")
    add_turn(db, writer, "之后改成简短定义")
    evidence = list(db.scalars(select(MemoryEvidence).where(MemoryEvidence.evidence_type != "turn")))
    assert sorted(row.evidence_type for row in evidence) == ["correction", "retired"]


def test_deleting_session_keeps_memory_but_removes_turn_receipts(db):
    writer = MemoryWriter(WriterLLM(lambda p: [candidate(db, p["new_user_ids"][0])]))
    add_turn(db, writer, "以后请举例")
    crud.delete_session(db, "s")
    assert len(crud.list_memory_records(db)) == 1
    assert db.get(MemoryProcessingState, "s") is None
    assert not list(db.scalars(select(MemoryEvidence).where(MemoryEvidence.evidence_type == "turn")))


def test_additive_table_creation_preserves_existing_memory():
    from sqlalchemy import inspect

    engine = create_engine("sqlite:///:memory:")
    MemoryRecord.__table__.create(engine)
    with Session(engine) as db:
        db.add(MemoryRecord(id="legacy", kind="preference", content="existing memory", importance=3))
        db.commit()
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        assert db.get(MemoryRecord, "legacy").content == "existing memory"
        assert {"memory_evidence", "memory_processing_states"}.issubset(inspect(engine).get_table_names())
    engine.dispose()
