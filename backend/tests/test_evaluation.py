import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.evaluation.schemas import AppendRequest, DatasetRequest, Review
from app.evaluation.service import EvaluationService, metrics
from app.evaluation.store import BusyError, EvaluationStore


CORPUS = [
    {"chunk_id": "c1", "document_id": "d1", "text": "注意力机制根据相关程度分配权重。",
     "source_file": "第一章.txt", "page": 1, "heading": "注意力"},
    {"chunk_id": "c2", "document_id": "d2", "text": "循环网络按顺序处理输入。",
     "source_file": "第二章.txt", "page": 2, "heading": "循环网络"},
]
RETRIEVED = [{**CORPUS[0], "vector_score": 0.2, "bm25_score": 1.0,
              "final_score": 1.0, "rerank_score": 0.05}]


class FakeLLM:
    base_url = "https://example.test/v1"

    def __init__(self, judge=False):
        self.is_judge = judge
        self.calls = []
        self.fail = False
        self.bad_quote = False
        self.omit_chunk = False

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("secret error text should not be persisted")
        payload = json.loads(kwargs["messages"][1]["content"])
        if not self.is_judge:
            source = payload["sources"][0]
            return json.dumps({
                "question": "这段内容解释了什么机制？",
                "key_points": ["解释机制"],
                "evidence": [{"chunk_id": source["chunk_id"], "quote": source["text"]}],
            })
        chunks = payload["contexts"]
        quote = [{"chunk_id": chunks[0]["chunk_id"],
                  "quote": "伪造引用" if self.bad_quote else chunks[0]["text"]}] if chunks else []
        judgment = {"label": "partial" if chunks else "irrelevant", "reason": "缺少完整说明", "evidence": quote}
        return json.dumps({
            "chunks": [] if self.omit_chunk else [
                {"chunk_id": c["chunk_id"], "label": "partial", "reason": "缺少完整说明",
                 "evidence": [{"chunk_id": c["chunk_id"], "quote": c["text"]}]} for c in chunks],
            "overall": judgment,
        })


@pytest.fixture
def service(tmp_path):
    settings = Settings(_env_file=None, eval_dir=tmp_path)
    return EvaluationService(settings, corpus_loader=lambda: list(CORPUS),
                             retriever=lambda *args: list(RETRIEVED),
                             generator=FakeLLM(), judge=FakeLLM(judge=True))


def create(service, **kwargs):
    return service.create(DatasetRequest(
        generation_model="test-flash", judge_model="test-pro", count=1, **kwargs))["id"]


def step(service, dataset_id):
    return asyncio.run(service.step(dataset_id))


def finish(service, dataset_id):
    for _ in range(10):
        result = step(service, dataset_id)
        assert result["status"] != "error", result
        if result["status"] == "idle":
            break


def test_resume_caches_each_stage_and_blinds_judge(service):
    dataset_id = create(service)
    assert step(service, dataset_id)["stage"] == "generate"
    assert step(service, dataset_id)["stage"] == "retrieve"
    # A new store connection sees checkpoints, as after a process/browser restart.
    assert EvaluationStore(service.settings.eval_dir).samples(dataset_id)[0]["retrieved"]
    assert step(service, dataset_id)["stage"] == "judge"
    finish(service, dataset_id)
    assert len(service.generator.calls) == 1
    assert len(service.judge.calls) == 2
    assert service.generator.calls[0]["model"] == "test-flash"
    assert service.judge.calls[0]["model"] == "test-pro"
    for call in service.judge.calls:
        prompt = call["messages"][1]["content"]
        assert "final_score" not in prompt and "rerank_score" not in prompt
        assert "key_points" not in prompt and "confidence" not in prompt
    data = service.detail(dataset_id)
    assert data["progress"]["completed"] == 1
    assert "疑似置信度虚高" in data["samples"][0]["reasons"]


def test_failure_retry_preserves_generation_and_retrieval(service):
    dataset_id = create(service)
    step(service, dataset_id)
    step(service, dataset_id)
    service.judge.fail = True
    assert step(service, dataset_id)["status"] == "error"
    sample = service.detail(dataset_id)["samples"][0]
    assert "secret" not in sample["error"]["message"]
    assert sample["generated"] and sample["retrieved"]
    assert step(service, dataset_id)["status"] == "idle"
    service.judge.fail = False
    service.retry(dataset_id, sample["id"])
    finish(service, dataset_id)
    assert len(service.generator.calls) == 1


@pytest.mark.parametrize("fault", ["bad_quote", "omit_chunk"])
def test_invalid_judgment_is_not_saved(service, fault):
    dataset_id = create(service)
    step(service, dataset_id)
    step(service, dataset_id)
    setattr(service.judge, fault, True)
    assert step(service, dataset_id)["status"] == "error"
    assert not service.detail(dataset_id)["samples"][0]["judgments"]


def test_no_context_is_irrelevant(service):
    service.retriever = lambda *args: []
    dataset_id = create(service, judge_passes=1)
    finish(service, dataset_id)
    sample = service.detail(dataset_id)["samples"][0]
    assert sample["confidence"] == "low"
    assert sample["judgments"][0]["overall"]["label"] == "irrelevant"


def test_corpus_change_requires_new_run(service):
    dataset_id = create(service)
    step(service, dataset_id)
    service.corpus_loader = lambda: CORPUS[:1]
    assert step(service, dataset_id)["status"] == "error"
    assert service.detail(dataset_id)["samples"][0]["retrieved"] is None


def test_configuration_change_requires_new_run(service):
    dataset_id = create(service)
    step(service, dataset_id)
    service.settings.enable_reranker = False
    assert step(service, dataset_id)["status"] == "error"


def test_review_keeps_machine_labels_and_history(service):
    dataset_id = create(service)
    finish(service, dataset_id)
    sample_id = service.detail(dataset_id)["samples"][0]["id"]
    service.review(dataset_id, sample_id, Review(status="accepted"))
    service.review(dataset_id, sample_id, Review(
        status="corrected", overall="sufficient", chunks={"c1": "sufficient"}, note="人工核实"))
    data = service.detail(dataset_id)
    assert data["report"]["all"]["machine"]["high_insufficient_rate"] == 1
    assert data["report"]["all"]["human"]["high_insufficient_rate"] == 0
    assert len(data["samples"][0]["review_history"]) == 2
    service.review(dataset_id, sample_id, Review(status="excluded"))
    data = service.detail(dataset_id)
    assert data["report"]["all"]["machine"]["labeled"] == 0
    assert data["report"]["all"]["human"]["labeled"] == 0


def test_incomplete_review_rejected(service):
    dataset_id = create(service)
    sample_id = service.detail(dataset_id)["samples"][0]["id"]
    with pytest.raises(ValueError):
        service.review(dataset_id, sample_id, Review(status="accepted"))
    finish(service, dataset_id)
    with pytest.raises(ValueError):
        service.review(dataset_id, sample_id, Review(status="corrected", overall="sufficient"))


def test_lease_prevents_concurrent_writers_and_releases_after_error(service):
    dataset_id = create(service)
    with pytest.raises(RuntimeError):
        with service.store.lease(dataset_id):
            with pytest.raises(BusyError):
                step(service, dataset_id)
            raise RuntimeError("test")
    assert not service.store.get(dataset_id)["busy"]
    assert step(service, dataset_id)["status"] == "processed"


def test_clone_reuses_questions_but_not_reviews(service):
    dataset_id = create(service)
    finish(service, dataset_id)
    sample = service.detail(dataset_id)["samples"][0]
    service.review(dataset_id, sample["id"], Review(status="accepted"))
    clone = service.create(DatasetRequest(
        name="对照", generation_model="unused", judge_model="new-judge", count=1), source_id=dataset_id)
    other = clone["samples"][0]
    assert other["case_id"] == sample["case_id"]
    assert other["split"] == sample["split"]
    assert other["generated"] == sample["generated"]
    assert other["review"] is None and other["retrieved"] is None and not other["judgments"]
    finish(service, clone["id"])
    assert len(service.generator.calls) == 1
    assert service.compare(clone["id"], dataset_id)["paired"] == 1


def test_stratified_sampling_and_split_grouping(service):
    result = service.create(DatasetRequest(
        generation_model="g", judge_model="j", count=12, seed=9))
    assert {s["group_id"] for s in result["samples"]} == {"d1", "d2"}
    for doc in ("d1", "d2"):
        assert len({s["split"] for s in result["samples"] if s["group_id"] == doc}) == 1
    assert sum(s["random_audit"] for s in result["samples"]) == 2
    assert {s["kind"] for s in result["samples"]} == {"fact", "paraphrase", "comparison", "missing_detail"}


def test_unknown_labels_are_not_treated_as_errors(service):
    dataset_id = create(service)
    finish(service, dataset_id)
    sample = service.detail(dataset_id)["samples"][0]
    sample["judgments"][0]["overall"]["label"] = "uncertain"
    result = metrics([sample], False)
    assert result["labeled"] == 1 and result["uncertain"] == 1
    assert result["high_insufficient_rate"] is None


def test_api_create_step_review_export_and_not_found(service):
    from app.api.evaluations import router

    app = FastAPI()
    app.include_router(router)
    # Preserve the real dependency's exception handling with a patched factory.
    from unittest.mock import patch
    with patch("app.api.evaluations.get_evaluation_service", return_value=service):
        client = TestClient(app)
        assert client.get("/api/evaluations/datasets/missing").status_code == 404
        response = client.post("/api/evaluations/datasets", json={
            "count": 1, "generation_model": "g", "judge_model": "j", "judge_passes": 1})
        assert response.status_code == 201
        dataset_id = response.json()["id"]
        base = "/api/evaluations/datasets/" + dataset_id
        for _ in range(3):
            assert client.post(base + "/step").json()["status"] == "processed"
        data = client.get(base).json()
        sample_id = data["samples"][0]["id"]
        assert client.put(base + "/samples/" + sample_id + "/review",
                          json={"status": "accepted"}).status_code == 200
        exported = client.get(base + "/export")
        records = [json.loads(line) for line in exported.text.splitlines()]
        assert [r["record_type"] for r in records] == ["metadata", "sample"]
        assert records[1]["review"]["status"] == "accepted"
        with service.store.lease(dataset_id):
            assert client.post(base + "/step").status_code == 409


def test_deepseek_evaluation_explicitly_disables_thinking(service):
    dataset = service.create(DatasetRequest(count=1, generation_model="deepseek-v4-flash",
                                           judge_model="deepseek-v4-pro", judge_passes=1))
    finish(service, dataset["id"])
    assert service.generator.calls[0]["thinking"] is False
    assert service.judge.calls[0]["thinking"] is False


def test_llm_thinking_extension_does_not_change_default_chat(monkeypatch):
    import httpx
    from app.core.llm_client import DeepSeekClient

    payloads = []
    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    client = DeepSeekClient("fake-key", "https://example.test/v1", "deepseek-v4-pro")
    asyncio.run(client.chat([{"role": "user", "content": "test"}]))
    asyncio.run(client.chat([{"role": "user", "content": "test"}], thinking=False))
    assert "thinking" not in payloads[0]
    assert payloads[1]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("quote", [
    "输入输出正好相反，目标函数也有所不同。",
    "输入输出 正好相\t反，目标函数也有所不同。",
])
def test_quote_whitespace_maps_back_to_exact_source(quote):
    from app.evaluation.schemas import Quote
    from app.evaluation.service import validate_quotes

    text = "• 输入输出正好相\n反，目标函数也有所不同。\nCBOW"
    evidence = Quote(chunk_id="c1", quote=quote)
    validate_quotes([evidence], [{"chunk_id": "c1", "text": text}])
    assert evidence.quote == "输入输出正好相\n反，目标函数也有所不同。"
    assert evidence.quote in text


@pytest.mark.parametrize("quote,chunk_id", [
    ("输入输出正好相同。", "c1"),
    ("输入输出正好相反.", "c1"),
    ("输入输出正好相反。", "wrong"),
    (" \n\t", "c1"),
    ("输入输出目标函数", "c1"),
])
def test_quote_rejects_rewrites_punctuation_wrong_source_and_splicing(quote, chunk_id):
    from app.evaluation.schemas import Quote
    from app.evaluation.service import QuoteValidationError, validate_quotes

    text = "输入输出正好相\n反。目标函数不同。"
    with pytest.raises(QuoteValidationError) as caught:
        validate_quotes([Quote(chunk_id=chunk_id, quote=quote)],
                        [{"chunk_id": "c1", "text": text}], "overall.evidence")
    detail = caught.value.details[0]
    assert detail["quote"] == quote and detail["chunk_id"] == chunk_id
    assert detail["scope"] == "overall.evidence"
    assert detail["source_text"] == (text if chunk_id == "c1" else None)


def test_failed_response_diagnostics_survive_retry_and_success(service):
    dataset_id = create(service)
    step(service, dataset_id)
    step(service, dataset_id)
    service.judge.bad_quote = True
    assert step(service, dataset_id)["status"] == "error"
    sample = service.detail(dataset_id)["samples"][0]
    error = sample["error"]
    assert error["judge_pass"] == 1
    assert json.loads(error["raw_response"])["overall"]["evidence"][0]["quote"] == "伪造引用"
    assert error["quote_failures"][0]["source_text"] == RETRIEVED[0]["text"]
    assert not sample["judgments"]
    service.retry(dataset_id, sample["id"])
    service.judge.bad_quote = False
    finish(service, dataset_id)
    sample = service.detail(dataset_id)["samples"][0]
    assert sample["error"] is None and sample["failure_history"] == [error]


def test_manual_review_of_failed_sample_finishes_without_machine_label(service):
    dataset_id = create(service)
    step(service, dataset_id)
    step(service, dataset_id)
    service.judge.bad_quote = True
    step(service, dataset_id)
    sample = service.detail(dataset_id)["samples"][0]
    with pytest.raises(ValueError):
        service.review(dataset_id, sample["id"], Review(status="accepted"))
    service.review(dataset_id, sample["id"], Review(
        status="corrected", overall="partial", chunks={"c1": "partial"}, note="手工核对原文"))
    data = service.detail(dataset_id)
    assert data["progress"]["completed"] == 0
    assert data["progress"]["finished"] == 1 and data["progress"]["pending"] == 0
    assert data["progress"]["errors"] == 0
    assert data["report"]["all"]["machine"]["labeled"] == 0
    assert data["report"]["all"]["human"]["labeled"] == 1
    assert data["samples"][0]["failure_history"]
    assert step(service, dataset_id)["status"] == "idle"
    assert len(service.judge.calls) == 1


def test_accept_first_valid_pass_when_second_pass_fails(service):
    dataset_id = create(service)
    for _ in range(3):
        step(service, dataset_id)
    service.judge.bad_quote = True
    step(service, dataset_id)
    sample = service.detail(dataset_id)["samples"][0]
    assert sample["error"]["judge_pass"] == 2
    service.review(dataset_id, sample["id"], Review(status="accepted"))
    assert step(service, dataset_id)["status"] == "idle"
    assert service.detail(dataset_id)["progress"]["reviewed"] == 1


def test_failed_question_does_not_block_next_question(service):
    dataset_id = service.create(DatasetRequest(
        count=2, generation_model="g", judge_model="j", judge_passes=1))["id"]
    step(service, dataset_id)
    first = service.store.samples(dataset_id)[0]
    first["generated"]["question"] += " 第一题"
    service.store.save_sample(first)
    step(service, dataset_id)
    service.judge.bad_quote = True
    assert step(service, dataset_id)["status"] == "error"
    service.judge.bad_quote = False
    result = step(service, dataset_id)
    assert result["stage"] == "generate" and result["sample_id"] != first["id"]
    finish(service, dataset_id)
    data = service.detail(dataset_id)
    assert data["progress"]["completed"] == 1 and data["progress"]["errors"] == 1


def test_exclude_failed_generation_and_preserve_legacy_error(service):
    dataset_id = create(service)
    sample = service.store.samples(dataset_id)[0]
    legacy = {"stage": "generate", "message": "旧错误", "at": "old"}
    sample["error"] = legacy
    sample.pop("failure_history", None)
    service.store.save_sample(sample)
    service.review(dataset_id, sample["id"], Review(status="excluded"))
    sample = service.store.samples(dataset_id)[0]
    assert sample["failure_history"] == [legacy]
    assert step(service, dataset_id)["status"] == "idle"


def test_api_exports_failure_diagnostics_and_manual_resolution(service):
    from unittest.mock import patch
    from app.api.evaluations import router

    app = FastAPI()
    app.include_router(router)
    with patch("app.api.evaluations.get_evaluation_service", return_value=service):
        client = TestClient(app)
        dataset_id = create(service)
        base = "/api/evaluations/datasets/" + dataset_id
        client.post(base + "/step")
        client.post(base + "/step")
        service.judge.bad_quote = True
        client.post(base + "/step")
        sample = client.get(base).json()["samples"][0]
        response = client.put(base + "/samples/" + sample["id"] + "/review",
                              json={"status": "corrected", "overall": "partial", "chunks": {"c1": "partial"}})
        assert response.status_code == 200
        exported = [json.loads(line) for line in client.get(base + "/export").text.splitlines()]
        assert exported[1]["failure_history"][0]["quote_failures"]
        assert exported[1]["review"]["status"] == "corrected"
        assert exported[1]["judgments"] == []


def test_append_preserves_existing_reviews_and_document_splits(service):
    dataset_id = create(service)
    finish(service, dataset_id)
    first = service.detail(dataset_id)["samples"][0]
    service.review(dataset_id, first["id"], Review(status="accepted"))
    original = service.detail(dataset_id)["samples"]
    for count in (3, 2):
        data = service.append(dataset_id, AppendRequest(count=count))
        assert data["samples"][:len(original)] == original
        assert not data["busy"]
    assert data["config"]["count"] == data["progress"]["total"] == 6
    assert data["progress"]["pending"] == 5
    assert len({s["case_id"] for s in data["samples"]}) == 6
    assert [x["count"] for x in data["append_history"]] == [3, 2]
    for doc in ("d1", "d2"):
        assert len({s["split"] for s in data["samples"] if s["group_id"] == doc}) == 1
    assert len(service.generator.calls) == 1  # Appending only creates plans.


def test_append_beyond_initial_limit_and_unfinished_dataset(service):
    dataset_id = create(service, document_ids=["d1"])
    service.append(dataset_id, AppendRequest(count=200))
    data = service.append(dataset_id, AppendRequest(count=1))
    assert data["config"]["count"] == 202
    assert data["progress"]["generated"] == 0
    assert {s["group_id"] for s in data["samples"]} == {"d1"}
    assert {s["split"] for s in data["samples"]} == {"calibration"}


def test_append_rejects_busy_and_changed_corpus_without_partial_update(service):
    dataset_id = create(service)
    before = service.detail(dataset_id)
    with service.store.lease(dataset_id):
        with pytest.raises(BusyError):
            service.append(dataset_id, AppendRequest(count=2))
    service.corpus_loader = lambda: CORPUS[:1]
    with pytest.raises(ValueError, match="资料库已改变"):
        service.append(dataset_id, AppendRequest(count=2))
    assert service.detail(dataset_id) == before


def test_append_transaction_rolls_back_metadata_on_insert_failure(service):
    import sqlite3
    dataset_id = create(service)
    before = service.store.get(dataset_id)
    modified = {**before, "config": {**before["config"], "count": 2}}
    with service.store.lease(dataset_id):
        with pytest.raises(sqlite3.IntegrityError):
            service.store.append(modified, service.store.samples(dataset_id))
    assert service.store.get(dataset_id) == before
    assert len(service.store.samples(dataset_id)) == 1


def test_appended_generation_retries_duplicates_against_old_and_new_questions(service):
    from app.evaluation.service import question_key
    dataset_id = create(service, judge_passes=1)
    finish(service, dataset_id)
    service.append(dataset_id, AppendRequest(count=2))
    questions = iter([" 这段内容解释了什么机制? ", "循环网络如何处理输入？",
                      "循环网络 如 何处理输入?", "注意力权重有何作用？"])
    original = service.generator.chat

    async def generate(**kwargs):
        result = json.loads(await original(**kwargs))
        result["question"] = next(questions)
        return json.dumps(result)

    service.generator.chat = generate
    finish(service, dataset_id)
    rows = service.store.samples(dataset_id)
    assert len({question_key(s["generated"]["question"]) for s in rows}) == 3
    assert len(service.generator.calls) == 5
    prompt = json.loads(service.generator.calls[-1]["messages"][1]["content"])
    assert rows[0]["generated"]["question"] in prompt["existing_questions"]
    assert rows[1]["generated"]["question"] in prompt["existing_questions"]


def test_exhausted_duplicate_attempts_remain_retryable(service):
    dataset_id = create(service, judge_passes=1)
    finish(service, dataset_id)
    service.append(dataset_id, AppendRequest(count=1))
    result = step(service, dataset_id)
    assert result["status"] == "error" and "重复" in result["error"]
    row = service.store.samples(dataset_id)[-1]
    assert row["generated"] is None
    assert len(service.generator.calls) == 4
    assert row["failure_history"][0]["stage"] == "generate"
    service.retry(dataset_id, row["id"])
    original = service.generator.chat

    async def unique(**kwargs):
        value = json.loads(await original(**kwargs))
        value["question"] = "新问题如何解答？"
        return json.dumps(value)

    service.generator.chat = unique
    finish(service, dataset_id)
    assert service.detail(dataset_id)["progress"]["completed"] == 2


@pytest.mark.parametrize("text", [" ＡＢＣ 是什么？", "abc是什么?", "ABC\n是什么。"])
def test_question_key_normalizes_typographic_variations(text):
    from app.evaluation.service import question_key
    assert question_key(text) == question_key("ABC是什么？")


def test_append_api_validation_and_export(service):
    from app.api.evaluations import router
    from unittest.mock import patch
    app = FastAPI()
    app.include_router(router)
    dataset_id = create(service)
    base = "/api/evaluations/datasets/" + dataset_id
    with patch("app.api.evaluations.get_evaluation_service", return_value=service):
        client = TestClient(app)
        for count in (0, -1, 201, 1.5, True):
            assert client.post(base + "/append", json={"count": count}).status_code == 422
        assert client.post("/api/evaluations/datasets/missing/append", json={"count": 1}).status_code == 404
        with service.store.lease(dataset_id):
            assert client.post(base + "/append", json={"count": 1}).status_code == 409
        response = client.post(base + "/append", json={"count": 2})
        assert response.status_code == 200
        assert response.json()["progress"]["total"] == 3
        records = [json.loads(line) for line in client.get(base + "/export").text.splitlines()]
        assert records[0]["config"]["count"] == 3
        assert len(records) == 4
