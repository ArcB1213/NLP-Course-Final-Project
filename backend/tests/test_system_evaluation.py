import asyncio
import copy
import json

import pytest

from app.config import Settings
from app.evaluation.system import SystemEvaluator, SystemRunConfig, read_export, select_cases, summarize


def bundle(dataset_id="d1", case_id="case1", human=False):
    return {"metadata": {"id": dataset_id, "config": {"name": dataset_id}}, "samples": [{
        "id": "s1", "dataset_id": dataset_id, "case_id": case_id, "kind": "fact", "split": "test",
        "generated": {"question": "注意力机制的作用是什么？", "key_points": ["分配权重"],
                      "evidence": [{"chunk_id": "c1", "quote": "注意力分配权重。"}]},
        "sources": [{"chunk_id": "c1", "text": "注意力分配权重。"}],
        "judgments": [{"overall": {"label": "sufficient"}}],
        "review": {"status": "corrected", "overall": "partial", "chunks": {}} if human else None,
    }]}


def config(**kwargs):
    return SystemRunConfig(judge_model="judge", **kwargs)


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def __call__(self, question, options):
        self.calls.append((question, options))
        if self.fail:
            raise RuntimeError("sensitive error body")
        return {"task_type": "qa", "answer": "注意力分配权重。",
                "sources": [{"chunk_id": "c1", "document_id": "doc", "source_file": "course.txt",
                             "text": "注意力分配权重。", "final_score": 1.0}],
                "confidence": "high", "message": None, "agent_trace": {"steps": []}}


class FakeJudge:
    base_url = "https://example.test/v1"

    def __init__(self):
        self.calls = []
        self.fail = False

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            return "not json"
        return json.dumps({"relevance": 2, "groundedness": 2, "coverage": 2,
                           "response_mode": "answered", "evidence_sufficiency": "sufficient",
                           "verdict": "acceptable", "reason": "原文支持答案", "issues": []})


@pytest.fixture
def evaluator(tmp_path):
    return SystemEvaluator(Settings(_env_file=None, eval_dir=tmp_path), runner=FakeRunner(),
                           judge=FakeJudge(), snapshot=lambda: {"version": 1})


def step(evaluator, run_id):
    return asyncio.run(evaluator.step(run_id))


def test_multiple_datasets_deduplicate_cases_preserve_label_provenance():
    first, second = bundle(), bundle("d2", human=True)
    third = bundle("d3", "case2")
    cases, skipped = select_cases([first, second, third], config())
    assert len(cases) == 2
    assert [m["annotation_source"] for m in cases[0]["memberships"]] == ["machine", "human"]
    assert len(cases[0]["memberships"]) == 2
    assert skipped == {"d1": {}, "d2": {}, "d3": {}}


def test_selection_filters_unfinished_excluded_and_unreviewed():
    data = bundle()
    excluded = copy.deepcopy(data["samples"][0])
    excluded.update(id="s2", case_id="case2", review={"status": "excluded"})
    unfinished = copy.deepcopy(data["samples"][0])
    unfinished.update(id="s3", case_id="case3", generated=None)
    data["samples"].extend([excluded, unfinished])
    cases, skipped = select_cases([data], config())
    assert len(cases) == 1 and skipped["d1"] == {"excluded": 1, "not_generated": 1}
    with pytest.raises(ValueError):
        select_cases([data], config(reviewed_only=True))
    with pytest.raises(ValueError):
        select_cases([data], config(split="calibration"))


def test_snapshot_conflicts_and_duplicate_dataset_selection_rejected():
    first, second = bundle(), bundle("d2")
    second["samples"][0]["generated"]["question"] = "不同的问题"
    with pytest.raises(ValueError):
        select_cases([first, second], config())
    with pytest.raises(ValueError):
        select_cases([first, first], config())


def test_resume_judging_does_not_regenerate_answer(evaluator):
    run = evaluator.create([bundle(), bundle("d2", human=True)], config())
    assert step(evaluator, run["id"])["stage"] == "system"
    evaluator.judge.fail = True
    assert step(evaluator, run["id"])["status"] == "error"
    assert step(evaluator, run["id"])["status"] == "idle"
    evaluator.retry_errors(run["id"])
    evaluator.judge.fail = False
    assert step(evaluator, run["id"])["stage"] == "judge"
    assert len(evaluator.runner.calls) == 1
    row = evaluator.store.samples(run["id"])[0]
    assert row["failure_history"][0]["raw_response"] == "not json"
    report = evaluator.report(run["id"])
    assert report["overall_unique_cases"]["total"] == 1
    assert all(x["assessed"] == 1 for x in report["by_dataset"].values())
    assert report["human_reviewed_cases"]["total"] == 1
    assert report["machine_only_cases"]["total"] == 0
    user_prompt = evaluator.judge.calls[-1]["messages"][1]["content"]
    assert "confidence" not in user_prompt and "final_score" not in user_prompt
    assert "annotation" not in user_prompt


def test_system_failure_does_not_become_a_scored_answer_and_next_case_runs(evaluator):
    run = evaluator.create([bundle(), bundle("d2", "case2")], config())
    evaluator.runner.fail = True
    assert step(evaluator, run["id"])["status"] == "error"
    assert evaluator.store.samples(run["id"])[0]["error"]["message"] == "sensitive error body"
    evaluator.runner.fail = False
    assert step(evaluator, run["id"])["stage"] == "system"
    assert step(evaluator, run["id"])["stage"] == "judge"
    report = evaluator.report(run["id"])["overall_unique_cases"]
    assert report["answered"] == 1 and report["assessed"] == 1
    assert report["errors"] == {"system": 1}
    assert report["scores"]["relevance"]["count"] == 1


def test_agent_llm_error_is_not_scored(evaluator):
    async def failed_response(question, options):
        return {"task_type": "qa", "answer": "接口错误", "sources": [],
                "confidence": "high", "message": "llm_error"}
    evaluator.runner = failed_response
    run = evaluator.create([bundle()], config())
    assert step(evaluator, run["id"])["status"] == "error"
    row = evaluator.store.samples(run["id"])[0]
    assert row["response"] is None
    assert row["error"]["message"] == "接口错误"
    assert row["error"]["response"]["message"] == "llm_error"
    assert row["failure_history"][0] == row["error"]
    assert not evaluator.judge.calls


def test_version_drift_blocks_resume(evaluator):
    run = evaluator.create([bundle()], config())
    evaluator.snapshot = lambda: {"version": 2}
    with pytest.raises(ValueError, match="已改变"):
        step(evaluator, run["id"])
    assert not evaluator.store.get(run["id"])["busy"]


def test_report_null_scores_and_zero_denominators():
    assert summarize([])["acceptable_rate"] is None
    row = {"response": {"sources": [], "confidence": "low"},
           "assessment": {"relevance": 2, "groundedness": None, "coverage": None,
                          "response_mode": "refused", "verdict": "uncertain"},
           "error": None, "kind": "missing_detail", "system_seconds": 1.0}
    report = summarize([row])
    assert report["scores"]["groundedness"] == {"mean_0_to_2": None, "count": 0}
    assert report["known_reference_hit_rate"] is None
    assert report["high_unacceptable_rate"] is None
    assert report["system_latency_seconds"]["p95"] == 1.0


def test_import_bom_jsonl_export_results_and_preserve_input(evaluator, tmp_path):
    source = bundle()
    path = tmp_path / "input.jsonl"
    rows = [{"record_type": "metadata", **source["metadata"]},
            {"record_type": "sample", **source["samples"][0]}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8-sig")
    loaded = read_export(path)
    original = copy.deepcopy(loaded)
    run = evaluator.create([loaded], config())
    step(evaluator, run["id"])
    step(evaluator, run["id"])
    evaluator.export(run["id"], tmp_path / "exports")
    assert loaded == original
    report = json.loads((tmp_path / "exports" / f"{run['id']}-report.json").read_text(encoding="utf-8"))
    assert report["overall_unique_cases"]["assessed"] == 1
    results = (tmp_path / "exports" / f"{run['id']}-results.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(results[1])["record_type"] == "result"


def test_run_agent_uses_real_handle_with_isolated_request(monkeypatch):
    from app.evaluation import system
    from app.db import database
    from app.core.schemas import ChatResponse

    captured = {}
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Agent:
        async def handle(self, db, request):
            captured["request"] = request
            return ChatResponse(task_type="qa", answer="answer", sources=[], confidence="low")
    monkeypatch.setattr(database, "SessionLocal", Session)
    monkeypatch.setattr(system, "isolated_agent", lambda: Agent())
    result = asyncio.run(system.run_agent("question", config(top_k=7).model_dump()))
    assert result["answer"] == "answer"
    assert captured["request"].task_type == "auto"
    assert captured["request"].session_id is None
    assert captured["request"].top_k == 7
    assert captured["request"].extra_context["debug_agent_trace"] is True
