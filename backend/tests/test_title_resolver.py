import asyncio
import json

import pytest

from app.config import Settings
from app.core.schemas import RawPage
from app.core.llm_client import IncompleteResponseError
from app.document.splitter import ChineseTextSplitter
from app.document.title_resolver import resolve_titles, title_inputs, validate_decisions


def page(number=1):
    return RawPage(page=number, text="## Explanation\n\nA body paragraph.\n\n$$x_i=y_i$$", metadata={
        "heading": "Explanation", "warnings": [], "title_candidates": [
            {"block_id": 0, "type": "header", "bbox": [0.1, 0.02, 0.3, 0.08], "text": "Topic Alpha"},
            {"block_id": 1, "type": "header", "bbox": [0.4, 0.02, 0.6, 0.08], "text": "Topic Beta"},
            {"block_id": 2, "type": "paragraph_title", "bbox": [0.1, 0.2, 0.4, 0.25], "text": "Explanation"},
        ]})


def decision(number=1, **changes):
    result = {"page": number, "page_title_ids": [0], "section_ids": [], "navigation_ids": [],
              "subtitle_ids": [1], "section_basis": "unknown", "reason": "Text and position evidence"}
    result.update(changes)
    return result


class FakeLLM:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.decisions, Exception):
            raise self.decisions
        return json.dumps({"pages": self.decisions})


def test_header_title_is_restored_and_context_reaches_formula_chunks(tmp_path):
    original = page()
    llm = FakeLLM([decision()])
    result = asyncio.run(resolve_titles([original], Settings(_env_file=None), llm=llm, cache_dir=tmp_path))[0]
    assert result.metadata["heading"] == "Topic Alpha"
    assert result.metadata["subtitle"] == "Topic Beta"
    assert result.text.startswith("# Topic Alpha\n")
    assert original.metadata["heading"] == "Explanation"
    chunks = ChineseTextSplitter(chunk_size=25, chunk_overlap=0).split_pages([result], "doc", "slides.pdf")
    assert all("页面标题：Topic Alpha" in c.text and c.heading == "Topic Alpha" for c in chunks)
    assert all("副标题：Topic Beta" in c.text for c in chunks)
    assert any("$$x_i=y_i$$" in c.text for c in chunks)
    assert llm.calls[0][1]["require_complete"] is True
    assert all(isinstance(m["content"], str) for m in llm.calls[0][0])
    assert "image_url" not in llm.calls[0][0][1]["content"]


def test_body_title_keeps_navigation_out_and_marks_section_as_inference():
    llm = FakeLLM([decision(page_title_ids=[2], subtitle_ids=[], navigation_ids=[0, 1],
                            section_ids=[1], section_basis="inferred")])
    result = asyncio.run(resolve_titles([page()], Settings(_env_file=None), llm=llm))[0]
    assert result.metadata["heading"] == "Explanation"
    assert result.metadata["section_heading"] == "Topic Beta"
    assert "Topic Alpha" not in result.text
    assert "inferred" in result.metadata["retrieval_context"]


@pytest.mark.parametrize("change", [
    {"page_title_ids": [999]}, {"page_title_ids": [True]},
    {"page_title_ids": [0, 0]}, {"navigation_ids": [0]},
    {"page": 9}, {"section_ids": [1], "subtitle_ids": []},
    {"title": "invented title"},
])
def test_invalid_model_output_is_rejected(change):
    with pytest.raises(ValueError):
        validate_decisions({"pages": [decision(**change)]}, title_inputs([page()]))


def test_duplicate_page_decisions_are_rejected():
    with pytest.raises(ValueError):
        validate_decisions({"pages": [decision(), decision()]}, title_inputs([page(), page(2)]))


def test_failure_keeps_text_and_candidates_without_false_title_or_cache(tmp_path):
    original = page()
    llm = FakeLLM(RuntimeError("unavailable"))
    result = asyncio.run(resolve_titles([original], Settings(_env_file=None), llm=llm, cache_dir=tmp_path))[0]
    assert result.text == original.text
    assert result.metadata["heading"] is None
    assert result.metadata["title_candidates"] == original.metadata["title_candidates"]
    assert "title_resolution_failed" in result.metadata["warnings"]
    assert not list(tmp_path.glob("*.json"))


def test_unknown_title_does_not_use_first_body_heading():
    llm = FakeLLM([decision(page_title_ids=[], subtitle_ids=[])])
    result = asyncio.run(resolve_titles([page()], Settings(_env_file=None), llm=llm))[0]
    assert result.metadata["heading"] is None
    assert result.metadata["title_resolution"]["status"] == "uncertain"


def test_cache_reuses_only_matching_input_and_model(tmp_path):
    llm = FakeLLM([decision()])
    settings = Settings(_env_file=None)
    for _ in range(2):
        asyncio.run(resolve_titles([page()], settings, llm=llm, cache_dir=tmp_path))
    assert len(llm.calls) == 1
    settings.deepseek_model = "another-model"
    asyncio.run(resolve_titles([page()], settings, llm=llm, cache_dir=tmp_path))
    changed = page()
    changed.metadata["title_candidates"][0]["text"] = "Changed title"
    asyncio.run(resolve_titles([changed], settings, llm=llm, cache_dir=tmp_path))
    assert len(llm.calls) == 3


def test_disabled_resolver_does_not_call_llm():
    llm = FakeLLM(RuntimeError("must not call"))
    assert asyncio.run(resolve_titles([page()], Settings(_env_file=None, pdf_title_resolution=False), llm=llm))[0].text == page().text
    assert not llm.calls


def test_invalid_roles_get_one_validated_repair_and_keep_attempts(tmp_path):
    class RepairLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            self.calls.append((list(messages), kwargs))
            value = decision(navigation_ids=[0]) if len(self.calls) == 1 else decision()
            return json.dumps({"pages": [value]})

    llm = RepairLLM([])
    result = asyncio.run(resolve_titles([page()], Settings(_env_file=None), llm=llm, cache_dir=tmp_path))[0]
    assert result.metadata["heading"] == "Topic Alpha"
    assert len(llm.calls) == 2 and "校验" in llm.calls[1][0][-1]["content"]
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "attempts").glob("*.json")]
    assert {a["status"] for a in attempts} == {"rejected", "validated"}
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_repeated_invalid_output_falls_back_after_two_calls(tmp_path):
    llm = FakeLLM([decision(page_title_ids=[999])])
    result = asyncio.run(resolve_titles([page()], Settings(_env_file=None), llm=llm, cache_dir=tmp_path))[0]
    assert result.metadata["title_resolution"]["status"] == "failed"
    assert len(llm.calls) == 2
    assert not list(tmp_path.glob("*.json"))


def test_candidate_free_pages_do_not_call_llm():
    plain = RawPage(text="Body", page=1)
    llm = FakeLLM(RuntimeError("must not call"))
    assert asyncio.run(resolve_titles([plain], Settings(_env_file=None), llm=llm)) == [plain]
    assert not llm.calls


def test_four_page_windows_remain_bounded_and_ordered():
    class WindowLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            rows = json.loads(messages[1]["content"])
            return json.dumps({"pages": [decision(number=r["page"]) for r in reversed(rows)]})

    llm = WindowLLM([])
    result = asyncio.run(resolve_titles([page(n) for n in range(1, 7)], Settings(_env_file=None), llm=llm))
    assert [p.page for p in result] == list(range(1, 7))
    assert [len(json.loads(call[0][1]["content"])) for call in llm.calls] == [4, 2]


@pytest.mark.parametrize("recover", [True, False])
def test_truncated_output_has_one_larger_retry_and_audit(tmp_path, recover):
    class TruncatedLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            self.calls.append((list(messages), kwargs))
            if len(self.calls) == 1 or not recover:
                raise IncompleteResponseError("budget exhausted", finish_reason="length",
                                              usage={"completion_tokens": kwargs["max_tokens"]},
                                              max_tokens=kwargs["max_tokens"])
            return json.dumps({"pages": [decision()]})

    llm = TruncatedLLM([])
    result = asyncio.run(resolve_titles([page()], Settings(_env_file=None), llm=llm, cache_dir=tmp_path))[0]
    assert [call[1]["max_tokens"] for call in llm.calls] == [6000, 12000]
    assert all(call[1]["require_complete"] and call[1]["thinking"] for call in llm.calls)
    assert llm.calls[0][0] == llm.calls[1][0]
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "attempts").glob("*.json")]
    first = next(a for a in attempts if a["attempt"] == 1)
    assert first["finish_reason"] == "length" and first["usage"]["completion_tokens"] == 6000
    assert len(list(tmp_path.glob("*.json"))) == int(recover)
    assert result.metadata["title_resolution"]["status"] == ("resolved" if recover else "failed")


def test_other_incomplete_reasons_do_not_increase_budget(tmp_path):
    llm = FakeLLM(IncompleteResponseError("filtered", finish_reason="content_filter"))
    result = asyncio.run(resolve_titles([page()], Settings(_env_file=None), llm=llm, cache_dir=tmp_path))[0]
    assert len(llm.calls) == 1
    assert result.metadata["title_resolution"]["status"] == "failed"
    audit = json.loads(next((tmp_path / "attempts").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["finish_reason"] == "content_filter"
