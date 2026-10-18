import asyncio

import pytest

from app.core.agent_service import AgentService, PreparedSubtask, _should_fallback_to_memory
from app.core.memory_service import LoadedMemoryContext, MemoryResolvedRequest
from app.core.memory_writer import WriteReport
from app.core.schemas import AgentState, AgentSubtask, ChatRequest, ChatResponse, RetrievedChunk


class ExplodingRouter:
    llm = None

    async def decide(self, *args, **kwargs):
        raise AssertionError("router should not be called for memory_only requests")


class FakeTool:
    name = "qa"
    retriever = object()
    force_pro_model = False


class FakeMemoryService:
    def __init__(self):
        self.remembered = []
        self.metadata = []

    def load_context(self, db, session_id, query):
        return LoadedMemoryContext(session_id=session_id or "session-1", recent_messages=[])

    async def resolve_request(self, context, query):
        return MemoryResolvedRequest(
            answer_query="recall previous generated questions",
            retrieval_query=None,
            mode="memory_only",
            referenced_turns=["assistant generated questions"],
            confidence="high",
            reason="recall previous conversation",
        )

    async def answer_from_memory(self, *, context, query, retrieval_note=None):
        return "这是基于会话记忆的回答"

    async def stream_answer_from_memory(self, **kwargs):
        yield "会话记忆回答"

    async def remember_turn(self, db, *, session_id, user_query, assistant_answer, task_type, **kwargs):
        self.remembered.append((session_id, user_query, assistant_answer, task_type))
        self.metadata.append(kwargs)
        return WriteReport()


def build_service(memory_service):
    tool = FakeTool()
    return AgentService(
        router=ExplodingRouter(),
        qa_tool=tool,
        summary_tool=tool,
        quiz_tool=tool,
        grading_tool=tool,
        memory_service=memory_service,
    )


def test_memory_only_handle_skips_rag_pipeline():
    memory_service = FakeMemoryService()
    service = build_service(memory_service)
    request = ChatRequest(query="刚才出过3道题，你不记得了吗", task_type="auto", session_id="session-1")

    response = asyncio.run(service.handle(None, request))

    assert response.task_type == "memory"
    assert response.answer == "这是基于会话记忆的回答"
    assert response.sources == []
    assert response.session_id == "session-1"
    assert memory_service.remembered


def test_memory_fallback_only_for_rag_with_memory_refusals():
    prepared = [
        PreparedSubtask(
            subtask=None,
            tool=None,
            chunks=[],
            confidence="low",
            use_pro_model=False,
            refusal_message="no evidence",
        )
    ]

    assert _should_fallback_to_memory(
        prepared,
        MemoryResolvedRequest(answer_query="explain previous question", mode="rag_with_memory"),
    )
    assert not _should_fallback_to_memory(
        prepared,
        MemoryResolvedRequest(answer_query="CRF 是什么", mode="normal"),
    )


def test_memory_only_stream_delivers_final_write_statistics():
    memory = FakeMemoryService()
    service = build_service(memory)

    async def run():
        return [event async for event in service.handle_stream(None, ChatRequest(
            query="回忆上文", extra_context={"debug_agent_trace": True},
        ))]

    events = asyncio.run(run())
    assert events[-1]["type"] == "done"
    output = events[-1]["agent_trace"]["steps"][-1]["output"]
    assert output["history_saved"] and output["added"] == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_both_transports_pass_grading_reliability_to_memory(stream, failed):
    class GradeMemory(FakeMemoryService):
        async def resolve_request(self, context, query):
            return MemoryResolvedRequest(answer_query=query, retrieval_query=query)

    memory = GradeMemory()
    service = build_service(memory)
    chunk = RetrievedChunk(chunk_id="source-1", document_id="doc", text="CRF全局归一化", source_file="course", final_score=1)

    class GradeTool:
        name = "grade"

        async def generate_with_chunks(self, **kwargs):
            return ChatResponse(task_type="grade", answer="批改内容", sources=[chunk], confidence="high",
                                message="llm_error" if failed else None)

        async def stream_with_chunks(self, **kwargs):
            yield {"type": "delta", "text": "批改内容"}
            yield {"type": "error", "message": "failed"} if failed else {"type": "done"}

    async def prepare(db, request):
        return AgentState(original_query=request.query, task_type="grade", retrieval_profile="qa", use_pro_model=False), [
            PreparedSubtask(AgentSubtask(task_type="grade", query=request.query), GradeTool(), [chunk], "high", False)
        ]

    service._prepare = prepare

    async def run():
        request = ChatRequest(query="批改答案", task_type="grade", extra_context={"debug_agent_trace": True})
        if stream:
            return [event async for event in service.handle_stream(None, request)]
        return await service.handle(None, request)

    asyncio.run(run())
    assert memory.metadata[0]["confidence"] == "high"
    assert memory.metadata[0]["sources"][0]["chunk_id"] == "source-1"
    assert memory.metadata[0]["message"] == ("llm_error" if failed else None)
