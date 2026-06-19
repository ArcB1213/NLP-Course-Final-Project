import asyncio

from app.core.agent_service import AgentService, PreparedSubtask, _should_fallback_to_memory
from app.core.memory_service import LoadedMemoryContext, MemoryResolvedRequest
from app.core.schemas import ChatRequest


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

    async def remember_turn(self, db, *, session_id, user_query, assistant_answer, task_type):
        self.remembered.append((session_id, user_query, assistant_answer, task_type))


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
