import asyncio

from app.core.memory_service import MemoryIntentResolver


class FakeLLM:
    api_key = "test-key"

    def __init__(self, content: str | Exception):
        self.content = content

    async def chat(self, **kwargs):
        if isinstance(self.content, Exception):
            raise self.content
        return self.content


def test_resolver_parses_memory_only_response():
    resolver = MemoryIntentResolver(
        FakeLLM(
            """
            {
              "mode": "memory_only",
              "answer_query": "回忆上一轮生成的三道题",
              "retrieval_query": "",
              "referenced_turns": ["assistant generated three quiz questions"],
              "confidence": "high",
              "reason": "user asks to recall previous conversation content"
            }
            """
        )
    )

    result = asyncio.run(resolver.resolve("assistant: generated three questions", "刚才出过3道题，你不记得了吗"))

    assert result.mode == "memory_only"
    assert result.memory_only is True
    assert result.retrieval_query is None
    assert result.referenced_turns == ["assistant generated three quiz questions"]
    assert result.confidence == "high"


def test_resolver_keeps_clean_rag_with_memory_query():
    resolver = MemoryIntentResolver(
        FakeLLM(
            """
            {
              "mode": "rag_with_memory",
              "answer_query": "解释第三题涉及的 CRF 分词知识点",
              "retrieval_query": "CRF 中文分词 序列标注 条件随机场",
              "referenced_turns": ["third quiz question about CRF segmentation"],
              "confidence": "high",
              "reason": "the user refers to a previous question and asks for course explanation"
            }
            """
        )
    )

    result = asyncio.run(resolver.resolve("assistant: third question was about CRF", "解释一下刚才出的第三道题"))

    assert result.mode == "rag_with_memory"
    assert result.retrieval_query == "CRF 中文分词 序列标注 条件随机场"


def test_resolver_downgrades_low_confidence_without_retrieval_query_to_memory_only():
    resolver = MemoryIntentResolver(
        FakeLLM(
            """
            {
              "mode": "rag_with_memory",
              "answer_query": "解释上一题",
              "retrieval_query": "",
              "referenced_turns": [],
              "confidence": "low",
              "reason": "cannot identify the course concept"
            }
            """
        )
    )

    result = asyncio.run(resolver.resolve("assistant: previous answer", "解释上一题"))

    assert result.mode == "memory_only"
    assert result.retrieval_query is None


def test_resolver_failure_falls_back_to_normal():
    resolver = MemoryIntentResolver(FakeLLM(RuntimeError("llm down")))

    result = asyncio.run(resolver.resolve("assistant: previous answer", "CRF 是什么"))

    assert result.mode == "normal"
    assert result.retrieval_query == "CRF 是什么"
    assert result.confidence == "low"
