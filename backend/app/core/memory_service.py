import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import httpx
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.llm_client import DeepSeekClient
from app.db import crud
from app.db.models import ChatMessage, MemoryRecord
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import FaissVectorStore


MEMORY_CONTEXT_TEMPLATE = """
[Conversation context]
{conversation_context}

Memory is only for understanding the user's context and learning preferences.
Do not treat memory as course evidence, and do not cite it as a source.

[Current user request]
{query}
""".strip()


SUMMARY_PROMPT = """
Update the conversation summary for a Chinese NLP course assistant.
Keep only information useful for future turns: discussed topics, user goals,
generated tasks, unresolved questions, and learning difficulties.
Return a concise Chinese summary.
""".strip()


EXTRACT_MEMORY_PROMPT = """
Extract durable learner memories from the latest conversation turn.
Only keep useful long-term facts such as preferences, learning goals, weak
knowledge points, repeated mistakes, or user notes. Do not store secrets,
API keys, passwords, or one-off wording.

Return JSON only:
{"memories":[{"kind":"preference|learning_goal|weakness|mistake|custom_note","content":"...","importance":1-5}]}
If there is no durable memory, return {"memories":[]}.
""".strip()


MEMORY_RESOLVE_SYSTEM_PROMPT = """
You are the memory intent resolver for a Chinese NLP course assistant.
Classify the current user request and resolve any references to prior dialogue.

Return JSON only:
{
  "mode": "normal|rag_with_memory|memory_only",
  "answer_query": "standalone request for answer generation",
  "retrieval_query": "short standalone course-knowledge query for RAG, or empty for memory_only",
  "referenced_turns": ["brief summaries of prior turns used, or message roles if exact turns are unclear"],
  "confidence": "high|medium|low",
  "reason": "brief reason"
}

Decision rules:
- normal: the request is self-contained and does not require conversation memory.
- memory_only: the user asks to recall, list, compare, confirm, rewrite, or continue prior conversation content.
- rag_with_memory: the user refers to prior conversation content and also asks for course knowledge explanation.

retrieval_query rules:
- For normal and rag_with_memory, write a standalone knowledge query suitable for course-document retrieval.
- Do not include deictic words such as "刚才", "上面", "之前", "第三题", "这道题", "它", or "你记得".
- For memory_only, retrieval_query must be an empty string.
""".strip()


MEMORY_ANSWER_PROMPT = """
You are a Chinese NLP course assistant with conversation memory.
Answer the user's follow-up using the conversation memory. If the user asks
about content generated earlier, recall it directly and explain it clearly.
Do not claim course-document citations from memory. If course retrieval was
insufficient, say the explanation is based on conversation memory and general
course knowledge rather than retrieved course evidence.
""".strip()


SECRET_PATTERNS = [
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]", re.IGNORECASE),
    re.compile(r"token\s*[:=]", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_\-]{40,}"),
]


DEICTIC_RETRIEVAL_TERMS = (
    "刚才",
    "上面",
    "之前",
    "前面",
    "刚刚",
    "上一",
    "第三题",
    "第三道",
    "第二题",
    "第二道",
    "第一题",
    "第一道",
    "这道题",
    "那道题",
    "这些题",
    "那三道",
    "它",
    "记得",
)


@dataclass
class LoadedMemoryContext:
    session_id: str
    summary: str | None = None
    recent_messages: list[ChatMessage] = field(default_factory=list)
    long_term_memories: list[MemoryRecord] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.summary or self.recent_messages or self.long_term_memories)


@dataclass
class MemoryResolvedRequest:
    answer_query: str
    retrieval_query: str | None = None
    mode: Literal["normal", "rag_with_memory", "memory_only"] = "normal"
    referenced_turns: list[str] = field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str | None = None

    @property
    def memory_only(self) -> bool:
        return self.mode == "memory_only"

class MemoryResolutionPayload(BaseModel):
    mode: Literal["normal", "rag_with_memory", "memory_only"] = "normal"
    answer_query: str
    retrieval_query: str = ""
    referenced_turns: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str = ""


class MemoryIntentResolver:
    def __init__(self, llm: DeepSeekClient):
        self.llm = llm

    async def resolve(self, context_text: str, query: str) -> MemoryResolvedRequest:
        if not self.llm.api_key:
            return MemoryResolvedRequest(
                answer_query=query,
                retrieval_query=query,
                mode="normal",
                confidence="low",
                reason="resolver_unavailable_no_api_key",
            )

        try:
            content = await self.llm.chat(
                messages=[
                    {"role": "system", "content": MEMORY_RESOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Conversation memory:\n{context_text}\n\nCurrent request:\n{query}"},
                ],
                temperature=0.0,
                max_tokens=700,
            )
            payload = MemoryResolutionPayload.model_validate(_parse_json_object(content))
        except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
            return MemoryResolvedRequest(
                answer_query=query,
                retrieval_query=query,
                mode="normal",
                confidence="low",
                reason="resolver_failed",
            )

        return _normalize_resolution(payload, query)


class MemoryService:
    def __init__(
        self,
        settings: Settings,
        llm: DeepSeekClient,
        embedder: Embedder,
        vector_store: FaissVectorStore | None = None,
        resolver: MemoryIntentResolver | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store or FaissVectorStore(settings.memory_index_dir)
        self.resolver = resolver or MemoryIntentResolver(llm)

    def load_context(self, db: Session, session_id: str | None, query: str) -> LoadedMemoryContext:
        session = crud.get_or_create_session(db, session_id)
        if not self.settings.memory_enabled:
            return LoadedMemoryContext(session_id=session.id)

        summary = crud.get_conversation_summary(db, session.id)
        recent = crud.list_recent_messages(db, session.id, self.settings.memory_recent_turns * 2)
        memories = self._recall_memories(db, query, self.settings.memory_recall_top_k)
        return LoadedMemoryContext(
            session_id=session.id,
            summary=summary.content if summary else None,
            recent_messages=recent,
            long_term_memories=memories,
        )

    async def resolve_request(self, context: LoadedMemoryContext, query: str) -> MemoryResolvedRequest:
        if not context.has_context:
            return MemoryResolvedRequest(answer_query=query, retrieval_query=query, mode="normal", confidence="high")
        return await self.resolver.resolve(self.build_context_text(context), query)

    def build_context_text(self, context: LoadedMemoryContext) -> str:
        conversation_parts: list[str] = []
        if context.summary:
            conversation_parts.append(f"Summary: {context.summary}")
        if context.recent_messages:
            recent_lines = [f"{msg.role}: {_trim(msg.content, 900)}" for msg in context.recent_messages]
            conversation_parts.append("Recent turns:\n" + "\n".join(recent_lines))

        memory_lines = [
            f"- ({memory.kind}, importance {memory.importance}) {memory.content}"
            for memory in context.long_term_memories
        ]
        if memory_lines:
            conversation_parts.append("Long-term learner memory:\n" + "\n".join(memory_lines))
        return "\n\n".join(conversation_parts) or "None"

    def build_contextual_query(self, context: LoadedMemoryContext, query: str) -> str:
        if not context.has_context:
            return query
        return MEMORY_CONTEXT_TEMPLATE.format(
            conversation_context=self.build_context_text(context),
            query=query,
        )

    async def answer_from_memory(
        self,
        *,
        context: LoadedMemoryContext,
        query: str,
        retrieval_note: str | None = None,
    ) -> str:
        note = f"\n\nRetrieval note: {retrieval_note}" if retrieval_note else ""
        return await self.llm.chat(
            messages=[
                {"role": "system", "content": MEMORY_ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation memory:\n{self.build_context_text(context)}\n\nUser request:\n{query}{note}",
                },
            ],
            temperature=0.2,
            max_tokens=1200,
            continue_on_length=True,
        )

    async def stream_answer_from_memory(
        self,
        *,
        context: LoadedMemoryContext,
        query: str,
        retrieval_note: str | None = None,
    ) -> AsyncIterator[str]:
        note = f"\n\nRetrieval note: {retrieval_note}" if retrieval_note else ""
        async for token in self.llm.chat_stream(
            messages=[
                {"role": "system", "content": MEMORY_ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation memory:\n{self.build_context_text(context)}\n\nUser request:\n{query}{note}",
                },
            ],
            temperature=0.2,
            max_tokens=1200,
            continue_on_length=True,
        ):
            yield token

    async def remember_turn(
        self,
        db: Session,
        *,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        task_type: str,
    ) -> None:
        user_message = crud.create_chat_message(
            db,
            session_id=session_id,
            role="user",
            content=user_query,
            task_type=task_type,
        )
        assistant_message = crud.create_chat_message(
            db,
            session_id=session_id,
            role="assistant",
            content=assistant_answer,
            task_type=task_type,
        )
        if not self.settings.memory_enabled:
            return

        await self._maybe_update_summary(db, session_id)
        await self._extract_long_term_memories(
            db,
            session_id=session_id,
            source_message_id=assistant_message.id,
            user_query=user_query,
            assistant_answer=assistant_answer,
            fallback_source_message_id=user_message.id,
        )

    def rebuild_memory_index(self, db: Session) -> None:
        memories = crud.list_memory_records(db)
        if not memories:
            self.vector_store.build([], None)
            return
        embeddings = self.embedder.encode_texts([memory.content for memory in memories])
        self.vector_store.build([memory.id for memory in memories], embeddings)

    def _recall_memories(self, db: Session, query: str, top_k: int) -> list[MemoryRecord]:
        memories = crud.list_memory_records(db)
        if not memories or top_k <= 0:
            return []
        try:
            query_embedding = self.embedder.encode_query(query)
            results = self.vector_store.search(query_embedding, top_k)
        except (RuntimeError, OSError, ValueError):
            self.rebuild_memory_index(db)
            try:
                query_embedding = self.embedder.encode_query(query)
                results = self.vector_store.search(query_embedding, top_k)
            except (RuntimeError, OSError, ValueError):
                results = []

        memory_ids = [memory_id for memory_id, _score in results]
        if not memory_ids:
            ranked = sorted(memories, key=lambda item: (item.importance, item.updated_at), reverse=True)
            memory_ids = [memory.id for memory in ranked[:top_k]]
        crud.mark_memories_used(db, memory_ids)
        return crud.get_memory_records_by_ids(db, memory_ids)

    async def _maybe_update_summary(self, db: Session, session_id: str) -> None:
        message_count = crud.count_session_messages(db, session_id)
        if message_count < self.settings.memory_summary_threshold:
            return

        current_summary = crud.get_conversation_summary(db, session_id)
        messages = crud.get_session_messages_after(
            db,
            session_id,
            current_summary.last_message_id if current_summary else None,
        )
        if not messages:
            return

        content = _format_messages(messages)
        previous = current_summary.content if current_summary else "None"
        try:
            summary = await self.llm.chat(
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {
                        "role": "user",
                        "content": f"Previous summary:\n{previous}\n\nNew messages:\n{content}",
                    },
                ],
                temperature=0.0,
                max_tokens=600,
            )
        except (RuntimeError, httpx.HTTPError):
            return

        crud.upsert_conversation_summary(
            db,
            session_id=session_id,
            content=summary.strip(),
            last_message_id=messages[-1].id,
        )

    async def _extract_long_term_memories(
        self,
        db: Session,
        *,
        session_id: str,
        source_message_id: str,
        user_query: str,
        assistant_answer: str,
        fallback_source_message_id: str,
    ) -> None:
        if _contains_secret(user_query) or _contains_secret(assistant_answer):
            return
        try:
            content = await self.llm.chat(
                messages=[
                    {"role": "system", "content": EXTRACT_MEMORY_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"User:\n{_trim(user_query, 1200)}\n\n"
                            f"Assistant:\n{_trim(assistant_answer, 1600)}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=500,
            )
            payload = _parse_json_object(content)
        except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
            return

        created = False
        for item in payload.get("memories") or []:
            kind = str(item.get("kind") or "custom_note")
            if kind not in {"preference", "learning_goal", "weakness", "mistake", "custom_note"}:
                kind = "custom_note"
            memory_content = str(item.get("content") or "").strip()
            if not memory_content or _contains_secret(memory_content):
                continue
            if crud.find_memory_by_content(db, memory_content) is not None:
                continue
            importance = item.get("importance") or 3
            crud.create_memory_record(
                db,
                kind=kind,
                content=memory_content,
                importance=int(importance),
                source_session_id=session_id,
                source_message_id=source_message_id or fallback_source_message_id,
            )
            created = True

        if created:
            self.rebuild_memory_index(db)


def _normalize_resolution(payload: MemoryResolutionPayload, original_query: str) -> MemoryResolvedRequest:
    answer_query = payload.answer_query.strip() or original_query
    retrieval_query = payload.retrieval_query.strip()
    mode = payload.mode
    reason = payload.reason

    if mode == "memory_only":
        retrieval_query = ""
    elif not retrieval_query:
        if payload.confidence == "low":
            mode = "memory_only"
            reason = reason or "resolver_low_confidence_without_retrieval_query"
        else:
            retrieval_query = answer_query

    if retrieval_query and _contains_deictic_term(retrieval_query):
        if payload.confidence == "low":
            mode = "memory_only"
            retrieval_query = ""
            reason = reason or "resolver_low_confidence_deictic_retrieval_query"
        else:
            retrieval_query = _strip_deictic_terms(retrieval_query)
            if not retrieval_query.strip():
                mode = "memory_only"
                retrieval_query = ""
                reason = reason or "resolver_invalid_retrieval_query"

    return MemoryResolvedRequest(
        answer_query=answer_query,
        retrieval_query=retrieval_query or None,
        mode=mode,
        referenced_turns=payload.referenced_turns,
        confidence=payload.confidence,
        reason=reason,
    )


def _format_messages(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {_trim(message.content, 1000)}" for message in messages)


def _trim(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in SECRET_PATTERNS)


def _contains_deictic_term(text: str) -> bool:
    return any(term in text for term in DEICTIC_RETRIEVAL_TERMS)


def _strip_deictic_terms(text: str) -> str:
    cleaned = text
    for term in DEICTIC_RETRIEVAL_TERMS:
        cleaned = cleaned.replace(term, "")
    return " ".join(cleaned.split())


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Memory output does not contain a JSON object")
    return json.loads(text[start : end + 1])
