import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import httpx
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.conversation_summary import (
    CONTEXT_CHARS, completed_turns, covered_count, read_summary, update_summary,
)
from app.core.llm_client import DeepSeekClient
from app.core.memory_writer import MemoryWriter, WriteReport
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


MEMORY_RESOLVE_SYSTEM_PROMPT = """
You are the memory intent resolver for a Chinese NLP course assistant.
Classify the current user request and resolve any references to prior dialogue.

Return JSON only:
{
  "mode": "normal|rag_with_memory|memory_only",
  "answer_query": "standalone request for answer generation",
  "retrieval_query": "short standalone course-knowledge query for RAG, or empty for memory_only",
  "referenced_turns": ["exact source message IDs used or needed from the history catalog"],
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
History catalog entries locate omitted originals. If original content is needed,
select its message ID in referenced_turns; do not invent its contents. If the
reference is ambiguous, ask for clarification in answer_query, not a guessed topic.
""".strip()


MEMORY_ANSWER_PROMPT = """
You are a Chinese NLP course assistant with conversation memory.
Answer the user's follow-up using the conversation memory. If the user asks
about content generated earlier, recall it directly and explain it clearly.
Do not claim course-document citations from memory. If course retrieval was
insufficient, say the explanation is based on conversation memory and general
course knowledge rather than retrieved course evidence.
Use the original message when available. If the requested original is absent,
ask for clarification instead of reconstructing exact questions from a summary.
""".strip()


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
    history: list[ChatMessage] = field(default_factory=list)
    referenced_messages: list[ChatMessage] = field(default_factory=list)

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
        self.writer = MemoryWriter(llm, settings.memory_batch_interval)
        self._index_snapshot = None

    def load_context(self, db: Session, session_id: str | None, query: str) -> LoadedMemoryContext:
        session = crud.get_or_create_session(db, session_id)
        if not self.settings.memory_enabled:
            return LoadedMemoryContext(session_id=session.id)

        summary = crud.get_conversation_summary(db, session.id)
        stored = read_summary(summary.content) if summary else None
        history = crud.list_session_messages(db, session.id)
        covered = covered_count(history, summary.last_message_id) if summary and stored else 0
        recent = history[covered:]
        memories = self._recall_memories(db, query, self.settings.memory_recall_top_k)
        return LoadedMemoryContext(
            session_id=session.id,
            summary=stored.state.model_dump_json() if stored and covered else None,
            recent_messages=recent,
            long_term_memories=memories,
            history=history,
        )

    async def resolve_request(self, context: LoadedMemoryContext, query: str) -> MemoryResolvedRequest:
        if not context.has_context:
            return MemoryResolvedRequest(answer_query=query, retrieval_query=query, mode="normal", confidence="high")
        initial_context = self.build_context_text(context)
        resolution = await self.resolver.resolve(initial_context, query)
        by_id = {message.id: message for message in context.history}
        context.referenced_messages = [by_id[mid] for mid in dict.fromkeys(resolution.referenced_turns) if mid in by_id]
        if context.referenced_messages:
            # Re-resolve after loading originals, so archive labels cannot become fabricated answers.
            if any(f"[message_id={msg.id} role={msg.role}]" not in initial_context for msg in context.referenced_messages):
                resolution = await self.resolver.resolve(self.build_context_text(context), query)
            if resolution.mode == "rag_with_memory":
                resolution.answer_query += "\n\nReferenced conversation originals:\n" + _format_messages(context.referenced_messages)
        return resolution

    def build_context_text(self, context: LoadedMemoryContext) -> str:
        conversation_parts: list[str] = []
        if context.summary:
            conversation_parts.append(f"Summary: {context.summary}")
        selected_ids = {msg.id for msg in context.referenced_messages}
        selected = _format_messages(context.referenced_messages)
        if selected:
            conversation_parts.append("Referenced originals:\n" + selected)
        budget = CONTEXT_CHARS - sum(len(part) for part in conversation_parts)
        recent_parts: list[str] = []
        for turn in reversed(completed_turns(context.recent_messages)):
            text = _format_messages([msg for msg in turn if msg.id not in selected_ids])
            if len(text) > budget:
                break
            recent_parts.insert(0, text)
            selected_ids.update(msg.id for msg in turn)
            budget -= len(text)
        if recent_parts:
            conversation_parts.append("Recent turns:\n" + "\n".join(recent_parts))
        omitted = [turn for turn in completed_turns(context.history) if turn[-1].id not in selected_ids]
        if omitted:
            conversation_parts.append("History catalog (originals available by message ID):\n" + "\n".join(
                f"turn={i + 1} user_message_id={turn[0].id} assistant_message_id={turn[-1].id} task={turn[-1].task_type}"
                for i, turn in enumerate(completed_turns(context.history)) if turn in omitted
            ))

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
        confidence: str | None = None,
        message: str | None = None,
        sources: list[dict] | None = None,
    ) -> WriteReport:
        user_message = crud.create_chat_message(
            db,
            session_id=session_id,
            role="user",
            content=user_query,
            task_type=task_type,
            commit=False,
        )
        assistant_message = crud.create_chat_message(
            db,
            session_id=session_id,
            role="assistant",
            content=assistant_answer,
            task_type=task_type,
            commit=False,
        )
        db.commit()
        if not self.settings.memory_enabled:
            return WriteReport(reasons=["memory_disabled"])

        await self._maybe_update_summary(db, session_id)
        report = await self.writer.process_turn(
            db, user_message, assistant_message,
            {"task_type": task_type, "confidence": confidence, "message": message, "sources": sources or []},
        )
        if report.changed:
            try:
                self.rebuild_memory_index(db)
            except (RuntimeError, OSError, ValueError):
                self._index_snapshot = None
                report.reasons.append("memory_index_rebuild_deferred")
        return report

    def rebuild_memory_index(self, db: Session) -> None:
        memories = crud.list_memory_records(db)
        if not memories:
            self.vector_store.build([], None)
            return
        embeddings = self.embedder.encode_texts([memory.content for memory in memories])
        self.vector_store.build([memory.id for memory in memories], embeddings)
        self._index_snapshot = tuple(sorted((m.id, m.content) for m in memories))

    def _recall_memories(self, db: Session, query: str, top_k: int) -> list[MemoryRecord]:
        memories = crud.list_memory_records(db)
        if not memories or top_k <= 0:
            return []
        try:
            if self._index_snapshot != tuple(sorted((m.id, m.content) for m in memories)):
                self.rebuild_memory_index(db)
            query_embedding = self.embedder.encode_query(query)
            results = self.vector_store.search(query_embedding, top_k)
        except (RuntimeError, OSError, ValueError):
            results = []

        valid_ids = {m.id for m in memories}
        memory_ids = [memory_id for memory_id, _score in results if memory_id in valid_ids]
        if not memory_ids:
            ranked = sorted(memories, key=lambda item: (item.importance, item.updated_at), reverse=True)
            memory_ids = [memory.id for memory in ranked[:top_k]]
        crud.mark_memories_used(db, memory_ids)
        return crud.get_memory_records_by_ids(db, memory_ids)

    async def _maybe_update_summary(self, db: Session, session_id: str) -> None:
        await update_summary(
            db, session_id, self.llm,
            first_turn=self.settings.memory_summary_threshold,
            interval=self.settings.memory_summary_interval,
            keep_turns=self.settings.memory_recent_turns,
        )

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
    return "\n".join(f"[message_id={message.id} role={message.role}]\n{message.content}" for message in messages)


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
