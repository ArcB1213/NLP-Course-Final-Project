import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.llm_client import DeepSeekClient
from app.core.prompts import QA_SYSTEM_PROMPT, build_qa_prompt
from app.core.schemas import ChatResponse
from app.retrieval.retriever import HybridRetriever, estimate_confidence


class QATool:
    def __init__(self, retriever: HybridRetriever, llm: DeepSeekClient, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def run(
        self,
        db: Session,
        query: str,
        use_pro_model: bool = False,
        top_k: int | None = None,
    ) -> ChatResponse:
        chunks = self.retriever.retrieve(db, query, top_k or self.settings.final_top_k)
        confidence = estimate_confidence(chunks)
        if confidence == "low":
            return ChatResponse(
                task_type="qa",
                answer="课程资料中未找到与该问题充分相关的内容。建议补充资料，或换一种更具体的问法。",
                sources=chunks,
                confidence="low",
                message="low_retrieval_confidence",
            )

        prompt = build_qa_prompt(query, chunks)
        model = self.settings.deepseek_pro_model if use_pro_model else self.settings.deepseek_model
        try:
            answer = await self.llm.chat(
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            return ChatResponse(
                task_type="qa",
                answer=f"调用 DeepSeek API 失败：{exc}",
                sources=chunks,
                confidence=confidence,
                message="llm_error",
            )

        return ChatResponse(task_type="qa", answer=answer, sources=chunks, confidence=confidence)
