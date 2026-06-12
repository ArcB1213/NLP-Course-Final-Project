import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.llm_client import DeepSeekClient
from app.core.prompts import GRADING_SYSTEM_PROMPT, build_grading_prompt
from app.core.schemas import ChatResponse
from app.retrieval.retriever import HybridRetriever, estimate_confidence


class GradingTool:
    def __init__(self, retriever: HybridRetriever, llm: DeepSeekClient, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def run(
        self,
        db: Session,
        query: str,
        use_pro_model: bool = True,
        top_k: int | None = None,
        retrieval_query: str | None = None,
        retrieval_profile: str = "summary",
    ) -> ChatResponse:
        chunks = self.retriever.retrieve(
            db,
            retrieval_query or query,
            top_k or self.settings.summary_final_top_k,
            profile=retrieval_profile,
        )
        confidence = estimate_confidence(chunks)
        prompt = build_grading_prompt(query, chunks)
        model = self.settings.deepseek_pro_model if use_pro_model else self.settings.deepseek_model

        try:
            answer = await self.llm.chat(
                messages=[
                    {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
                max_tokens=1600,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            return ChatResponse(
                task_type="grade",
                answer=f"调用 DeepSeek API 失败：{exc}",
                sources=chunks,
                confidence=confidence,
                message="llm_error",
            )

        message = "low_retrieval_confidence" if confidence == "low" else None
        return ChatResponse(task_type="grade", answer=answer, sources=chunks, confidence=confidence, message=message)
