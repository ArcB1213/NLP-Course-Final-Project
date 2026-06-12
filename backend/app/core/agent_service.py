from sqlalchemy.orm import Session

from app.core.router_agent import RouterAgent, TaskType
from app.core.schemas import ChatRequest, ChatResponse
from app.tools.grading_tool import GradingTool
from app.tools.qa_tool import QATool
from app.tools.quiz_tool import QuizTool
from app.tools.summary_tool import SummaryTool


class AgentService:
    def __init__(
        self,
        router: RouterAgent,
        qa_tool: QATool,
        summary_tool: SummaryTool,
        quiz_tool: QuizTool,
        grading_tool: GradingTool,
    ):
        self.router = router
        self.qa_tool = qa_tool
        self.summary_tool = summary_tool
        self.quiz_tool = quiz_tool
        self.grading_tool = grading_tool

    async def handle(self, db: Session, request: ChatRequest) -> ChatResponse:
        decision = await self.router.decide(request.query, request.task_type)
        task_type = TaskType(decision.task_type)
        retrieval_query = decision.rewritten_query or request.query
        retrieval_profile = decision.retrieval_profile
        use_pro_model = request.use_pro_model or decision.needs_pro_model

        if task_type == TaskType.SUMMARY:
            return await self.summary_tool.run(
                db,
                request.query,
                use_pro_model,
                request.top_k,
                retrieval_query=retrieval_query,
                retrieval_profile=retrieval_profile,
            )
        if task_type == TaskType.QUIZ:
            return await self.quiz_tool.run(
                db,
                request.query,
                use_pro_model,
                request.top_k,
                retrieval_query=retrieval_query,
                retrieval_profile=retrieval_profile,
            )
        if task_type == TaskType.GRADE:
            return await self.grading_tool.run(
                db,
                request.query,
                True,
                request.top_k,
                retrieval_query=retrieval_query,
                retrieval_profile=retrieval_profile,
            )
        return await self.qa_tool.run(
            db,
            request.query,
            use_pro_model,
            request.top_k,
            retrieval_query=retrieval_query,
            retrieval_profile=retrieval_profile,
        )
