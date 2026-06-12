from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.agent_service import AgentService
from app.core.schemas import ChatRequest, ChatResponse
from app.db.database import get_db
from app.dependencies import get_agent_service


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    agent: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    return await agent.handle(db, request)
