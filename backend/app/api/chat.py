from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.schemas import ChatRequest, ChatResponse
from app.db.database import get_db
from app.dependencies import get_qa_tool
from app.tools.qa_tool import QATool


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    qa_tool: QATool = Depends(get_qa_tool),
) -> ChatResponse:
    return await qa_tool.run(
        db,
        query=request.query,
        use_pro_model=request.use_pro_model,
        top_k=request.top_k,
    )

