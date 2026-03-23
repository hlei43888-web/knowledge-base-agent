"""Query API route."""

import logging

from fastapi import APIRouter

from app.models.schemas import QueryRequest, QueryResponse
from app.services.query_engine import process_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Ask a question to the knowledge base agent.

    Returns structured response with answer, sources, confidence level, and fallback flag.
    """
    result = await process_query(request.question)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        fallback=result["fallback"],
    )
