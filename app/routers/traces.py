"""Trace log API routes."""

from fastapi import APIRouter, HTTPException, Query

from app.services.trace_logger import get_trace, list_traces, count_traces

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("/")
async def list_all_traces(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List recent trace records with pagination."""
    traces = list_traces(limit=limit, offset=offset)
    total = count_traces()
    return {"total": total, "limit": limit, "offset": offset, "traces": traces}


@router.get("/{request_id}")
async def get_single_trace(request_id: str):
    """Get a single trace record by request_id."""
    trace = get_trace(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
