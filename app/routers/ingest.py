"""Data ingestion API routes."""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestURLRequest, IngestResponse, StatsResponse
from app.services.parser import parse_file, fetch_url
from app.services.chunker import chunk_text
from app.services.vectorstore import add_chunks, get_stats, list_sources, delete_by_source, delete_all, source_exists

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/document", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a PDF or Word document into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_extensions = (".pdf", ".docx")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        text = parse_file(file_bytes, file.filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")

        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from the file")

        # Remove old data if re-ingesting the same file
        if source_exists(file.filename):
            delete_by_source(file.filename)
            logger.info(f"Replaced existing source: {file.filename}")

        add_chunks(chunks, source=file.filename, source_type="document")

        return IngestResponse(
            source=file.filename,
            chunks_count=len(chunks),
            message=f"Successfully ingested {file.filename} ({len(chunks)} chunks)",
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to ingest document: {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest):
    """Ingest content from a URL into the knowledge base."""
    try:
        text = await fetch_url(request.url)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No content could be extracted from the URL")

        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from the URL content")

        # Remove old data if re-ingesting the same URL
        if source_exists(request.url):
            delete_by_source(request.url)
            logger.info(f"Replaced existing source: {request.url}")

        add_chunks(chunks, source=request.url, source_type="url")

        return IngestResponse(
            source=request.url,
            chunks_count=len(chunks),
            message=f"Successfully ingested URL ({len(chunks)} chunks)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to ingest URL: {request.url}")
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Get knowledge base statistics."""
    return StatsResponse(**get_stats())


@router.get("/sources")
async def sources():
    """List all ingested sources with chunk counts."""
    return list_sources()


@router.delete("/source")
async def delete_source(name: str):
    """Delete all chunks from a specific source (document name or URL)."""
    deleted = delete_by_source(name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    return {"source": name, "deleted_chunks": deleted}


@router.delete("/all")
async def delete_all_data():
    """Delete all data from the knowledge base."""
    deleted = delete_all()
    return {"deleted_chunks": deleted, "message": "Knowledge base cleared"}
