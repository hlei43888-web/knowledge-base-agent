"""FastAPI application entry point."""

import logging

from fastapi import FastAPI

from app.routers import ingest, query, traces

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Knowledge Base Agent",
    description="Enterprise knowledge base Q&A system powered by Claude",
    version="0.1.0",
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(traces.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
