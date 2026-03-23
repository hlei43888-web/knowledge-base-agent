"""Query engine: orchestrates intent classification, retrieval, and answer generation."""

import logging
import time

from app.models.schemas import QueryResponse
from app.services.llm import classify_intent, generate_rag_answer, generate_url_answer
from app.services.vectorstore import query_chunks
from app.services.parser import fetch_url
from app.services.response_builder import (
    build_rag_response,
    build_rag_empty_response,
    build_url_response,
    build_url_error_response,
    build_chitchat_response,
    build_error_response,
)
from app.services.trace_logger import log_trace
from app.config import TOP_K

logger = logging.getLogger(__name__)


async def process_query(user_query: str) -> dict:
    """Process a user query through the full pipeline.

    Returns a dict containing QueryResponse fields plus trace metadata.
    """
    start_time = time.time()
    intent = "unknown"
    retrieved_chunks: list[str] = []
    error = None

    try:
        # Step 1: Intent classification via Function Calling
        result = classify_intent(user_query)

        if result.tool_name is None:
            intent = "chitchat"
            msg = result.text or "您好，有什么可以帮您的吗？"
            qr = build_chitchat_response(msg)

        elif result.tool_name == "rag_search":
            intent = "rag"
            search_query = result.tool_args.get("query", user_query)
            qr, retrieved_chunks = await _handle_rag(search_query, user_query)

        elif result.tool_name == "url_fetch":
            intent = "url"
            url = result.tool_args.get("url", "")
            qr = await _handle_url(url, user_query)

        elif result.tool_name == "chitchat_reply":
            intent = "chitchat"
            msg = result.tool_args.get("message", "您好，有什么可以帮您的吗？")
            qr = build_chitchat_response(msg)

        else:
            intent = "unknown"
            qr = build_error_response()

    except Exception as e:
        logger.exception("Query processing failed")
        error = str(e)
        qr = build_error_response(error)

    latency_ms = int((time.time() - start_time) * 1000)

    # Build output dict for trace
    output = {
        "answer": qr.answer,
        "sources": qr.sources,
        "confidence": qr.confidence.value if hasattr(qr.confidence, "value") else qr.confidence,
        "fallback": qr.fallback,
    }

    # Write trace to SQLite
    request_id = log_trace(
        user_query=user_query,
        intent=intent,
        retrieved_chunks=retrieved_chunks,
        llm_prompt=user_query,
        llm_response=qr.answer,
        output=output,
        latency_ms=latency_ms,
        error=error,
    )

    return {
        "request_id": request_id,
        # QueryResponse fields
        "answer": qr.answer,
        "sources": qr.sources,
        "confidence": qr.confidence,
        "fallback": qr.fallback,
        # Trace metadata
        "intent": intent,
        "retrieved_chunks": retrieved_chunks,
        "latency_ms": latency_ms,
        "error": error,
    }


async def _handle_rag(search_query: str, user_query: str) -> tuple[QueryResponse, list[str]]:
    """Handle RAG intent: search ChromaDB and generate answer."""
    results = query_chunks(search_query, top_k=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return build_rag_empty_response(), []

    sources = [m.get("source", "unknown") for m in metadatas]
    answer = generate_rag_answer(user_query, documents, sources)
    qr = build_rag_response(answer, sources, distances)
    return qr, documents


async def _handle_url(url: str, user_query: str) -> QueryResponse:
    """Handle URL intent: fetch page and generate answer."""
    if not url:
        return build_url_error_response("", "请提供一个有效的URL")

    try:
        page_content = await fetch_url(url)
        if not page_content.strip():
            return build_url_error_response(url, "无法从该URL提取到有效内容")
        answer = generate_url_answer(user_query, page_content)
        return build_url_response(answer, url)
    except Exception as e:
        logger.exception(f"URL fetch failed: {url}")
        return build_url_error_response(url, str(e))
