"""Centralized response builder to ensure consistent structured output across all intent paths."""

from app.models.schemas import QueryResponse, ConfidenceLevel

# Confidence thresholds based on cosine distance (lower = more similar)
CONFIDENCE_HIGH_THRESHOLD = 0.5
CONFIDENCE_MEDIUM_THRESHOLD = 0.8


def determine_confidence(distances: list[float]) -> ConfidenceLevel:
    """Determine confidence level from ChromaDB cosine distances."""
    if not distances:
        return ConfidenceLevel.LOW
    best = min(distances)
    if best <= CONFIDENCE_HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    elif best <= CONFIDENCE_MEDIUM_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


_UNABLE_KEYWORDS = [
    "无法回答", "无法找到", "没有找到", "未找到",
    "不确定", "没有相关", "无相关", "不在知识库",
    "无法确定", "抱歉",
]


def _answer_indicates_unable(answer: str) -> bool:
    """Check if the answer text indicates the LLM could not answer the question."""
    return any(kw in answer for kw in _UNABLE_KEYWORDS)


def build_rag_response(
    answer: str,
    sources: list[str],
    distances: list[float],
) -> QueryResponse:
    """Build response for RAG intent."""
    confidence = determine_confidence(distances)
    unable = _answer_indicates_unable(answer)
    fallback = confidence == ConfidenceLevel.LOW or not sources or unable

    if unable and confidence != ConfidenceLevel.LOW:
        confidence = ConfidenceLevel.LOW

    if fallback and answer and not unable:
        answer = "知识库中找到了一些可能相关的内容，但相关度较低，以下回答仅供参考：\n\n" + answer
    return QueryResponse(
        answer=answer,
        sources=list(set(sources)),
        confidence=confidence,
        fallback=fallback,
    )


def build_rag_empty_response() -> QueryResponse:
    """Build response when RAG retrieval returns no results."""
    return QueryResponse(
        answer="抱歉，知识库中没有找到与您问题相关的信息。",
        sources=[],
        confidence=ConfidenceLevel.LOW,
        fallback=True,
    )


def build_url_response(answer: str, url: str) -> QueryResponse:
    """Build response for URL fetch intent."""
    return QueryResponse(
        answer=answer,
        sources=[url],
        confidence=ConfidenceLevel.HIGH,
        fallback=False,
    )


def build_url_error_response(url: str, error: str) -> QueryResponse:
    """Build response when URL fetch fails."""
    return QueryResponse(
        answer=f"获取URL内容失败: {error}",
        sources=[url] if url else [],
        confidence=ConfidenceLevel.LOW,
        fallback=True,
    )


def build_chitchat_response(message: str) -> QueryResponse:
    """Build response for chitchat intent."""
    return QueryResponse(
        answer=message,
        sources=[],
        confidence=ConfidenceLevel.HIGH,
        fallback=False,
    )


def build_error_response(error: str | None = None) -> QueryResponse:
    """Build response when an error occurs."""
    return QueryResponse(
        answer="抱歉，处理您的请求时出现了错误，请稍后重试。",
        sources=[],
        confidence=ConfidenceLevel.LOW,
        fallback=True,
    )
