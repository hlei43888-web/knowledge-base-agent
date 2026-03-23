"""Pydantic models for API request/response schemas."""

from enum import Enum

from pydantic import BaseModel, Field


# --- Ingest ---

class IngestURLRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    source: str
    chunks_count: int
    message: str


class StatsResponse(BaseModel):
    total_chunks: int


# --- Query ---

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["公司的请假制度是什么？"])


class QueryResponse(BaseModel):
    """Structured response from the knowledge base agent."""

    answer: str = Field(..., description="回答内容")
    sources: list[str] = Field(default_factory=list, description="来源文档名或URL列表")
    confidence: ConfidenceLevel = Field(..., description="置信度: high/medium/low")
    fallback: bool = Field(..., description="是否触发降级（知识库未命中时为true）")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "根据公司制度，年假为每年15天，需提前3个工作日申请。",
                    "sources": ["员工手册.pdf"],
                    "confidence": "high",
                    "fallback": False,
                },
                {
                    "answer": "抱歉，知识库中没有找到与您问题相关的信息。",
                    "sources": [],
                    "confidence": "low",
                    "fallback": True,
                },
            ]
        }
    }
