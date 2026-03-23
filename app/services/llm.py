"""LLM service with Function Calling for intent routing (OpenAI-compatible, supports DeepSeek)."""

import json
import logging

from openai import OpenAI, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, REQUEST_TIMEOUT, MAX_RETRIES

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


# Tool definitions (OpenAI function calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the internal knowledge base for information. "
                "Use this when the user asks questions that can be answered from "
                "existing documents in the knowledge base, such as company policies, "
                "product documentation, technical guides, or any previously ingested content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information in the knowledge base.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "url_fetch",
            "description": (
                "Fetch and read content from a specific URL in real-time. "
                "Use this when the user provides a URL and asks about its content, "
                "or when the user asks about information that needs to be fetched "
                "from a specific web page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chitchat_reply",
            "description": (
                "Handle casual conversation, greetings, or questions unrelated to the knowledge base. "
                "Use this for small talk, greetings like 'hello', 'how are you', "
                "or any question that does not require searching documents or fetching URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "A friendly reply to the user's casual message.",
                    },
                },
                "required": ["message"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个知识库问答助手。知识库中可能包含各种类型的内容（技术文档、诗词、百科、文章等）。

路由规则（严格按优先级执行）：
1. 如果用户提供了URL并要求了解其内容 → 使用 url_fetch
2. 如果用户只是打招呼（如"你好"、"hi"）或纯闲聊 → 使用 chitchat_reply
3. 其他所有提问 → 一律使用 rag_search 搜索知识库，不要自行判断知识库里有没有相关内容

每次只调用一个工具。"""


class IntentResult:
    """Parsed result from intent classification."""
    def __init__(self, tool_name: str | None, tool_args: dict, text: str | None = None):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.text = text


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError)),
    reraise=True,
)
def classify_intent(user_query: str) -> IntentResult:
    """Use LLM with Function Calling to classify user intent and route the query."""
    client = get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        tc = message.tool_calls[0]
        args = json.loads(tc.function.arguments)
        return IntentResult(tool_name=tc.function.name, tool_args=args, text=message.content)

    return IntentResult(tool_name=None, tool_args={}, text=message.content)


RAG_ANSWER_PROMPT = """你是一个企业内部知识库问答助手。请根据以下检索到的参考内容回答用户的问题。

要求：
- 仅基于提供的参考内容回答，不要编造信息
- 如果参考内容不足以回答问题，明确说明
- 回答要准确、简洁、有条理
- 使用中文回答

参考内容：
{context}

用户问题：{query}"""


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError)),
    reraise=True,
)
def generate_rag_answer(query: str, context_chunks: list[str], sources: list[str]) -> str:
    """Generate answer based on retrieved chunks."""
    context = "\n\n---\n\n".join(
        f"[来源: {src}]\n{chunk}" for chunk, src in zip(context_chunks, sources)
    )
    prompt = RAG_ANSWER_PROMPT.format(context=context, query=query)

    client = get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


URL_ANSWER_PROMPT = """你是一个企业内部知识库问答助手。用户要求你分析以下网页内容并回答问题。

网页内容：
{content}

用户问题：{query}

请基于网页内容给出准确、简洁的回答。使用中文回答。"""


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError)),
    reraise=True,
)
def generate_url_answer(query: str, page_content: str) -> str:
    """Generate answer based on fetched URL content."""
    if len(page_content) > 15000:
        page_content = page_content[:15000] + "\n\n[内容已截断...]"

    prompt = URL_ANSWER_PROMPT.format(content=page_content, query=query)

    client = get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
