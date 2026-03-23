import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
CHROMA_PERSIST_DIR = str(DATA_DIR / "chromadb")
SQLITE_DB_PATH = str(DATA_DIR / "trace.db")
COLLECTION_NAME = "knowledge_base"

# Chunking config
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# Retrieval config
TOP_K = 3

# Timeout & retry
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
