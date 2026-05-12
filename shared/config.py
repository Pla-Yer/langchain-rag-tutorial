"""
Configuration module for LangChain RAG Tutorial.
Centralizes provider selection, API keys, paths, and default parameters.
"""

import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

# Set User Agent for HTTP requests
USER_AGENT = os.getenv("USER_AGENT", "LangChain-RAG-Tutorial/1.0")
os.environ["USER_AGENT"] = USER_AGENT

# ============================================================================
# API KEYS AND PROVIDERS
# ============================================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "vllm"}
VLLM_DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RAW_LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_KEY = RAW_LLM_API_KEY
if not LLM_API_KEY:
    if LLM_PROVIDER == "deepseek":
        LLM_API_KEY = DEEPSEEK_API_KEY or OPENAI_API_KEY
    elif LLM_PROVIDER in OPENAI_COMPATIBLE_PROVIDERS:
        LLM_API_KEY = OPENAI_API_KEY
        if LLM_PROVIDER == "vllm":
            LLM_API_KEY = LLM_API_KEY or "EMPTY"

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RAW_LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_BASE_URL = RAW_LLM_BASE_URL
if not LLM_BASE_URL:
    if LLM_PROVIDER == "deepseek":
        LLM_BASE_URL = DEEPSEEK_BASE_URL
    elif LLM_PROVIDER == "openai":
        LLM_BASE_URL = OPENAI_BASE_URL
    elif LLM_PROVIDER == "vllm":
        LLM_BASE_URL = OPENAI_BASE_URL or VLLM_DEFAULT_BASE_URL

if not LLM_API_KEY:
    if LLM_PROVIDER == "deepseek":
        print("WARNING: DEEPSEEK_API_KEY not found in environment")
        print("  Please create .env with: DEEPSEEK_API_KEY=your-key-here")
    elif LLM_PROVIDER == "openai":
        print("WARNING: OPENAI_API_KEY not found in environment")
        print("  Please create .env with: OPENAI_API_KEY=your-key-here")
    elif LLM_PROVIDER == "vllm":
        print("WARNING: OPENAI_API_KEY not found in environment")
        print("  Falling back to api_key=EMPTY for local vLLM server compatibility")

# Compatibility for OpenAI-compatible SDKs such as langchain_openai.ChatOpenAI.
if LLM_API_KEY:
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
if LLM_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL

# Optional: HuggingFace API Key (not needed for local embeddings)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

# Optional: Tavily API Key (for premium web search in CRAG)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Optional: LangSmith API Key (for tracing and monitoring)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "langchain-rag-tutorial")

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
CACHE_DIR = DATA_DIR / "cache"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_VECTOR_STORE_PATH = VECTOR_STORE_DIR / "openai_embeddings"
HF_VECTOR_STORE_PATH = VECTOR_STORE_DIR / "huggingface_embeddings"
DEFAULT_VECTOR_STORE_PATH = (
    HF_VECTOR_STORE_PATH if EMBEDDING_PROVIDER == "huggingface" else OPENAI_VECTOR_STORE_PATH
)

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
DEFAULT_MMR_FETCH_K = int(os.getenv("DEFAULT_MMR_FETCH_K", "20"))
DEFAULT_MMR_LAMBDA = float(os.getenv("DEFAULT_MMR_LAMBDA", "0.5"))

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0"))
DEFAULT_VISION_MODEL = os.getenv("DEFAULT_VISION_MODEL", DEFAULT_MODEL)

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

DEFAULT_LANGCHAIN_URLS = [
    "https://python.langchain.com/docs/use_cases/question_answering/",
    "https://python.langchain.com/docs/modules/data_connection/retrievers/",
    "https://python.langchain.com/docs/modules/model_io/llms/",
    "https://python.langchain.com/docs/use_cases/chatbots/",
]


def _resolve_llm_api_key(provider: str) -> str | None:
    if RAW_LLM_API_KEY:
        return RAW_LLM_API_KEY
    if provider == "deepseek":
        return DEEPSEEK_API_KEY or OPENAI_API_KEY
    if provider == "vllm":
        return OPENAI_API_KEY or "EMPTY"
    if provider == "openai":
        return OPENAI_API_KEY
    raise ValueError(
        f"Unsupported LLM provider={provider!r}. Use 'deepseek', 'vllm', or 'openai'."
    )


def _resolve_llm_base_url(provider: str) -> str | None:
    if RAW_LLM_BASE_URL:
        return RAW_LLM_BASE_URL
    if provider == "deepseek":
        return DEEPSEEK_BASE_URL
    if provider == "vllm":
        return OPENAI_BASE_URL or VLLM_DEFAULT_BASE_URL
    if provider == "openai":
        return OPENAI_BASE_URL
    raise ValueError(
        f"Unsupported LLM provider={provider!r}. Use 'deepseek', 'vllm', or 'openai'."
    )

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

SECTION_WIDTH = int(os.getenv("SECTION_WIDTH", "80"))
PREVIEW_LENGTH = int(os.getenv("PREVIEW_LENGTH", "300"))


class LocalHashEmbeddings:
    """
    Lightweight deterministic local embeddings fallback.

    This avoids external model downloads when HuggingFace access is unavailable.
    Retrieval quality is lower than sentence-transformers, but it keeps the
    tutorial runnable in restricted environments.
    """

    def __init__(self, n_features: int = 512) -> None:
        self.n_features = n_features

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.n_features
        for token in self._tokenize(text):
            vector[hash(token) % self.n_features] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


class DirectSentenceTransformerEmbeddings:
    """
    Direct SentenceTransformer wrapper that avoids langchain_huggingface init issues.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, **(model_kwargs or {}))
        self.encode_kwargs = {"normalize_embeddings": True}
        if encode_kwargs:
            self.encode_kwargs.update(encode_kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, **self.encode_kwargs)
        return embedding.tolist()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def verify_api_key() -> bool:
    """
    Verify that the active LLM provider key is loaded.
    """
    if LLM_API_KEY:
        print(f"OK {LLM_PROVIDER} API key: LOADED")
        print(f"  Preview: {LLM_API_KEY[:7]}...{LLM_API_KEY[-4:]}")
        if LLM_BASE_URL:
            print(f"  Base URL: {LLM_BASE_URL}")
        return True

    print(f"X {LLM_PROVIDER} API key: NOT LOADED")
    print("\nSetup instructions:")
    print("  1. Create .env file in project root")
    if LLM_PROVIDER == "deepseek":
        print("  2. Add: DEEPSEEK_API_KEY=your-key")
        print("  3. Add: LLM_BASE_URL=https://api.deepseek.com")
        print("  4. Get key from: https://platform.deepseek.com/")
    elif LLM_PROVIDER == "vllm":
        print("  2. Add: LLM_PROVIDER=vllm")
        print("  3. Add: OPENAI_BASE_URL=http://localhost:8000/v1")
        print("  4. Optional: OPENAI_API_KEY=EMPTY")
    else:
        print("  2. Add: OPENAI_API_KEY=sk-proj-...")
        print("  3. Get key from: https://platform.openai.com/api-keys")
    print("  4. Restart kernel after updating .env")
    return False


def create_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create a chat model using the configured or requested OpenAI-compatible provider.

    Examples:
        create_chat_model()
        create_chat_model(provider="deepseek", model="deepseek-v4-flash")
        create_chat_model(provider="vllm", model="Qwen/Qwen2.5-7B-Instruct")
    """
    from langchain_openai import ChatOpenAI

    selected_provider = (provider or LLM_PROVIDER).lower()
    api_key = kwargs.pop("api_key", None) or _resolve_llm_api_key(selected_provider)
    base_url = kwargs.pop("base_url", None) or _resolve_llm_base_url(selected_provider)

    init_kwargs: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "api_key": api_key,
        "http_client": httpx.Client(trust_env=False),
        "http_async_client": httpx.AsyncClient(trust_env=False),
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    init_kwargs.update(kwargs)
    return ChatOpenAI(**init_kwargs)


def create_embeddings(**kwargs: Any) -> Any:
    """
    Create the configured embeddings backend.

    DeepSeek's official docs currently document chat-completions compatibility,
    so this project defaults to local HuggingFace embeddings when DeepSeek is the LLM.
    """
    if EMBEDDING_PROVIDER == "huggingface":
        init_kwargs: dict[str, Any] = {"model_name": HF_EMBEDDING_MODEL}
        init_kwargs.update(kwargs)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(**init_kwargs)
        except Exception as exc:
            print("WARNING: failed to initialize langchain_huggingface embeddings.")
            print(f"  Reason: {exc}")
            try:
                return DirectSentenceTransformerEmbeddings(**init_kwargs)
            except Exception as direct_exc:
                print(
                    "WARNING: failed to initialize direct SentenceTransformer embeddings; "
                    "falling back to LocalHashEmbeddings."
                )
                print(f"  Reason: {direct_exc}")
                return LocalHashEmbeddings()

    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        init_kwargs: dict[str, Any] = {
            "model": OPENAI_EMBEDDING_MODEL,
            "api_key": OPENAI_API_KEY or LLM_API_KEY,
            "http_client": httpx.Client(trust_env=False),
            "http_async_client": httpx.AsyncClient(trust_env=False),
        }
        if OPENAI_BASE_URL:
            init_kwargs["base_url"] = OPENAI_BASE_URL
        init_kwargs.update(kwargs)
        return OpenAIEmbeddings(**init_kwargs)

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. "
        "Use 'huggingface', 'openai', or configure local fallback via huggingface."
    )


def get_project_info() -> dict[str, Any]:
    """
    Get project configuration information.
    """
    return {
        "environment": ENVIRONMENT,
        "debug_mode": DEBUG_MODE,
        "log_level": LOG_LEVEL,
        "project_root": str(PROJECT_ROOT),
        "vector_store_dir": str(VECTOR_STORE_DIR),
        "cache_dir": str(CACHE_DIR),
        "llm_provider": LLM_PROVIDER,
        "embedding_provider": EMBEDDING_PROVIDER,
        "llm_api_key_loaded": bool(LLM_API_KEY),
        "llm_base_url": LLM_BASE_URL,
        "openai_api_key_loaded": bool(OPENAI_API_KEY),
        "deepseek_api_key_loaded": bool(DEEPSEEK_API_KEY),
        "huggingface_api_key_loaded": bool(HUGGINGFACE_API_KEY),
        "tavily_api_key_loaded": bool(TAVILY_API_KEY),
        "langsmith_api_key_loaded": bool(LANGSMITH_API_KEY),
        "default_model": DEFAULT_MODEL,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_vector_store_path": str(DEFAULT_VECTOR_STORE_PATH),
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "hf_embedding_model": HF_EMBEDDING_MODEL,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "k": DEFAULT_K,
        "mmr_fetch_k": DEFAULT_MMR_FETCH_K,
        "mmr_lambda": DEFAULT_MMR_LAMBDA,
        "section_width": SECTION_WIDTH,
        "preview_length": PREVIEW_LENGTH,
    }


if __name__ == "__main__":
    print("=" * SECTION_WIDTH)
    print("LANGCHAIN RAG - CONFIGURATION")
    print("=" * SECTION_WIDTH)

    info = get_project_info()
    for key, value in info.items():
        print(f"{key:.<30} {value}")

    print("\n" + "=" * SECTION_WIDTH)
    verify_api_key()
    print("=" * SECTION_WIDTH)
