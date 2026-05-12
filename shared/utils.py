"""
Utility functions for LangChain RAG Tutorial.
Provides reusable functions for document formatting, vector store management, and display.
"""

import warnings
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import PREVIEW_LENGTH, SECTION_WIDTH


def suppress_warnings() -> None:
    """
    Suppress common non-critical warnings for cleaner notebook output.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api.deprecation")
    warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


suppress_warnings()


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string for use in prompts.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def load_vector_store(
    path: str | Path,
    embeddings: Embeddings,
    verbose: bool = True,
) -> Optional[FAISS]:
    """
    Load a FAISS vector store from disk.
    """
    try:
        vectorstore = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        if verbose:
            print(f"Loaded vector store from {path}")
        return vectorstore
    except Exception as exc:
        if verbose:
            print(f"Error loading vector store from {path}: {exc}")
        return None


def save_vector_store(
    vectorstore: FAISS,
    path: str | Path,
    verbose: bool = True,
) -> None:
    """
    Save a FAISS vector store to disk.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(path))
        if verbose:
            print(f"Saved vector store to {path}")
    except Exception as exc:
        if verbose:
            print(f"Error saving vector store to {path}: {exc}")
        raise


def print_section_header(title: str, width: int = SECTION_WIDTH) -> None:
    """
    Print a formatted section header.
    """
    print("\n" + "=" * width)
    print(title.upper())
    print("=" * width + "\n")


def print_results(
    docs: List[Document],
    title: str = "Retrieved Documents",
    max_docs: Optional[int] = None,
    preview_length: int = PREVIEW_LENGTH,
) -> None:
    """
    Print formatted results from document retrieval.
    """
    print(f"\n{title}")
    print("-" * SECTION_WIDTH)

    display_docs = docs[:max_docs] if max_docs else docs

    for i, doc in enumerate(display_docs, 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'N/A')}")

        if "source_type" in doc.metadata:
            print(f"   Type: {doc.metadata['source_type']}")
        if "process_date" in doc.metadata:
            print(f"   Date: {doc.metadata['process_date']}")

        content = doc.page_content[:preview_length]
        if len(doc.page_content) > preview_length:
            content += "..."
        print(f"   Content: {content}")

    if max_docs and len(docs) > max_docs:
        print(f"\n... and {len(docs) - max_docs} more documents")


def print_comparison_table(
    data: List[List[str]],
    headers: Optional[List[str]] = None,
) -> None:
    """
    Print a formatted comparison table.
    """
    if not data:
        return

    if headers is None and data:
        headers = data[0]
        data = data[1:]

    all_rows = [headers] + data if headers else data
    col_widths = [max(len(str(row[i])) for row in all_rows) + 2 for i in range(len(all_rows[0]))]

    if headers:
        print("".join(str(item).ljust(col_widths[j]) for j, item in enumerate(headers)))
        print("-" * sum(col_widths))

    for row in data:
        print("".join(str(item).ljust(col_widths[j]) for j, item in enumerate(row)))


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in a text string.
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        return len(text) // 4
    except Exception:
        return len(text) // 4


def estimate_embedding_cost(
    texts: List[str],
    model: str = "text-embedding-3-small",
    cost_per_million: float = 0.02,
) -> tuple[int, float]:
    """
    Estimate the cost of embedding a list of texts.
    """
    total_tokens = sum(estimate_tokens(text) for text in texts)
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million
    return total_tokens, estimated_cost


if __name__ == "__main__":
    print_section_header("Testing Utilities")

    data = [
        ["Feature", "OpenAI", "HuggingFace"],
        ["Dimension", "1536", "384"],
        ["Cost", "Paid", "Free"],
        ["Speed", "Fast", "Medium"],
    ]
    print_comparison_table(data)

    test_text = "This is a test string for token estimation."
    tokens = estimate_tokens(test_text)
    print(f"\nTest text tokens: {tokens}")
