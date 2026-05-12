"""
Document loading utilities for LangChain RAG Tutorial
Provides functions for loading and splitting documents from various sources.
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_LANGCHAIN_URLS,
    PROJECT_ROOT,
    USER_AGENT,
)


def download_langchain_docs_to_local(
    urls: Optional[List[str]] = None,
    target_dir: str | Path | None = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Download LangChain documentation pages into local text files.

    Args:
        urls: List of URLs to download (defaults to DEFAULT_LANGCHAIN_URLS)
        target_dir: Directory for downloaded text files
        verbose: Whether to print status messages

    Returns:
        List[Path]: Paths to downloaded local text files
    """
    if urls is None:
        urls = DEFAULT_LANGCHAIN_URLS

    if target_dir is None:
        target_dir = PROJECT_ROOT / "data" / "source_docs" / "langchain"

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading {len(urls)} documents to local directory...")
        print(f"  - Target: {target_path}")

    local_paths: List[Path] = []

    for index, url in enumerate(urls, start=1):
        slug = url.rstrip("/").split("/")[-1] or f"doc_{index}"
        local_file = target_path / f"{index:02d}_{slug}.txt"

        if verbose:
            print(f"  - {url}")

        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        lines = [line.strip() for line in soup.get_text("\n").splitlines()]
        text = "\n".join(line for line in lines if line)
        local_file.write_text(text, encoding="utf-8")
        local_paths.append(local_file)

    if verbose:
        print(f"OK Downloaded {len(local_paths)} local documents")

    return local_paths


def load_local_text_docs(
    local_paths: List[str | Path],
    original_urls: Optional[List[str]] = None,
    add_metadata: bool = True,
    verbose: bool = True,
) -> List[Document]:
    """
    Load locally saved text files as LangChain Documents.

    Args:
        local_paths: Paths to local text files
        original_urls: Optional source URLs aligned by index
        add_metadata: Whether to add custom metadata fields
        verbose: Whether to print status messages

    Returns:
        List[Document]: Loaded documents with metadata
    """
    docs: List[Document] = []
    current_date = datetime.date.today().isoformat()

    for index, local_path in enumerate(local_paths):
        path_obj = Path(local_path)
        text = path_obj.read_text(encoding="utf-8")
        metadata = {
            "source": original_urls[index] if original_urls and index < len(original_urls) else str(path_obj),
            "local_path": str(path_obj),
        }

        if add_metadata:
            metadata.update(
                {
                    "source_type": "local_text_download",
                    "process_date": current_date,
                    "domain": "langchain",
                }
            )

        docs.append(Document(page_content=text, metadata=metadata))

    if verbose:
        print(f"OK Loaded {len(docs)} local documents")
        if add_metadata:
            print("OK Added custom metadata to all documents")

    return docs


def load_langchain_docs(
    urls: Optional[List[str]] = None,
    add_metadata: bool = True,
    verbose: bool = True
) -> List[Document]:
    """
    Load LangChain documentation from web URLs.

    Args:
        urls: List of URLs to load (defaults to DEFAULT_LANGCHAIN_URLS)
        add_metadata: Whether to add custom metadata fields
        verbose: Whether to print status messages

    Returns:
        List[Document]: Loaded documents with metadata

    Example:
        >>> docs = load_langchain_docs()
        >>> print(f"Loaded {len(docs)} documents")
    """
    if urls is None:
        urls = DEFAULT_LANGCHAIN_URLS

    if verbose:
        print(f"Loading {len(urls)} documents from web...")
        for url in urls:
            print(f"  - {url}")

    try:
        # Initialize WebBaseLoader and load documents
        loader = WebBaseLoader(urls)
        docs = loader.load()

        if verbose:
            print(f"✓ Loaded {len(docs)} documents")

        # Add custom metadata
        if add_metadata:
            current_date = datetime.date.today().isoformat()
            for doc in docs:
                doc.metadata['source_type'] = 'web_documentation'
                doc.metadata['process_date'] = current_date
                doc.metadata['domain'] = 'langchain'

            if verbose:
                print("✓ Added custom metadata to all documents")

        return docs

    except Exception as e:
        if verbose:
            print(f"✗ Error loading documents: {e}")
        raise


def split_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = True
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.

    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        verbose: Whether to print status messages

    Returns:
        List[Document]: Split document chunks

    Example:
        >>> chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        >>> print(f"Created {len(chunks)} chunks")
    """
    if verbose:
        print("Splitting documents...")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Chunk overlap: {chunk_overlap}")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(docs)

        if verbose:
            print(f"✓ Created {len(chunks)} chunks")

            # Show sample chunk info
            if chunks:
                sample_chunk = chunks[0]
                print("\n  Sample chunk:")
                print(f"    - Length: {len(sample_chunk.page_content)} chars")
                print(f"    - Source: {sample_chunk.metadata.get('source', 'N/A')}")
                print(f"    - Preview: {sample_chunk.page_content[:150]}...")

        return chunks

    except Exception as e:
        if verbose:
            print(f"✗ Error splitting documents: {e}")
        raise


def compare_splitting_strategies(
    docs: List[Document],
    strategies: List[Tuple[int, int]],
    verbose: bool = True
) -> Dict:
    """
    Compare different text splitting strategies.

    Args:
        docs: Documents to split
        strategies: List of (chunk_size, chunk_overlap) tuples
        verbose: Whether to print comparison table

    Returns:
        dict: Results for each strategy

    Example:
        >>> strategies = [(1000, 200), (500, 100), (2000, 400)]
        >>> results = compare_splitting_strategies(docs, strategies)
    """
    results = {}

    for chunk_size, chunk_overlap in strategies:
        chunks = split_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=False
        )

        strategy_name = f"{chunk_size}/{chunk_overlap}"
        results[strategy_name] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks),
            "chunks": chunks
        }

    if verbose:
        print("\n=== Splitting Strategy Comparison ===\n")
        print(f"{'Strategy':<15} {'Chunk Size':<15} {'Overlap':<15} {'Chunks':<10}")
        print("-" * 60)

        for strategy_name, result in results.items():
            print(f"{strategy_name:<15} "
                  f"{result['chunk_size']:<15} "
                  f"{result['chunk_overlap']:<15} "
                  f"{result['num_chunks']:<10}")

        print("\n💡 Larger chunks = more context, fewer chunks")
        print("💡 Smaller chunks = more precise, more chunks")

    return results


def load_and_split(
    urls: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = True
) -> Tuple[List[Document], List[Document]]:
    """
    Convenience function to load and split documents in one call.

    Args:
        urls: URLs to load (defaults to DEFAULT_LANGCHAIN_URLS)
        chunk_size: Chunk size for splitting
        chunk_overlap: Overlap between chunks
        verbose: Whether to print status messages

    Returns:
        tuple: (original_docs, chunks)

    Example:
        >>> docs, chunks = load_and_split()
        >>> print(f"Loaded {len(docs)} docs, created {len(chunks)} chunks")
    """
    docs = load_langchain_docs(urls=urls, verbose=verbose)
    chunks = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose
    )

    return docs, chunks


if __name__ == "__main__":
    # Test document loading
    print("=" * 80)
    print("TESTING DOCUMENT LOADERS")
    print("=" * 80)

    # Load documents
    docs = load_langchain_docs()
    print(f"\nLoaded {len(docs)} documents")

    # Split with default settings
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Compare strategies
    strategies = [(1000, 200), (500, 100)]
    results = compare_splitting_strategies(docs, strategies)
