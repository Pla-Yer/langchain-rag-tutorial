"""  
Streamlit Production Template for LangChain RAG
Interactive UI for querying the RAG system

Note: This template requires Streamlit dependencies.
Install with: pip install -r requirements.txt
"""

import streamlit as st
import sys
from pathlib import Path
import time

from typing import Tuple, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from shared import (
    load_vector_store,
    format_docs,
    DEFAULT_MODEL,
    DEFAULT_VECTOR_STORE_PATH,
    create_chat_model,
    create_embeddings,
)
from shared.prompts import RAG_PROMPT_TEMPLATE

# Page configuration
st.set_page_config(
    page_title="LangChain RAG Tutorial",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag() -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize RAG components (cached)"""
    try:
        # Load embeddings and vector store
        embeddings = create_embeddings()
        vectorstore = load_vector_store(DEFAULT_VECTOR_STORE_PATH, embeddings)

        # Initialize LLM
        llm = create_chat_model(model=DEFAULT_MODEL, temperature=0)

        return vectorstore, llm

    except Exception as e:
        st.error(f"Failed to initialize RAG: {e}")
        return None, None


def main() -> None:
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">🦜 LangChain RAG Tutorial</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Architecture selection
        architecture = st.selectbox(
            "RAG Architecture",
            ["Simple RAG", "Contextual RAG", "Fusion RAG"],
            help="Select the RAG architecture to use"
        )

        # Number of documents
        k = st.slider(
            "Documents to retrieve (k)",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of documents to retrieve from vector store"
        )

        # Temperature
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls randomness in responses"
        )

        st.divider()

        # Info
        st.subheader("📚 About")
        st.markdown("""
        This is a production-ready RAG application built with:
        - **LangChain** for RAG pipelines
        - **DeepSeek** for the chat model
        - **HuggingFace** for local embeddings
        - **FAISS** for vector search
        - **Streamlit** for UI

        [View on GitHub](https://github.com/gianlucamazza/langchain-rag-tutorial)
        """)

    # Initialize RAG
    with st.spinner("🔄 Loading RAG components..."):
        vectorstore, llm = initialize_rag()

    if vectorstore is None or llm is None:
        st.error("❌ Failed to load RAG components. Check your configuration.")
        return

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Ask a Question")

        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is RAG and how does it work?",
            key="query_input"
        )

        # Submit button
        submit = st.button("🚀 Get Answer", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Statistics")

        # Placeholder for statistics
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Architecture", architecture.split()[0])
        with stat_col2:
            st.metric("Documents (k)", k)

    # Process query
    if submit and query:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()

            try:
                # Create retriever
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": k}
                )

                # Update LLM temperature
                llm.temperature = temperature

                # Build chain
                chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | RAG_PROMPT_TEMPLATE
                    | llm
                    | StrOutputParser()
                )

                # Get answer
                answer = chain.invoke(query)

                # Get source documents
                docs = retriever.invoke(query)

                # Calculate latency
                latency = (time.time() - start_time) * 1000

                # Display results
                st.divider()

                st.subheader("✅ Answer")
                st.markdown(answer)

                st.divider()

                # Source documents
                with st.expander("📄 Source Documents", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Document {i}** (Source: {doc.metadata.get('source', 'unknown')})")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Latency", f"{latency:.0f} ms")
                with col2:
                    st.metric("📚 Sources", len(docs))
                with col3:
                    st.metric("🎯 Architecture", architecture.split()[0])

                # Success message
                st.success("✅ Query completed successfully!")

            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif submit and not query:
        st.warning("⚠️ Please enter a question.")

    # Sample queries
    st.divider()
    st.subheader("💡 Try These Sample Queries")

    sample_queries = [
        "What is RAG and how does it work?",
        "How do I create a FAISS vector store?",
        "What's the difference between similarity and MMR retrieval?",
        "How does Adaptive RAG routing work?",
        "What are the cost optimization strategies for RAG?"
    ]

    cols = st.columns(len(sample_queries))
    for i, col in enumerate(cols):
        if col.button(f"📝 Query {i+1}", use_container_width=True):
            st.session_state.query_input = sample_queries[i]
            st.rerun()


if __name__ == "__main__":
    main()
