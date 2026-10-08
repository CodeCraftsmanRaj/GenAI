import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(page_title="SPIT RAG Chatbot", page_icon="🎓", layout="wide")
st.title("SPIT Domain-Specific RAG Chatbot")
st.caption("Retrieval-first answers with visible evidence and source attribution.")
st.warning("Use only official public SPIT documents and instructor-approved anonymized/mock ERP data. The included sample document is explicitly illustrative, not official policy.")
with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=3)
    st.markdown("**Before first use:** run `uv run main.py ingest` in the project terminal.")
question = st.text_area("Ask a question", placeholder="Ask something supported by the documents in your knowledge base…", height=90)
if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving evidence and generating an answer…"):
        try:
            result = RAGPipeline().ask(question.strip(), top_k=top_k)
            st.subheader("Answer")
            st.write(result["answer"])
            st.caption(f"Generation latency: {result['latency_seconds']:.2f}s")
            st.subheader("Retrieved evidence")
            if not result["retrieved_chunks"]:
                st.info("No chunks were retrieved.")
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                with st.expander(f"{i}. {chunk['source']} · similarity {chunk['score']:.3f}", expanded=(i == 1)):
                    st.caption(f"Path: {chunk.get('source_path')} · Section: {chunk.get('page_section')}")
                    st.write(chunk["text"])
        except Exception as e:
            st.error(str(e))
            st.info("Check that ingestion is complete and the configured LLM is running. See README.md.")
