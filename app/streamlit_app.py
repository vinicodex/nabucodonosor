"""Streamlit chat interface for Nabucodonosor RAG."""

import time

import streamlit as st

from nabucodonosor.config import Settings
from nabucodonosor.pipeline import RAGPipeline

st.set_page_config(
    page_title="Nabucodonosor — Ask about Marcos's CV",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Nabucodonosor")
st.caption("Ask anything about Marcos Vinicius Nascimento's professional background.")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings=Settings())


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about Marcos's experience, skills, or projects..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()
            response = pipeline.query(question)
            latency_ms = (time.time() - t0) * 1000

        st.markdown(response.answer)

        with st.expander("Details", expanded=False):
            cols = st.columns(3)
            cols[0].metric("Retrieval score", f"{response.retrieval_score:.2f}")
            cols[1].metric("Latency", f"{latency_ms:.0f} ms")
            cols[2].metric("Cache hit", "✅" if response.cache_hit else "❌")

            if response.rewritten_query and response.rewritten_query != question:
                st.info(f"**Rewritten query:** {response.rewritten_query}")

            if response.sources:
                st.write("**Sources:**", ", ".join(set(response.sources)))

            if response.blocked:
                st.warning(f"**Blocked:** {response.block_reason}")

    st.session_state.messages.append({"role": "assistant", "content": response.answer})
