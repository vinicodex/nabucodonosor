"""HuggingFace Spaces entry point — Gradio chatbot interface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from nabucodonosor.config import Settings
from nabucodonosor.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(settings=Settings())
    return _pipeline


def respond(message: str, history: list[list[str | None]]) -> str:  # noqa: ARG001
    response = get_pipeline().query(message)

    details: list[str] = []
    if response.cache_hit:
        details.append("⚡ cache hit")
    if response.rewritten_query and response.rewritten_query != message:
        details.append(f"🔍 rewritten: *{response.rewritten_query}*")
    if response.sources:
        details.append(f"📄 sources: {', '.join(set(response.sources))}")
    if response.blocked:
        return response.answer

    footer = "  \n".join(details)
    if footer:
        return f"{response.answer}\n\n---\n{footer}"
    return response.answer


demo = gr.ChatInterface(
    fn=respond,
    title="🤖 Nabucodonosor",
    description=(
        "Ask anything about **Marcos Vinicius Nascimento**'s professional background — "
        "skills, experience, projects, and more. Powered by RAG + Groq."
    ),
    examples=[
        "What is Marcos's current role?",
        "What cloud platforms has Marcos worked with?",
        "Does Marcos have experience with vector search?",
        "What programming languages does Marcos know?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
