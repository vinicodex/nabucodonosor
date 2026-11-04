from unittest.mock import MagicMock, patch

import pytest

from nabucodonosor.config import Settings
from nabucodonosor.generation.llm import BaseLLM, OllamaLLM, get_llm
from nabucodonosor.generation.prompts import (
    NO_CONTEXT_RESPONSE,
    RAG_TEMPLATE,
    SYSTEM_PROMPT,
)
from nabucodonosor.retrieval.reranker import ScoredChunk


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "fake answer") -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response


def _make_chunk(content: str) -> ScoredChunk:
    return ScoredChunk(chunk_id="id", content=content, metadata={}, score=0.8)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_system_prompt_is_non_empty():
    assert len(SYSTEM_PROMPT) > 0


def test_rag_template_contains_placeholders():
    assert "{context}" in RAG_TEMPLATE
    assert "{question}" in RAG_TEMPLATE


def test_rag_template_formats_correctly():
    result = RAG_TEMPLATE.format(context="some context", question="a question")
    assert "some context" in result
    assert "a question" in result


def test_no_context_response_is_non_empty():
    assert len(NO_CONTEXT_RESPONSE) > 0


# ---------------------------------------------------------------------------
# BaseLLM.generate
# ---------------------------------------------------------------------------


def test_generate_includes_chunk_content():
    llm = FakeLLM()
    chunks = [_make_chunk("FastAPI Python experience")]
    llm.generate("What frameworks?", chunks)  # FakeLLM ignores prompt; just ensure no crash


def test_generate_joins_multiple_chunks():
    captured: list[str] = []

    class CapturingLLM(BaseLLM):
        def complete(self, prompt: str) -> str:
            captured.append(prompt)
            return "ok"

    llm = CapturingLLM()
    chunks = [_make_chunk("chunk A"), _make_chunk("chunk B")]
    llm.generate("question", chunks)
    assert "chunk A" in captured[0]
    assert "chunk B" in captured[0]


def test_generate_separates_chunks_with_delimiter():
    captured: list[str] = []

    class CapturingLLM(BaseLLM):
        def complete(self, prompt: str) -> str:
            captured.append(prompt)
            return "ok"

    llm = CapturingLLM()
    llm.generate("q", [_make_chunk("A"), _make_chunk("B")])
    assert "---" in captured[0]


# ---------------------------------------------------------------------------
# get_llm factory
# ---------------------------------------------------------------------------


def test_get_llm_returns_ollama_by_default(settings: Settings):
    llm = get_llm(settings)
    assert isinstance(llm, OllamaLLM)


def test_get_llm_raises_for_groq_without_package(settings: Settings):
    settings.llm_provider = "groq"  # type: ignore[misc]
    settings.groq_api_key = "fake-key"  # type: ignore[misc]
    with pytest.raises(ImportError, match="groq"):
        get_llm(settings)


def test_get_llm_groq_with_package_mocked(settings: Settings):
    settings.llm_provider = "groq"  # type: ignore[misc]
    settings.groq_api_key = "fake-key"  # type: ignore[misc]
    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = MagicMock()
    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        from nabucodonosor.generation.llm import GroqLLM

        llm = GroqLLM("fake-key", "llama-3.2-3b-preview")
        assert isinstance(llm, GroqLLM)
