from abc import ABC, abstractmethod

from nabucodonosor.config import Settings
from nabucodonosor.generation.prompts import RAG_TEMPLATE, SYSTEM_PROMPT
from nabucodonosor.retrieval.reranker import ScoredChunk


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str:
        context = "\n\n---\n\n".join(c.content for c in chunks)
        prompt = RAG_TEMPLATE.format(context=context, question=question)
        return self.complete(prompt)


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str, model: str) -> None:
        import ollama

        self._client = ollama.Client(host=base_url)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.message.content or ""  # type: ignore[union-attr,unused-ignore]


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str) -> None:
        try:
            from groq import Groq  # type: ignore[import-untyped,unused-ignore]

            self._client = Groq(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "groq package is required for LLM_PROVIDER=groq. Install with: uv sync --extra groq"
            ) from exc
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


def get_llm(settings: Settings) -> BaseLLM:
    if settings.llm_provider == "groq":
        return GroqLLM(settings.groq_api_key, settings.groq_model)
    return OllamaLLM(settings.ollama_base_url, settings.llm_model)
