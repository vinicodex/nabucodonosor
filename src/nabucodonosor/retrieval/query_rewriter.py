from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


_REWRITE_PROMPT = """\
You are a query optimizer for a CV/resume search engine.
Rewrite the question below to maximize recall. Focus on synonyms,
related skills, and job titles. Output only the rewritten question — no explanation.

Question: {question}
Rewritten:"""


class QueryRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def rewrite(self, question: str) -> str:
        try:
            rewritten = self._llm.complete(_REWRITE_PROMPT.format(question=question))
            rewritten = rewritten.strip()
            return rewritten if rewritten else question
        except Exception:
            return question
