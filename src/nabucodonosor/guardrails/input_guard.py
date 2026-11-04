import re
from dataclasses import dataclass

from nabucodonosor.config import Settings
from nabucodonosor.embedding.embedder import BaseEmbedder, cosine_similarity

_INJECTION_PATTERNS = re.compile(
    r"ignore (previous|all|prior)|disregard|forget (previous|all)|"
    r"pretend (you are|to be)|you are now|system\s*:|<\||\|>|"
    r"new instructions|override instructions",
    re.IGNORECASE,
)

_TOPIC_ANCHORS = [
    "professional experience career skills technology background",
    "software engineer developer programming job work",
    "resume CV portfolio projects education qualifications",
]


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def check_input(question: str, embedder: BaseEmbedder, settings: Settings) -> GuardResult:
    if len(question) > settings.max_question_length:
        return GuardResult(
            allowed=False,
            reason=f"Question exceeds {settings.max_question_length} characters.",
        )

    if _INJECTION_PATTERNS.search(question):
        return GuardResult(allowed=False, reason="Potential prompt injection detected.")

    q_emb = embedder.embed(question)
    anchor_embs = [embedder.embed(anchor) for anchor in _TOPIC_ANCHORS]
    max_sim = max(cosine_similarity(q_emb, a) for a in anchor_embs)

    if max_sim < settings.topic_guard_threshold:
        return GuardResult(
            allowed=False,
            reason="Question appears unrelated to Marcos's professional background.",
        )

    return GuardResult(allowed=True)
