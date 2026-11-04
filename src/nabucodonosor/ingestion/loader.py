from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    content: str
    metadata: dict[str, str]


def load(knowledge_dir: Path) -> list[Document]:
    """Load all .md files from knowledge_dir as Documents."""
    if not knowledge_dir.exists():
        return []

    documents = []
    for path in sorted(knowledge_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append(Document(content=content, metadata={"source": path.name}))

    return documents
