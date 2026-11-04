import re
from dataclasses import dataclass, field

from nabucodonosor.ingestion.loader import Document


@dataclass
class Chunk:
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sliding-window split for text that exceeds chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks


def chunk(document: Document, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    """
    Split a Document into Chunks, preserving section context.

    Strategy:
    1. Split by markdown headers (# / ## / ###)
    2. If a section fits in chunk_size, keep it as one chunk
    3. Otherwise accumulate paragraphs into a buffer; flush when full
    4. Paragraphs that still exceed chunk_size are split by chars with overlap
    """
    source = document.metadata.get("source", "")
    chunks: list[Chunk] = []
    chunk_index = 0

    raw_sections = re.split(r"(?=\n#{1,3} )", document.content)

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        first_line = section.split("\n")[0].strip()
        heading = first_line.lstrip("#").strip() if first_line.startswith("#") else ""

        def _make_chunk(content: str) -> Chunk:
            nonlocal chunk_index
            c = Chunk(
                content=content,
                metadata={"source": source, "section": heading, "chunk_index": str(chunk_index)},
            )
            chunk_index += 1
            return c

        if len(section) <= chunk_size:
            chunks.append(_make_chunk(section))
            continue

        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        buffer = ""

        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph

            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(_make_chunk(buffer))

                if len(paragraph) > chunk_size:
                    for sub in _split_by_chars(paragraph, chunk_size, chunk_overlap):
                        chunks.append(_make_chunk(sub))
                    buffer = ""
                else:
                    buffer = paragraph

        if buffer:
            chunks.append(_make_chunk(buffer))

    return chunks
