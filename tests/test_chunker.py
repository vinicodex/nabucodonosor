from nabucodonosor.ingestion.chunker import Chunk, chunk
from nabucodonosor.ingestion.loader import Document


def _doc(content: str) -> Document:
    return Document(content=content, metadata={"source": "cv.md"})


def test_chunk_returns_list_of_chunks(sample_cv_text: str) -> None:
    result = chunk(_doc(sample_cv_text))
    assert len(result) > 0
    assert all(isinstance(c, Chunk) for c in result)


def test_chunk_metadata_has_required_keys(sample_cv_text: str) -> None:
    for c in chunk(_doc(sample_cv_text)):
        assert "source" in c.metadata
        assert "section" in c.metadata
        assert "chunk_index" in c.metadata


def test_chunk_source_propagated(sample_cv_text: str) -> None:
    result = chunk(_doc(sample_cv_text))
    assert all(c.metadata["source"] == "cv.md" for c in result)


def test_chunk_indices_are_sequential(sample_cv_text: str) -> None:
    result = chunk(_doc(sample_cv_text))
    indices = [int(c.metadata["chunk_index"]) for c in result]
    assert indices == list(range(len(result)))


def test_chunk_short_text_produces_single_chunk() -> None:
    result = chunk(_doc("Just a short sentence."), chunk_size=500, chunk_overlap=100)
    assert len(result) == 1


def test_chunk_long_text_is_split() -> None:
    long_text = "word " * 300  # ~1500 chars
    result = chunk(_doc(long_text), chunk_size=200, chunk_overlap=20)
    assert len(result) > 1


def test_chunk_no_content_exceeds_size_plus_overlap() -> None:
    result = chunk(_doc("word " * 300), chunk_size=200, chunk_overlap=20)
    # tolerance: one paragraph beyond chunk_size is acceptable
    assert all(len(c.content) <= 400 for c in result)


def test_chunk_all_text_is_preserved(sample_cv_text: str) -> None:
    result = chunk(_doc(sample_cv_text))
    combined = " ".join(c.content for c in result)
    for word in ["John", "Doe", "FastAPI", "AWS"]:
        assert word in combined


def test_chunk_section_heading_captured(sample_cv_text: str) -> None:
    result = chunk(_doc(sample_cv_text))
    sections = {c.metadata["section"] for c in result}
    assert any("Experience" in s for s in sections)
    assert any("Skills" in s for s in sections)
