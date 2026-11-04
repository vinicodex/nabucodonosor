from pathlib import Path

import pytest

from nabucodonosor.ingestion.loader import Document, load


def test_load_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    result = load(tmp_path / "nonexistent")
    assert result == []


def test_load_reads_md_files(tmp_path: Path) -> None:
    (tmp_path / "cv.md").write_text("# John Doe\nSenior Engineer", encoding="utf-8")
    result = load(tmp_path)
    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].metadata["source"] == "cv.md"
    assert "John Doe" in result[0].content


def test_load_skips_empty_files(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n  ", encoding="utf-8")
    result = load(tmp_path)
    assert result == []


def test_load_ignores_non_md_files(tmp_path: Path) -> None:
    (tmp_path / "cv.md").write_text("# CV", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("some text", encoding="utf-8")
    result = load(tmp_path)
    assert len(result) == 1


def test_load_sorts_files_alphabetically(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    result = load(tmp_path)
    assert [r.metadata["source"] for r in result] == ["a.md", "b.md"]


@pytest.mark.parametrize(
    "content",
    [
        "# Simple heading",
        "## Section\nSome body text.",
        "No heading at all, just plain text.",
    ],
)
def test_load_stores_full_content(tmp_path: Path, content: str) -> None:
    (tmp_path / "file.md").write_text(content, encoding="utf-8")
    result = load(tmp_path)
    assert result[0].content == content
