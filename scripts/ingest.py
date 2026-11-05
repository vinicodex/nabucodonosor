#!/usr/bin/env python
"""Ingest knowledge/*.md files into ChromaDB.

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --knowledge-dir path/to/docs
    uv run python scripts/ingest.py --reset   # drop and recreate the collection
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabucodonosor.cache import exact as exact_cache
from nabucodonosor.config import settings
from nabucodonosor.embedding.embedder import get_embedder
from nabucodonosor.ingestion.chunker import chunk
from nabucodonosor.ingestion.loader import load
from nabucodonosor.vectorstore.chroma import build_collection, upsert_chunks


def main(knowledge_dir: Path, reset: bool) -> None:
    print(f"Loading documents from {knowledge_dir} ...")
    documents = load(knowledge_dir)
    if not documents:
        print("No documents found. Exiting.")
        sys.exit(1)

    all_chunks = []
    for doc in documents:
        chunks = chunk(doc)
        print(f"  {doc.metadata['source']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    if reset:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_path)
        try:
            client.delete_collection(settings.collection_name)
            print(f"Dropped collection '{settings.collection_name}'")
        except Exception:
            pass

    collection = build_collection(settings)
    embedder = get_embedder(settings)

    print(f"\nEmbedding and upserting into '{settings.collection_name}' ...")
    inserted = upsert_chunks(collection, all_chunks, embedder)
    print(f"Done. {inserted} chunks stored in ChromaDB at {settings.chroma_path}")

    deleted = exact_cache.purge_expired(settings.cache_db_path, ttl_hours=0)
    if deleted:
        print(f"Cache invalidated: {deleted} stale entr{'y' if deleted == 1 else 'ies'} purged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CV documents into ChromaDB")
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path(__file__).parent.parent / "knowledge",
        help="Directory containing .md files to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the ChromaDB collection before ingesting",
    )
    args = parser.parse_args()
    main(args.knowledge_dir, args.reset)
