"""RAG knowledge base: kitchen SOPs and policies stored in ChromaDB.

Pipeline
1. Load:     Markdown files in knowledge/ with a small front-matter header (doc_id, title, version...).
2. Chunk:    one chunk per "## " section, prefixed with the document title so it stands alone.
3. Metadata: doc_id, title, section, version, updated, source file, content checksum.
4. Embed:    ChromaDB's default local embedding model (all-MiniLM-L6-v2, no API key needed).
5. Index:    upsert into a persistent Chroma collection (cosine distance). Re-index when files change.
6. Retrieve: top-k chunks above a minimum similarity score, returned with citation ids "SOP-01#2".
"""

import hashlib
import logging
import re
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_collection = None
_lock = threading.Lock()


class KnowledgeBaseError(Exception):
    """Raised when the vector store is unavailable."""


# --- Load and chunk ------------------------------------------------------------------------

def _parse_front_matter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def load_chunks(knowledge_dir: str | None = None) -> list[dict]:
    """Split every Markdown document into section chunks with metadata."""
    chunks = []
    for path in sorted(Path(knowledge_dir or settings.knowledge_dir).glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("doc_id", path.stem)
        title = meta.get("title", path.stem)
        sections = re.split(r"^## +", body, flags=re.MULTILINE)
        number = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading, _, content = section.partition("\n")
            content = " ".join(content.split())
            if not content:
                continue
            number += 1
            text = f"{title} - {heading.strip()}: {content}"
            chunks.append({
                "id": f"{doc_id}#{number}",
                "text": text,
                "metadata": {
                    "doc_id": doc_id, "title": title, "section": heading.strip(),
                    "version": meta.get("version", ""), "updated": meta.get("updated", ""),
                    "source": path.name,
                    "checksum": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                },
            })
    return chunks


# --- Index ---------------------------------------------------------------------------------

def _get_collection():
    global _collection
    with _lock:
        if _collection is None:
            try:
                import chromadb
            except ImportError as error:
                raise KnowledgeBaseError("chromadb is not installed. Run: pip install -r requirements.txt") from error
            Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=settings.chroma_path)
            _collection = client.get_or_create_collection(
                name=settings.chroma_collection, metadata={"hnsw:space": "cosine"})
        return _collection


def reset_cache() -> None:
    global _collection
    with _lock:
        _collection = None


def ingest(rebuild: bool = False) -> dict:
    """Index the knowledge folder. Unchanged chunks are skipped; removed chunks are deleted."""
    collection = _get_collection()
    chunks = load_chunks()
    existing = collection.get(include=["metadatas"])
    existing_checksums = {cid: (meta or {}).get("checksum")
                          for cid, meta in zip(existing.get("ids", []), existing.get("metadatas", []))}
    if rebuild and existing_checksums:
        collection.delete(ids=list(existing_checksums))
        existing_checksums = {}

    changed = [c for c in chunks if existing_checksums.get(c["id"]) != c["metadata"]["checksum"]]
    if changed:
        collection.upsert(ids=[c["id"] for c in changed], documents=[c["text"] for c in changed],
                          metadatas=[c["metadata"] for c in changed])
    removed = sorted(set(existing_checksums) - {c["id"] for c in chunks})
    if removed:
        collection.delete(ids=removed)
    summary = {"chunks": len(chunks), "updated": len(changed), "removed": len(removed)}
    logger.info("Knowledge base indexed: %s", summary)
    return summary


def ensure_indexed() -> None:
    """Index on first use so the app works without a separate ingest step."""
    try:
        if _get_collection().count() == 0:
            ingest()
    except KnowledgeBaseError:
        raise
    except Exception as error:  # noqa: BLE001 - embedding model download or disk problems
        raise KnowledgeBaseError(f"Knowledge base unavailable: {error}") from error


# --- Retrieve ------------------------------------------------------------------------------

def search(query: str, top_k: int | None = None, min_score: float | None = None) -> list[dict]:
    """Return the most relevant chunks with a similarity score (1 - cosine distance)."""
    if not query.strip():
        return []
    ensure_indexed()
    top_k = top_k or settings.rag_top_k
    min_score = settings.rag_min_score if min_score is None else min_score
    try:
        result = _get_collection().query(query_texts=[query], n_results=top_k)
    except Exception as error:  # noqa: BLE001
        raise KnowledgeBaseError(f"Knowledge search failed: {error}") from error

    hits = []
    for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                              result["metadatas"][0], result["distances"][0]):
        score = round(1 - float(distance), 3)
        if score >= min_score:
            hits.append({"id": chunk_id, "doc_id": meta.get("doc_id"), "title": meta.get("title"),
                         "section": meta.get("section"), "version": meta.get("version"),
                         "score": score, "text": text})
    return hits


def search_many(queries: list[str], top_k: int | None = None) -> list[dict]:
    """Run several queries and merge the results, keeping each chunk's best score."""
    best: dict[str, dict] = {}
    for query in queries:
        for hit in search(query, top_k=top_k):
            if hit["id"] not in best or hit["score"] > best[hit["id"]]["score"]:
                best[hit["id"]] = hit
    return sorted(best.values(), key=lambda h: h["score"], reverse=True)[: (top_k or settings.rag_top_k)]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Index or search the kitchen knowledge base")
    parser.add_argument("--rebuild", action="store_true", help="delete and re-index everything")
    parser.add_argument("--search", help="run a test query")
    args = parser.parse_args()
    if args.search:
        print(json.dumps(search(args.search), indent=2))
    else:
        print(json.dumps(ingest(rebuild=args.rebuild), indent=2))
