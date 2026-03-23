"""ChromaDB vector store service."""

import uuid
import logging

import chromadb

from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(
    chunks: list[str],
    source: str,
    source_type: str = "document",
    extra_metadata: dict | None = None,
) -> list[str]:
    """Add text chunks to ChromaDB. Returns list of chunk IDs."""
    collection = get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = {
            "source": source,
            "source_type": source_type,
            "chunk_index": i,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        metadatas.append(meta)

    # ChromaDB handles embedding via its default embedding function
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info(f"Added {len(chunks)} chunks from '{source}' to vector store")
    return ids


def query_chunks(query: str, top_k: int = 3) -> dict:
    """Query ChromaDB for similar chunks. Deduplicates results by content."""
    collection = get_collection()
    # Fetch extra results to have enough after dedup
    results = collection.query(query_texts=[query], n_results=top_k * 3)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    # Deduplicate by content
    seen: set[str] = set()
    deduped = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
        content_key = doc.strip()[:200]
        if content_key in seen:
            continue
        seen.add(content_key)
        deduped["ids"][0].append(id_)
        deduped["documents"][0].append(doc)
        deduped["metadatas"][0].append(meta)
        deduped["distances"][0].append(dist)
        if len(deduped["documents"][0]) >= top_k:
            break

    return deduped


def source_exists(source: str) -> bool:
    """Check if a source already exists in the collection."""
    collection = get_collection()
    results = collection.get(where={"source": source}, limit=1)
    return len(results["ids"]) > 0


def get_stats() -> dict:
    """Get collection statistics."""
    collection = get_collection()
    return {"total_chunks": collection.count()}


def list_sources() -> list[dict]:
    """List all ingested sources with chunk counts."""
    collection = get_collection()
    all_meta = collection.get()["metadatas"]
    source_map: dict[str, dict] = {}
    for meta in all_meta:
        src = meta.get("source", "unknown")
        if src not in source_map:
            source_map[src] = {"source": src, "source_type": meta.get("source_type", ""), "chunks": 0}
        source_map[src]["chunks"] += 1
    return sorted(source_map.values(), key=lambda x: x["source"])


def delete_by_source(source: str) -> int:
    """Delete all chunks from a specific source. Returns number of deleted chunks."""
    collection = get_collection()
    results = collection.get(where={"source": source})
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks from source '{source}'")
    return len(ids)


def delete_all() -> int:
    """Delete all chunks. Returns number of deleted chunks."""
    global _collection
    client = get_client()
    count = get_collection().count()
    client.delete_collection(COLLECTION_NAME)
    _collection = None  # Force re-creation on next access
    logger.info(f"Deleted all {count} chunks")
    return count
