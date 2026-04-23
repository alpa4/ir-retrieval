import asyncio
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from app.summarizer import DocumentSummarizer
from app.models import AppConfig
from app.files import DocumentFile, scan_documents
from app.splitter import split_text
from app.embeddings import embed_single, embed_texts
from app.sparse import build_sparse_vector
from app import qdrant_store as store

logger = logging.getLogger("uvicorn.error")


def index_document(
    doc: DocumentFile,
    config: AppConfig,
    client: QdrantClient,
    embed_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    summarizer: DocumentSummarizer,
    doc_collection: str,
    chunk_collection: str,
    index_hash: str,
    summary: Optional[str] = None,
) -> bool:
    """Returns True if indexed successfully, False if skipped due to a Qdrant error."""
    doc_text = summary if summary is not None else summarizer.summarize(doc.content)
    doc_vector = embed_single(embed_model, doc_text)

    try:
        store.upsert_document(
            client=client,
            collection=doc_collection,
            doc_id=doc.doc_id_int,
            vector=doc_vector,
            payload={
                "doc_id": doc.doc_id,
                "file_path": doc.file_path,
                "file_name": doc.relative_path,
                "summary": doc_text,
                "content_hash": doc.content_hash,
                "index_hash": index_hash,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to upsert document {doc.relative_path}: {e}")
        return False

    chunks = split_text(doc.doc_id, doc.content, config.splitting)
    if not chunks:
        return True

    texts = [c.chunk_text for c in chunks]
    dense_vectors = embed_texts(embed_model, texts, config.embeddings.batch_size)

    points = []
    for chunk, dense_vector in zip(chunks, dense_vectors):
        sparse = build_sparse_vector(sparse_model, chunk.chunk_text)
        points.append({
            "id": chunk.chunk_id_int,
            "dense_vector": dense_vector,
            "sparse_indices": sparse.indices,
            "sparse_values": sparse.values,
            "payload": {
                "chunk_id": chunk.chunk_id,
                "doc_id": doc.doc_id,
                "file_path": doc.file_path,
                "chunk_text": chunk.chunk_text,
                "chunk_index": chunk.chunk_index,
                "content_hash": doc.content_hash,
                "index_hash": index_hash,
            },
        })

    try:
        store.upsert_chunks(client, chunk_collection, points)
    except Exception as e:
        logger.warning(f"Failed to upsert chunks for {doc.relative_path}: {e}")
        store.delete_document(client, doc_collection, doc.doc_id_int)
        return False

    return True


def delete_document(
    doc_id: str,
    doc_id_int: int,
    client: QdrantClient,
    doc_collection: str,
    chunk_collection: str,
) -> None:
    store.delete_document(client, doc_collection, doc_id_int)
    store.delete_chunks_by_doc(client, chunk_collection, doc_id)


async def _prefetch_summaries(
    docs: list[DocumentFile],
    summarizer: DocumentSummarizer,
    concurrency: int,
) -> dict[str, str]:
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_one(doc: DocumentFile) -> tuple[str, str]:
        async with semaphore:
            summary = await summarizer.summarize_async(doc.content)
            return doc.doc_id, summary

    pairs = await asyncio.gather(*[fetch_one(doc) for doc in docs])
    return dict(pairs)


async def sync_documents(
    config: AppConfig,
    client: QdrantClient,
    embed_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    summarizer: DocumentSummarizer,
    doc_collection: str,
    chunk_collection: str,
    index_hash: str,
) -> None:
    docs = scan_documents(config.index.docs_path)
    logger.info(f"Syncing {len(docs)} document(s)...")

    pending: list[tuple[DocumentFile, bool]] = []
    for doc in docs:
        existing = store.get_document(client, doc_collection, doc.doc_id_int)
        if existing is None:
            pending.append((doc, False))
        elif existing.payload.get("content_hash") != doc.content_hash:
            pending.append((doc, True))

    if not pending:
        logger.info(f"Sync complete. Indexed/updated: 0, skipped: {len(docs)}")
        return

    concurrency = config.doc_summary.concurrency
    logger.info(f"Indexing {len(pending)} document(s) in batches of {concurrency}...")

    indexed = 0
    for batch_start in range(0, len(pending), concurrency):
        batch = pending[batch_start: batch_start + concurrency]
        batch_docs = [doc for doc, _ in batch]

        summaries = await _prefetch_summaries(batch_docs, summarizer, concurrency)

        for doc, needs_delete in batch:
            if needs_delete:
                logger.info(f"Reindexing (changed): {doc.relative_path}")
                delete_document(doc.doc_id, doc.doc_id_int, client, doc_collection, chunk_collection)
            else:
                logger.info(f"Indexing: {doc.relative_path}")
            if index_document(
                doc, config, client, embed_model, sparse_model, summarizer,
                doc_collection, chunk_collection, index_hash,
                summary=summaries.get(doc.doc_id),
            ):
                indexed += 1

    failed = len(pending) - indexed
    logger.info(f"Sync complete. Indexed/updated: {indexed}, skipped: {len(docs) - len(pending)}"
                + (f", failed: {failed}" if failed else ""))
