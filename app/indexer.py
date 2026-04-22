import logging
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
) -> None:
    doc_text = summarizer.summarize(doc.content)
    doc_vector = embed_single(embed_model, doc_text)

    # 2. Upsert document into doc_level collection
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

    # 3. Split into chunks
    chunks = split_text(doc.doc_id, doc.content, config.splitting)
    if not chunks:
        return

    # 4. Embed all chunks at once
    texts = [c.chunk_text for c in chunks]
    dense_vectors = embed_texts(embed_model, texts, config.embeddings.batch_size)

    # 5. Build sparse vectors and prepare points
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

    # 6. Upsert chunks into chunk_level collection
    store.upsert_chunks(client, chunk_collection, points)


def delete_document(
    doc_id: str,
    doc_id_int: int,
    client: QdrantClient,
    doc_collection: str,
    chunk_collection: str,
) -> None:
    store.delete_document(client, doc_collection, doc_id_int)
    store.delete_chunks_by_doc(client, chunk_collection, doc_id)


def sync_documents(
    config: AppConfig,
    client: QdrantClient,
    embed_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    summarizer: DocumentSummarizer,
    doc_collection: str,
    chunk_collection: str,
    index_hash: str,
) -> None:
    """Index all documents that are not yet in the doc_level collection."""
    docs = scan_documents(config.index.docs_path)
    logger.info(f"Syncing {len(docs)} document(s)...")

    indexed = 0
    for doc in docs:
        existing = store.get_document(client, doc_collection, doc.doc_id_int)

        if existing is None:
            logger.info(f"Indexing: {doc.relative_path}")
            index_document(doc, config, client, embed_model, sparse_model, summarizer,
                           doc_collection, chunk_collection, index_hash)
            indexed += 1
        elif existing.payload.get("content_hash") != doc.content_hash:
            logger.info(f"Reindexing (changed): {doc.relative_path}")
            delete_document(doc.doc_id, doc.doc_id_int, client, doc_collection, chunk_collection)
            index_document(doc, config, client, embed_model, sparse_model, summarizer,
                           doc_collection, chunk_collection, index_hash)
            indexed += 1

    logger.info(f"Sync complete. Indexed/updated: {indexed}, skipped: {len(docs) - indexed}")
