from dataclasses import dataclass, field
from typing import Optional

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.embeddings import embed_single
from app.sparse import build_sparse_vector
from app import qdrant_store as store


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    file_path: str
    chunk_index: int
    chunk_text: str
    scores: dict = field(default_factory=dict)


def search(
    query: str,
    client: QdrantClient,
    embed_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    doc_collection: str,
    chunk_collection: str,
    top_k_doc: int = 5,
    top_k_dense: int = 10,
    top_k_sparse: int = 10,
    final_top_k: int = 10,
    use_cross_encoder: bool = False,
    cross_encoder=None,
    mode: str = "default",
) -> list[SearchResult]:
    # 1. Encode query
    dense_q = embed_single(embed_model, query)
    sparse_q = build_sparse_vector(sparse_model, query)

    # 2. Doc-level coarse retrieval
    doc_hits = store.search_docs(client, doc_collection, dense_q, top_k_doc)
    if not doc_hits:
        return []

    doc_ids = [h.payload["doc_id"] for h in doc_hits]
    doc_scores = {h.payload["doc_id"]: h.score for h in doc_hits}

    # 3. Chunk-level hybrid retrieval (dense + sparse RRF)
    chunk_hits = store.search_chunks_hybrid(
        client, chunk_collection, dense_q, sparse_q, doc_ids, top_k_dense, top_k_sparse
    )

    # 4. Assemble results
    results = []
    for hit in chunk_hits:
        p = hit.payload
        results.append(SearchResult(
            chunk_id=p["chunk_id"],
            doc_id=p["doc_id"],
            file_path=p["file_path"],
            chunk_index=p["chunk_index"],
            chunk_text=p["chunk_text"],
            scores={
                "doc_score": doc_scores.get(p["doc_id"]),
                "qdrant_fusion_score": hit.score,
                "cross_encoder_score": None,
            },
        ))

    # 5. Optional cross-encoder rerank
    if use_cross_encoder and cross_encoder is not None:
        pairs = [(query, r.chunk_text) for r in results]
        ce_scores = cross_encoder.predict(pairs).tolist()
        for r, score in zip(results, ce_scores):
            r.scores["cross_encoder_score"] = score
        results.sort(key=lambda r: r.scores["cross_encoder_score"], reverse=True)

    return results[:final_top_k]
