from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Prefetch,
    FusionQuery,
    Fusion,
)
from app.models import QdrantConfig

# Dimension of BAAI/bge-small-en-v1.5 embeddings
DENSE_DIM = 384


def get_client(config: QdrantConfig) -> QdrantClient:
    return QdrantClient(host=config.host, port=config.port)


def collection_exists(client: QdrantClient, name: str) -> bool:
    return client.collection_exists(name)


def create_doc_collection(client: QdrantClient, name: str) -> None:
    """One point per document. Only dense vector (summary embedding)."""
    if collection_exists(client, name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    )


def create_chunk_collection(client: QdrantClient, name: str) -> None:
    """One point per chunk. Dense + sparse vectors."""
    if collection_exists(client, name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams())
        },
    )


def upsert_document(client: QdrantClient, collection: str, doc_id: int,
                    vector: list[float], payload: dict) -> None:
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=doc_id, vector=vector, payload=payload)],
    )


def upsert_chunks(client: QdrantClient, collection: str,
                  points: list[dict]) -> None:
    """
    Each point dict: {id, dense_vector, sparse_indices, sparse_values, payload}
    """
    structs = []
    for p in points:
        structs.append(PointStruct(
            id=p["id"],
            vector={
                "dense": p["dense_vector"],
                "sparse": SparseVector(
                    indices=p["sparse_indices"],
                    values=p["sparse_values"],
                ),
            },
            payload=p["payload"],
        ))
    client.upsert(collection_name=collection, points=structs)


def delete_document(client: QdrantClient, collection: str, doc_id: int) -> None:
    client.delete(
        collection_name=collection,
        points_selector=[doc_id],
    )


def delete_chunks_by_doc(client: QdrantClient, collection: str, doc_id: str) -> None:
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )


def get_document(client: QdrantClient, collection: str, doc_id: str):
    results = client.retrieve(collection_name=collection, ids=[doc_id], with_payload=True)
    return results[0] if results else None


def count_documents(client: QdrantClient, collection: str) -> int:
    result = client.count(collection_name=collection)
    return result.count


def search_docs(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int,
) -> list:
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return result.points


def search_chunks_hybrid(
    client: QdrantClient,
    collection: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    doc_ids: list[str],
    top_k_dense: int,
    top_k_sparse: int,
) -> list:
    doc_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
    )
    result = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k_dense, filter=doc_filter),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k_sparse, filter=doc_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k_dense + top_k_sparse,
        with_payload=True,
    )
    return result.points
