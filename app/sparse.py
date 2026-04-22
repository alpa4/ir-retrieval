from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


def load_sparse_model(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


def build_sparse_vector(model: SparseTextEmbedding, text: str) -> SparseVector:
    result = next(model.embed([text]))
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
