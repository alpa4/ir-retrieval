from typing import Optional
from pydantic import BaseModel


class IndexConfig(BaseModel):
    docs_path: str
    state_path: str


class SplittingConfig(BaseModel):
    class_: str = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 100

    model_config = {"populate_by_name": True}

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if isinstance(obj, dict) and "class" in obj:
            obj = dict(obj)
            obj["class_"] = obj.pop("class")
        return super().model_validate(obj, **kwargs)


class EmbeddingsConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    vector_size: int = 384
    device: Optional[str] = None
    batch_size: int = 32


class CrossEncoderConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    enabled_by_default: bool = True
    model_name: str
    device: Optional[str] = None
    batch_size: int = 16


class SparseConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qdrant/bm25"


class DocSummaryConfig(BaseModel):
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0
    max_tokens: int = 300
    prompt_version: str = "v1"


class QdrantConfig(BaseModel):
    host: str = "qdrant"
    port: int = 6333


class SearchDefaultsConfig(BaseModel):
    top_k_doc: int = 5
    top_k_dense: int = 10
    top_k_sparse: int = 10
    final_top_k: int = 10
    mode: str = "default"


class AppConfig(BaseModel):
    index: IndexConfig
    splitting: SplittingConfig
    embeddings: EmbeddingsConfig
    cross_encoder: CrossEncoderConfig
    sparse: SparseConfig
    doc_summary: DocSummaryConfig
    qdrant: QdrantConfig
    search_defaults: SearchDefaultsConfig
