import hashlib
import json
from app.models import AppConfig


def compute_index_hash(config: AppConfig) -> str:
    """
    Compute a hash from config parameters that affect the index content.
    If any of these change, the index must be rebuilt.
    """
    payload = {
        "chunk_size": config.splitting.chunk_size,
        "chunk_overlap": config.splitting.chunk_overlap,
        "splitting_class": config.splitting.class_,
        "embedding_model": config.embeddings.model_name,
        "summary_enabled": config.doc_summary.enabled,
        "summary_model": config.doc_summary.model,
        "summary_temperature": config.doc_summary.temperature,
        "summary_max_tokens": config.doc_summary.max_tokens,
        "summary_prompt_version": config.doc_summary.prompt_version,
        "sparse_enabled": config.sparse.enabled,
        "schema_version": 1,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]
