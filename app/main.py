import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config_loader import load_config
from app.hashing import compute_index_hash
from app.state import load_state, save_state, IndexState
from app.qdrant_store import get_client, create_doc_collection, create_chunk_collection
from app.embeddings import load_embedding_model
from app.summarizer import DocumentSummarizer
from app.sparse import load_sparse_model
from app.reranker import load_cross_encoder
from app.indexer import sync_documents

logger = logging.getLogger("uvicorn.error")


async def _wait_for_qdrant(config, retries: int = 15, delay: float = 3.0):
    for attempt in range(1, retries + 1):
        try:
            client = get_client(config.qdrant)
            client.get_collections()
            return client
        except Exception:
            if attempt == retries:
                raise
            logger.info(f"Qdrant not ready, retrying ({attempt}/{retries})...")
            await asyncio.sleep(delay)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config, env = load_config()
    index_hash = compute_index_hash(config)
    doc_collection = f"doc_level_{index_hash}"
    chunk_collection = f"chunk_level_{index_hash}"
    logger.info(f"Index hash: {index_hash}")

    existing_state = load_state(config.index.state_path)

    if existing_state is None:
        logger.info("Scenario A: no state file → indexing all documents")
    elif existing_state.index_hash != index_hash:
        logger.info("Scenario C: config changed → reindexing everything")
    else:
        logger.info("Scenario B: hash matches → checking for missing documents")

    client = await _wait_for_qdrant(config)
    vector_size = config.embeddings.vector_size
    create_doc_collection(client, doc_collection, vector_size)
    create_chunk_collection(client, chunk_collection, vector_size)

    logger.info("Loading embedding model...")
    embed_model = load_embedding_model(config.embeddings)
    logger.info("Embedding model ready")

    logger.info("Loading sparse model...")
    sparse_model = load_sparse_model(config.sparse.model_name)
    logger.info("Sparse model ready")

    logger.info("Initializing document summarizer...")
    summarizer = DocumentSummarizer(
        enabled=config.doc_summary.enabled,
        api_key=env.openai_api_key,
        base_url=env.openai_base_url,
        model=config.doc_summary.model,
        temperature=config.doc_summary.temperature,
        max_tokens=config.doc_summary.max_tokens,
        prompt_version=config.doc_summary.prompt_version,
    )
    logger.info("Summarizer ready")

    logger.info("Loading cross-encoder...")
    cross_encoder = load_cross_encoder(config.cross_encoder)
    logger.info("Cross-encoder ready" if cross_encoder else "Cross-encoder disabled")

    await sync_documents(config, client, embed_model, sparse_model, summarizer,
                         doc_collection, chunk_collection, index_hash)

    state = IndexState(
        index_hash=index_hash,
        doc_collection=doc_collection,
        chunk_collection=chunk_collection,
    )
    save_state(config.index.state_path, state)

    app.state.config = config
    app.state.client = client
    app.state.embed_model = embed_model
    app.state.sparse_model = sparse_model
    app.state.summarizer = summarizer
    app.state.cross_encoder = cross_encoder
    app.state.doc_collection = doc_collection
    app.state.chunk_collection = chunk_collection
    app.state.index_hash = index_hash

    yield


app = FastAPI(title="IR Retrieval Service", lifespan=lifespan)

from app.api import router  # noqa: E402 — imported after app is defined
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
