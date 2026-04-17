import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config_loader import load_config
from app.hashing import compute_index_hash
from app.state import load_state, save_state, IndexState
from app.qdrant_store import get_client, create_doc_collection, create_chunk_collection
from app.embeddings import load_embedding_model
from app.summarizer import DocumentSummarizer
from app.indexer import sync_documents

logger = logging.getLogger("uvicorn.error")


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

    client = get_client(config.qdrant)
    create_doc_collection(client, doc_collection)
    create_chunk_collection(client, chunk_collection)

    logger.info("Loading embedding model...")
    embed_model = load_embedding_model(config.embeddings)
    logger.info("Embedding model ready")
    
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

    sync_documents(config, client, embed_model, summarizer,
        doc_collection, chunk_collection, index_hash)

    state = IndexState(
        index_hash=index_hash,
        doc_collection=doc_collection,
        chunk_collection=chunk_collection,
    )
    save_state(config.index.state_path, state)

    yield


app = FastAPI(title="IR Retrieval Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}
