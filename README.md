# IR Retrieval Service

Two-level retrieval pipeline for searching text documents. Each document is summarized by an LLM and embedded at the doc level for broad candidate selection, then split into chunks for hybrid dense+sparse retrieval with RRF fusion and optional cross-encoder reranking.

## Stack

- **API**: FastAPI + uvicorn
- **Vector DB**: Qdrant 1.11
- **Embeddings**: `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers)
- **Sparse**: BM25 (fastembed)
- **Fusion**: Reciprocal Rank Fusion (Qdrant native)
- **Cross-encoder**: `BAAI/bge-reranker-v2-m3` (optional)
- **Summarization**: OpenAI-compatible API with text fallback
- **Config**: pydantic-settings + PyYAML
- **Runtime**: Docker Compose (cpu / gpu profiles)

## How it works

At indexing time each document is summarized via LLM (or falls back to first 4000 chars), the summary is embedded into the `doc_level` collection, and the full text is chunked with dense + sparse vectors stored in `chunk_level`.

At query time: doc-level dense retrieval picks the top candidate documents, then chunk-level hybrid retrieval (dense + sparse → RRF) runs filtered to those documents, and an optional cross-encoder reranks the final results.

## Setup

### 1. Create `.env`

```bash
cp .env.example .env
```

Point `OPENAI_BASE_URL` at your LLM backend. If running vLLM or similar locally, use `host.docker.internal` instead of `localhost` so the container can reach it:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=http://host.docker.internal:37555/v1
```

Set the served model name in `config/config.yaml` → `doc_summary.model` to match your backend's `--served-model-name`. To disable LLM summarization entirely: `doc_summary.enabled: false`.

### 2. Add documents

Drop `.txt` or `.md` files into `data/documents/`. Subdirectories are supported. Alternatively use the `/upload-file` API or the Streamlit UI.

### 3. Run

```bash
# CPU
docker compose --profile cpu up --build

# GPU (NVIDIA)
docker compose --profile gpu up --build
```

API available at `http://localhost:8000`. On first start all documents in `data/documents/` are indexed automatically.

### 4. UI (optional)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Tabs: Search, List Files, Upload File, Delete File, System Status.

## Configuration

All tunable parameters live in `config/config.yaml`, secrets in `.env`.

```yaml
splitting:
  chunk_size: 1000
  chunk_overlap: 100

embeddings:
  model_name: Qwen/Qwen3-Embedding-0.6B
  vector_size: 1024

cross_encoder:
  enabled_by_default: true
  model_name: BAAI/bge-reranker-v2-m3

doc_summary:
  enabled: true
  model: "qwen"           # must match --served-model-name in your LLM backend
  concurrency: 16         # parallel summary requests during indexing

search_defaults:
  top_k_doc: 100
  top_k_dense: 40
  top_k_sparse: 40
  final_top_k: 20
```

Changing `chunk_size`, `chunk_overlap`, `embeddings.model_name`, `doc_summary.*`, or `sparse.enabled` triggers a full reindex on next startup (index hash changes → new Qdrant collections are created).

## API

`GET /health` — liveness check.

`GET /index-info` — returns doc count on disk vs. in index.

`GET /list-files?page=1&page_size=50&search=` — paginated file listing.

`POST /search` — main search endpoint:
```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "your query", "final_top_k": 10, "use_cross_encoder": true}'
```
All fields except `query` are optional and fall back to `search_defaults` in config.

`POST /upload-file` — upload a `.txt` or `.md` file (multipart), saves to the documents volume and indexes immediately. Returns `indexed`, `reindexed`, or `already_indexed`.

`POST /delete-file` — removes a document from the index and from disk:
```bash
curl -X POST http://localhost:8000/delete-file \
  -H 'Content-Type: application/json' \
  -d '{"path": "/app/data/documents/old_doc.txt"}'
```

Full interactive docs at `http://localhost:8000/docs`.

## Evaluation

Download and convert a BEIR dataset:
```bash
python3 scripts/prepare_dataset.py --dataset scifact
# also available: nfcorpus, fiqa, arguana
```

Run evaluation against the live API:
```bash
python3 -m app.evaluator --queries eval/queries.jsonl --qrels eval/qrels.jsonl

# with cross-encoder
python3 -m app.evaluator --cross-encoder

# custom k values
python3 -m app.evaluator --k 5 10 20
```

### Results on SciFact (BEIR, 5183 docs, 300 queries)

Config: `Qwen3-Embedding-0.6B`, BM25 sparse, hybrid RRF, `Qwen2.5-14B-Instruct-GPTQ-Int8` summaries.

Without cross-encoder:

| Metric | Value |
|---|---|
| Recall@5 | 0.7425 |
| Recall@10 | 0.8033 |
| Precision@5 | 0.1647 |
| Precision@10 | 0.0907 |
| MRR | 0.6585 |
| nDCG@5 | 0.6661 |
| nDCG@10 | 0.6876 |

With cross-encoder (`BAAI/bge-reranker-v2-m3`):

| Metric | Value |
|---|---|
| Recall@5 | 0.7856 |
| Recall@10 | 0.8226 |
| Precision@5 | 0.1740 |
| Precision@10 | 0.0923 |
| MRR | 0.6900 |
| nDCG@5 | 0.7025 |
| nDCG@10 | 0.7164 |

## Startup scenarios

On each startup the service computes an index hash from the current config and compares it to the saved state:

- **No state file** — index all documents
- **Hash matches** — check for new or changed documents only
- **Hash changed** (config modified) — create new Qdrant collections and reindex everything
