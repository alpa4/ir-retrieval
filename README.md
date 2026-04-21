# IR Retrieval Service

Локальный двухуровневый retrieval pipeline для поиска по текстовым документам с гибридным поиском (dense + sparse) и опциональным cross-encoder reranking.

## Стек

| Компонент | Технология |
|---|---|
| API | FastAPI + uvicorn |
| Векторная БД | Qdrant 1.11 |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Sparse | TF-weighted bag of words |
| Hybrid fusion | Qdrant RRF (Reciprocal Rank Fusion) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (опционально) |
| Summarization | OpenAI-compatible API (OpenAI / LM Studio / Ollama) |
| Конфиг | pydantic-settings + PyYAML |
| Запуск | Docker Compose (cpu / gpu профили) |

## Как работает

```
Запрос
  │
  ▼
Doc-level dense retrieval (top_k_doc документов)
  │
  ▼
Chunk-level hybrid retrieval (dense + sparse → RRF fusion)
  │  фильтрация по найденным doc_id
  ▼
[Optional] Cross-encoder reranking
  │
  ▼
Результаты с полными score-ами
```

При индексации каждый документ:
1. Суммаризуется через LLM (или первые 4000 символов как fallback)
2. Его summary эмбеддится → одна точка в `doc_level` коллекции
3. Текст режется на чанки → dense + sparse векторы → `chunk_level` коллекция

## Структура проекта

```
ir-retrieval/
├── app/
│   ├── main.py           # точка входа, lifespan startup
│   ├── api.py            # FastAPI endpoints
│   ├── search.py         # поисковый pipeline
│   ├── indexer.py        # индексация документов
│   ├── reranker.py       # cross-encoder reranking
│   ├── summarizer.py     # LLM summarization с fallback
│   ├── qdrant_store.py   # операции с Qdrant
│   ├── embeddings.py     # dense embeddings
│   ├── sparse.py         # sparse векторы (TF-weighted)
│   ├── splitter.py       # разбиение текста на чанки
│   ├── files.py          # обход файлов, doc_id, content_hash
│   ├── hashing.py        # вычисление index_hash
│   ├── models.py         # AppConfig pydantic модели
│   ├── settings.py       # EnvSettings (читает .env)
│   ├── config_loader.py  # загрузка config.yaml
│   ├── state.py          # index_state.json управление
│   └── evaluator.py      # CLI evaluation: Recall, Precision, MRR, nDCG
├── scripts/
│   └── prepare_dataset.py  # скачать BEIR датасет и конвертировать
├── config/
│   └── config.yaml       # параметры системы
├── data/documents/       # сюда кладём .txt / .md файлы
├── state/                # index_state.json (генерируется автоматически)
├── eval/
│   ├── queries.jsonl     # запросы для evaluation
│   └── qrels.jsonl       # релевантные документы для evaluation
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Быстрый старт

### 1. Создать `.env`

```bash
cp .env.example .env
```

Для OpenAI:
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

Для LM Studio (локальная LLM):
```env
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://host.docker.internal:1234/v1
```

> LM Studio: запустить локальный сервер на `0.0.0.0:1234`, загрузить модель.
> Имя модели указать в `config/config.yaml` → `doc_summary.model`.

Если LLM не нужна — выставить `doc_summary.enabled: false` в `config.yaml`.

### 2. Положить документы

```bash
cp my_docs/*.txt data/documents/
```

Поддерживаются `.txt` и `.md` файлы.

### 3. Запустить

```bash
docker compose --profile cpu up --build
```

Для NVIDIA GPU:
```bash
docker compose --profile gpu up --build
```

Сервис поднимется на `http://localhost:8000`.

## Конфигурация

Все параметры в `config/config.yaml`, секреты в `.env`.

Ключевые параметры:

```yaml
splitting:
  chunk_size: 800       # размер чанка в символах
  chunk_overlap: 100    # перекрытие

embeddings:
  model_name: BAAI/bge-small-en-v1.5

cross_encoder:
  enabled_by_default: false   # включить для лучшего качества (медленнее)
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2

doc_summary:
  enabled: false        # true = вызов LLM для summary каждого документа

search_defaults:
  top_k_doc: 5          # сколько документов отбирается на doc-level
  top_k_dense: 10       # dense кандидаты на chunk-level
  top_k_sparse: 10      # sparse кандидаты на chunk-level
  final_top_k: 10       # итоговое число результатов
```

Параметры влияющие на `index_hash` (изменение = полная переиндексация):
`chunk_size`, `chunk_overlap`, `embeddings.model_name`, `doc_summary.*`, `sparse.enabled`

## API

### `GET /health`
```json
{"status": "ok"}
```

### `GET /index-info`
```json
{"docs_on_disk": 5183, "docs_in_index": 5183, "status": "ok"}
```

### `POST /search`
```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "your query here", "final_top_k": 5}'
```

Параметры (все опциональны, defaults из config):
- `top_k_doc`, `top_k_dense`, `top_k_sparse`, `final_top_k`
- `use_cross_encoder: true/false`

### `POST /add-file`
```bash
curl -X POST http://localhost:8000/add-file \
  -H 'Content-Type: application/json' \
  -d '{"path": "/app/data/documents/new_doc.txt"}'
```
Возвращает `status`: `indexed` / `reindexed` / `already_indexed`

### `POST /delete-file`
```bash
curl -X POST http://localhost:8000/delete-file \
  -H 'Content-Type: application/json' \
  -d '{"path": "/app/data/documents/old_doc.txt"}'
```

## Evaluation

### Подготовка датасета BEIR

```bash
# Скачать и конвертировать SciFact (5183 документа, 300 запросов)
python3 scripts/prepare_dataset.py --dataset scifact

# Доступные датасеты: scifact, nfcorpus, fiqa, arguana
python3 scripts/prepare_dataset.py --dataset nfcorpus
```

Скрипт скачивает датасет, сохраняет документы в `data/documents/`,
обновляет `eval/queries.jsonl` и `eval/qrels.jsonl`.

### Запуск evaluation

```bash
# После индексации
python3 -m app.evaluator \
  --queries eval/queries.jsonl \
  --qrels eval/qrels.jsonl \
  --api http://localhost:8000
```

### Результаты на SciFact (BEIR)

Конфигурация: `BAAI/bge-small-en-v1.5`, hybrid RRF, без cross-encoder, без LLM summary.

| Метрика | Значение |
|---|---|
| Recall@5 | 0.765 |
| Recall@10 | 0.765 |
| Precision@5 | 0.170 |
| MRR | 0.650 |
| nDCG@5 | 0.672 |
| nDCG@10 | 0.672 |

## Сценарии при старте

| Сценарий | Условие | Действие |
|---|---|---|
| A | нет state файла | индексировать все документы |
| B | hash совпадает | доиндексировать только новые/изменённые |
| C | hash изменился | создать новые коллекции, переиндексировать всё |
