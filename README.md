# IR Retrieval Service

Локальный двухуровневый retrieval pipeline для поиска по текстовым документам.

## Что это

Сервис индексирует `.txt` и `.md` файлы и позволяет искать по ним с использованием:
- **Doc-level retrieval** — сначала находим релевантные документы
- **Chunk-level retrieval** — затем ищем чанки внутри этих документов (dense + sparse)
- **Optional cross-encoder reranking** — финальная пересортировка результатов

## Стек

| Компонент | Технология |
|---|---|
| API | FastAPI + uvicorn |
| Векторная БД | Qdrant |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Sparse | TF-weighted bag of words |
| Конфиг | pydantic-settings + PyYAML |
| Запуск | Docker Compose (cpu / gpu профили) |

## Структура проекта

```
ir-retrieval/
├── app/
│   ├── main.py           # точка входа, startup логика
│   ├── api.py            # FastAPI endpoints
│   ├── settings.py       # EnvSettings (читает .env)
│   ├── config_loader.py  # загрузка config.yaml
│   ├── models.py         # AppConfig pydantic модели
│   ├── hashing.py        # вычисление index_hash
│   ├── files.py          # обход файлов, doc_id, content_hash
│   ├── splitter.py       # разбиение текста на чанки
│   ├── embeddings.py     # текст → dense вектор
│   ├── sparse.py         # текст → sparse вектор
│   ├── qdrant_store.py   # операции с Qdrant
│   ├── indexer.py        # индексация документов
│   ├── search.py         # поисковый pipeline
│   └── evaluator.py      # evaluation метрики
├── config/
│   └── config.yaml       # параметры системы
├── data/documents/       # сюда кладём .txt / .md файлы
├── state/                # index_state.json (генерируется автоматически)
├── eval/
│   ├── queries.jsonl     # запросы для evaluation
│   └── qrels.jsonl       # релевантные документы для evaluation
├── .env.example          # шаблон для .env
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd ir-retrieval
```

### 2. Создать `.env`

```bash
cp .env.example .env
```

Заполнить `.env` своими значениями:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
CONFIG_PATH=/app/config/config.yaml
```

### 3. Положить документы

Скопировать `.txt` или `.md` файлы в папку `data/documents/`.

### 4. Запустить

**CPU:**
```bash
docker compose --profile cpu up --build
```

**GPU (NVIDIA):**
```bash
docker compose --profile gpu up --build
```

### 5. Проверить

- API: `http://localhost:8000/health`
- Qdrant UI: `http://localhost:6333/dashboard`

## Как работает индексация

При каждом запуске система:

1. Вычисляет `index_hash` из параметров конфига (chunk_size, модель и др.)
2. Читает `state/index_state.json`
3. Выбирает сценарий:
   - **A** — state не найден → индексировать всё
   - **B** — hash совпадает → доиндексировать только новые/изменённые файлы
   - **C** — hash изменился → создать новые коллекции и переиндексировать всё
4. Создаёт две коллекции в Qdrant: `doc_level_{hash}` и `chunk_level_{hash}`
5. Для каждого документа: считает embedding, режет на чанки, заливает в Qdrant

## Конфигурация

Все параметры в `config/config.yaml`. Секреты в `.env`.

Параметры влияющие на `index_hash` (изменение = полная переиндексация):
- `splitting.chunk_size`, `chunk_overlap`
- `embeddings.model_name`
- `doc_summary.*`
- `sparse.enabled`

## API

| Метод | Endpoint | Описание |
|---|---|---|
| GET | `/health` | статус сервиса |
| GET | `/index-info` | информация об индексе |
| POST | `/search` | поиск по запросу |
| POST | `/add-file` | добавить файл в индекс |
| POST | `/delete-file` | удалить файл из индекса |

## Что сделано (часть 1 — инфраструктура)

- [x] Docker Compose с профилями cpu и gpu
- [x] Конфигурация через `config.yaml` + `.env` + pydantic
- [x] Вычисление `index_hash` из параметров конфига
- [x] Рекурсивный обход файлов, `doc_id`, `content_hash`
- [x] Recursive chunk splitter с overlap
- [x] State management — сценарии A/B/C при старте
- [x] Создание коллекций Qdrant (`doc_level` + `chunk_level`)
- [x] Dense embeddings через sentence-transformers
- [x] Sparse vectors (TF-weighted)
- [x] Полная индексация документов в Qdrant

## Что предстоит сделать

- [ ] `summarizer.py` — summary документов через OpenAI (часть 2)
- [ ] `search.py` — поисковый pipeline: doc-level → chunk-level → cross-encoder (часть 3)
- [ ] `api.py` — все endpoints (часть 4)
- [ ] `evaluator.py` — Recall@K, Precision@K, MRR, nDCG@K (часть 4)
