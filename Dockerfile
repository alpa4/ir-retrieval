FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/app/model_cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/model_cache/fastembed

RUN pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config/ ./config/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
