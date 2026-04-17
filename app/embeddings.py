import torch
from sentence_transformers import SentenceTransformer
from app.models import EmbeddingsConfig


def get_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(config: EmbeddingsConfig) -> SentenceTransformer:
    device = get_device(config.device)
    model = SentenceTransformer(config.model_name, device=device)
    return model


def embed_texts(model: SentenceTransformer, texts: list[str],
                batch_size: int) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_single(model: SentenceTransformer, text: str) -> list[float]:
    return embed_texts(model, [text], batch_size=1)[0]
