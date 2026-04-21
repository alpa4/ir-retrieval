from typing import Optional
from sentence_transformers import CrossEncoder
from app.embeddings import get_device
from app.models import CrossEncoderConfig


def load_cross_encoder(config: CrossEncoderConfig) -> Optional[CrossEncoder]:
    if not config.enabled_by_default:
        return None
    device = get_device(config.device)
    return CrossEncoder(config.model_name, device=device)
