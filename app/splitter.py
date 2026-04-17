import hashlib
from dataclasses import dataclass
from app.models import SplittingConfig


@dataclass
class Chunk:
    chunk_id: str      # string key stored in payload
    chunk_id_int: int  # integer id for Qdrant point
    chunk_index: int
    chunk_text: str


def _chunk_int_id(doc_id: str, index: int) -> int:
    """Stable integer id for a chunk derived from doc_id and index."""
    raw = f"{doc_id}:{index}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


def split_text(doc_id: str, text: str, config: SplittingConfig) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    chunk_id = doc_id:chunk_index
    """
    size = config.chunk_size
    overlap = config.chunk_overlap
    step = size - overlap

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{index}",
                chunk_id_int=_chunk_int_id(doc_id, index),
                chunk_index=index,
                chunk_text=chunk_text,
            ))
            index += 1
        start += step

    return chunks
