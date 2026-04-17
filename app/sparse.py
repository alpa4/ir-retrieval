import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _term_to_id(term: str) -> int:
    """Map term string to a stable integer index via hash."""
    return hash(term) % (2 ** 24)


def build_sparse_vector(text: str) -> SparseVector:
    """
    Build a simple TF-IDF-like sparse vector from text.
    Uses term frequency weighted by log(1 + tf).
    """
    tokens = _tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    total = len(tokens)

    index_value: dict[int, float] = {}
    for term, count in counts.items():
        idx = _term_to_id(term)
        tf = count / total
        weight = math.log(1 + tf * 10)
        # Accumulate in case of hash collision
        index_value[idx] = index_value.get(idx, 0.0) + weight

    indices = list(index_value.keys())
    values = list(index_value.values())
    return SparseVector(indices=indices, values=values)
