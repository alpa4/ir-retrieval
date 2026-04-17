import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class IndexState:
    index_hash: str
    doc_collection: str
    chunk_collection: str


def load_state(state_path: str) -> Optional[IndexState]:
    path = Path(state_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IndexState(**data)


def save_state(state_path: str, state: IndexState) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
