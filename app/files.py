import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentFile:
    doc_id: str       # hex string, used in payloads and chunk_id
    doc_id_int: int   # integer, used as Qdrant point id
    file_path: str    # absolute path
    relative_path: str  # relative to docs_path
    content: str
    content_hash: str


def compute_doc_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()


def _doc_id_to_int(doc_id: str) -> int:
    return int(doc_id[:15], 16)


def _compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def load_document(file_path: Path, docs_path: Path) -> "DocumentFile | None":
    """Load a single document file; returns None if empty or unsupported."""
    if file_path.suffix not in (".txt", ".md") or not file_path.is_file():
        return None
    content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return None
    relative_path = str(file_path.relative_to(docs_path))
    doc_id = compute_doc_id(relative_path)
    return DocumentFile(
        doc_id=doc_id,
        doc_id_int=_doc_id_to_int(doc_id),
        file_path=str(file_path),
        relative_path=relative_path,
        content=content,
        content_hash=_compute_content_hash(content),
    )


def scan_documents(docs_path: str) -> list[DocumentFile]:
    """Recursively scan docs_path for .txt and .md files."""
    root = Path(docs_path)
    documents = []
    for file_path in sorted(root.rglob("*")):
        doc = load_document(file_path, root)
        if doc is not None:
            documents.append(doc)
    return documents
