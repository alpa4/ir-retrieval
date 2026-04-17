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


def _compute_doc_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()


def _doc_id_to_int(doc_id: str) -> int:
    return int(doc_id[:15], 16)


def _compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def scan_documents(docs_path: str) -> list[DocumentFile]:
    """
    Recursively scan docs_path for .txt and .md files.
    Returns a list of DocumentFile objects.
    """
    root = Path(docs_path)
    documents = []

    for file_path in sorted(root.rglob("*")):
        if file_path.suffix not in (".txt", ".md"):
            continue
        if not file_path.is_file():
            continue

        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue

        relative_path = str(file_path.relative_to(root))
        doc_id = _compute_doc_id(relative_path)
        content_hash = _compute_content_hash(content)

        documents.append(DocumentFile(
            doc_id=doc_id,
            doc_id_int=_doc_id_to_int(doc_id),
            file_path=str(file_path),
            relative_path=relative_path,
            content=content,
            content_hash=content_hash,
        ))

    return documents
