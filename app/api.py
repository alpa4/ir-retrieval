import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

from app.search import search
from app.indexer import index_document, delete_document
from app.files import load_document, compute_doc_id, scan_documents
from app import qdrant_store as store

logger = logging.getLogger("uvicorn.error")
router = APIRouter()


# --- Request / response models ---

class SearchRequest(BaseModel):
    query: str
    top_k_doc: Optional[int] = None
    top_k_dense: Optional[int] = None
    top_k_sparse: Optional[int] = None
    final_top_k: Optional[int] = None
    use_cross_encoder: Optional[bool] = None
    mode: Optional[str] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    file_path: str
    chunk_index: int
    chunk_text: str
    scores: dict


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class FilePathRequest(BaseModel):
    path: str


class UploadFileResponse(BaseModel):
    filename: str
    status: str  # "indexed" | "reindexed" | "already_indexed"


class DeleteFileResponse(BaseModel):
    path: str
    status: str  # "deleted" | "not_found"


class FileInfo(BaseModel):
    filename: str
    size_bytes: int


class ListFilesResponse(BaseModel):
    files: list[FileInfo]
    total: int
    page: int
    page_size: int


class IndexInfoResponse(BaseModel):
    docs_on_disk: int
    docs_in_index: int
    status: str  # "ok" | "partial"


# --- Helpers ---

def _resolve_docs_path(request: Request) -> Path:
    return Path(request.app.state.config.index.docs_path).resolve()


def _validate_path(raw_path: str, docs_path: Path) -> Path:
    p = Path(raw_path).resolve()
    try:
        p.relative_to(docs_path)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Path must be inside {docs_path}")
    if p.suffix not in (".txt", ".md"):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported")
    return p


# --- Endpoints ---

@router.post("/search", response_model=SearchResponse)
def search_endpoint(body: SearchRequest, request: Request):
    s = request.app.state
    cfg = s.config.search_defaults

    results = search(
        query=body.query,
        client=s.client,
        embed_model=s.embed_model,
        sparse_model=s.sparse_model,
        doc_collection=s.doc_collection,
        chunk_collection=s.chunk_collection,
        top_k_doc=body.top_k_doc if body.top_k_doc is not None else cfg.top_k_doc,
        top_k_dense=body.top_k_dense if body.top_k_dense is not None else cfg.top_k_dense,
        top_k_sparse=body.top_k_sparse if body.top_k_sparse is not None else cfg.top_k_sparse,
        final_top_k=body.final_top_k if body.final_top_k is not None else cfg.final_top_k,
        use_cross_encoder=body.use_cross_encoder if body.use_cross_encoder is not None else s.config.cross_encoder.enabled_by_default,
        cross_encoder=s.cross_encoder,
        mode=body.mode if body.mode is not None else cfg.mode,
    )

    return SearchResponse(
        query=body.query,
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                file_path=r.file_path,
                chunk_index=r.chunk_index,
                chunk_text=r.chunk_text,
                scores=r.scores,
            )
            for r in results
        ],
    )


@router.post("/upload-file", response_model=UploadFileResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename or Path(filename).suffix not in (".txt", ".md"):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported")

    s = request.app.state
    docs_path = _resolve_docs_path(request)
    dest = docs_path / filename

    content_bytes = await file.read()
    dest.write_bytes(content_bytes)

    doc = load_document(dest, docs_path)
    if doc is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is empty or could not be read")

    existing = store.get_document(s.client, s.doc_collection, doc.doc_id_int)

    if existing is None:
        index_document(doc, s.config, s.client, s.embed_model, s.sparse_model, s.summarizer,
                       s.doc_collection, s.chunk_collection, s.index_hash)
        return UploadFileResponse(filename=file.filename, status="indexed")

    if existing.payload.get("content_hash") == doc.content_hash:
        return UploadFileResponse(filename=file.filename, status="already_indexed")

    delete_document(doc.doc_id, doc.doc_id_int, s.client, s.doc_collection, s.chunk_collection)
    index_document(doc, s.config, s.client, s.embed_model, s.sparse_model, s.summarizer,
                   s.doc_collection, s.chunk_collection, s.index_hash)
    return UploadFileResponse(filename=file.filename, status="reindexed")


@router.post("/delete-file", response_model=DeleteFileResponse)
def delete_file(body: FilePathRequest, request: Request):
    s = request.app.state
    docs_path = _resolve_docs_path(request)
    file_path = _validate_path(body.path, docs_path)

    relative_path = str(file_path.relative_to(docs_path))
    doc_id = compute_doc_id(relative_path)
    doc_id_int = int(doc_id[:15], 16)

    existing = store.get_document(s.client, s.doc_collection, doc_id_int)
    if existing is None:
        return DeleteFileResponse(path=body.path, status="not_found")

    file_path.unlink(missing_ok=True)
    delete_document(doc_id, doc_id_int, s.client, s.doc_collection, s.chunk_collection)
    return DeleteFileResponse(path=body.path, status="deleted")


@router.get("/list-files", response_model=ListFilesResponse)
def list_files(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str = Query("", description="Filter by filename"),
):
    docs_path = _resolve_docs_path(request)
    all_docs = scan_documents(str(docs_path))

    if search:
        all_docs = [d for d in all_docs if search.lower() in d.relative_path.lower()]

    total = len(all_docs)
    start = (page - 1) * page_size
    page_docs = all_docs[start: start + page_size]

    files = [
        FileInfo(
            filename=d.relative_path,
            size_bytes=Path(d.file_path).stat().st_size,
        )
        for d in page_docs
    ]

    return ListFilesResponse(files=files, total=total, page=page, page_size=page_size)


@router.get("/index-info", response_model=IndexInfoResponse)
def index_info(request: Request):
    s = request.app.state
    docs_on_disk = len(scan_documents(s.config.index.docs_path))
    docs_in_index = store.count_documents(s.client, s.doc_collection)
    status = "ok" if docs_on_disk == docs_in_index else "partial"
    return IndexInfoResponse(
        docs_on_disk=docs_on_disk,
        docs_in_index=docs_in_index,
        status=status,
    )
