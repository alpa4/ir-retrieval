"""
Download a BEIR dataset and convert it to the project's format.

Usage:
    python scripts/prepare_dataset.py --dataset scifact
    python scripts/prepare_dataset.py --dataset nfcorpus
    python scripts/prepare_dataset.py --dataset fiqa

What it does:
  1. Downloads the dataset zip from the BEIR public mirror
  2. Saves each corpus document as data/documents/<id>.txt
  3. Writes eval/queries.jsonl and eval/qrels.jsonl in the project format

After running, restart the service to reindex:
    docker compose --profile cpu up --build
Then evaluate:
    docker exec ir-retrieval-app-1 python3 -m app.evaluator \
        --queries eval/queries.jsonl --qrels eval/qrels.jsonl
"""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DATASETS = {
    "scifact":   {"split": "test",  "max_docs": None, "max_queries": 300},
    "nfcorpus":  {"split": "test",  "max_docs": None, "max_queries": 323},
    "fiqa":      {"split": "test",  "max_docs": 2000, "max_queries": 100},
    "arguana":   {"split": "test",  "max_docs": 2000, "max_queries": 100},
}

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "documents"
EVAL_DIR = ROOT / "eval"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def compute_doc_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()


def prepare(dataset: str, tmp_dir: Path) -> None:
    cfg = DATASETS[dataset]
    zip_path = tmp_dir / f"{dataset}.zip"
    extract_dir = tmp_dir / dataset

    # Download
    url = f"{BEIR_BASE_URL}/{dataset}.zip"
    download(url, zip_path)

    # Extract
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    # Locate corpus / queries / qrels
    # BEIR zip may extract to dataset/ or dataset/dataset/
    candidate = extract_dir / "corpus.jsonl"
    if not candidate.exists():
        # try nested
        nested = extract_dir / dataset / "corpus.jsonl"
        if nested.exists():
            extract_dir = extract_dir / dataset
        else:
            raise FileNotFoundError(f"Cannot find corpus.jsonl under {extract_dir}")

    corpus_path  = extract_dir / "corpus.jsonl"
    queries_path = extract_dir / "queries.jsonl"
    qrels_dir    = extract_dir / "qrels"
    split        = cfg["split"]
    qrels_path   = qrels_dir / f"{split}.tsv"

    # Clear old sample documents, keep nothing
    if DOCS_DIR.exists():
        for f in DOCS_DIR.glob("*.txt"):
            f.unlink()
        for f in DOCS_DIR.glob("*.md"):
            f.unlink()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print("Loading corpus ...")
    corpus: dict[str, dict] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = row

    max_docs = cfg["max_docs"]
    if max_docs:
        corpus = dict(list(corpus.items())[:max_docs])

    # Write documents as .txt files
    print(f"Writing {len(corpus)} documents to {DOCS_DIR} ...")
    id_to_relpath: dict[str, str] = {}
    for orig_id, doc in corpus.items():
        filename = f"{orig_id}.txt"
        content = ""
        if doc.get("title"):
            content = doc["title"].strip() + "\n\n"
        content += doc.get("text", "").strip()
        (DOCS_DIR / filename).write_text(content, encoding="utf-8")
        id_to_relpath[orig_id] = filename

    # Load qrels first to know which query IDs exist in this split
    raw_qrels: list[tuple] = []
    with open(qrels_path, encoding="utf-8") as f:
        first = f.readline()
        # skip header only if it looks like a header (non-numeric first field)
        if first.split("\t")[0].strip().lstrip("-").isdigit():
            raw_qrels_lines = [first] + f.readlines()
        else:
            raw_qrels_lines = f.readlines()
    for line in raw_qrels_lines:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        qid, did, rel = parts[0], parts[1], int(parts[2])
        if rel >= 1 and did in id_to_relpath:
            raw_qrels.append((qid, did))

    split_query_ids = {qid for qid, _ in raw_qrels}

    # Load queries — only those that appear in the qrels for this split
    max_queries = cfg["max_queries"]
    queries: list[dict] = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in split_query_ids:
                queries.append({"query_id": row["_id"], "text": row["text"]})
    if max_queries:
        queries = queries[:max_queries]
    query_ids = {q["query_id"] for q in queries}

    # Build final qrels filtered to kept queries
    qrels: list[dict] = []
    for qid, did in raw_qrels:
        if qid not in query_ids:
            continue
        qrels.append({"query_id": qid, "doc_id": compute_doc_id(id_to_relpath[did])})

    # Write eval files
    out_queries = EVAL_DIR / "queries.jsonl"
    out_qrels   = EVAL_DIR / "qrels.jsonl"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_queries, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    with open(out_qrels, "w", encoding="utf-8") as f:
        for r in qrels:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone!")
    print(f"  Documents : {len(corpus):>6}  →  {DOCS_DIR}")
    print(f"  Queries   : {len(queries):>6}  →  {out_queries}")
    print(f"  Qrel pairs: {len(qrels):>6}  →  {out_qrels}")
    print()
    print("Next steps:")
    print("  1. docker compose --profile cpu up --build   # reindex")
    print("  2. docker exec ir-retrieval-app-1 python3 -m app.evaluator \\")
    print("       --queries eval/queries.jsonl --qrels eval/qrels.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact",
                        choices=list(DATASETS.keys()),
                        help="BEIR dataset name (default: scifact)")
    parser.add_argument("--tmp", default="/tmp/beir_download",
                        help="Temporary directory for downloads")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepare(args.dataset, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
