"""
Standalone evaluator: python -m app.evaluator --queries eval/queries.jsonl --qrels eval/qrels.jsonl
"""
import argparse
import json
import math
import os
import urllib.request
import urllib.error


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def search_api(api_base: str, query: str, final_top_k: int) -> list[str]:
    """Return ordered list of doc_ids from /search (deduped, first-occurrence wins)."""
    payload = json.dumps({"query": query, "final_top_k": final_top_k, "use_cross_encoder": False}).encode()
    req = urllib.request.Request(
        f"{api_base}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    seen: set[str] = set()
    doc_ids: list[str] = []
    for r in data.get("results", []):
        d = r["doc_id"]
        if d not in seen:
            seen.add(d)
            doc_ids.append(d)
    return doc_ids


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(retrieved[:k])
        if d in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(queries_path: str, qrels_path: str, api_base: str, ks: list[int]) -> None:
    queries = load_jsonl(queries_path)
    qrels_raw = load_jsonl(qrels_path)

    qrels: dict[str, set[str]] = {}
    for row in qrels_raw:
        qid = str(row["query_id"])
        qrels.setdefault(qid, set()).add(str(row["doc_id"]))

    max_k = max(ks)
    metrics: dict[str, list[float]] = {
        f"Recall@{k}": [] for k in ks
    }
    metrics.update({f"Precision@{k}": [] for k in ks})
    metrics["MRR"] = []
    metrics.update({f"nDCG@{k}": [] for k in ks})

    for q in queries:
        qid = str(q["query_id"])
        query_text = q["text"]
        relevant = qrels.get(qid, set())

        try:
            retrieved = search_api(api_base, query_text, max_k)
        except urllib.error.URLError as e:
            print(f"[WARN] Query {qid} failed: {e}")
            retrieved = []

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            metrics[f"Precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            metrics[f"nDCG@{k}"].append(ndcg_at_k(retrieved, relevant, k))
        metrics["MRR"].append(reciprocal_rank(retrieved, relevant))

    n = len(queries)
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for name, values in metrics.items():
        avg = sum(values) / len(values) if values else 0.0
        print(f"{name:<20} {avg:>10.4f}")
    print(f"\nEvaluated {n} queries against {api_base}")


def main():
    parser = argparse.ArgumentParser(description="IR Retrieval Evaluator")
    parser.add_argument("--queries", required=True, help="Path to queries.jsonl")
    parser.add_argument("--qrels", required=True, help="Path to qrels.jsonl")
    parser.add_argument("--api", default=os.getenv("API_BASE", "http://localhost:8000"),
                        help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10],
                        help="K values for metrics (default: 5 10)")
    args = parser.parse_args()

    evaluate(args.queries, args.qrels, args.api, sorted(set(args.k)))


if __name__ == "__main__":
    main()
