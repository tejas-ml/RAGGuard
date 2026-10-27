"""Run the RAGGuard synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragguard.data import Query, as_records, build_documents, build_queries
from ragguard.firewall import RAGGuardFirewall
from ragguard.llm_sim import generate_baseline_response, generate_ragguard_response
from ragguard.metrics import contains_any_secret, leakage_fraction, summarize_rows, utility_score
from ragguard.retrieval import TfidfRetriever


def run_once(threshold: float, top_k: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    docs = build_documents()
    queries = build_queries()
    retriever = TfidfRetriever(docs)
    firewall = RAGGuardFirewall(threshold=threshold)

    rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []

    for query in queries:
        retrievals = retriever.search(query.text, top_k=top_k)
        retrieved_docs = [item.document for item in retrievals]

        baseline_answer = generate_baseline_response(query, retrieved_docs)
        rows.append(_result_row("Baseline RAG", query, baseline_answer, retrieved_docs, [], threshold))

        decisions = firewall.filter_context(retrievals, query.text)
        ragguard_answer = generate_ragguard_response(query, decisions)
        rows.append(_result_row("RAGGuard", query, ragguard_answer, retrieved_docs, decisions, threshold))

        for rank, (retrieval, decision) in enumerate(zip(retrievals, decisions), start=1):
            decision_rows.append(
                {
                    "threshold": threshold,
                    "query_id": query.query_id,
                    "query_type": query.query_type,
                    "rank": rank,
                    "doc_id": retrieval.document.doc_id,
                    "title": retrieval.document.title,
                    "retrieval_score": round(retrieval.score, 4),
                    "access_level": retrieval.document.access_level,
                    "category": retrieval.document.category,
                    "firewall_action": decision.action,
                    "sensitivity_score": decision.sensitivity_score,
                    "reasons": ";".join(decision.reasons),
                }
            )

    return rows, decision_rows


def _result_row(system: str, query: Query, answer: str, retrieved_docs, decisions, threshold: float) -> Dict[str, object]:
    leaked_any = contains_any_secret(answer, query.attack_targets)
    frac = leakage_fraction(answer, query.attack_targets)
    utility = utility_score(answer, query.gold_facts)
    return {
        "system": system,
        "threshold": threshold,
        "query_id": query.query_id,
        "query": query.text,
        "query_type": query.query_type,
        "user_role": query.user_role,
        "retrieved_doc_ids": ",".join(doc.doc_id for doc in retrieved_docs),
        "answer": answer,
        "leaked_any": int(leaked_any),
        "leakage_fraction": frac,
        "utility": utility,
        "blocked_chunks": sum(1 for d in decisions if getattr(d, "action", None) == "private-blocked"),
        "redacted_chunks": sum(1 for d in decisions if getattr(d, "action", None) == "sensitive-redacted"),
    }


def make_summary(rows: List[Dict[str, object]]) -> pd.DataFrame:
    summaries = []
    for system in sorted({row["system"] for row in rows}):
        sys_rows = [row for row in rows if row["system"] == system]
        summary = summarize_rows(sys_rows)
        summary["system"] = system
        summaries.append(summary)
    return pd.DataFrame(summaries)[
        [
            "system",
            "leakage_rate_any_sensitive",
            "leakage_fraction_sensitive",
            "leakage_rate_private_attacks",
            "answer_utility_benign",
            "answer_utility_all",
            "mean_blocked_chunks",
            "mean_redacted_chunks",
        ]
    ]


def threshold_sweep(thresholds: List[float], top_k: int) -> pd.DataFrame:
    sweep_rows = []
    for threshold in thresholds:
        rows, _ = run_once(threshold=threshold, top_k=top_k)
        ragguard_rows = [row for row in rows if row["system"] == "RAGGuard"]
        summary = summarize_rows(ragguard_rows)
        summary["threshold"] = threshold
        sweep_rows.append(summary)
    return pd.DataFrame(sweep_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows, decision_rows = run_once(threshold=args.threshold, top_k=args.top_k)
    pd.DataFrame(rows).to_csv(args.out / "query_results.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(args.out / "firewall_decisions.csv", index=False)
    make_summary(rows).to_csv(args.out / "metrics_summary.csv", index=False)

    thresholds = [round(x, 2) for x in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]]
    threshold_sweep(thresholds, top_k=args.top_k).to_csv(args.out / "threshold_sweep.csv", index=False)

    docs = build_documents()
    queries = build_queries()
    pd.DataFrame(as_records(docs)).to_json(args.out / "synthetic_documents.json", orient="records", indent=2)
    pd.DataFrame(as_records(queries)).to_json(args.out / "synthetic_queries.json", orient="records", indent=2)

    with open(args.out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": args.threshold, "top_k": args.top_k}, f, indent=2)

    print(f"Wrote experiment artifacts to {args.out}")
    print(make_summary(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
