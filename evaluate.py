"""Score retrieval and answer quality independently on a versioned corpus."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from service import INDEX_FORMAT, DocumentService
from workspace import ROOT, ollama_request


def covers(hits, support):
    return any(
        hit["document_name"] == support["document"]
        and hit["page"] == support["page"]
        and support["contains"].casefold() in hit["excerpt"].casefold()
        for hit in hits
    )


def score_case(service, case, limit):
    question, supports = case["question"], case["supports"]
    start = time.perf_counter()
    hits = service.retrieve(question, limit=limit)
    retrieval_ms = round(1000 * (time.perf_counter() - start), 1)
    start = time.perf_counter()
    result = service.ask(question, limit=limit)
    answer_ms = round(1000 * (time.perf_counter() - start), 1)
    citations = result["citations"]
    return {
        "id": case["id"],
        "answerable": bool(supports),
        "matched_passages": sum(covers(hits, support) for support in supports),
        "expected_passages": len(supports),
        "citation_support": bool(supports) and all(covers(citations, support) for support in supports),
        "citation_count": len(citations),
        "supporting_citations": sum(any(covers([citation], support) for support in supports) for citation in citations),
        "answer_facts_match": bool(supports) and not result["abstained"] and all(
            fact.casefold() in result["answer"].casefold() for fact in case["answer_contains"]
        ),
        "abstained": result["abstained"],
        "abstention_reason": result["reason"],
        "retrieval_ms": retrieval_ms,
        "generation_ms": result["generation_ms"],
        "end_to_end_answer_ms": answer_ms,
        "answer": result["answer"],
        "retrieved": [{key: hit[key] for key in ("document_name", "page", "score")} for hit in hits],
        "citations": [{key: hit[key] for key in ("document_name", "page", "excerpt")} for hit in citations],
    }


def summarize(rows):
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    def ratio(count, total):
        return round(count / total, 4) if total else None

    def latency(key):
        values = sorted(row[key] for row in rows)
        return {"median": statistics.median(values), "p95": values[math.ceil(len(values) * 0.95) - 1]}

    return {
        "retrieval_recall_at_k": ratio(sum(row["matched_passages"] for row in answerable),
                                       sum(row["expected_passages"] for row in answerable)),
        "citation_support_rate": ratio(sum(row["citation_support"] for row in answerable), len(answerable)),
        "citation_precision": ratio(sum(row["supporting_citations"] for row in rows),
                                    sum(row["citation_count"] for row in rows)),
        "answer_fact_match_rate": ratio(sum(row["answer_facts_match"] for row in answerable), len(answerable)),
        "correct_abstention_rate": ratio(sum(row["abstained"] for row in unanswerable), len(unanswerable)),
        "false_abstention_rate": ratio(sum(row["abstained"] for row in answerable), len(answerable)),
        "latency_ms": {key: latency(key) for key in ("retrieval_ms", "generation_ms", "end_to_end_answer_ms")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluations" / "v1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--hardware-note", required=True, help="CPU/GPU and memory of the Ollama host")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_bytes())
    with tempfile.TemporaryDirectory() as directory:
        service = DocumentService(Path(directory) / "eval.sqlite3")
        models = service.engine.metadata()
        runtime_start = ollama_request("version")["version"]
        corpus_hashes = {}
        for filename in dataset["corpus"]:
            data = (ROOT / "examples" / filename).read_bytes()
            corpus_hashes[filename] = hashlib.sha256(data).hexdigest()
            job = service.enqueue(filename, data)
            service.process_next()
            result = service.job(job["id"])
            if result["status"] != "ready":
                raise RuntimeError(f"{filename}: {result['error']}")
        rows = [score_case(service, case, args.limit) for case in dataset["questions"]]
        if service.engine.metadata() != models or ollama_request("version")["version"] != runtime_start:
            raise RuntimeError("Ollama models or version changed during evaluation. Run again.")
        report = {
            "dataset": dataset["version"],
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "corpus_sha256": corpus_hashes,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "settings": {"k": args.limit, "min_score": service.min_score, "index_format": INDEX_FORMAT},
            "models": models,
            "ollama_version": runtime_start,
            "hardware": {
                "ollama_host": args.hardware_note,
                "client_platform": platform.platform(),
                "client_logical_cpus": os.cpu_count(),
            },
            "metrics": summarize(rows),
            "questions": rows,
        }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
