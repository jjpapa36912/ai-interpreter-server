#!/usr/bin/env python3
"""Evaluate the resident translation endpoint on a bilingual golden manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path


def character_fscore(reference: str, prediction: str) -> float:
    reference_chars = Counter(re.sub(r"\s+", "", reference.casefold()))
    prediction_chars = Counter(re.sub(r"\s+", "", prediction.casefold()))
    overlap = sum((reference_chars & prediction_chars).values())
    precision = overlap / max(1, sum(prediction_chars.values()))
    recall = overlap / max(1, sum(reference_chars.values()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def normalized(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", text.casefold())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    token = os.getenv("AI_INTERPRETER_SERVER_TOKEN", "")
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()][:args.limit]
    results = []
    for index, row in enumerate(rows):
        body = json.dumps({
            "text": row["source_text"],
            "source_language": row["source_language"],
            "target_language": row["target_language"],
            "session_id": f"quality-{index}",
        }).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(args.url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        prediction = payload["translation"]
        entities = row.get("named_entities", [])
        entity_hits = sum(normalized(entity) in normalized(prediction) for entity in entities)
        result = {
            "id": row["id"],
            "source_language": row["source_language"],
            "target_language": row["target_language"],
            "reference": row["target_text"],
            "prediction": prediction,
            "character_fscore": round(character_fscore(row["target_text"], prediction), 4),
            "entity_hits": entity_hits,
            "entity_total": len(entities),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "examples": len(results),
        "mean_character_fscore": statistics.mean(row["character_fscore"] for row in results),
        "entity_retention": sum(row["entity_hits"] for row in results) / max(1, sum(row["entity_total"] for row in results)),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in results),
        "maximum_latency_ms": max(row["latency_ms"] for row in results),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0 if summary["mean_character_fscore"] >= 0.72 and summary["entity_retention"] >= 0.92 else 2


if __name__ == "__main__":
    raise SystemExit(main())
