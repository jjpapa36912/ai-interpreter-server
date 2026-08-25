#!/usr/bin/env python3
"""Fail fast unless a public deployment exposes the verified final models."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request


def request(url: str, token: str, body: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    method = "GET"
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
        method = "POST"
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers, method=method),
        timeout=30,
    ) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    token = os.getenv("AI_INTERPRETER_SERVER_TOKEN", "")
    ready = request(f"{base}/ready", token)
    expected = {
        "status": "ready",
        "translation_model": "google/madlad400-3b-mt",
        "asr_model": "turbo",
        "translation_loaded": True,
        "asr_loaded": True,
    }
    mismatches = {
        key: {"expected": value, "actual": ready.get(key)}
        for key, value in expected.items()
        if ready.get(key) != value
    }
    translations = []
    for index, (text, source, target) in enumerate((
        ("We need to review the budget.", "en", "ko"),
        ("일정을 확인해 주세요.", "ko", "en"),
    )):
        translations.append(request(f"{base}/v1/translate", token, {
            "text": text,
            "source_language": source,
            "target_language": target,
            "session_id": f"deploy-smoke-{index}",
        }))
    report = {"ready": ready, "mismatches": mismatches, "translations": translations}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if mismatches or any(not row.get("translation") for row in translations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
