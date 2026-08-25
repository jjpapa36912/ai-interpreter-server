#!/usr/bin/env python3
"""Pace WAV files into the GPU WebSocket and enforce streaming ASR gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
import wave
from pathlib import Path

import websockets


def words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)


def edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, expected in enumerate(left, 1):
        next_row = [i]
        for j, actual in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def read_pcm16(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as source:
        spec = (source.getnchannels(), source.getsampwidth(), source.getframerate())
        if spec != (1, 2, 16_000):
            raise ValueError(f"{path}: expected mono PCM16 16kHz, got {spec}")
        frames = source.readframes(source.getnframes())
        return frames, source.getnframes() / 16_000


async def evaluate_clip(url: str, token: str, item: dict, pace: bool) -> dict:
    pcm, duration = read_pcm16(Path(item["audio_path"]))
    language = item.get("source_language", "en")
    endpoint = f"{url.rstrip('/')}?language={language}&sample_rate=16000"
    headers = {"Authorization": f"Bearer {token}"} if token else None
    chunks = [pcm[offset:offset + 3_200] for offset in range(0, len(pcm), 3_200)]
    messages: list[dict] = []
    started = time.perf_counter()

    async with websockets.connect(endpoint, additional_headers=headers, max_size=2**22) as socket:
        async def receive() -> None:
            async for raw in socket:
                message = json.loads(raw)
                messages.append({**message, "wall_seconds": time.perf_counter() - started})
                if message.get("type") == "finished":
                    break

        receiver = asyncio.create_task(receive())
        deadline = started
        for chunk in chunks:
            await socket.send(chunk)
            if pace:
                deadline += 0.1
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        await socket.send("finish")
        await receiver

    asr = [message for message in messages if message.get("type") == "asr"]
    hypotheses = [message for message in asr if message.get("hypothesis")]
    committed = [message for message in asr if message.get("committed_delta")]
    transcript = " ".join(message["committed_delta"] for message in committed).strip()
    reference = item.get("clean") or item.get("reference") or ""
    reference_words = words(reference)
    hypothesis_words = words(transcript)
    protocol_ok = all(
        message.get("committed_words", 0) >= len(words(" ".join(m["committed_delta"] for m in committed[:index + 1])))
        for index, message in enumerate(committed)
    )
    return {
        "id": item.get("id", Path(item["audio_path"]).stem),
        "audio_seconds": round(duration, 3),
        "first_hypothesis_seconds": round(hypotheses[0]["wall_seconds"], 3) if hypotheses else None,
        "first_commit_seconds": round(committed[0]["wall_seconds"], 3) if committed else None,
        "final_wall_seconds": round(messages[-1]["wall_seconds"], 3),
        "backlog_seconds": round(max(0.0, messages[-1]["wall_seconds"] - duration), 3) if pace else None,
        "decode_latency_ms_p95": round(sorted(m["latency_ms"] for m in asr)[max(0, int(len(asr) * .95) - 1)], 1) if asr else None,
        "wer": round(edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words)), 4),
        "protocol_monotonic": protocol_ok,
        "reference": reference,
        "transcript": transcript,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="wss://host/v1/asr/stream")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    token = os.getenv("AI_INTERPRETER_SERVER_TOKEN", "")
    items = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()][:args.limit]
    results = []
    for item in items:
        result = await evaluate_clip(args.url, token, item, not args.unpaced)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    summary = {
        "clips": len(results),
        "first_hypothesis_p50": statistics.median(r["first_hypothesis_seconds"] for r in results if r["first_hypothesis_seconds"] is not None),
        "first_commit_p50": statistics.median(r["first_commit_seconds"] for r in results if r["first_commit_seconds"] is not None),
        "wer_mean": statistics.mean(r["wer"] for r in results),
        "backlog_max": max((r["backlog_seconds"] or 0) for r in results),
        "protocol_monotonic": all(r["protocol_monotonic"] for r in results),
    }
    report = {"summary": summary, "results": results}
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    passed = (
        summary["first_hypothesis_p50"] <= 1.2
        and summary["first_commit_p50"] <= 1.8
        and summary["wer_mean"] <= 0.25
        and summary["backlog_max"] <= 1.0
        and summary["protocol_monotonic"]
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
