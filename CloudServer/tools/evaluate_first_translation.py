#!/usr/bin/env python3
"""Measure paced audio-to-first-committed-translation latency."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import websockets


def read_pcm16(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as source:
        spec = (source.getnchannels(), source.getsampwidth(), source.getframerate())
        if spec != (1, 2, 16_000):
            raise ValueError(f"expected mono PCM16 16kHz, got {spec}")
        pcm = source.readframes(source.getnframes())
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frame = 320
    active = [
        np.sqrt(np.mean(samples[i:i + frame] ** 2)) > 10 ** (-42 / 20)
        for i in range(0, len(samples), frame)
    ]
    onset = 0.0
    for index in range(max(1, len(active) - 4)):
        if sum(active[index:index + 5]) >= 3:
            onset = index * .02
            break
    return pcm, onset


def translate(url: str, text: str, source: str, target: str) -> dict:
    body = json.dumps({
        "text": text,
        "source_language": source,
        "target_language": target,
        "session_id": f"first-{source}-{time.time_ns()}",
    }).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


async def evaluate(ws_url: str, translate_url: str, audio: Path, source: str) -> dict:
    pcm, onset = read_pcm16(audio)
    started = time.perf_counter()
    first_commit = None
    translated = None
    translation_payload = None
    source_chunk = None
    target = "ko" if source == "en" else "en"
    separator = "&" if "?" in ws_url else "?"
    endpoint = f"{ws_url}{separator}language={source}&sample_rate=16000"
    async with websockets.connect(endpoint, max_size=2**22) as socket:
        async def receive() -> None:
            nonlocal first_commit, translated, translation_payload, source_chunk
            async for raw in socket:
                message = json.loads(raw)
                delta = message.get("committed_delta", "").strip()
                if delta and first_commit is None:
                    first_commit = time.perf_counter() - started
                    source_chunk = delta
                    if message.get("translation"):
                        translation_payload = {
                            "translation": message["translation"],
                            "latency_ms": message.get("translation_latency_ms"),
                        }
                    else:
                        translation_payload = await asyncio.to_thread(
                            translate, translate_url, delta, source, target
                        )
                    translated = time.perf_counter() - started
                if message.get("type") == "finished":
                    break

        receiver = asyncio.create_task(receive())
        deadline = started
        for offset in range(0, len(pcm), 3200):
            await socket.send(pcm[offset:offset + 3200])
            deadline += .1
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            if translated is not None:
                break
        await socket.send("finish")
        await receiver
    return {
        "source_language": source,
        "speech_onset_seconds": round(onset, 3),
        "first_commit_after_speech_seconds": round(first_commit - onset, 3),
        "first_translation_after_speech_seconds": round(translated - onset, 3),
        "first_source_chunk": source_chunk,
        "first_translation": translation_payload and translation_payload.get("translation"),
        "translation_latency_ms": translation_payload and translation_payload.get("latency_ms"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8001/v1/asr/stream")
    parser.add_argument("--translate-url", default="http://127.0.0.1:8001/v1/translate")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--source", choices=("en", "ko"), required=True)
    args = parser.parse_args()
    print(json.dumps(await evaluate(
        args.ws_url, args.translate_url, Path(args.audio), args.source
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
