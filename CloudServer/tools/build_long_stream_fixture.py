#!/usr/bin/env python3
"""Concatenate manifest WAVs into a reproducible long streaming fixture."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--minimum-seconds", type=float, default=120.0)
    parser.add_argument("--silence-seconds", type=float, default=.35)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    frames: list[bytes] = []
    references: list[str] = []
    total_frames = 0
    silence = b"\0\0" * int(16_000 * args.silence_seconds)
    available = [row for row in rows if Path(row["audio_path"]).exists()]
    if not available:
        raise FileNotFoundError("manifest contains no locally available WAV files")
    index = 0
    while total_frames / 16_000 < args.minimum_seconds:
        row = available[index % len(available)]
        index += 1
        with wave.open(row["audio_path"], "rb") as source:
            spec = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if spec != (1, 2, 16_000):
                continue
            payload = source.readframes(source.getnframes())
        frames.extend((payload, silence))
        total_frames += len(payload) // 2 + len(silence) // 2
        references.append(row.get("clean") or row.get("reference") or "")
    with wave.open(args.output_wav, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"".join(frames))
    item = {
        "id": "long-stream",
        "source_language": "en",
        "audio_path": str(Path(args.output_wav).resolve()),
        "clean": " ".join(references),
    }
    Path(args.output_manifest).write_text(json.dumps(item) + "\n")
    print(json.dumps({"audio_seconds": total_frames / 16_000, "clips": len(references)}))


if __name__ == "__main__":
    main()
