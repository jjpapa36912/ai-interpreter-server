#!/usr/bin/env python3
"""Bake all public model weights into the deployment image."""

from __future__ import annotations

import os
from pathlib import Path

from faster_whisper.utils import download_model
from huggingface_hub import snapshot_download


def main() -> None:
    cache = Path(os.environ.get("HF_HOME", "/models/huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=os.environ.get(
            "AI_INTERPRETER_TRANSLATION_MODEL", "google/madlad400-3b-mt"
        ),
        cache_dir=str(cache),
    )
    download_model(
        os.environ.get("AI_INTERPRETER_ASR_MODEL", "turbo"),
        cache_dir=str(cache),
    )
    snapshot_download(
        repo_id=os.environ.get(
            "AI_INTERPRETER_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        ),
        cache_dir=str(cache),
    )
    print(f"preloaded public models in {cache}")


if __name__ == "__main__":
    main()
