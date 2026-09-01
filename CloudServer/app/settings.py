from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    service_mode: str = os.getenv(
        "AI_INTERPRETER_SERVICE_MODE", "full"
    ).strip().lower()
    translation_model: str = os.getenv(
        "AI_INTERPRETER_TRANSLATION_MODEL", "Qwen/Qwen3-8B"
    )
    api_token: str = os.getenv("AI_INTERPRETER_SERVER_TOKEN", "")
    max_new_tokens: int = int(os.getenv("AI_INTERPRETER_MAX_NEW_TOKENS", "128"))
    model_max_memory: str = os.getenv("AI_INTERPRETER_MODEL_MAX_MEMORY", "18GiB")
    translation_adapter: str = os.getenv("AI_INTERPRETER_TRANSLATION_ADAPTER", "")
    asr_model: str = os.getenv("AI_INTERPRETER_ASR_MODEL", "turbo")
    asr_compute_type: str = os.getenv("AI_INTERPRETER_ASR_COMPUTE_TYPE", "float16")
    asr_beam_size: int = int(os.getenv("AI_INTERPRETER_ASR_BEAM_SIZE", "1"))
    tts_model: str = os.getenv(
        "AI_INTERPRETER_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    )
    tts_korean_voice: str = os.getenv("AI_INTERPRETER_TTS_KOREAN_VOICE", "Sohee")
    tts_english_voice: str = os.getenv("AI_INTERPRETER_TTS_ENGLISH_VOICE", "Ryan")
    # The CUDA-graph backend preallocates its KV cache to this length.  The app
    # sends short simultaneous-interpretation clauses, so reserving the library
    # default of 2048 tokens wastes VRAM and makes repeated requests much more
    # vulnerable to allocator fragmentation on 16 GB cards.
    tts_max_sequence_length: int = int(os.getenv(
        "AI_INTERPRETER_TTS_MAX_SEQUENCE_LENGTH", "768"
    ))
    # A clause should never need the library's 2048-token runaway ceiling.
    # Bound generation so a missed EOS cannot consume the worker indefinitely.
    tts_max_new_tokens: int = int(os.getenv(
        "AI_INTERPRETER_TTS_MAX_NEW_TOKENS", "384"
    ))
    # FasterQwen's CUDA streaming generator can wedge before yielding its first
    # chunk and hold the global generation lock indefinitely.  Keep the proven
    # whole-utterance CUDA path as the production default; streaming remains an
    # explicit opt-in for isolated compatibility testing.
    tts_experimental_streaming: bool = os.getenv(
        "AI_INTERPRETER_TTS_EXPERIMENTAL_STREAMING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


settings = Settings()
