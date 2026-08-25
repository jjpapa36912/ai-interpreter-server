from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
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


settings = Settings()
