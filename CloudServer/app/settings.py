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


settings = Settings()

