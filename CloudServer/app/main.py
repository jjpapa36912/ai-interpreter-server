from __future__ import annotations

import secrets
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .settings import settings
from .translation import CUDATranslator


app = FastAPI(title="AI Interpreter CUDA Server", version="0.1.0")


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    source_language: str
    target_language: str
    session_id: str = Field(default="default", min_length=1, max_length=128)


class TranslationResponse(BaseModel):
    translation: str
    model: str
    latency_ms: float


@lru_cache(maxsize=1)
def translator() -> CUDATranslator:
    return CUDATranslator(settings)


def authorize(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    expected = f"Bearer {settings.api_token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", dependencies=[Depends(authorize)])
def ready() -> dict[str, object]:
    import torch

    return {
        "status": "ready" if torch.cuda.is_available() else "no_cuda",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "translation_model": settings.translation_model,
        "translation_loaded": translator().loaded,
    }


@app.post(
    "/v1/translate", response_model=TranslationResponse,
    dependencies=[Depends(authorize)],
)
def translate(request: TranslationRequest) -> TranslationResponse:
    started = time.perf_counter()
    try:
        result = translator().translate(
            request.text,
            request.source_language,
            request.target_language,
            request.session_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return TranslationResponse(
        translation=result,
        model=settings.translation_model,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.delete("/v1/sessions/{session_id}", dependencies=[Depends(authorize)])
def reset_session(session_id: str) -> dict[str, str]:
    translator().reset(session_id)
    return {"status": "reset", "session_id": session_id}

