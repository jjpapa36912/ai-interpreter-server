from __future__ import annotations

import secrets
import time
import asyncio
import threading
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .settings import settings
from .translation import CUDATranslator
from .asr import (
    CUDASpeechRecognizer, ConfirmedPhraseAccumulator, StreamCommitLedger,
    StreamingASRSession,
)
from .tts import CUDANeuralTTS


app = FastAPI(title="AI Interpreter CUDA Server", version="0.2.0")

_preload_lock = threading.Lock()
_preload_started = False
_preload_complete = False
_preload_error: str | None = None


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    source_language: str
    target_language: str
    session_id: str = Field(default="default", min_length=1, max_length=128)


class TranslationResponse(BaseModel):
    translation: str
    model: str
    latency_ms: float


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: str
    voice_id: str | None = None


def next_tts_chunk(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


@lru_cache(maxsize=1)
def translator() -> CUDATranslator:
    return CUDATranslator(settings)


@lru_cache(maxsize=1)
def speech_recognizer() -> CUDASpeechRecognizer:
    return CUDASpeechRecognizer(settings)


@lru_cache(maxsize=1)
def neural_tts() -> CUDANeuralTTS:
    return CUDANeuralTTS(settings)


def _preload_models() -> None:
    """Load ASR first, then translation, without blocking the health endpoint."""
    global _preload_complete, _preload_error
    try:
        speech_recognizer().load()
        speech_recognizer().warmup()
        translator().load()
        translator().warmup()
        neural_tts().load()
        neural_tts().warmup()
        _preload_complete = True
    except Exception as error:  # surfaced verbatim by /ready for deployment diagnosis
        _preload_error = f"{type(error).__name__}: {error}"


def ensure_preload_started() -> None:
    global _preload_started
    with _preload_lock:
        if _preload_started:
            return
        _preload_started = True
        threading.Thread(target=_preload_models, name="model-preload", daemon=True).start()


@app.on_event("startup")
def start_preload() -> None:
    ensure_preload_started()


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
        "status": (
            "error" if _preload_error else
            "ready" if torch.cuda.is_available() and _preload_complete else
            "loading" if torch.cuda.is_available() else
            "no_cuda"
        ),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "translation_model": settings.translation_model,
        "translation_adapter": settings.translation_adapter or None,
        "translation_loaded": translator().loaded,
        "asr_model": settings.asr_model,
        "asr_loaded": speech_recognizer().loaded,
        "tts_model": settings.tts_model,
        "tts_loaded": neural_tts().loaded,
        "preload_error": _preload_error,
    }


@app.post("/v1/tts/stream", dependencies=[Depends(authorize)])
def synthesize_speech(request: TTSRequest) -> StreamingResponse:
    def chunks():
        yield from (
            pcm for pcm, _sample_rate in neural_tts().synthesize_stream(
                request.text, request.language, request.voice_id
            )
        )

    return StreamingResponse(
        chunks(),
        media_type="audio/L16;rate=24000;channels=1",
        headers={"X-Audio-Sample-Rate": "24000"},
    )


@app.websocket("/v1/tts/ws")
async def stream_speech(websocket: WebSocket) -> None:
    """Stream neural speech without G-Cube's unreliable HTTP upstream."""
    if not websocket_authorized(websocket):
        await websocket.close(code=4401, reason="invalid bearer token")
        return
    ensure_preload_started()
    if _preload_error:
        await websocket.close(code=1011, reason="model preload failed")
        return
    if not neural_tts().loaded:
        await websocket.close(code=1013, reason="TTS model is still loading")
        return
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                request = TTSRequest.model_validate(payload)
            except Exception as error:
                await websocket.send_json({"type": "tts_error", "error": str(error)})
                continue
            await websocket.send_json({"type": "tts_start", "sample_rate": 24_000})
            try:
                iterator = neural_tts().synthesize_stream(
                    request.text, request.language, request.voice_id
                )
                while True:
                    generated = await asyncio.to_thread(next_tts_chunk, iterator)
                    if generated is None:
                        break
                    pcm, _sample_rate = generated
                    await websocket.send_bytes(pcm)
                await websocket.send_json({"type": "tts_end"})
            except Exception as error:
                await websocket.send_json({"type": "tts_error", "error": str(error)})
    except WebSocketDisconnect:
        pass


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


def websocket_authorized(websocket: WebSocket) -> bool:
    if not settings.api_token:
        return True
    supplied = websocket.headers.get("authorization")
    if supplied is None:
        supplied = f"Bearer {websocket.query_params.get('token', '')}"
    return secrets.compare_digest(supplied, f"Bearer {settings.api_token}")


@app.websocket("/v1/asr/stream")
async def stream_asr(websocket: WebSocket) -> None:
    if not websocket_authorized(websocket):
        await websocket.close(code=4401, reason="invalid bearer token")
        return
    ensure_preload_started()
    if _preload_error:
        await websocket.close(code=1011, reason="model preload failed")
        return
    if not speech_recognizer().loaded:
        await websocket.close(code=1013, reason="ASR model is still loading")
        return
    await websocket.accept()
    language = websocket.query_params.get("language", "en")
    try:
        sample_rate = int(websocket.query_params.get("sample_rate", "16000"))
    except ValueError:
        await websocket.close(code=4400, reason="invalid sample_rate")
        return
    if sample_rate != 16_000 or language not in {"en", "ko"}:
        await websocket.close(code=4400, reason="PCM16 16kHz mono en/ko required")
        return
    session = StreamingASRSession(sample_rate=sample_rate)
    ledger = StreamCommitLedger()
    phrase_accumulator = ConfirmedPhraseAccumulator()
    translation_session_id = f"ws-{id(websocket)}"
    target_language = "ko" if language == "en" else "en"
    # One item may be synthesizing while at most two fresh utterances wait.
    # Anything older is obsolete for live interpretation and is discarded.
    tts_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=2)
    send_lock = asyncio.Lock()

    async def send_json(message: dict) -> None:
        async with send_lock:
            await websocket.send_json(message)

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(payload)

    async def stream_translated_speech() -> None:
        while True:
            item = await tts_queue.get()
            translated_text, translated_language = item
            try:
                await send_json({
                    "type": "tts_start", "sample_rate": 24_000,
                })
                iterator = neural_tts().synthesize_stream(
                    translated_text, translated_language
                )
                while True:
                    generated = await asyncio.to_thread(next_tts_chunk, iterator)
                    if generated is None:
                        break
                    pcm, _sample_rate = generated
                    await send_bytes(pcm)
                await send_json({"type": "tts_end"})
            except Exception as error:
                await send_json({
                    "type": "tts_error", "error": str(error),
                })

    tts_task = asyncio.create_task(stream_translated_speech())

    async def decode(final: bool) -> None:
        if not session.pcm:
            return
        text, latency_ms = await asyncio.to_thread(
            speech_recognizer().transcribe_pcm16, bytes(session.pcm), language
        )
        session.mark_decoded()
        decision = ledger.apply(session.committer.update(text, final=final))
        stable_delta = str(decision.get("committed_delta", "")).strip()
        committed_delta = phrase_accumulator.append(stable_delta, final=final)
        decision["committed_delta"] = committed_delta
        translation = None
        translation_latency_ms = None
        if committed_delta:
            translation_started = time.perf_counter()
            translation = await asyncio.to_thread(
                translator().translate,
                committed_delta,
                language,
                target_language,
                translation_session_id,
            )
            translation_latency_ms = round(
                (time.perf_counter() - translation_started) * 1000, 1
            )
        await send_json({
            "type": "asr",
            **decision,
            "translation": translation,
            "translation_latency_ms": translation_latency_ms,
            "inline_tts": translation is not None,
            "audio_seconds": round(session.audio_seconds, 3),
            "latency_ms": round(latency_ms, 1),
            "model": settings.asr_model,
        })
        if translation:
            try:
                tts_queue.put_nowait((translation, target_language))
            except asyncio.QueueFull:
                # Preserve live speech: discard the oldest voice request, never
                # accumulate obsolete translated audio behind the speaker.
                try:
                    tts_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                tts_queue.put_nowait((translation, target_language))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            payload = message.get("bytes")
            command = message.get("text")
            if payload is not None:
                if session.append(payload):
                    await decode(final=False)
                if session.audio_seconds >= session.maximum_audio_seconds:
                    await decode(final=True)
                    ledger.begin_new_buffer()
                    session = StreamingASRSession(sample_rate=sample_rate)
            elif command == "finish":
                await decode(final=True)
                await send_json({"type": "finished"})
                break
            elif command == "reset":
                session = StreamingASRSession(sample_rate=sample_rate)
                phrase_accumulator.reset()
                await send_json({"type": "reset"})
            else:
                await send_json({"type": "error", "error": "unknown message"})
    except WebSocketDisconnect:
        pass
    finally:
        # Never keep a disconnected client alive while obsolete speech drains.
        tts_task.cancel()
        await asyncio.gather(tts_task, return_exceptions=True)
        translator().reset(translation_session_id)
