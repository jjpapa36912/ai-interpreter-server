from __future__ import annotations

import threading

import numpy as np

from .settings import Settings


class CUDANeuralTTS:
    """Bilingual CUDA TTS boundary returning raw mono PCM16."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.loaded = False
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self.streaming_available = False

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            import torch
            try:
                from faster_qwen3_tts import FasterQwen3TTS
            except ImportError:
                FasterQwen3TTS = None
            from qwen_tts import Qwen3TTSModel

            if FasterQwen3TTS is not None:
                self.model = FasterQwen3TTS.from_pretrained(
                    self.settings.tts_model,
                    device="cuda",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                self.streaming_available = True
            else:
                kwargs = {
                    "device_map": "cuda:0",
                    "dtype": torch.bfloat16,
                }
                try:
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.settings.tts_model,
                        attn_implementation="flash_attention_2",
                        **kwargs,
                    )
                except (ImportError, ValueError):
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.settings.tts_model,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
            self.loaded = True

    @staticmethod
    def _pcm(waveform) -> bytes:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        return (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    @staticmethod
    def _remove_excess_silence(waveform, sample_rate: int, state: dict) -> np.ndarray:
        """Remove model padding and cap unnatural pauses across stream chunks.

        Qwen can place 0.5--1.1 s of digital/near-digital silence at the start
        or inside a short live utterance. That silence was audible as repeated
        speech drop-outs and also delayed the first actual phoneme. Keep a
        natural 160 ms pause between voiced regions and discard leading and
        trailing model padding. State is shared across generated chunks so a
        pause split at a chunk boundary is handled as one pause.
        """
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        threshold = 0.001
        maximum_pause = max(1, int(sample_rate * 0.16))
        output: list[float] = []
        for sample in samples:
            if abs(float(sample)) < threshold:
                if state["started"]:
                    state["pending_silence"] += 1
                continue
            if state["started"] and state["pending_silence"]:
                output.extend([0.0] * min(state["pending_silence"], maximum_pause))
            state["pending_silence"] = 0
            state["started"] = True
            output.append(float(sample))
        return np.asarray(output, dtype=np.float32)

    def synthesize_stream(self, text: str, language: str, voice_id: str | None = None):
        """Yield playable PCM while CUDA generation is still in progress."""
        if language not in {"en", "ko"}:
            raise ValueError("TTS language must be en or ko")
        clean = text.strip()
        if not clean:
            raise ValueError("TTS text must not be empty")
        self.load()
        speaker = voice_id or (
            self.settings.tts_korean_voice if language == "ko"
            else self.settings.tts_english_voice
        )
        silence_state = {"started": False, "pending_silence": 0}
        model_language = "Korean" if language == "ko" else "English"
        instruct = (
            "자연스럽고 또렷한 대화체로 말해 주세요." if language == "ko" else
            "Speak naturally and clearly in a conversational tone."
        )
        with self._generation_lock:
            if self.streaming_available:
                for waveform, sample_rate, _timing in self.model.generate_custom_voice_streaming(
                    text=clean,
                    language=model_language,
                    speaker=speaker,
                    instruct=instruct,
                    chunk_size=4,
                    non_streaming_mode=True,
                ):
                    cleaned = self._remove_excess_silence(
                        waveform, int(sample_rate), silence_state
                    )
                    pcm = self._pcm(cleaned)
                    if pcm:
                        yield pcm, int(sample_rate)
                return
            wavs, sample_rate = self.model.generate_custom_voice(
                text=clean,
                language=model_language,
                speaker=speaker,
                instruct=instruct,
            )
            cleaned = self._remove_excess_silence(
                wavs[0], int(sample_rate), silence_state
            )
            pcm = self._pcm(cleaned)
            if pcm:
                yield pcm, int(sample_rate)

    def synthesize(self, text: str, language: str, voice_id: str | None = None) -> tuple[bytes, int]:
        chunks = list(self.synthesize_stream(text, language, voice_id))
        if not chunks:
            raise RuntimeError("TTS returned no audio")
        return b"".join(chunk for chunk, _ in chunks), chunks[0][1]

    def warmup(self) -> None:
        self.load()
        if self.streaming_available and hasattr(self.model, "warmup"):
            self.model.warmup(prefill_len=100)
        self.synthesize("준비됐습니다.", "ko")
