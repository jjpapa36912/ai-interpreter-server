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

    def _use_accelerated_backend(self, backend_available: bool) -> bool:
        # FasterQwen also accelerates the proven whole-utterance API. Keep its
        # experimental streaming generator independently opt-in below.
        return backend_available

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

            # Use FasterQwen's accelerated whole-utterance implementation in
            # production. Its experimental streaming generator remains a
            # separate explicit opt-in; the app now lets an accepted request
            # finish instead of disconnecting mid-generation.
            if self._use_accelerated_backend(FasterQwen3TTS is not None):
                self.model = FasterQwen3TTS.from_pretrained(
                    self.settings.tts_model,
                    device="cuda",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                self.streaming_available = self.settings.tts_experimental_streaming
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
        model_language = "Korean" if language == "ko" else "English"
        instruct = (
            "실제 전문 통역사가 바로 옆에서 말하듯 자연스럽고 또렷하게 말하세요. "
            "입력된 문장만 정확히 읽고, 어·음·에 같은 군더더기 소리나 설명을 절대 추가하지 마세요. "
            "문장 중간을 끊지 말고 의미 단위가 자연스럽게 이어지게 발음하세요."
            if language == "ko" else
            "Speak like a natural professional interpreter in a clear conversational voice. "
            "Say only the supplied text exactly; never add fillers, commentary, or extra sounds. "
            "Keep each thought connected and do not stop in the middle of a phrase."
        )
        with self._generation_lock:
            if self.streaming_available:
                for waveform, sample_rate, _timing in self.model.generate_custom_voice_streaming(
                    text=clean,
                    language=model_language,
                    speaker=speaker,
                    instruct=instruct,
                    chunk_size=4,
                    # The old value waited for the complete utterance before
                    # yielding anything, despite the streaming HTTP endpoint.
                    # Emit codec chunks as they are generated so first audio is
                    # not tied to total sentence duration.
                    non_streaming_mode=False,
                ):
                    pcm = self._pcm(waveform)
                    if pcm:
                        yield pcm, int(sample_rate)
                return
            wavs, sample_rate = self.model.generate_custom_voice(
                text=clean,
                language=model_language,
                speaker=speaker,
                instruct=instruct,
            )
            pcm = self._pcm(wavs[0])
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
