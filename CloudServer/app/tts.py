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

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            import torch
            from qwen_tts import Qwen3TTSModel

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

    def synthesize(self, text: str, language: str, voice_id: str | None = None) -> tuple[bytes, int]:
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
        with self._generation_lock:
            wavs, sample_rate = self.model.generate_custom_voice(
                text=clean,
                language=model_language,
                speaker=speaker,
                instruct="자연스럽고 또렷한 대화체로 말해 주세요."
                if language == "ko" else
                "Speak naturally and clearly in a conversational tone.",
            )
        waveform = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
        return pcm.tobytes(), int(sample_rate)

    def warmup(self) -> None:
        self.synthesize("준비됐습니다.", "ko")
