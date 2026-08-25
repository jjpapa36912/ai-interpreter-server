from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .settings import Settings


def _words(text: str) -> list[str]:
    return " ".join(text.split()).split()


def _common_prefix_length(left: list[str], right: list[str]) -> int:
    length = 0
    for expected, actual in zip(left, right):
        if expected.casefold().strip(".,!?;:\"") != actual.casefold().strip(".,!?;:\""):
            break
        length += 1
    return length


@dataclass
class StableASRCommitter:
    """Commit only words preserved by two consecutive cumulative decodes."""

    lookahead_words: int = 2
    previous: list[str] = field(default_factory=list)
    committed_count: int = 0

    def update(self, text: str, final: bool = False) -> dict[str, object]:
        current = _words(text)
        stable_count = len(current) if final else _common_prefix_length(self.previous, current)
        commit_count = len(current) if final else max(0, stable_count - self.lookahead_words)
        commit_count = max(self.committed_count, commit_count)
        delta = current[self.committed_count:commit_count]
        self.committed_count = commit_count
        self.previous = current
        return {
            "hypothesis": " ".join(current),
            "committed_delta": " ".join(delta),
            "committed_words": self.committed_count,
            "is_final": final,
        }


class CUDASpeechRecognizer:
    """Resident faster-whisper model shared by bounded WebSocket sessions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._load_lock = threading.Lock()
        self._decode_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.settings.asr_model,
                device="cuda",
                compute_type=self.settings.asr_compute_type,
            )

    def transcribe_pcm16(self, pcm: bytes, language: str) -> tuple[str, float]:
        if language not in {"en", "ko"}:
            raise ValueError("only en/ko ASR is supported")
        self.load()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        started = time.perf_counter()
        with self._decode_lock:
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=1,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, (time.perf_counter() - started) * 1000


@dataclass
class StreamingASRSession:
    sample_rate: int = 16_000
    minimum_decode_seconds: float = 0.8
    decode_interval_seconds: float = 0.6
    maximum_audio_seconds: float = 20.0
    pcm: bytearray = field(default_factory=bytearray)
    last_decode_bytes: int = 0
    committer: StableASRCommitter = field(default_factory=StableASRCommitter)

    def append(self, payload: bytes) -> bool:
        if len(payload) % 2:
            raise ValueError("PCM16 payload must contain complete samples")
        self.pcm.extend(payload)
        minimum = int(self.sample_rate * self.minimum_decode_seconds) * 2
        interval = int(self.sample_rate * self.decode_interval_seconds) * 2
        return len(self.pcm) >= minimum and len(self.pcm) - self.last_decode_bytes >= interval

    def mark_decoded(self) -> None:
        self.last_decode_bytes = len(self.pcm)

    @property
    def audio_seconds(self) -> float:
        return len(self.pcm) / (self.sample_rate * 2)

