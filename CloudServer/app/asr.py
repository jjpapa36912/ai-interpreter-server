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


def _normalized(word: str) -> str:
    return word.casefold().strip(".,!?;:\"")


def _uncommitted_tail(committed: list[str], current: list[str]) -> list[str] | None:
    """Align a revised cumulative hypothesis after already-spoken words.

    Whisper can remove fillers or rewrite an early phrase. In that case a raw
    word index repeats or skips text. Align the longest committed suffix found
    in the new hypothesis and only consider words after it.
    """
    if not committed:
        return current
    normalized_current = [_normalized(word) for word in current]
    normalized_committed = [_normalized(word) for word in committed]
    for overlap in range(min(len(committed), len(current)), 0, -1):
        suffix = normalized_committed[-overlap:]
        for start in range(0, len(current) - overlap + 1):
            if normalized_current[start:start + overlap] == suffix:
                return current[start + overlap:]
    return None


@dataclass
class StableASRCommitter:
    """Commit only words preserved by two consecutive cumulative decodes."""

    lookahead_words: int = 2
    # Two matching incremental decodes still make this confirmed-only, while
    # avoiding the extra 300 ms turn that pushed first translated speech near
    # two seconds even on a warm GPU.
    agreement_decodes: int = 2
    uncommitted_history: list[list[str]] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)

    def update(self, text: str, final: bool = False) -> dict[str, object]:
        current = _words(text)
        tail = _uncommitted_tail(self.committed, current)
        if tail is None:
            self.uncommitted_history = []
            delta: list[str] = []
        else:
            if final:
                stable_count = len(tail)
            elif len(self.uncommitted_history) < self.agreement_decodes - 1:
                stable_count = 0
            else:
                stable_count = len(tail)
                for prior in self.uncommitted_history:
                    stable_count = min(
                        stable_count, _common_prefix_length(prior, tail)
                    )
            # Always allow one stable leading word for sub-1.5 s startup, but
            # retain two lookahead words once the hypothesis becomes longer.
            hold = min(self.lookahead_words, max(0, stable_count - 1))
            commit_count = len(tail) if final else max(0, stable_count - hold)
            delta = tail[:commit_count]
            self.committed.extend(delta)
            history = [prior[commit_count:] for prior in self.uncommitted_history]
            history.append(tail[commit_count:])
            self.uncommitted_history = history[-(self.agreement_decodes - 1):]
        return {
            "hypothesis": " ".join(current),
            "committed_delta": " ".join(delta),
            "committed_words": len(self.committed),
            "is_final": final,
        }


@dataclass
class StreamCommitLedger:
    """Keep monotonic counts and remove duplicated words at buffer seams."""

    committed: list[str] = field(default_factory=list)
    seam_pending: bool = False

    def begin_new_buffer(self) -> None:
        self.seam_pending = True

    def apply(self, decision: dict[str, object]) -> dict[str, object]:
        delta = _words(str(decision.get("committed_delta", "")))
        if delta and self.seam_pending:
            previous = [_normalized(word) for word in self.committed[-8:]]
            incoming = [_normalized(word) for word in delta]
            overlap = 0
            for size in range(min(len(previous), len(incoming)), 0, -1):
                if previous[-size:] == incoming[:size]:
                    overlap = size
                    break
            delta = delta[overlap:]
            self.seam_pending = False
        self.committed.extend(delta)
        return {
            **decision,
            "committed_delta": " ".join(delta),
            "committed_words": len(self.committed),
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
                beam_size=self.settings.asr_beam_size,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, (time.perf_counter() - started) * 1000

    def warmup(self) -> None:
        """Run real CUDA inference so the first user utterance isn't the warmup."""
        silence = np.zeros(16_000, dtype="<i2").tobytes()
        self.transcribe_pcm16(silence, "en")
        self.transcribe_pcm16(silence, "ko")


@dataclass
class StreamingASRSession:
    sample_rate: int = 16_000
    minimum_decode_seconds: float = 0.4
    decode_interval_seconds: float = 0.2
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
