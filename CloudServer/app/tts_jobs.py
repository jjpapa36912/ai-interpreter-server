from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TTSJobSnapshot:
    chunk: bytes | None
    done: bool
    final_chunk: bool
    error: str | None


@dataclass
class TTSJob:
    job_id: str
    sample_rate: int
    request_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    first_chunk_at: float | None = None
    last_accessed_at: float = field(default_factory=time.monotonic)
    chunks: list[bytes] = field(default_factory=list)
    generation_started: bool = False
    done: bool = False
    error: str | None = None
    _condition: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )

    def claim_generation(self) -> bool:
        """Return true exactly once, including idempotent create retries."""
        with self._condition:
            if self.generation_started:
                return False
            self.generation_started = True
            return True

    def append(self, pcm: bytes, maximum_chunk_bytes: int) -> None:
        if not pcm:
            return
        with self._condition:
            for offset in range(0, len(pcm), maximum_chunk_bytes):
                chunk = pcm[offset:offset + maximum_chunk_bytes]
                if chunk:
                    if self.first_chunk_at is None:
                        self.first_chunk_at = time.monotonic()
                    self.chunks.append(chunk)
            self.last_accessed_at = time.monotonic()
            self._condition.notify_all()

    def finish(self) -> None:
        with self._condition:
            self.done = True
            self.last_accessed_at = time.monotonic()
            self._condition.notify_all()

    def fail(self, message: str) -> None:
        with self._condition:
            self.error = message
            self.done = True
            self.last_accessed_at = time.monotonic()
            self._condition.notify_all()

    def wait_for(self, sequence: int, timeout: float) -> TTSJobSnapshot:
        with self._condition:
            self.last_accessed_at = time.monotonic()
            if sequence >= len(self.chunks) and not self.done:
                self._condition.wait_for(
                    lambda: sequence < len(self.chunks) or self.done,
                    timeout=max(0.0, timeout),
                )
            self.last_accessed_at = time.monotonic()
            chunk = self.chunks[sequence] if sequence < len(self.chunks) else None
            return TTSJobSnapshot(
                chunk=chunk,
                done=self.done,
                final_chunk=self.done and chunk is not None
                    and sequence == len(self.chunks) - 1,
                error=self.error,
            )


class TTSJobStore:
    """Small, retry-safe PCM mailbox for gateways that buffer HTTP streams."""

    def __init__(
        self,
        sample_rate: int = 24_000,
        chunk_milliseconds: int = 200,
        maximum_jobs: int = 32,
        retention_seconds: float = 120.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.maximum_chunk_bytes = max(
            2, int(sample_rate * chunk_milliseconds / 1_000) * 2
        )
        self.maximum_jobs = maximum_jobs
        self.retention_seconds = retention_seconds
        self._lock = threading.Lock()
        self._jobs: dict[str, TTSJob] = {}
        self._request_jobs: dict[str, str] = {}

    def create(self, request_id: str | None = None) -> TTSJob | None:
        with self._lock:
            self._prune_locked()
            if request_id:
                existing_id = self._request_jobs.get(request_id)
                if existing_id:
                    existing = self._jobs.get(existing_id)
                    if existing is not None:
                        return existing
            active_jobs = sum(1 for job in self._jobs.values() if not job.done)
            if active_jobs >= self.maximum_jobs:
                return None
            job = TTSJob(
                job_id=uuid.uuid4().hex,
                sample_rate=self.sample_rate,
                request_id=request_id,
            )
            self._jobs[job.job_id] = job
            if request_id:
                self._request_jobs[request_id] = job.job_id
            return job

    def get(self, job_id: str) -> TTSJob | None:
        with self._lock:
            self._prune_locked()
            return self._jobs.get(job_id)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [
            job_id for job_id, job in self._jobs.items()
            if job.done and now - job.last_accessed_at > self.retention_seconds
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id, None)
            if job is not None and job.request_id:
                self._request_jobs.pop(job.request_id, None)
