"""In-memory batch transcription job queue."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.model import TranscriptionResult, WhisperModelManager


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchJob:
    id: str
    status: JobStatus
    model_id: str
    language: str | None
    prompt: str | None
    temperature: float | None
    task: str
    audio_path: str
    created_at: float
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore:
    def __init__(self, model_manager: WhisperModelManager) -> None:
        self._jobs: dict[str, BatchJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._model_manager = model_manager
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None

    def _result_dict(self, result: TranscriptionResult) -> dict[str, Any]:
        return {
            "text": result.text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in result.segments
            ],
        }

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue

            job.status = JobStatus.PROCESSING
            try:
                result = await asyncio.to_thread(
                    self._model_manager.transcribe,
                    job.model_id,
                    job.audio_path,
                    job.language,
                    job.prompt,
                    job.temperature,
                    job.task,
                )
                job.result = self._result_dict(result)
                job.status = JobStatus.COMPLETED
            except Exception as exc:  # noqa: BLE001
                job.error = str(exc)
                job.status = JobStatus.FAILED
            finally:
                job.completed_at = time.time()
                if os.path.exists(job.audio_path):
                    os.unlink(job.audio_path)

    async def submit(
        self,
        audio_path: str,
        model_id: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str = "transcribe",
    ) -> BatchJob:
        job = BatchJob(
            id=uuid.uuid4().hex[:12],
            status=JobStatus.PENDING,
            model_id=model_id,
            language=language,
            prompt=prompt,
            temperature=temperature,
            task=task,
            audio_path=audio_path,
            created_at=time.time(),
        )
        self._jobs[job.id] = job
        await self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def to_dict(self, job: BatchJob) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": job.id,
            "status": job.status.value,
            "model": job.model_id,
            "task": job.task,
            "created_at": job.created_at,
        }
        if job.completed_at is not None:
            d["completed_at"] = job.completed_at
        if job.result is not None:
            d["result"] = job.result
        if job.error is not None:
            d["error"] = job.error
        return d
