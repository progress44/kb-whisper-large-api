"""Model lifecycle and transcription execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import pipeline

from app.config import Config


@dataclass
class ModelStatus:
    initialized: bool
    initializing: bool
    model_id: str
    device: str
    error: str | None


class WhisperModel:
    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._lock = threading.Lock()
        self._initializing = False
        self._error: str | None = None
        self._resolved_device = self._resolve_device()

    def _resolve_device(self) -> str:
        configured = Config.DEVICE.lower()
        if configured != "auto":
            return configured

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _pipeline_device(self) -> int:
        return 0 if self._resolved_device == "cuda" else -1

    def _torch_dtype(self) -> torch.dtype:
        return torch.float16 if self._resolved_device == "cuda" else torch.float32

    def initialize(self) -> None:
        if self._pipe is not None:
            return

        with self._lock:
            if self._pipe is not None:
                return

            self._initializing = True
            self._error = None
            try:
                self._pipe = pipeline(
                    "automatic-speech-recognition",
                    model=Config.MODEL_ID,
                    device=self._pipeline_device(),
                    torch_dtype=self._torch_dtype(),
                    model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": True},
                )
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                raise
            finally:
                self._initializing = False

    def status(self) -> ModelStatus:
        return ModelStatus(
            initialized=self._pipe is not None,
            initializing=self._initializing,
            model_id=Config.MODEL_ID,
            device=self._resolved_device,
            error=self._error,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
    ) -> str:
        self.initialize()

        if self._pipe is None:
            raise RuntimeError("Whisper pipeline is not initialized")

        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        elif Config.DEFAULT_LANGUAGE:
            generate_kwargs["language"] = Config.DEFAULT_LANGUAGE

        if prompt:
            generate_kwargs["prompt"] = prompt

        if temperature is not None:
            generate_kwargs["temperature"] = temperature

        result = self._pipe(
            audio_path,
            return_timestamps=False,
            chunk_length_s=30,
            batch_size=8,
            generate_kwargs=generate_kwargs,
        )

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip()


whisper_model = WhisperModel()
