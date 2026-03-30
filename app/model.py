"""Model lifecycle and transcription execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
        self._model: Any | None = None
        self._processor: Any | None = None
        self._init_lock = threading.Lock()
        self._infer_lock = threading.Lock()
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

    def _torch_dtype(self) -> torch.dtype:
        return torch.float16 if self._resolved_device == "cuda" else torch.float32

    def _device(self) -> torch.device:
        return torch.device(self._resolved_device)

    def initialize(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._init_lock:
            if self._model is not None and self._processor is not None:
                return

            self._initializing = True
            self._error = None
            try:
                self._processor = AutoProcessor.from_pretrained(Config.MODEL_ID)
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    Config.MODEL_ID,
                    torch_dtype=self._torch_dtype(),
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                self._model.to(self._device())
                self._model.eval()
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                raise
            finally:
                self._initializing = False

    def status(self) -> ModelStatus:
        return ModelStatus(
            initialized=self._model is not None and self._processor is not None,
            initializing=self._initializing,
            model_id=Config.MODEL_ID,
            device=self._resolved_device,
            error=self._error,
        )

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        if self._processor is None:
            raise RuntimeError("Whisper processor is not initialized")

        audio, sample_rate = torchaudio.load(audio_path)
        if audio.ndim == 2 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)

        target_sample_rate = self._processor.feature_extractor.sampling_rate
        if sample_rate != target_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        return audio.cpu(), sample_rate

    def _prompt_ids(self, prompt: str) -> torch.Tensor | None:
        if self._processor is None:
            return None

        getter = getattr(self._processor, "get_prompt_ids", None)
        if getter is None and hasattr(self._processor, "tokenizer"):
            getter = getattr(self._processor.tokenizer, "get_prompt_ids", None)
        if getter is None:
            return None

        try:
            prompt_ids = getter(prompt, return_tensors="pt")
        except TypeError:
            prompt_ids = getter(prompt)

        if isinstance(prompt_ids, torch.Tensor):
            return prompt_ids.reshape(-1).to(dtype=torch.long)

        return torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)

    def _decode(self, generated: Any) -> str:
        if self._processor is None:
            raise RuntimeError("Whisper processor is not initialized")

        sequences = generated.get("sequences") if isinstance(generated, dict) else generated
        if sequences is None:
            raise RuntimeError("Whisper generate() returned no sequences")

        decoded = self._processor.batch_decode(sequences, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def transcribe(
        self,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
    ) -> str:
        self.initialize()

        if self._model is None or self._processor is None:
            raise RuntimeError("Whisper model is not initialized")

        audio, sample_rate = self._load_audio(audio_path)
        inputs = self._processor(
            audio.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,
            return_attention_mask=True,
        )

        device = self._device()
        model_inputs: dict[str, Any] = {"input_features": inputs["input_features"].to(device)}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device)

        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "return_timestamps": True,
        }
        if language:
            generate_kwargs["language"] = language
        elif Config.DEFAULT_LANGUAGE:
            generate_kwargs["language"] = Config.DEFAULT_LANGUAGE

        if prompt:
            prompt_ids = self._prompt_ids(prompt)
            if prompt_ids is not None:
                generate_kwargs["prompt_ids"] = prompt_ids.to(device)

        if temperature is not None:
            generate_kwargs["temperature"] = temperature
            if temperature > 0:
                generate_kwargs["do_sample"] = True

        with self._infer_lock:
            try:
                with torch.inference_mode():
                    generated = self._model.generate(**model_inputs, **generate_kwargs)
                self._error = None
                return self._decode(generated)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                if self._resolved_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise


whisper_model = WhisperModel()
