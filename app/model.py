"""Model lifecycle and transcription execution."""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from app.config import Config


class ModelLoadError(RuntimeError):
    """Raised when a model id cannot be loaded for transcription."""


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    processor: Any
    infer_lock: threading.Lock


@dataclass
class ModelStatus:
    initialized: bool
    initializing: bool
    model_id: str
    device: str
    error: str | None
    loaded_model_count: int
    max_models_in_memory: int
    loaded_models: list[str]


class WhisperModelManager:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._model_load_locks: dict[str, threading.Lock] = {}
        self._loading_models = 0
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

    def _clear_device_cache(self) -> None:
        if self._resolved_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_error(self, message: str) -> None:
        with self._cache_lock:
            self._error = message

    def _set_loading(self, loading: bool) -> None:
        with self._cache_lock:
            if loading:
                self._loading_models += 1
            else:
                self._loading_models = max(0, self._loading_models - 1)

    def _load_bundle(self, model_id: str) -> LoadedModel:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self._torch_dtype(),
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self._device())
        model.eval()
        return LoadedModel(
            model_id=model_id,
            model=model,
            processor=processor,
            infer_lock=threading.Lock(),
        )

    def _dispose_bundle(self, bundle: LoadedModel) -> None:
        with suppress(Exception):
            if hasattr(bundle.model, "to"):
                bundle.model.to("cpu")

        del bundle.model
        del bundle.processor
        gc.collect()
        self._clear_device_cache()

    def _get_or_load_model(self, model_id: str) -> LoadedModel:
        with self._cache_lock:
            cached = self._cache.get(model_id)
            if cached is not None:
                self._cache.move_to_end(model_id)
                return cached
            load_lock = self._model_load_locks.setdefault(model_id, threading.Lock())

        with load_lock:
            with self._cache_lock:
                cached = self._cache.get(model_id)
                if cached is not None:
                    self._cache.move_to_end(model_id)
                    return cached

            self._set_loading(True)
            try:
                bundle = self._load_bundle(model_id)
                self._set_error("")
            except Exception as exc:  # noqa: BLE001
                message = f"Unable to load model '{model_id}': {exc}"
                self._set_error(message)
                with self._cache_lock:
                    self._model_load_locks.pop(model_id, None)
                raise ModelLoadError(message) from exc
            finally:
                self._set_loading(False)

            evicted: list[LoadedModel] = []
            with self._cache_lock:
                self._cache[model_id] = bundle
                self._cache.move_to_end(model_id)
                while len(self._cache) > Config.MAX_MODELS_IN_MEMORY:
                    _, old_bundle = self._cache.popitem(last=False)
                    evicted.append(old_bundle)
                self._model_load_locks.pop(model_id, None)

            for old_bundle in evicted:
                self._dispose_bundle(old_bundle)

            return bundle

    def initialize(self) -> None:
        # Keeps backward compatibility with startup warm-up flows.
        self._get_or_load_model(Config.MODEL_ID)

    def loaded_model_ids(self) -> list[str]:
        with self._cache_lock:
            return list(self._cache.keys())

    def status(self) -> ModelStatus:
        with self._cache_lock:
            loaded_models = list(self._cache.keys())
            return ModelStatus(
                initialized=bool(loaded_models),
                initializing=self._loading_models > 0,
                model_id=Config.MODEL_ID,
                device=self._resolved_device,
                error=self._error or None,
                loaded_model_count=len(loaded_models),
                max_models_in_memory=Config.MAX_MODELS_IN_MEMORY,
                loaded_models=loaded_models,
            )

    def _load_audio(self, audio_path: str, processor: Any) -> tuple[torch.Tensor, int]:
        audio, sample_rate = torchaudio.load(audio_path)
        if audio.ndim == 2 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)

        target_sample_rate = processor.feature_extractor.sampling_rate
        if sample_rate != target_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        return audio.cpu(), sample_rate

    def _prompt_ids(self, processor: Any, prompt: str) -> torch.Tensor | None:
        getter = getattr(processor, "get_prompt_ids", None)
        if getter is None and hasattr(processor, "tokenizer"):
            getter = getattr(processor.tokenizer, "get_prompt_ids", None)
        if getter is None:
            return None

        try:
            prompt_ids = getter(prompt, return_tensors="pt")
        except TypeError:
            prompt_ids = getter(prompt)

        if isinstance(prompt_ids, torch.Tensor):
            return prompt_ids.reshape(-1).to(dtype=torch.long)

        return torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)

    def _decode(self, processor: Any, generated: Any) -> str:
        sequences = generated.get("sequences") if isinstance(generated, dict) else generated
        if sequences is None:
            raise RuntimeError("Whisper generate() returned no sequences")

        decoded = processor.batch_decode(sequences, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def _model_compute_dtype(self, model: Any) -> torch.dtype:
        """Resolve runtime model dtype, falling back to configured default."""
        with suppress(Exception):
            parameter = next(model.parameters())
            if isinstance(parameter, torch.Tensor):
                return parameter.dtype
        return self._torch_dtype()

    def transcribe(
        self,
        model_id: str,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
    ) -> str:
        bundle = self._get_or_load_model(model_id)

        audio, sample_rate = self._load_audio(audio_path, bundle.processor)
        inputs = bundle.processor(
            audio.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,
            return_attention_mask=True,
        )

        device = self._device()
        model_dtype = self._model_compute_dtype(bundle.model)
        # Keep input features aligned with model dtype to avoid float32/float16 mismatch on CUDA.
        model_inputs: dict[str, Any] = {
            "input_features": inputs["input_features"].to(device=device, dtype=model_dtype)
        }
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
            prompt_ids = self._prompt_ids(bundle.processor, prompt)
            if prompt_ids is not None:
                generate_kwargs["prompt_ids"] = prompt_ids.to(device)

        if temperature is not None:
            generate_kwargs["temperature"] = temperature
            if temperature > 0:
                generate_kwargs["do_sample"] = True

        with bundle.infer_lock:
            try:
                with torch.inference_mode():
                    generated = bundle.model.generate(**model_inputs, **generate_kwargs)
                self._set_error("")
                return self._decode(bundle.processor, generated)
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                self._clear_device_cache()
                raise


whisper_model = WhisperModelManager()
