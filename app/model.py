"""Model lifecycle and transcription execution."""

from __future__ import annotations

import gc
import re
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Generator

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from app.config import Config

_TIMESTAMP_RE = re.compile(r"<\|(\d+\.\d+)\|>")


class ModelLoadError(RuntimeError):
    """Raised when a model id cannot be loaded for transcription."""


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)


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


@dataclass
class Metrics:
    transcriptions_total: int
    translations_total: int
    inference_seconds_total: float
    errors_total: int
    active_inferences: int


class WhisperModelManager:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._model_load_locks: dict[str, threading.Lock] = {}
        self._loading_models = 0
        self._error: str | None = None
        self._resolved_device = self._resolve_device()
        # Metrics
        self._metrics_lock = threading.Lock()
        self._transcription_count = 0
        self._translation_count = 0
        self._inference_seconds = 0.0
        self._error_count = 0
        self._active_inferences = 0

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

    def metrics(self) -> Metrics:
        with self._metrics_lock:
            return Metrics(
                transcriptions_total=self._transcription_count,
                translations_total=self._translation_count,
                inference_seconds_total=self._inference_seconds,
                errors_total=self._error_count,
                active_inferences=self._active_inferences,
            )

    def load_model(self, model_id: str) -> None:
        self._get_or_load_model(model_id)

    def unload_model(self, model_id: str) -> bool:
        with self._cache_lock:
            bundle = self._cache.pop(model_id, None)
        if bundle is not None:
            self._dispose_bundle(bundle)
            return True
        return False

    def supported_languages(self, model_id: str) -> list[str]:
        bundle = self._get_or_load_model(model_id)
        tokenizer = bundle.processor.tokenizer
        langs = []
        for token in getattr(tokenizer, "additional_special_tokens", []):
            if token.startswith("<|") and token.endswith("|>") and len(token) == 6:
                langs.append(token[2:-2])
        return sorted(langs)

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

    def _parse_segments(self, raw: str) -> list[TranscriptionSegment]:
        matches = list(_TIMESTAMP_RE.finditer(raw))
        segments = []
        i = 0
        while i < len(matches) - 1:
            start = float(matches[i].group(1))
            end = float(matches[i + 1].group(1))
            text = raw[matches[i].end() : matches[i + 1].start()].strip()
            if text:
                segments.append(TranscriptionSegment(start=start, end=end, text=text))
                i += 2
            else:
                i += 1
        return segments

    def _model_compute_dtype(self, model: Any) -> torch.dtype:
        """Resolve runtime model dtype, falling back to configured default."""
        with suppress(Exception):
            parameter = next(model.parameters())
            if isinstance(parameter, torch.Tensor):
                return parameter.dtype
        return self._torch_dtype()

    def _transcribe_tensor(
        self,
        bundle: LoadedModel,
        audio: torch.Tensor,
        sample_rate: int,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str = "transcribe",
        time_offset: float = 0.0,
    ) -> TranscriptionResult:
        inputs = bundle.processor(
            audio.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,
            return_attention_mask=True,
        )

        device = self._device()
        model_dtype = self._model_compute_dtype(bundle.model)
        model_inputs: dict[str, Any] = {
            "input_features": inputs["input_features"].to(device=device, dtype=model_dtype)
        }
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device)

        generate_kwargs: dict[str, Any] = {
            "task": task,
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

        del inputs, audio  # Free CPU tensors before inference

        with bundle.infer_lock:
            try:
                with torch.inference_mode():
                    generated = bundle.model.generate(**model_inputs, **generate_kwargs)
                self._set_error("")

                sequences = generated.get("sequences") if isinstance(generated, dict) else generated
                if sequences is None:
                    raise RuntimeError("Whisper generate() returned no sequences")

                text = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
                text = text[0].strip() if text else ""

                # Parse timestamped segments
                segments: list[TranscriptionSegment] = []
                with suppress(Exception):
                    raw = bundle.processor.tokenizer.decode(
                        sequences[0], skip_special_tokens=False, decode_with_timestamps=True
                    )
                    segments = self._parse_segments(raw)
                    if time_offset > 0:
                        segments = [
                            TranscriptionSegment(s.start + time_offset, s.end + time_offset, s.text)
                            for s in segments
                        ]

                return TranscriptionResult(text=text, segments=segments)
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                raise
            finally:
                del model_inputs
                if "prompt_ids" in generate_kwargs:
                    del generate_kwargs["prompt_ids"]
                gc.collect()
                self._clear_device_cache()

    def transcribe(
        self,
        model_id: str,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        bundle = self._get_or_load_model(model_id)
        audio, sample_rate = self._load_audio(audio_path, bundle.processor)

        with self._metrics_lock:
            self._active_inferences += 1

        t0 = time.monotonic()
        try:
            result = self._transcribe_tensor(
                bundle, audio, sample_rate, language, prompt, temperature, task
            )
            with self._metrics_lock:
                if task == "translate":
                    self._translation_count += 1
                else:
                    self._transcription_count += 1
                self._inference_seconds += time.monotonic() - t0
            return result
        except Exception:
            with self._metrics_lock:
                self._error_count += 1
                self._inference_seconds += time.monotonic() - t0
            raise
        finally:
            with self._metrics_lock:
                self._active_inferences -= 1

    def transcribe_chunks(
        self,
        model_id: str,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str = "transcribe",
    ) -> Generator[TranscriptionResult, None, None]:
        """Yield a TranscriptionResult per ~30 s audio chunk for streaming."""
        bundle = self._get_or_load_model(model_id)
        audio, sample_rate = self._load_audio(audio_path, bundle.processor)

        chunk_duration = 30  # seconds
        chunk_samples = chunk_duration * sample_rate
        total_samples = audio.shape[0]

        if total_samples <= chunk_samples:
            yield self.transcribe(model_id, audio_path, language, prompt, temperature, task)
            return

        with self._metrics_lock:
            self._active_inferences += 1

        t0 = time.monotonic()
        try:
            for start in range(0, total_samples, chunk_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = audio[start:end]
                time_offset = start / sample_rate
                result = self._transcribe_tensor(
                    bundle, chunk, sample_rate, language, prompt, temperature, task, time_offset
                )
                yield result

            with self._metrics_lock:
                if task == "translate":
                    self._translation_count += 1
                else:
                    self._transcription_count += 1
                self._inference_seconds += time.monotonic() - t0
        except Exception:
            with self._metrics_lock:
                self._error_count += 1
                self._inference_seconds += time.monotonic() - t0
            raise
        finally:
            with self._metrics_lock:
                self._active_inferences -= 1


whisper_model = WhisperModelManager()
