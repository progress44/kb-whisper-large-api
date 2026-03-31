from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch

from app.model import LoadedModel, TranscriptionResult, WhisperModelManager


def make_bundle(model_id: str) -> LoadedModel:
    model = MagicMock()
    processor = MagicMock()
    return LoadedModel(
        model_id=model_id,
        model=model,
        processor=processor,
        infer_lock=threading.Lock(),
    )


class _FakeTokenizer:
    additional_special_tokens = ["<|en|>", "<|sv|>", "<|notranslate|>"]

    def decode(self, _sequence, skip_special_tokens=True, decode_with_timestamps=False):  # noqa: ARG002
        if decode_with_timestamps:
            return "<|0.00|> hej varlden<|2.50|>"
        return "hej varlden"

    def get_prompt_ids(self, _prompt, return_tensors="pt"):  # noqa: ARG002
        return torch.tensor([10, 11, 12], dtype=torch.long)


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = _FakeFeatureExtractor()
        self.tokenizer = _FakeTokenizer()

    def __call__(self, *_args, **_kwargs):
        return {
            "input_features": torch.randn(1, 80, 300, dtype=torch.float32),
            "attention_mask": torch.ones(1, 300, dtype=torch.long),
        }

    def batch_decode(self, _sequences, skip_special_tokens=True):  # noqa: ARG002
        return ["hej varlden"]

    def get_prompt_ids(self, _prompt, return_tensors="pt"):  # noqa: ARG002
        return torch.tensor([10, 11, 12], dtype=torch.long)


class _FakeModel:
    def __init__(self, dtype: torch.dtype) -> None:
        self._param = torch.zeros(1, dtype=dtype)
        self.last_generate_kwargs = None

    def parameters(self):
        return iter([self._param])

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def make_transcribe_bundle(model_id: str, model_dtype: torch.dtype) -> LoadedModel:
    return LoadedModel(
        model_id=model_id,
        model=_FakeModel(model_dtype),
        processor=_FakeProcessor(),
        infer_lock=threading.Lock(),
    )


class WhisperModelManagerTests(unittest.TestCase):
    def test_reuses_loaded_model(self) -> None:
        manager = WhisperModelManager()
        with patch("app.model.Config.MAX_MODELS_IN_MEMORY", 2):
            with patch.object(manager, "_load_bundle", return_value=make_bundle("m1")) as load_bundle:
                first = manager._get_or_load_model("m1")
                second = manager._get_or_load_model("m1")

        self.assertIs(first, second)
        self.assertEqual(load_bundle.call_count, 1)
        self.assertEqual(manager.loaded_model_ids(), ["m1"])

    def test_evicts_lru_when_capacity_exceeded(self) -> None:
        manager = WhisperModelManager()
        bundles = {
            "m1": make_bundle("m1"),
            "m2": make_bundle("m2"),
        }

        with patch("app.model.Config.MAX_MODELS_IN_MEMORY", 1):
            with patch.object(manager, "_load_bundle", side_effect=lambda model_id: bundles[model_id]):
                with patch.object(manager, "_dispose_bundle") as dispose_bundle:
                    manager._get_or_load_model("m1")
                    manager._get_or_load_model("m2")

        self.assertEqual(manager.loaded_model_ids(), ["m2"])
        dispose_bundle.assert_called_once_with(bundles["m1"])

    def test_concurrent_first_load_uses_single_initializer(self) -> None:
        manager = WhisperModelManager()
        load_count = 0
        load_count_lock = threading.Lock()

        def slow_load(model_id: str) -> LoadedModel:
            nonlocal load_count
            with load_count_lock:
                load_count += 1
            time.sleep(0.05)
            return make_bundle(model_id)

        results: list[LoadedModel] = []
        errors: list[Exception] = []

        with patch("app.model.Config.MAX_MODELS_IN_MEMORY", 2):
            with patch.object(manager, "_load_bundle", side_effect=slow_load):
                def worker() -> None:
                    try:
                        results.append(manager._get_or_load_model("m1"))
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)

                threads = [threading.Thread(target=worker) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

        self.assertFalse(errors)
        self.assertEqual(load_count, 1)
        self.assertTrue(results)
        first = results[0]
        self.assertTrue(all(bundle is first for bundle in results))

    def test_transcribe_returns_result_with_text(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float16)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            with patch.object(manager, "_device", return_value=torch.device("cpu")):
                with patch.object(
                    manager,
                    "_load_audio",
                    return_value=(torch.randn(16000, dtype=torch.float32), 16000),
                ):
                    result = manager.transcribe("m1", "/tmp/audio.wav", "sv", None, None)

        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "hej varlden")

    def test_transcribe_casts_input_features_to_float16_when_model_is_float16(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float16)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            with patch.object(manager, "_device", return_value=torch.device("cpu")):
                with patch.object(
                    manager,
                    "_load_audio",
                    return_value=(torch.randn(16000, dtype=torch.float32), 16000),
                ):
                    manager.transcribe("m1", "/tmp/audio.wav", "sv", None, None)

        self.assertIsNotNone(bundle.model.last_generate_kwargs)
        self.assertEqual(bundle.model.last_generate_kwargs["input_features"].dtype, torch.float16)

    def test_transcribe_keeps_input_features_float32_when_model_is_float32(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float32)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            with patch.object(manager, "_device", return_value=torch.device("cpu")):
                with patch.object(
                    manager,
                    "_load_audio",
                    return_value=(torch.randn(16000, dtype=torch.float32), 16000),
                ):
                    manager.transcribe("m1", "/tmp/audio.wav", "sv", None, None)

        self.assertIsNotNone(bundle.model.last_generate_kwargs)
        self.assertEqual(bundle.model.last_generate_kwargs["input_features"].dtype, torch.float32)

    def test_transcribe_keeps_attention_mask_and_prompt_ids_integer_dtypes(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float16)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            with patch.object(manager, "_device", return_value=torch.device("cpu")):
                with patch.object(
                    manager,
                    "_load_audio",
                    return_value=(torch.randn(16000, dtype=torch.float32), 16000),
                ):
                    manager.transcribe("m1", "/tmp/audio.wav", "sv", "ledtext", None)

        self.assertIsNotNone(bundle.model.last_generate_kwargs)
        attention_mask = bundle.model.last_generate_kwargs["attention_mask"]
        prompt_ids = bundle.model.last_generate_kwargs["prompt_ids"]
        self.assertIn(attention_mask.dtype, (torch.long, torch.int64, torch.bool))
        self.assertEqual(prompt_ids.dtype, torch.long)

    def test_transcribe_with_translate_task(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float32)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            with patch.object(manager, "_device", return_value=torch.device("cpu")):
                with patch.object(
                    manager,
                    "_load_audio",
                    return_value=(torch.randn(16000, dtype=torch.float32), 16000),
                ):
                    manager.transcribe("m1", "/tmp/audio.wav", "sv", None, None, task="translate")

        self.assertEqual(bundle.model.last_generate_kwargs["task"], "translate")

    def test_unload_model(self) -> None:
        manager = WhisperModelManager()
        with patch("app.model.Config.MAX_MODELS_IN_MEMORY", 2):
            with patch.object(manager, "_load_bundle", return_value=make_bundle("m1")):
                manager._get_or_load_model("m1")

        with patch.object(manager, "_dispose_bundle"):
            self.assertTrue(manager.unload_model("m1"))
            self.assertFalse(manager.unload_model("m1"))

        self.assertEqual(manager.loaded_model_ids(), [])

    def test_metrics_tracks_transcriptions(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float32)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            with patch.object(manager, "_device", return_value=torch.device("cpu")):
                with patch.object(
                    manager,
                    "_load_audio",
                    return_value=(torch.randn(16000, dtype=torch.float32), 16000),
                ):
                    manager.transcribe("m1", "/tmp/audio.wav", "sv", None, None)
                    manager.transcribe("m1", "/tmp/audio.wav", "sv", None, None, task="translate")

        m = manager.metrics()
        self.assertEqual(m.transcriptions_total, 1)
        self.assertEqual(m.translations_total, 1)
        self.assertGreater(m.inference_seconds_total, 0)
        self.assertEqual(m.active_inferences, 0)

    def test_supported_languages(self) -> None:
        manager = WhisperModelManager()
        bundle = make_transcribe_bundle("m1", torch.float32)

        with patch.object(manager, "_get_or_load_model", return_value=bundle):
            langs = manager.supported_languages("m1")

        self.assertEqual(langs, ["en", "sv"])

    def test_parse_segments(self) -> None:
        manager = WhisperModelManager()
        raw = "<|0.00|> Hello world.<|3.20|><|3.20|> How are you?<|5.40|>"
        segments = manager._parse_segments(raw)

        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 3.2)
        self.assertEqual(segments[0].text, "Hello world.")
        self.assertAlmostEqual(segments[1].start, 3.2)
        self.assertAlmostEqual(segments[1].end, 5.4)
        self.assertEqual(segments[1].text, "How are you?")


if __name__ == "__main__":
    unittest.main()
