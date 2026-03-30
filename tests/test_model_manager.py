from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from app.model import LoadedModel, WhisperModelManager


def make_bundle(model_id: str) -> LoadedModel:
    model = MagicMock()
    processor = MagicMock()
    return LoadedModel(
        model_id=model_id,
        model=model,
        processor=processor,
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


if __name__ == "__main__":
    unittest.main()
