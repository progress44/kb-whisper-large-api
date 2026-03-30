from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import Config
from app.main import app
from app.model import ModelStatus


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_accepts_different_models_sequentially(self) -> None:
        with patch("app.main._transcribe_upload", new=AsyncMock(return_value="ok")) as transcribe_upload:
            first = self.client.post(
                "/v1/audio/transcriptions",
                data={"model": "KBLab/kb-whisper-large", "response_format": "json"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )
            second = self.client.post(
                "/v1/audio/transcriptions",
                data={"model": "openai/whisper-large-v3", "response_format": "json"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(transcribe_upload.await_count, 2)
        self.assertEqual(transcribe_upload.await_args_list[0].args[1], "KBLab/kb-whisper-large")
        self.assertEqual(transcribe_upload.await_args_list[1].args[1], "openai/whisper-large-v3")

    def test_uses_default_model_when_model_omitted(self) -> None:
        with patch("app.main._transcribe_upload", new=AsyncMock(return_value="ok")) as transcribe_upload:
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"response_format": "json"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(transcribe_upload.await_count, 1)
        self.assertEqual(transcribe_upload.await_args_list[0].args[1], Config.MODEL_ID)

    def test_returns_invalid_model_for_blank_model(self) -> None:
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "   ", "response_format": "json"},
            files={"file": ("sample.wav", b"fake", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["detail"]["error"]["type"], "invalid_model")

    def test_models_endpoint_returns_loaded_models(self) -> None:
        with patch("app.main.whisper_model.loaded_model_ids", return_value=["model-a", "model-b"]):
            response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["id"] for item in response.json()["data"]],
            ["model-a", "model-b"],
        )

    def test_models_endpoint_falls_back_to_default(self) -> None:
        with patch("app.main.whisper_model.loaded_model_ids", return_value=[]):
            response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["data"]], [Config.MODEL_ID])

    def test_health_includes_cache_metadata(self) -> None:
        status = ModelStatus(
            initialized=True,
            initializing=False,
            model_id=Config.MODEL_ID,
            device="cuda",
            error=None,
            loaded_model_count=2,
            max_models_in_memory=3,
            loaded_models=["m1", "m2"],
        )
        with patch("app.main.whisper_model.status", return_value=status):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["loaded_model_count"], 2)
        self.assertEqual(payload["max_models_in_memory"], 3)


if __name__ == "__main__":
    unittest.main()
