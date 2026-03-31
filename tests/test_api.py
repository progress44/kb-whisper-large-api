from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.config import Config
from app.main import app
from app.model import ModelStatus, TranscriptionResult, TranscriptionSegment


def _mock_result(text: str = "ok") -> TranscriptionResult:
    return TranscriptionResult(text=text, segments=[])


def _mock_result_with_segments() -> TranscriptionResult:
    return TranscriptionResult(
        text="hello world",
        segments=[
            TranscriptionSegment(start=0.0, end=2.5, text="hello"),
            TranscriptionSegment(start=2.5, end=5.0, text="world"),
        ],
    )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_accepts_different_models_sequentially(self) -> None:
        with patch(
            "app.main._transcribe_upload", new=AsyncMock(return_value=_mock_result())
        ) as transcribe_upload:
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
        with patch(
            "app.main._transcribe_upload", new=AsyncMock(return_value=_mock_result())
        ) as transcribe_upload:
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
        status_obj = ModelStatus(
            initialized=True,
            initializing=False,
            model_id=Config.MODEL_ID,
            device="cuda",
            error=None,
            loaded_model_count=2,
            max_models_in_memory=3,
            loaded_models=["m1", "m2"],
        )
        with patch("app.main.whisper_model.status", return_value=status_obj):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["loaded_model_count"], 2)
        self.assertEqual(payload["max_models_in_memory"], 3)

    # --- Translation endpoint ---

    def test_translation_endpoint(self) -> None:
        with patch(
            "app.main._transcribe_upload", new=AsyncMock(return_value=_mock_result("translated"))
        ) as transcribe_upload:
            response = self.client.post(
                "/v1/audio/translations",
                data={"response_format": "json"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "translated")
        call_kwargs = transcribe_upload.await_args
        self.assertEqual(call_kwargs.kwargs.get("task"), "translate")

    # --- SRT/VTT response formats ---

    def test_srt_response_format(self) -> None:
        with patch(
            "app.main._transcribe_upload",
            new=AsyncMock(return_value=_mock_result_with_segments()),
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"response_format": "srt"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("00:00:00,000 --> 00:00:02,500", body)
        self.assertIn("hello", body)

    def test_vtt_response_format(self) -> None:
        with patch(
            "app.main._transcribe_upload",
            new=AsyncMock(return_value=_mock_result_with_segments()),
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"response_format": "vtt"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("WEBVTT", body)
        self.assertIn("00:00:00.000 --> 00:00:02.500", body)

    def test_verbose_json_includes_segments(self) -> None:
        with patch(
            "app.main._transcribe_upload",
            new=AsyncMock(return_value=_mock_result_with_segments()),
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"response_format": "verbose_json"},
                files={"file": ("sample.wav", b"fake", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "hello world")
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(payload["segments"][0]["text"], "hello")

    # --- Metrics endpoint ---

    def test_metrics_endpoint(self) -> None:
        with patch("app.main.whisper_model.metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(
                transcriptions_total=5,
                translations_total=2,
                inference_seconds_total=10.5,
                errors_total=1,
                active_inferences=0,
            )
            with patch("app.main.whisper_model.status") as mock_status:
                mock_status.return_value = MagicMock(loaded_model_count=1)
                response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("whisper_transcriptions_total 5", body)
        self.assertIn("whisper_translations_total 2", body)
        self.assertIn("whisper_errors_total 1", body)

    # --- Batch endpoint ---

    def test_batch_job_not_found(self) -> None:
        response = self.client.get("/v1/audio/transcriptions/batch/nonexistent")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
