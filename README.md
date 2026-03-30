# KB Whisper Large API

FastAPI wrapper that exposes OpenAI-style transcription endpoints backed by
Whisper-compatible models from Hugging Face.

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions` (OpenAI-compatible multipart form)
- `POST /transcribe` (simple alias)

## OpenAI-style request example

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=KBLab/kb-whisper-large" \
  -F "file=@./sample.wav" \
  -F "language=sv" \
  -F "response_format=json"
```

`model` is selected per request. On first use, the model is downloaded from
Hugging Face and cached under `HF_HOME`.

## Environment Variables

- `WHISPER_MODEL_ID` (default: `KBLab/kb-whisper-large`)
- `WHISPER_MAX_MODELS_IN_MEMORY` (default: `2`)
- `WHISPER_DEVICE` (`auto|cpu|cuda|mps`, default: `cuda` in container)
- `WHISPER_DEFAULT_LANGUAGE` (default: `sv`)
- `WHISPER_MAX_UPLOAD_SIZE_MB` (default: `200`)
- `WHISPER_ENABLE_DOCS` (`true|false`, default: `true`)
- `HF_TOKEN` and `HF_ENDPOINT` are supported through Hugging Face libraries.

## Runtime model behavior

- If `model` is omitted, the API uses `WHISPER_MODEL_ID`.
- Models are loaded lazily on first request.
- The API keeps up to `WHISPER_MAX_MODELS_IN_MEMORY` active models in an LRU cache.
- If a model cannot be loaded, the API returns `400 invalid_model`.
