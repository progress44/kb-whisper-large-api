# KB Whisper Large API

FastAPI wrapper that exposes OpenAI-style transcription endpoints backed by
[`KBLab/kb-whisper-large`](https://huggingface.co/KBLab/kb-whisper-large).

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

## Environment Variables

- `WHISPER_MODEL_ID` (default: `KBLab/kb-whisper-large`)
- `WHISPER_DEVICE` (`auto|cpu|cuda|mps`, default: `cuda` in container)
- `WHISPER_DEFAULT_LANGUAGE` (default: `sv`)
- `WHISPER_MAX_UPLOAD_SIZE_MB` (default: `200`)
- `WHISPER_ENABLE_DOCS` (`true|false`, default: `true`)
- `HF_TOKEN` and `HF_ENDPOINT` are supported through Hugging Face libraries.
