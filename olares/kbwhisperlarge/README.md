# KB Whisper Large for Olares

This package deploys the published image:

- `ghcr.io/progress44/rpi-system-kb-whisper-large:latest`

The app exposes OpenAI-compatible transcription at:

- `http://kbwhisperlarge-svc:8000`

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /transcribe`

## Request example

```bash
curl -X POST http://kbwhisperlarge-svc:8000/v1/audio/transcriptions \
  -F "model=openai/whisper-large-v3" \
  -F "file=@./sample.wav" \
  -F "language=en" \
  -F "response_format=json"
```

## Notes

- The first request for each model may be slower while that model downloads and caches.
- Hugging Face cache persists under `userspace.appData`.
- If `model` is omitted, `WHISPER_MODEL_ID` is used as the default model.
- Models are loaded lazily and kept in an in-memory LRU cache sized by
  `WHISPER_MAX_MODELS_IN_MEMORY`.
- Use Olares env variables `OLARES_USER_HUGGINGFACE_TOKEN` and
  `OLARES_USER_HUGGINGFACE_SERVICE` if needed for your environment.
