# KB Whisper Large Shared for Olares

This package deploys the published image:

- `ghcr.io/progress44/rpi-system-kb-whisper-large:latest`

This package models KB Whisper Large as a shared Olares app:

- `kbwhisperlargeserver` deploys one administrator-owned shared backend.
- `kbwhisperlarge` deploys a lightweight per-user proxy so each installed user
  gets a normal user-space API entrance.

## Olares endpoints

- User endpoint: `https://kbwhisperlarge.<OlaresID>.olares.com`
  - Use this from browsers, user automations, and tools that need a normal
    Olares user route.
- Shared endpoint: `http://kbwhisperlarge.shared.olares.com`
  - Use this hidden internal endpoint for cluster app-to-app API calls.

The user endpoint proxies to the shared backend at
`sharedentrances-api.kbwhisperlargeserver-shared:80`. The shared endpoint is
declared through `sharedEntrances` and is not shown as a user-facing icon.

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /transcribe`

## Request example

```bash
curl -X POST https://kbwhisperlarge.<OlaresID>.olares.com/v1/audio/transcriptions \
  -F "model=openai/whisper-large-v3" \
  -F "file=@./sample.wav" \
  -F "language=en" \
  -F "response_format=json"
```

## Notes

- The first request for each model may be slower while that model downloads and caches.
- Hugging Face cache persists under the administrator shared service app data.
- If `model` is omitted, `WHISPER_MODEL_ID` is used as the default model.
- Models are loaded lazily and kept in an in-memory LRU cache sized by
  `WHISPER_MAX_MODELS_IN_MEMORY`.
- The administrator install can use Olares env variables
  `OLARES_USER_HUGGINGFACE_TOKEN` and `OLARES_USER_HUGGINGFACE_SERVICE` if
  needed for the shared backend environment.
