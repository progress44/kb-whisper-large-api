"""FastAPI app for KB Whisper Large transcription."""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import Config
from app.model import ModelLoadError, whisper_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


def _docs_url(path: str) -> str | None:
    return path if Config.ENABLE_DOCS else None


app = FastAPI(
    title="KB Whisper Large API",
    description="OpenAI-compatible transcription API powered by Hugging Face Whisper models",
    version="1.6.0",
    docs_url=_docs_url("/docs"),
    redoc_url=_docs_url("/redoc"),
    lifespan=lifespan,
)

cors_origins = Config.CORS_ORIGINS
allowed_origins = ["*"] if cors_origins == "*" else [o.strip() for o in cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    s = whisper_model.status()
    return {
        "name": "KB Whisper Large API",
        "model": s.model_id,
        "device": s.device,
        "initialized": s.initialized,
        "loaded_model_count": s.loaded_model_count,
        "max_models_in_memory": s.max_models_in_memory,
        "docs": "/docs" if Config.ENABLE_DOCS else None,
    }


@app.get("/health")
def health() -> dict[str, object]:
    s = whisper_model.status()
    return {
        "status": "ok",
        "model": s.model_id,
        "initialized": s.initialized,
        "initializing": s.initializing,
        "device": s.device,
        "error": s.error,
        "loaded_model_count": s.loaded_model_count,
        "max_models_in_memory": s.max_models_in_memory,
    }


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    loaded_models = whisper_model.loaded_model_ids()
    model_ids = loaded_models if loaded_models else [Config.MODEL_ID]
    return {"data": [{"id": model_id, "object": "model"} for model_id in model_ids]}


def _final_language(language: str | None) -> str | None:
    if language:
        return language.strip() or None
    return None


def _final_model(model: str | None) -> str:
    candidate = model if model is not None else Config.MODEL_ID
    normalized = candidate.strip()
    if normalized:
        return normalized
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": {
                "type": "invalid_model",
                "message": "Field 'model' must be a non-empty Hugging Face model id.",
            }
        },
    )


async def _save_upload(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename or "audio.bin")[1]

    with tempfile.NamedTemporaryFile(prefix="kb-whisper-", suffix=suffix, delete=False) as tmp:
        total = 0
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > Config.MAX_UPLOAD_SIZE_BYTES:
                tmp_path = tmp.name
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error": {
                            "type": "request_too_large",
                            "message": (
                                f"Uploaded file exceeds max size of {Config.MAX_UPLOAD_SIZE_MB} MB"
                            ),
                        }
                    },
                )
            tmp.write(chunk)

        return tmp.name


async def _transcribe_upload(
    file: UploadFile,
    model_id: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
) -> str:
    tmp_path = await _save_upload(file)
    try:
        try:
            text = await asyncio.to_thread(
                whisper_model.transcribe,
                model_id,
                tmp_path,
                _final_language(language),
                prompt,
                temperature,
            )
            return text
        except ModelLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "type": "invalid_model",
                        "message": str(exc),
                    }
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "type": "transcription_unavailable",
                        "message": str(exc),
                    }
                },
            ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=Config.MODEL_ID),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
):
    model_id = _final_model(model)

    text = await _transcribe_upload(file, model_id, language, prompt, temperature)

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format in {"json", "verbose_json"}:
        payload: dict[str, object] = {"text": text}
        if response_format == "verbose_json":
            payload["model"] = model_id
            payload["language"] = language or Config.DEFAULT_LANGUAGE
        return JSONResponse(content=payload)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": {
                "type": "invalid_request_error",
                "message": "response_format must be one of: json, verbose_json, text",
            }
        },
    )


@app.post("/transcribe")
async def transcribe_alias(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
):
    text = await _transcribe_upload(file, Config.MODEL_ID, language, prompt, None)
    return {"text": text}
