"""FastAPI app for KB Whisper Large transcription."""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import Config
from app.model import whisper_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_task = asyncio.create_task(asyncio.to_thread(whisper_model.initialize))
    try:
        yield
    finally:
        if not init_task.done():
            init_task.cancel()
            with suppress(asyncio.CancelledError):
                await init_task


def _docs_url(path: str) -> str | None:
    return path if Config.ENABLE_DOCS else None


app = FastAPI(
    title="KB Whisper Large API",
    description="OpenAI-compatible transcription API powered by KBLab/kb-whisper-large",
    version="1.0.0",
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
    }


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": Config.MODEL_ID, "object": "model"}]}


def _final_language(language: str | None) -> str | None:
    if language:
        return language.strip() or None
    return None


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
    language: str | None,
    prompt: str | None,
    temperature: float | None,
) -> str:
    tmp_path = await _save_upload(file)
    try:
        text = await asyncio.to_thread(
            whisper_model.transcribe,
            tmp_path,
            _final_language(language),
            prompt,
            temperature,
        )
        return text
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
    if model != Config.MODEL_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "type": "invalid_model",
                    "message": f"Only model '{Config.MODEL_ID}' is available in this deployment.",
                }
            },
        )

    text = await _transcribe_upload(file, language, prompt, temperature)

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format in {"json", "verbose_json"}:
        payload: dict[str, object] = {"text": text}
        if response_format == "verbose_json":
            payload["model"] = Config.MODEL_ID
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
    text = await _transcribe_upload(file, language, prompt, None)
    return {"text": text}
