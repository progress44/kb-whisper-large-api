"""FastAPI app for KB Whisper Large transcription."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.config import Config
from app.jobs import JobStore
from app.model import ModelLoadError, TranscriptionResult, whisper_model

job_store = JobStore(whisper_model)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    job_store.start()
    yield
    await job_store.stop()


def _docs_url(path: str) -> str | None:
    return path if Config.ENABLE_DOCS else None


app = FastAPI(
    title="KB Whisper Large API",
    description="OpenAI-compatible transcription API powered by Hugging Face Whisper models",
    version="2.0.0",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


async def _download_url(url: str) -> str:
    def _fetch() -> str:
        parsed = urllib.parse.urlparse(url)
        suffix = os.path.splitext(parsed.path)[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(prefix="kb-whisper-url-", suffix=suffix, delete=False)
        try:
            urllib.request.urlretrieve(url, tmp.name)  # noqa: S310
            return tmp.name
        except Exception:
            os.unlink(tmp.name)
            raise

    return await asyncio.to_thread(_fetch)


async def _transcribe_upload(
    file: UploadFile,
    model_id: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    task: str = "transcribe",
) -> TranscriptionResult:
    tmp_path = await _save_upload(file)
    try:
        try:
            result = await asyncio.to_thread(
                whisper_model.transcribe,
                model_id,
                tmp_path,
                _final_language(language),
                prompt,
                temperature,
                task,
            )
            return result
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


def _format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _to_srt(result: TranscriptionResult) -> str:
    lines = []
    for idx, seg in enumerate(result.segments, 1):
        lines.append(str(idx))
        lines.append(f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _to_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _format_response(
    result: TranscriptionResult,
    response_format: str,
    model_id: str,
    language: str | None,
):
    if response_format == "text":
        return PlainTextResponse(result.text)

    if response_format == "srt":
        return PlainTextResponse(_to_srt(result), media_type="text/plain")

    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(result), media_type="text/vtt")

    if response_format in {"json", "verbose_json"}:
        payload: dict[str, object] = {"text": result.text}
        if response_format == "verbose_json":
            payload["model"] = model_id
            payload["language"] = language or Config.DEFAULT_LANGUAGE
            payload["segments"] = [
                {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
            ]
        return JSONResponse(content=payload)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": {
                "type": "invalid_request_error",
                "message": "response_format must be one of: json, verbose_json, text, srt, vtt",
            }
        },
    )


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    loaded_models = whisper_model.loaded_model_ids()
    model_ids = loaded_models if loaded_models else [Config.MODEL_ID]
    return {"data": [{"id": model_id, "object": "model"} for model_id in model_ids]}


@app.post("/v1/models/load")
async def load_model(model: str = Form(...)):
    model_id = _final_model(model)
    try:
        await asyncio.to_thread(whisper_model.load_model, model_id)
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"type": "invalid_model", "message": str(exc)}},
        ) from exc
    return {"status": "loaded", "model": model_id}


@app.delete("/v1/models/{model_id:path}")
async def unload_model(model_id: str):
    removed = await asyncio.to_thread(whisper_model.unload_model, model_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"type": "not_found", "message": f"Model '{model_id}' is not loaded"}},
        )
    return {"status": "unloaded", "model": model_id}


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------


@app.get("/v1/audio/languages")
async def languages(model: str | None = None):
    model_id = _final_model(model)
    langs = await asyncio.to_thread(whisper_model.supported_languages, model_id)
    return {"model": model_id, "languages": langs}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


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
    result = await _transcribe_upload(file, model_id, language, prompt, temperature)
    return _format_response(result, response_format, model_id, language)


@app.post("/transcribe")
async def transcribe_alias(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
):
    result = await _transcribe_upload(file, Config.MODEL_ID, language, prompt, None)
    return {"text": result.text}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: str = Form(default=Config.MODEL_ID),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
):
    model_id = _final_model(model)
    result = await _transcribe_upload(file, model_id, None, prompt, temperature, task="translate")
    return _format_response(result, response_format, model_id, None)


# ---------------------------------------------------------------------------
# URL-based transcription
# ---------------------------------------------------------------------------


@app.post("/v1/audio/transcriptions/url")
async def transcribe_url(
    url: str = Form(...),
    model: str = Form(default=Config.MODEL_ID),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
    task: str = Form(default="transcribe"),
):
    model_id = _final_model(model)
    tmp_path = await _download_url(url)
    try:
        try:
            result = await asyncio.to_thread(
                whisper_model.transcribe,
                model_id,
                tmp_path,
                _final_language(language),
                prompt,
                temperature,
                task,
            )
        except ModelLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"type": "invalid_model", "message": str(exc)}},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": {"type": "transcription_unavailable", "message": str(exc)}},
            ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return _format_response(result, response_format, model_id, language)


# ---------------------------------------------------------------------------
# Streaming transcription (SSE)
# ---------------------------------------------------------------------------


@app.post("/v1/audio/transcriptions/stream")
async def transcription_stream(
    file: UploadFile = File(...),
    model: str = Form(default=Config.MODEL_ID),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    task: str = Form(default="transcribe"),
):
    model_id = _final_model(model)
    tmp_path = await _save_upload(file)

    async def _generate():
        q: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                for chunk_result in whisper_model.transcribe_chunks(
                    model_id,
                    tmp_path,
                    _final_language(language),
                    prompt,
                    temperature,
                    task,
                ):
                    payload = {
                        "event": "segment",
                        "text": chunk_result.text,
                        "segments": [
                            {"start": s.start, "end": s.end, "text": s.text}
                            for s in chunk_result.segments
                        ],
                    }
                    asyncio.run_coroutine_threadsafe(q.put(("segment", payload)), loop).result()
                asyncio.run_coroutine_threadsafe(q.put(("done", None)), loop).result()
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(q.put(("error", str(exc))), loop).result()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        loop.run_in_executor(None, _worker)

        while True:
            kind, value = await q.get()
            if kind == "segment":
                yield f"data: {json.dumps(value)}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'event': 'error', 'error': value})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
                break

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Batch transcription
# ---------------------------------------------------------------------------


@app.post("/v1/audio/transcriptions/batch")
async def batch_submit(
    file: UploadFile = File(...),
    model: str = Form(default=Config.MODEL_ID),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    task: str = Form(default="transcribe"),
):
    model_id = _final_model(model)
    tmp_path = await _save_upload(file)
    job = await job_store.submit(
        tmp_path, model_id, _final_language(language), prompt, temperature, task
    )
    return JSONResponse(content=job_store.to_dict(job), status_code=status.HTTP_202_ACCEPTED)


@app.get("/v1/audio/transcriptions/batch/{job_id}")
async def batch_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"type": "not_found", "message": f"Job '{job_id}' not found"}},
        )
    return job_store.to_dict(job)


# ---------------------------------------------------------------------------
# Metrics (Prometheus text format)
# ---------------------------------------------------------------------------


@app.get("/metrics")
def prometheus_metrics():
    m = whisper_model.metrics()
    s = whisper_model.status()
    lines = [
        "# HELP whisper_transcriptions_total Total transcription requests completed.",
        "# TYPE whisper_transcriptions_total counter",
        f"whisper_transcriptions_total {m.transcriptions_total}",
        "# HELP whisper_translations_total Total translation requests completed.",
        "# TYPE whisper_translations_total counter",
        f"whisper_translations_total {m.translations_total}",
        "# HELP whisper_inference_seconds_total Cumulative inference time in seconds.",
        "# TYPE whisper_inference_seconds_total counter",
        f"whisper_inference_seconds_total {m.inference_seconds_total:.3f}",
        "# HELP whisper_errors_total Total inference errors.",
        "# TYPE whisper_errors_total counter",
        f"whisper_errors_total {m.errors_total}",
        "# HELP whisper_active_inferences Currently running inferences.",
        "# TYPE whisper_active_inferences gauge",
        f"whisper_active_inferences {m.active_inferences}",
        "# HELP whisper_loaded_models Number of models currently loaded.",
        "# TYPE whisper_loaded_models gauge",
        f"whisper_loaded_models {s.loaded_model_count}",
        "",
    ]
    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")
