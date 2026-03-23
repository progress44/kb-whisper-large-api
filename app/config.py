"""Configuration for KB Whisper Large API."""

from __future__ import annotations

import os


class Config:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    MODEL_ID = os.getenv("WHISPER_MODEL_ID", "KBLab/kb-whisper-large")
    DEVICE = os.getenv("WHISPER_DEVICE", "auto")
    DEFAULT_LANGUAGE = os.getenv("WHISPER_DEFAULT_LANGUAGE", "sv")

    MAX_UPLOAD_SIZE_MB = int(os.getenv("WHISPER_MAX_UPLOAD_SIZE_MB", "200"))
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    ENABLE_DOCS = os.getenv("WHISPER_ENABLE_DOCS", "true").lower() == "true"

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    @classmethod
    def validate(cls) -> None:
        if cls.PORT <= 0:
            raise ValueError("PORT must be positive")
        if cls.MAX_UPLOAD_SIZE_MB <= 0:
            raise ValueError("WHISPER_MAX_UPLOAD_SIZE_MB must be positive")
        if not cls.MODEL_ID.strip():
            raise ValueError("WHISPER_MODEL_ID cannot be empty")

