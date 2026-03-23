#!/usr/bin/env python3
"""KB Whisper Large API entrypoint."""

from __future__ import annotations

import uvicorn

from app.config import Config


def main() -> None:
    Config.validate()

    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
