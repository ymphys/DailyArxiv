from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "dailyarxiv.log"


def configure_logging(log_file: Optional[Path | str] = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    target = Path(log_file) if log_file else DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(target, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        handlers=handlers,
    )
