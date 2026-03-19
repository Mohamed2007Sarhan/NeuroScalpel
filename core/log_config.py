"""
log_config.py
=============
Centralized logging configuration for NeuroScalpel.

Log folder layout
-----------------
logs/
  pipeline/     <- GeneratedLog human-readable pipeline run files
                   neuroscalpel_YYYYMMDD.log  (daily rotating)
                   latest.log                 (always the last run)
  app/          <- Python logging module captures (all loggers)
                   app_YYYYMMDD.log           (daily rotating)
                   app_latest.log             (always the last run)
  web/          <- Flask / werkzeug access + error log
                   web_YYYYMMDD.log

Usage
-----
Call setup_logging() once, as early as possible in main.py / web_main.py:

    from core.log_config import setup_logging, LOGS_ROOT
    setup_logging(mode="desktop")    # or mode="web"
"""

import logging
import logging.handlers
import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGS_ROOT     = Path(__file__).resolve().parent.parent / "logs"
PIPELINE_DIR  = LOGS_ROOT / "pipeline"
APP_DIR       = LOGS_ROOT / "app"
WEB_DIR       = LOGS_ROOT / "web"

# Ensure all sub-folders exist the moment this module is imported
for _d in (PIPELINE_DIR, APP_DIR, WEB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _today() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


def setup_logging(mode: str = "desktop", level: int = logging.INFO) -> None:
    """
    Configure the root Python logger to write to:
        logs/app/app_YYYYMMDD.log   (daily rolling)
        logs/app/app_latest.log     (overwritten each run)

    For the web server, werkzeug logs also go to:
        logs/web/web_YYYYMMDD.log

    Parameters
    ----------
    mode  : "desktop" | "web"
    level : logging level (default INFO)
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(name)-30s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # ── Console handler ───────────────────────────────────────────────────────
    if not any(isinstance(h, logging.StreamHandler) and
               not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # ── App daily file ────────────────────────────────────────────────────────
    app_daily  = APP_DIR / f"app_{_today()}.log"
    app_latest = APP_DIR / "app_latest.log"

    fh_daily = logging.FileHandler(app_daily, encoding="utf-8")
    fh_daily.setLevel(level)
    fh_daily.setFormatter(fmt)
    root.addHandler(fh_daily)

    fh_latest = logging.FileHandler(app_latest, mode="w", encoding="utf-8")
    fh_latest.setLevel(level)
    fh_latest.setFormatter(fmt)
    root.addHandler(fh_latest)

    # ── Web server logs (Flask / werkzeug) ────────────────────────────────────
    if mode == "web":
        web_daily = WEB_DIR / f"web_{_today()}.log"
        wfh = logging.FileHandler(web_daily, encoding="utf-8")
        wfh.setLevel(level)
        wfh.setFormatter(fmt)
        for name in ("werkzeug", "flask.app"):
            logging.getLogger(name).addHandler(wfh)

    logging.getLogger("NeuroScalpel").info(
        f"Logging initialised | mode={mode} | "
        f"app log -> logs/app/app_{_today()}.log"
    )
