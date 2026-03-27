"""
__init__.py — NeuroScalpel core package.

Configures application-wide logging on import so all submodules
(session_manager, edit_engine, model_backend, etc.) emit consistent,
readable log output with timestamps.
"""

import logging
import sys

def setup_logging(level=logging.INFO):
    """Call once at startup to configure root logger."""
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    root = logging.getLogger("NeuroScalpel")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


# Auto-configure when the package is loaded
setup_logging()
