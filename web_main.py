"""
web_main.py -- NeuroScalpel Web Interface Entry Point
======================================================
Run this instead of main.py to launch the web UI (no PyQt required).

Usage:
    python web_main.py

Then open http://localhost:5000 in your browser.
"""

import io
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout so print() never crashes on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Centralized logging → logs/app/ and logs/web/ ────────────────────────
from core.log_config import setup_logging
setup_logging(mode="web")


from web.app import create_app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 56)
    print("  NeuroScalpel  -  Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
