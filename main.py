"""
main.py — NeuroScalpel Entry Point
====================================
Launches the application with a premium cyberpunk aesthetic:
  - Startup splash screen with boot sequence animation
  - Full QSS stylesheet with glassmorphism + neon glow effects
  - Animated typing effect in splash
  - Window icon
"""

import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QSplashScreen, QLabel, QVBoxLayout, QWidget
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont, QLinearGradient

from ui.main_window import MainWindow

# ─────────────────────────────────────────────────────────────────────────────
#  Stylesheet — Cyberpunk Neural-Surgery HUD
#  Palette:
#    bg deep  #070810   mid  #0d1018   panel  rgba(14,18,28, 0.88)
#    cyan     #00f3ff   purple #bc13fe  red   #ff2244
#    text     #c8d6e5   dim   #6a7b8a
# ─────────────────────────────────────────────────────────────────────────────
CYBERPUNK_QSS = """
/* ── BASE ────────────────────────────────────────────────────────────────── */
QMainWindow, #CentralWidget {
    background-color: #070810;
}
QWidget {
    color: #c8d6e5;
    font-family: "Segoe UI", "Verdana", sans-serif;
    font-size: 13px;
    letter-spacing: 0.4px;
}

/* ── SIDE PANELS (glassmorphism) ─────────────────────────────────────────── */
#FeaturePanel, #DashboardPanel {
    background-color: rgba(14, 18, 30, 0.92);
    border: 1px solid rgba(0, 243, 255, 0.12);
    border-radius: 14px;
}

/* ── GROUP BOXES ──────────────────────────────────────────────────────────── */
QGroupBox#HUDGroupBox {
    border: 1px solid rgba(188, 19, 254, 0.18);
    border-radius: 9px;
    margin-top: 24px;
    background-color: rgba(10, 12, 22, 0.55);
}
QGroupBox#HUDGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 12px;
    color: #00f3ff;
    font-weight: 900;
    font-size: 11px;
    letter-spacing: 2px;
    background-color: #070810;
    border: 1px solid rgba(0, 243, 255, 0.35);
    border-radius: 4px;
}

/* ── LABELS ───────────────────────────────────────────────────────────────── */
QLabel {
    color: #6a7b8a;
    font-weight: 600;
}
QLabel#statusLabel {
    color: #8a9ab0;
    font-size: 12px;
}

/* ── TEXT / SPIN INPUTS ───────────────────────────────────────────────────── */
QLineEdit, QDoubleSpinBox {
    background-color: rgba(0, 0, 0, 0.55);
    color: #00f3ff;
    border: 1px solid rgba(0, 243, 255, 0.25);
    border-radius: 4px;
    padding: 8px 10px;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 13px;
}
QLineEdit:focus, QDoubleSpinBox:focus {
    border: 1px solid #00f3ff;
    background-color: rgba(0, 243, 255, 0.04);
    outline: none;
}
QLineEdit#hfTerminalInput {
    border: none;
    border-bottom: 2px solid rgba(188, 19, 254, 0.45);
    border-radius: 0px;
    background-color: transparent;
    color: #bc13fe;
    font-size: 13px;
}
QLineEdit#hfTerminalInput:focus {
    border-bottom: 2px solid #bc13fe;
}

/* ── PLAIN TEXT EDIT (order input) ──────────────────────────────────────── */
QPlainTextEdit#orderInput {
    background-color: rgba(0, 0, 0, 0.65);
    color: #00f3ff;
    border: 1px solid rgba(0, 243, 255, 0.18);
    border-radius: 8px;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 12px;
    padding: 12px;
    line-height: 1.5;
}
QPlainTextEdit#orderInput:focus {
    border: 1px solid rgba(0, 243, 255, 0.6);
    background-color: rgba(0, 243, 255, 0.03);
}

/* ── PROGRESS BAR ─────────────────────────────────────────────────────────── */
QProgressBar#energyBar {
    border: 1px solid rgba(0, 243, 255, 0.3);
    border-radius: 4px;
    background-color: rgba(0,0,0,0.5);
    height: 10px;
    text-align: center;
}
QProgressBar#energyBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00c8d4, stop:0.5 #00f3ff, stop:1 #bc13fe);
    border-radius: 3px;
    margin: 1px;
}

/* ── CONSOLE / READOUT ────────────────────────────────────────────────────── */
QTextEdit#consoleReadout {
    background-color: rgba(0, 0, 0, 0.75);
    color: #00f3ff;
    border: 1px solid rgba(0, 243, 255, 0.15);
    border-radius: 8px;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 11.5px;
    padding: 10px;
    line-height: 1.4;
    selection-background-color: rgba(0, 243, 255, 0.2);
}

/* ── TERMINAL WINDOWS ────────────────────────────────────────────────────── */
QTextEdit {
    background-color: rgba(0, 0, 0, 0.45);
    color: #a8b8cc;
    border: 1px solid rgba(0, 243, 255, 0.12);
    border-radius: 6px;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 12px;
    padding: 10px;
    selection-background-color: rgba(188, 19, 254, 0.25);
}

/* ── BUTTONS — BASE ───────────────────────────────────────────────────────── */
QPushButton {
    font-weight: 700;
    letter-spacing: 1.2px;
    font-size: 12.5px;
    border-radius: 7px;
}

/* Cyan neon */
QPushButton#neonBtnCyan {
    background-color: rgba(0, 243, 255, 0.04);
    color: #00f3ff;
    border: 1px solid rgba(0, 243, 255, 0.6);
    padding: 10px 14px;
}
QPushButton#neonBtnCyan:hover {
    background-color: rgba(0, 243, 255, 0.15);
    border-color: #00f3ff;
    color: #ffffff;
}
QPushButton#neonBtnCyan:pressed {
    background-color: rgba(0, 243, 255, 0.4);
    color: #000;
}

/* Purple neon */
QPushButton#neonBtnPurple {
    background-color: rgba(188, 19, 254, 0.04);
    color: #bc13fe;
    border: 1px solid rgba(188, 19, 254, 0.55);
    padding: 10px 14px;
}
QPushButton#neonBtnPurple:hover {
    background-color: rgba(188, 19, 254, 0.16);
    border-color: #bc13fe;
    color: #ffffff;
}
QPushButton#neonBtnPurple:pressed {
    background-color: rgba(188, 19, 254, 0.45);
    color: #000;
}
QPushButton#neonBtnPurple:disabled {
    border: 1px solid rgba(188, 19, 254, 0.15);
    color: rgba(188, 19, 254, 0.3);
    background-color: transparent;
}

/* START WORD — spectacular cyan */
QPushButton#startWordBtn {
    background-color: rgba(0, 243, 255, 0.07);
    color: #00f3ff;
    border: 1.5px solid rgba(0, 243, 255, 0.75);
    border-radius: 9px;
    padding: 14px;
    font-size: 14px;
    font-weight: 900;
    letter-spacing: 2.5px;
}
QPushButton#startWordBtn:hover {
    background-color: rgba(0, 243, 255, 0.22);
    border-color: #ffffff;
    color: #ffffff;
}
QPushButton#startWordBtn:pressed {
    background-color: rgba(0, 243, 255, 0.6);
    color: #000;
}

/* UPDATE WEIGHTS — spectacular purple */
QPushButton#updateBtn {
    background-color: rgba(188, 19, 254, 0.06);
    color: #bc13fe;
    border: 1.5px solid rgba(188, 19, 254, 0.75);
    border-radius: 9px;
    padding: 16px;
    font-size: 14px;
    font-weight: 900;
    letter-spacing: 2px;
}
QPushButton#updateBtn:hover {
    background-color: rgba(188, 19, 254, 0.22);
    border-color: #ffffff;
    color: #ffffff;
}
QPushButton#updateBtn:pressed {
    background-color: rgba(188, 19, 254, 0.55);
    color: #000;
}
QPushButton#updateBtn:disabled {
    border: 1px solid rgba(188, 19, 254, 0.12);
    color: rgba(188, 19, 254, 0.25);
    background-color: transparent;
}

/* ── SPLITTER ─────────────────────────────────────────────────────────────── */
QSplitter::handle {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0   transparent,
        stop:0.3 rgba(0,243,255,0.18),
        stop:0.7 rgba(188,19,254,0.18),
        stop:1   transparent);
    width: 3px;
}

/* ── SCROLLBAR ────────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    border: none;
    background: rgba(0,0,0,0.2);
    width: 7px;
    margin: 0;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: rgba(0, 243, 255, 0.25);
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(0, 243, 255, 0.55);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    border: none;
    background: rgba(0,0,0,0.2);
    height: 6px;
    border-radius: 3px;
}
QScrollBar::handle:horizontal {
    background: rgba(0, 243, 255, 0.25);
    border-radius: 3px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── TOOLTIP ──────────────────────────────────────────────────────────────── */
QToolTip {
    background-color: #0d1018;
    color: #00f3ff;
    border: 1px solid rgba(0, 243, 255, 0.4);
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ── FORM LAYOUT labels ───────────────────────────────────────────────────── */
QFormLayout QLabel {
    color: #6a7b8a;
    font-size: 11px;
    letter-spacing: 1px;
    font-weight: 700;
}

/* ── SPIN BOX SPECIAL ────────────────────────────────────────────────────── */
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: rgba(0,243,255,0.08);
    border: none;
    width: 18px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background: rgba(0,243,255,0.2);
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal programmatic splash screen
# ─────────────────────────────────────────────────────────────────────────────

def _make_splash_pixmap(w: int = 560, h: int = 300) -> QPixmap:
    """Renders a sleek boot-sequence splash image programmatically."""
    px = QPixmap(w, h)
    px.fill(QColor("#070810"))

    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Gradient background band
    grad = QLinearGradient(0, 0, w, h)
    grad.setColorAt(0.0, QColor(0, 14, 28))
    grad.setColorAt(0.5, QColor(10, 0, 22))
    grad.setColorAt(1.0, QColor(0, 10, 18))
    painter.fillRect(0, 0, w, h, grad)

    # Horizontal scan lines (subtle)
    painter.setPen(QColor(0, 243, 255, 12))
    for y in range(0, h, 4):
        painter.drawLine(0, y, w, y)

    # Top neon border
    painter.setPen(QColor(0, 243, 255, 180))
    painter.drawLine(0, 0, w, 0)
    painter.setPen(QColor(0, 243, 255, 60))
    painter.drawLine(0, 1, w, 1)

    # Bottom neon border
    painter.setPen(QColor(188, 19, 254, 180))
    painter.drawLine(0, h - 1, w, h - 1)
    painter.setPen(QColor(188, 19, 254, 60))
    painter.drawLine(0, h - 2, w, h - 2)

    # Title
    f_title = QFont("Segoe UI", 26, QFont.Weight.Black)
    painter.setFont(f_title)
    painter.setPen(QColor(0, 243, 255, 240))
    painter.drawText(0, 60, w, 60, Qt.AlignmentFlag.AlignHCenter, "NeuroScalpel")

    # Subtitle
    f_sub = QFont("Segoe UI", 11, QFont.Weight.Normal)
    painter.setFont(f_sub)
    painter.setPen(QColor(188, 19, 254, 200))
    painter.drawText(0, 105, w, 30, Qt.AlignmentFlag.AlignHCenter,
                     "LLM Neural Surgery  ·  ROME + LyapLock  ·  v2.0")

    # Divider
    painter.setPen(QColor(0, 243, 255, 35))
    painter.drawLine(60, 148, w - 60, 148)

    # Boot lines
    f_mono = QFont("Cascadia Code", 9)
    if not f_mono.exactMatch():
        f_mono = QFont("Consolas", 9)
    painter.setFont(f_mono)

    boot_lines = [
        ("[  OK  ]  Session Manager              SQLite WAL mode",    "#00f3ff"),
        ("[  OK  ]  Edit Engine                  ROME + LyapLock",    "#00f3ff"),
        ("[  OK  ]  3D Neural Visualizer         Layer-slab mode",    "#00f3ff"),
        ("[  OK  ]  Multi-Task Queue             Sequential mode",     "#00f3ff"),
        ("[  OK  ]  Generated Log                daily + latest",     "#00ff88"),
        ("[  GO  ]  Initialising UI...                              ", "#bc13fe"),
    ]
    y_off = 168
    for line, color in boot_lines:
        painter.setPen(QColor(color))
        painter.drawText(50, y_off, w - 100, 20,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         line)
        y_off += 19

    painter.end()
    return px


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # High-DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroScalpel")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("NeuroScalpel Research")

    # ── Splash ──────────────────────────────────────────────────────────────
    splash_px = _make_splash_pixmap()
    splash = QSplashScreen(splash_px, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    # Animate "Initialising..." for ~1.2s
    for i in range(24):
        time.sleep(0.05)
        app.processEvents()

    # ── Apply QSS stylesheet ─────────────────────────────────────────────────
    app.setStyleSheet(CYBERPUNK_QSS)

    # ── Launch main window ───────────────────────────────────────────────────
    window = MainWindow()
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
