import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # ADVANCED SCIFI CYBERPUNK QSS
    # - Deep #090A0F base background
    # - Glassmorphism approximation for side panels
    # - Glowing cyan (#00f3ff) and purple (#bc13fe) accents
    # - High-tech typography
    
    cyberpunk_qss = """
    /* --- GENERAL BASE --- */
    QMainWindow, #CentralWidget {
        background-color: #090A0F;
    }
    QWidget {
        color: #d1d8e0;
        font-family: "Segoe UI", "Verdana", sans-serif;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    
    /* --- SIDE PANELS (HUD) --- */
    #FeaturePanel, #DashboardPanel {
        background-color: rgba(14, 18, 28, 0.85); /* Glassmorphism simulation */
        border: 1px solid rgba(0, 243, 255, 0.15);
        border-radius: 12px;
    }
    
    /* --- GROUP BOXES --- */
    QGroupBox#HUDGroupBox {
        border: 1px solid rgba(188, 19, 254, 0.2); /* Subtle purple edge */
        border-radius: 8px;
        margin-top: 25px;
        background-color: transparent;
    }
    QGroupBox#HUDGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 10px;
        color: #00f3ff; /* Neon cyan headers */
        font-weight: 900;
        font-size: 12px;
        letter-spacing: 1.5px;
        background-color: #090A0F;
        border: 1px solid rgba(0, 243, 255, 0.4);
        border-radius: 4px;
    }
    
    /* --- LABELS --- */
    QLabel {
        color: #8392a5;
        font-weight: 600;
    }
    
    /* --- INPUTS / SPINBOXES --- */
    QLineEdit, QDoubleSpinBox {
        background-color: rgba(0, 0, 0, 0.5);
        color: #00f3ff;
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 4px;
        padding: 8px;
        font-family: monospace;
        font-size: 14px;
    }
    QLineEdit:focus, QDoubleSpinBox:focus {
        border: 1px solid #00f3ff;
        background-color: rgba(0, 243, 255, 0.05);
    }
    QLineEdit#hfTerminalInput {
        border: none;
        border-bottom: 2px solid rgba(188, 19, 254, 0.5);
        border-radius: 0px;
        background-color: transparent;
        color: #bc13fe;
    }
    QLineEdit#hfTerminalInput:focus {
        border-bottom: 2px solid #bc13fe;
    }
    
    /* --- PROGRESS BAR (ENERGY METER) --- */
    QProgressBar#energyBar {
        border: 1px solid rgba(0, 243, 255, 0.4);
        border-radius: 3px;
        background-color: #000;
        height: 12px;
    }
    QProgressBar#energyBar::chunk {
        background-color: #00f3ff;
        width: 10px;
        margin: 1px;
    }
    
    /* --- TERMINAL READOUT AND INPUTS --- */
    QTextEdit#consoleReadout {
        background-color: rgba(0, 0, 0, 0.7);
        color: #00f3ff;
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-radius: 6px;
        font-family: Consolas, "Fira Code", monospace;
        font-size: 12px;
        padding: 10px;
    }
    
    QPlainTextEdit#orderInput {
        background-color: rgba(0, 0, 0, 0.6);
        color: #00f3ff;
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-radius: 6px;
        font-family: Consolas, "Fira Code", monospace;
        font-size: 13px;
        padding: 12px;
    }
    QPlainTextEdit#orderInput:focus {
        border: 1px solid #00f3ff;
        background-color: rgba(0, 243, 255, 0.05);
    }
    
    /* --- BUTTONS --- */
    QPushButton {
        font-weight: bold;
        letter-spacing: 1px;
        font-size: 13px;
    }
    
    /* Cyan Neon Buttons */
    QPushButton#neonBtnCyan {
        background-color: rgba(0, 243, 255, 0.05);
        color: #00f3ff;
        border: 1px solid #00f3ff;
        border-radius: 6px;
        padding: 12px;
    }
    QPushButton#neonBtnCyan:hover {
        background-color: rgba(0, 243, 255, 0.2);
        border: 1px solid #ffffff;
        color: #ffffff;
    }
    QPushButton#neonBtnCyan:pressed {
        background-color: #00f3ff;
        color: #000000;
    }
    
    /* Purple Neon Buttons */
    QPushButton#neonBtnPurple {
        background-color: rgba(188, 19, 254, 0.05);
        color: #bc13fe;
        border: 1px solid #bc13fe;
        border-radius: 6px;
        padding: 12px;
    }
    QPushButton#neonBtnPurple:hover {
        background-color: rgba(188, 19, 254, 0.2);
        border: 1px solid #ffffff;
        color: #ffffff;
    }
    QPushButton#neonBtnPurple:pressed {
        background-color: #bc13fe;
        color: #000000;
    }
    QPushButton#neonBtnPurple:disabled {
        border: 1px solid rgba(188, 19, 254, 0.2);
        color: rgba(188, 19, 254, 0.4);
    }
    
    /* SPECTACULAR START WORD BUTTON */
    QPushButton#startWordBtn {
        background-color: rgba(0, 243, 255, 0.1);
        color: #00f3ff;
        border: 2px solid #00f3ff;
        border-radius: 8px;
        padding: 16px;
        font-size: 15px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    QPushButton#startWordBtn:hover {
        background-color: #00f3ff;
        color: #090A0F;
        border: 2px solid #ffffff;
    }
    QPushButton#startWordBtn:pressed {
        background-color: rgba(0, 243, 255, 0.5);
    }
    
    /* SPECTACULAR UPDATE BUTTON */
    QPushButton#updateBtn {
        background-color: rgba(188, 19, 254, 0.1);
        color: #bc13fe;
        border: 2px solid #bc13fe;
        border-radius: 8px;
        padding: 20px;
        font-size: 16px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    QPushButton#updateBtn:hover {
        background-color: #bc13fe;
        color: #ffffff;
        border: 2px solid #ffffff;
        /* Simulated glow by changing bg and border intensely */
    }
    QPushButton#updateBtn:pressed {
        background-color: rgba(188, 19, 254, 0.5);
    }
    QPushButton#updateBtn:disabled {
        border: 1px solid #333;
        color: #555;
        background-color: transparent;
    }
    
    /* --- SPLITTER & SCROLLBAR --- */
    QSplitter::handle {
        background-color: transparent;
        width: 2px;
    }
    QScrollBar:vertical {
        border: none;
        background: transparent;
        width: 8px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: rgba(0, 243, 255, 0.3);
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background: #00f3ff;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """
    app.setStyleSheet(cyberpunk_qss)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
