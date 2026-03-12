from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor

class OrderTerminalWindow(QWidget):
    """
    An independent secondary window spawned from the main application.
    Features a sleek Cyberpunk aesthetic with deep dark backgrounds and neon accents.
    Contains a robust QTextEdit to safely handle real-time log streaming.
    """
    def __init__(self, title="TERMINAL", parent=None):
        super().__init__(parent)
        
        # Ensure it acts as an independent window
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.resize(500, 400)
        
        # Apply the specific window ID for QSS styling in main
        self.setObjectName("OrderTerminalWindow")
        
        self.init_ui(title)
        
    def init_ui(self, title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Futuristic glowing centered title
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("neonTitle")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        
        # Secure, read-only terminal area for the Matrix streams
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(0, 243, 255, 0.2);
                border-radius: 6px;
                font-family: Consolas, "Fira Code", monospace;
                font-size: 13px;
                padding: 10px;
            }
        """)
        
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.text_area)
        
    def append_text(self, text, color="#00f3ff"):
        """
        Safely appends text to the terminal in the specified hex color.
        Ensures the scrollbar stays automatically anchored to the bottom.
        """
        self.text_area.moveCursor(QTextCursor.MoveOperation.End)
        
        # Apply strict color formatting to the inserted chunk
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        self.text_area.setCurrentCharFormat(fmt)
        
        self.text_area.insertPlainText(text)
        
        # Auto-scroll to track rapid log injections
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
