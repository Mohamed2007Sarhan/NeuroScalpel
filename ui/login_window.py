import sys
import threading
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QApplication, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor

from core.auth_manager import auth_manager, AuthException

class LoginDialog(QDialog):
    login_success = pyqtSignal()
    login_failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NexCore App Authentication")
        self.setFixedSize(450, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        # Consistent setup
        self.setStyleSheet("""
            QDialog {
                background-color: #0d0f1a;
                border: 1px solid rgba(0, 243, 255, 0.4);
                border-radius: 12px;
            }
            QLabel { color: #f8fafc; font-family: "Segoe UI"; }
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255,255,255, 0.2);
                border-radius: 8px;
                padding: 10px;
                color: #00f3ff;
                font-family: Consolas, monospace;
                font-size: 14px;
            }
            QLineEdit:focus { border: 1px solid #00f3ff; }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #6366f1);
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #818cf8);
            }
            QPushButton:disabled {
                background: #334; color: #889;
            }
        """)

        self.init_ui()
        
        self.login_success.connect(self.accept)
        self.login_failed.connect(self.show_error)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title
        lbl_title = QLabel("CONNECT TO NEXCORE")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 900; color: #00f3ff; letter-spacing: 2px;")
        
        lbl_desc = QLabel("Please enter your account credentials to bind this device to your active subscription.")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 12px; margin-bottom: 10px;")

        # Inputs
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(15)
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("you@example.com")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("••••••••")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)

        # Style labels for form
        lbl_email = QLabel("Email Address")
        lbl_email.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")
        lbl_pass = QLabel("Password")
        lbl_pass.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")

        form_layout.addRow(lbl_email, self.email_input)
        form_layout.addRow(lbl_pass, self.password_input)

        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: bold;")
        self.lbl_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_error.hide()

        self.btn_submit = QPushButton("INITIALIZE CONNECTION")
        self.btn_submit.clicked.connect(self.handle_login)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_desc)
        layout.addLayout(form_layout)
        layout.addWidget(self.lbl_error)
        layout.addStretch()
        layout.addWidget(self.btn_submit)

    def handle_login(self):
        email = self.email_input.text().strip()
        pwd = self.password_input.text().strip()

        if not email or not pwd:
            self.show_error("Credentials cannot be empty")
            return

        self.lbl_error.hide()
        self.btn_submit.setEnabled(False)
        self.btn_submit.setText("AUTHENTICATING...")
        self.email_input.setEnabled(False)
        self.password_input.setEnabled(False)

        # Run auth in background to not block UI
        threading.Thread(target=self._async_login, args=(email, pwd), daemon=True).start()

    def _async_login(self, email, pwd):
        try:
            auth_manager.login(email, pwd)
            self.login_success.emit()
        except AuthException as e:
            self.login_failed.emit(str(e))
        except Exception as e:
            self.login_failed.emit(f"Unknown error: {e}")

    def show_error(self, message):
        self.lbl_error.setText(f"⚠ {message}")
        self.lbl_error.show()
        self.btn_submit.setEnabled(True)
        self.btn_submit.setText("INITIALIZE CONNECTION")
        self.email_input.setEnabled(True)
        self.password_input.setEnabled(True)

def show_login_flow():
    """
    Shows login window if needed, attempts to activate session.
    If success, it returns True. Otherwise False.
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    def do_activate():
        try:
            auth_manager.activate()
            auth_manager.fetch_settings()
            auth_manager.apply_settings_to_env()
            return True
        except AuthException as e:
            print(f"[AUTH] Activate failed: {e}")
            auth_manager.delete_session()
            return False

    has_session = auth_manager.load_session()
    
    if has_session:
        print("[AUTH] Decrypted local session, attempting silent activation...")
        if do_activate():
            return True

    print("[AUTH] No valid session or activation failed. Showing Login UI.")
    dialog = LoginDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        # After login dialog accepts, the session was saved successfully. Note: we still need to activate to fetch settings, etc.
        auth_manager.load_session()
        if do_activate():
            return True
    
    return False
