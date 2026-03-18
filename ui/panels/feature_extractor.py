import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, 
    QLineEdit, QGroupBox, QTextEdit, QPushButton, QFileDialog, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, QTimer

class FeatureExtractorPanel(QWidget):
    """
    Left sidebar HUD panel for loading the LLM model.
    Designed with a Cyberpunk/Sci-Fi aesthetic in mind.
    """
    
    # Custom signal emitted when a model is successfully loaded
    modelLoaded = pyqtSignal(str)
    
    def __init__(self, backend_manager, parent=None):
        super().__init__(parent)
        self.backend = backend_manager
        
        self.setFixedWidth(300)
        self.setObjectName("FeaturePanel")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(25)
        
        # --- Local Model Group (Option A) ---
        group_local = QGroupBox("LOCAL DIRECTORY")
        group_local.setObjectName("HUDGroupBox")
        local_layout = QVBoxLayout()
        local_layout.setContentsMargins(15, 25, 15, 15)
        local_layout.setSpacing(12)
        
        self.lbl_local_path = QLabel("// NO TARGET DIR...")
        self.lbl_local_path.setWordWrap(True)
        self.lbl_local_path.setObjectName("statusLabel")
        
        self.btn_browse = QPushButton("BROWSE UPLINK")
        self.btn_browse.setObjectName("neonBtnCyan")
        self.btn_browse.clicked.connect(self.on_browse_clicked)
        
        self.btn_load_local = QPushButton("INITIALIZE SECURE LOAD")
        self.btn_load_local.setObjectName("neonBtnPurple")
        self.btn_load_local.clicked.connect(self.on_load_local_clicked)
        self.btn_load_local.setEnabled(False)
        
        local_layout.addWidget(self.btn_browse)
        local_layout.addWidget(self.lbl_local_path)
        local_layout.addWidget(self.btn_load_local)
        group_local.setLayout(local_layout)
        
        # --- Hugging Face Group (Option B) ---
        group_hf = QGroupBox("HUGGINGFACE REPO INIT")
        group_hf.setObjectName("HUDGroupBox")
        hf_layout = QVBoxLayout()
        hf_layout.setContentsMargins(15, 25, 15, 15)
        hf_layout.setSpacing(12)
        
        self.input_hf = QLineEdit()
        self.input_hf.setObjectName("hfTerminalInput")
        self.input_hf.setPlaceholderText("> model_id (e.g. mistralai/Mistral-7B)")
        
        self.btn_download_hf = QPushButton("DOWNLOAD & INJECT")
        self.btn_download_hf.setObjectName("neonBtnCyan")
        self.btn_download_hf.clicked.connect(self.on_download_hf_clicked)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("energyBar")
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        
        hf_layout.addWidget(self.input_hf)
        hf_layout.addWidget(self.btn_download_hf)
        hf_layout.addWidget(self.progress_bar)
        group_hf.setLayout(hf_layout)
        
        # --- Real-time Readout Group ---
        group_readout = QGroupBox("DEBUG CONSOLE")
        group_readout.setObjectName("HUDGroupBox")
        readout_layout = QVBoxLayout()
        readout_layout.setContentsMargins(15, 25, 15, 15)
        
        self.text_readout = QTextEdit()
        self.text_readout.setObjectName("consoleReadout")
        self.text_readout.setReadOnly(True)
        
        readout_layout.addWidget(self.text_readout)
        group_readout.setLayout(readout_layout)
        
        layout.addWidget(group_local)
        layout.addWidget(group_hf)
        layout.addWidget(group_readout)
        
    def on_browse_clicked(self):
        """Triggered to open folder dialog for Local model selection."""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if folder:
            self.lbl_local_path.setText(f"DIR: {folder}")
            self.lbl_local_path.setStyleSheet("color: #00f3ff; font-weight: bold;")
            self.btn_load_local.setEnabled(True)
            self.update_readout(f"Uplink path locked:\n{folder}")
            
    def on_load_local_clicked(self):
        """Loads the selected local model using the backend."""
        path = self.lbl_local_path.text()
        self.update_readout("Initializing local neural load...")
        self.btn_load_local.setEnabled(False)
        
        # Simulate local loading
        success = self.backend.load_local_model(path)
        if success:
            model_name = os.path.basename(path.replace("DIR: ", "")) or "LOCAL_MODEL"
            self.update_readout(f"Load complete. Model '{model_name}' active.")
            self.modelLoaded.emit(model_name)
        self.btn_load_local.setEnabled(True)
            
    def on_download_hf_clicked(self):
        """Initiates a download progress bar for the requested HuggingFace model."""
        model_id = self.input_hf.text().strip()
        if not model_id:
            self.update_readout("ERROR: Missing HuggingFace repo identifier.")
            return
            
        self.update_readout(f"Establishing HF API Link:\n{model_id}...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.btn_download_hf.setEnabled(False)
        self.input_hf.setEnabled(False)
        
        # Simulate background download
        self.progress = 0
        self.timer = QTimer(self)
        self.timer_model_id = model_id
        self.timer.timeout.connect(self._simulate_download)
        self.timer.start(40) # 40ms resolution
        
    def _simulate_download(self):
        """Timer callback simulating a download pipeline."""
        self.progress += 2
        self.progress_bar.setValue(self.progress)
        
        if self.progress >= 100:
            self.timer.stop()
            success = self.backend.load_hf_model(self.timer_model_id)
            if success:
                self.update_readout("Injection complete. Synthesizing latent space...")
                self.modelLoaded.emit(self.timer_model_id)
            # Reset UI
            self.btn_download_hf.setEnabled(True)
            self.input_hf.setEnabled(True)
            self.progress_bar.hide()
            
    def update_readout(self, text: str):
        """Appends a new line to the terminal emulation text edit (plain white)."""
        if not isinstance(text, str):
            text = str(text)
        self.text_readout.append(f"> {text}")
        scrollbar = self.text_readout.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_readout_colored(self, text: str, color: str = "#00f3ff"):
        """
        Appends coloured text to the panel console.
        Used by ModelManager log callbacks.
        """
        from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor
        if not isinstance(text, str):
            text = str(text)
        cursor = self.text_readout.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"> {text}")
        scrollbar = self.text_readout.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
