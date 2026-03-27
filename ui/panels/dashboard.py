"""
dashboard.py
============
Right-panel HUD for NeuroScalpel.

Enhancements over original
--------------------------
- Task Queue status display (shows all tasks with ⏳/⚙️/✅/❌ icons)
- Task progress counter "TASK 1 / 3"
- "APPLY ROME EDIT" button enabled after Phase 3 TARGET LOCKED
- Bias vector controls preserved
- All signals preserved for backward compatibility
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QDoubleSpinBox, QGroupBox, QPushButton, QDialog, QDialogButtonBox, QSpinBox,
    QPlainTextEdit, QLabel, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, Qt, QLocale
from PyQt6.QtGui import QFont

class EditTargetDialog(QDialog):
    def __init__(self, current_layer, current_neuron, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Target Lock")
        self.setFixedSize(250, 150)
        self.setStyleSheet("background-color: #1e1e2e; color: #fff;")
        
        layout = QFormLayout(self)
        
        # Force English locale so numbers appear as 1, 2, 3 instead of Arabic letters
        english_locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        
        self.spin_layer = QSpinBox()
        self.spin_layer.setLocale(english_locale)
        self.spin_layer.setRange(0, 500)
        self.spin_layer.setValue(current_layer if current_layer is not None else 0)
        self.spin_layer.setStyleSheet("background-color: #2e2e3e; color: #00f3ff; border: 1px solid #555;")
        
        self.spin_neuron = QSpinBox()
        self.spin_neuron.setLocale(english_locale)
        self.spin_neuron.setRange(0, 1000000)
        self.spin_neuron.setValue(current_neuron if current_neuron is not None else 0)
        self.spin_neuron.setStyleSheet("background-color: #2e2e3e; color: #00f3ff; border: 1px solid #555;")
        
        layout.addRow("Layer:", self.spin_layer)
        layout.addRow("Neuron:", self.spin_neuron)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_values(self):
        return self.spin_layer.value(), self.spin_neuron.value()


class DashboardPanel(QWidget):
    """
    Right Controller panel (HUD-style).
    Signals:
        updateRequested(int, tuple, tuple)  — update weights (dashboard drag)
        startWordRequested(str)             — user clicked START WORD
        applyRomeRequested()                — user clicked APPLY ROME EDIT
    """

    startWordRequested = pyqtSignal(str)
    applyRomeRequested = pyqtSignal()
    targetLockEdited  = pyqtSignal(int, int)
    runAbliterationRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setObjectName("DashboardPanel")

        self.selected_point_id = None
        self.last_coords = (0.0, 0.0, 0.0)
        self._target_layer: int = None
        self._target_point: int = None

        self.init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(14)

        # ---- ORDER INPUT ----
        group_order = QGroupBox("ORDER TERMINAL")
        group_order.setObjectName("HUDGroupBox")
        order_layout = QVBoxLayout()
        order_layout.setContentsMargins(12, 22, 12, 12)
        order_layout.setSpacing(10)

        self.input_order = QPlainTextEdit()
        self.input_order.setObjectName("orderInput")
        self.input_order.setPlaceholderText(
            "> Describe the hallucination(s) to correct...\n"
            "> You can list multiple issues — they will be\n"
            "> queued and handled one by one automatically."
        )
        self.input_order.setMinimumHeight(110)

        self.btn_start_word = QPushButton("⚡  START WORD")
        self.btn_start_word.setObjectName("startWordBtn")
        self.btn_start_word.clicked.connect(self.on_start_word_clicked)

        order_layout.addWidget(self.input_order)
        order_layout.addWidget(self.btn_start_word)
        group_order.setLayout(order_layout)

        # ---- TASK QUEUE STATUS ----
        group_queue = QGroupBox("TASK QUEUE")
        group_queue.setObjectName("HUDGroupBox")
        queue_layout = QVBoxLayout()
        queue_layout.setContentsMargins(12, 22, 12, 12)
        queue_layout.setSpacing(8)

        self.lbl_task_progress = QLabel("No active tasks")
        self.lbl_task_progress.setObjectName("statusLabel")
        self.lbl_task_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_task_progress.setStyleSheet(
            "color: #00f3ff; font-weight: bold; font-size: 13px;"
        )

        self.task_bar = QProgressBar()
        self.task_bar.setObjectName("energyBar")
        self.task_bar.setValue(0)
        self.task_bar.setTextVisible(False)
        self.task_bar.hide()

        self.lbl_task_list = QLabel("")
        self.lbl_task_list.setObjectName("statusLabel")
        self.lbl_task_list.setWordWrap(True)
        self.lbl_task_list.setStyleSheet(
            "color: #7a8a9a; font-size: 11px; font-family: Consolas, monospace;"
        )

        queue_layout.addWidget(self.lbl_task_progress)
        queue_layout.addWidget(self.task_bar)
        queue_layout.addWidget(self.lbl_task_list)
        group_queue.setLayout(queue_layout)

        # ---- TARGET LOCK INFO ----
        group_target = QGroupBox("TARGET LOCK")
        group_target.setObjectName("HUDGroupBox")
        target_layout = QVBoxLayout()
        target_layout.setContentsMargins(12, 22, 12, 12)
        target_layout.setSpacing(8)

        self.lbl_target = QLabel("// AWAITING SCAN...")
        self.lbl_target.setObjectName("statusLabel")
        self.lbl_target.setWordWrap(True)
        self.lbl_target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_target.setStyleSheet(
            "color: #555; font-family: Consolas, monospace; font-size: 12px;"
        )

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.lbl_target, stretch=1)
        
        self.btn_edit_target = QPushButton("✏️")
        self.btn_edit_target.setFixedSize(24, 24)
        self.btn_edit_target.setToolTip("Edit Target Layer & Neuron")
        self.btn_edit_target.setStyleSheet("background-color: #333; border: 1px solid #555; border-radius: 4px;")
        self.btn_edit_target.clicked.connect(self.on_edit_target_clicked)
        self.btn_edit_target.hide()
        
        h_layout.addWidget(self.btn_edit_target, alignment=Qt.AlignmentFlag.AlignRight)

        self.btn_rome_edit = QPushButton("🧬  APPLY NEURAL EDIT")
        self.btn_rome_edit.setObjectName("neonBtnPurple")
        self.btn_rome_edit.setEnabled(False)
        self.btn_rome_edit.setToolTip(
            "Available after Target Lock is confirmed.\n"
            "Runs the neural weight correction pipeline."
        )
        self.btn_rome_edit.clicked.connect(self.on_rome_edit_clicked)

        target_layout.addLayout(h_layout)
        target_layout.addWidget(self.btn_rome_edit)
        group_target.setLayout(target_layout)

        # ---- FAST OPTIONS ----
        group_fast_options = QGroupBox("FAST OPTIONS")
        group_fast_options.setObjectName("HUDGroupBox")
        fast_layout = QVBoxLayout()
        fast_layout.setContentsMargins(12, 22, 12, 12)
        fast_layout.setSpacing(12)

        self.btn_run_abliteration = QPushButton("Run Abliteration")
        self.btn_run_abliteration.setObjectName("neonBtnPurple")
        self.btn_run_abliteration.setToolTip("Apply current Abliteration refusal_dir to the model")
        self.btn_run_abliteration.clicked.connect(self.on_run_abliteration_clicked)

        fast_layout.addWidget(self.btn_run_abliteration)
        group_fast_options.setLayout(fast_layout)

        # ---- Copyright ----
        self.lbl_copyright = QLabel(
            "© 2026 NextCore. All rights reserved.\n"
            "© 2026 Mohamed Sarhan. All rights reserved."
        )
        self.lbl_copyright.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_copyright.setStyleSheet(
            "color: #4a5a6a; font-size: 11px; font-family: Consolas, monospace; letter-spacing: 1px; padding-top: 10px;"
        )
        # self.lbl_copyright2 = QLabel("© 2026 mo sar han ham ed. All rights reserved.")
        # self.lbl_copyright2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.lbl_copyright2.setStyleSheet(
        #     "color: #4a5a6a; font-size: 11px; font-family: Consolas, monospace; letter-spacing: 1px; padding-top: 10px;"
        # )

        # ---- Assemble ----
        layout.addWidget(group_order, stretch=2)
        layout.addWidget(group_queue)
        layout.addWidget(group_target)
        layout.addWidget(group_fast_options)
        layout.addWidget(self.lbl_copyright)
        # layout.addWidget(self.lbl_copyright2)

    # ------------------------------------------------------------------
    # Public update methods (called from main_window)
    # ------------------------------------------------------------------

    def update_task_queue(self, summary_text: str, current: int, total: int):
        """Refreshes the task queue display."""
        self.lbl_task_list.setText(summary_text)
        if total > 0:
            self.lbl_task_progress.setText(f"TASK  {current} / {total}")
            self.task_bar.setMaximum(total)
            self.task_bar.setValue(current - 1)
            self.task_bar.show()
        else:
            self.lbl_task_progress.setText("No active tasks")
            self.task_bar.hide()

    def set_target_locked(self, layer_idx: int, vector_point: int):
        """Called after Phase 3 emits TARGET LOCKED confirmation."""
        self._target_layer = layer_idx
        self._target_point = vector_point

        self.lbl_target.setText(
            f"🎯 LAYER  {layer_idx}\n"
            f"   NEURON  #{vector_point}"
        )
        self.lbl_target.setStyleSheet(
            "color: #ff2244; font-weight: bold; font-size: 13px;"
            "font-family: Consolas, monospace; letter-spacing: 1px;"
        )
        self.btn_rome_edit.setEnabled(True)
        self.btn_edit_target.show()

    def clear_target_lock(self):
        """Resets target info (called at pipeline start)."""
        self._target_layer = None
        self._target_point = None
        self.lbl_target.setText("// AWAITING SCAN...")
        self.lbl_target.setStyleSheet(
            "color: #555; font-family: Consolas, monospace; font-size: 12px;"
        )
        self.btn_rome_edit.setEnabled(False)
        self.btn_edit_target.hide()

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------

    def on_start_word_clicked(self):
        order_text = self.input_order.toPlainText().strip()
        if order_text:
            self.clear_target_lock()
            self.startWordRequested.emit(order_text)

    def on_run_abliteration_clicked(self):
        self.runAbliterationRequested.emit()

    def on_rome_edit_clicked(self):
        self.btn_rome_edit.setEnabled(False)
        self.btn_rome_edit.setText("⚙️  EDITING...")
        self.applyRomeRequested.emit()

    def on_edit_target_clicked(self):
        if self._target_layer is None or self._target_point is None:
            return
        dialog = EditTargetDialog(self._target_layer, self._target_point, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_layer, new_neuron = dialog.get_values()
            self._target_layer = new_layer
            self._target_point = new_neuron
            self.lbl_target.setText(
                f"🎯 LAYER  {new_layer}\n"
                f"   NEURON  #{new_neuron} (Manual)"
            )
            self.targetLockEdited.emit(new_layer, new_neuron)
