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
    QWidget, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QGroupBox, QPushButton,
    QPlainTextEdit, QLabel, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class DashboardPanel(QWidget):
    """
    Right Controller panel (HUD-style).
    Signals:
        updateRequested(int, tuple, tuple)  — update weights (dashboard drag)
        startWordRequested(str)             — user clicked START WORD
        applyRomeRequested()                — user clicked APPLY ROME EDIT
    """

    updateRequested   = pyqtSignal(int, tuple, tuple)
    startWordRequested = pyqtSignal(str)
    applyRomeRequested = pyqtSignal()

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

        self.btn_rome_edit = QPushButton("🧬  APPLY NEURAL EDIT")
        self.btn_rome_edit.setObjectName("neonBtnPurple")
        self.btn_rome_edit.setEnabled(False)
        self.btn_rome_edit.setToolTip(
            "Available after Target Lock is confirmed.\n"
            "Runs the neural weight correction pipeline."
        )
        self.btn_rome_edit.clicked.connect(self.on_rome_edit_clicked)

        target_layout.addWidget(self.lbl_target)
        target_layout.addWidget(self.btn_rome_edit)
        group_target.setLayout(target_layout)

        # ---- BIAS VECTOR ----
        group_bias = QGroupBox("IN-FLIGHT STEERING BIAS")
        group_bias.setObjectName("HUDGroupBox")
        form_bias = QFormLayout()
        form_bias.setContentsMargins(12, 22, 12, 12)
        form_bias.setVerticalSpacing(12)

        self.bias_x = QDoubleSpinBox()
        self.bias_y = QDoubleSpinBox()
        self.bias_z = QDoubleSpinBox()
        for spin in (self.bias_x, self.bias_y, self.bias_z):
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.1)
            spin.setValue(0.0)

        form_bias.addRow("X-AXIS:", self.bias_x)
        form_bias.addRow("Y-AXIS:", self.bias_y)
        form_bias.addRow("Z-AXIS:", self.bias_z)
        group_bias.setLayout(form_bias)

        # ---- UPDATE WEIGHTS (manual drag) ----
        self.btn_update = QPushButton("UPDATE WEIGHTS")
        self.btn_update.setObjectName("updateBtn")
        self.btn_update.setEnabled(False)
        self.btn_update.setToolTip(
            "Ctrl+Click a point in the 3D view, then click here\n"
            "to apply the bias-vector shift."
        )
        self.btn_update.clicked.connect(self.on_update_clicked)

        # ---- Assemble ----
        layout.addWidget(group_order, stretch=2)
        layout.addWidget(group_queue)
        layout.addWidget(group_target)
        layout.addWidget(group_bias)
        layout.addWidget(self.btn_update)

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
        self.btn_rome_edit.setStyleSheet(
            "QPushButton { animation: none; }"
        )

    def clear_target_lock(self):
        """Resets target info (called at pipeline start)."""
        self._target_layer = None
        self._target_point = None
        self.lbl_target.setText("// AWAITING SCAN...")
        self.lbl_target.setStyleSheet(
            "color: #555; font-family: Consolas, monospace; font-size: 12px;"
        )
        self.btn_rome_edit.setEnabled(False)

    # ------------------------------------------------------------------
    # Original API (backward compat)
    # ------------------------------------------------------------------

    def set_selected_point(self, point_id: int, coords: tuple):
        self.selected_point_id = point_id
        self.last_coords = coords
        self.btn_update.setText(f"UPDATE WEIGHTS\n[ TARGET: P{point_id} ]")
        self.btn_update.setEnabled(True)

    def get_bias_vector(self) -> tuple:
        return (self.bias_x.value(), self.bias_y.value(), self.bias_z.value())

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------

    def on_start_word_clicked(self):
        order_text = self.input_order.toPlainText().strip()
        if order_text:
            self.clear_target_lock()
            self.startWordRequested.emit(order_text)

    def on_update_clicked(self):
        if self.selected_point_id is not None:
            self.updateRequested.emit(
                self.selected_point_id,
                self.last_coords,
                self.get_bias_vector()
            )

    def on_rome_edit_clicked(self):
        self.btn_rome_edit.setEnabled(False)
        self.btn_rome_edit.setText("⚙️  EDITING...")
        self.applyRomeRequested.emit()
