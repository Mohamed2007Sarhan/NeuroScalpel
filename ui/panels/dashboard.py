from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, 
    QDoubleSpinBox, QGroupBox, QPushButton, QPlainTextEdit
)
from PyQt6.QtCore import pyqtSignal

class DashboardPanel(QWidget):
    """
    Right Controller panel (HUD-style).
    Features a high-tech terminal input for "Orders", a "Start Word" action,
    In-Flight Steering Bias controls, and the critical Update Weights Action.
    """
    
    # Custom signals for actions
    updateRequested = pyqtSignal(int, tuple, tuple) # point_id, new_coords, bias_vector
    startWordRequested = pyqtSignal(str) # order_text
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setFixedWidth(280)
        self.selected_point_id = None
        self.last_coords = (0.0, 0.0, 0.0) 
        
        self.setObjectName("DashboardPanel")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(20)
        
        # --- ORDER TERMINAL GROUP ---
        group_order = QGroupBox("ORDER")
        group_order.setObjectName("HUDGroupBox")
        order_layout = QVBoxLayout()
        order_layout.setContentsMargins(15, 25, 15, 15)
        order_layout.setSpacing(15)
        
        self.input_order = QPlainTextEdit()
        self.input_order.setObjectName("orderInput")
        self.input_order.setPlaceholderText("> Awaiting instructions...")
        
        self.btn_start_word = QPushButton("START WORD")
        self.btn_start_word.setObjectName("startWordBtn")
        self.btn_start_word.clicked.connect(self.on_start_word_clicked)
        
        order_layout.addWidget(self.input_order) # Expands naturally
        order_layout.addWidget(self.btn_start_word)
        group_order.setLayout(order_layout)
        
        # --- Bias Vector Adjustment Group ---
        group_bias = QGroupBox("IN-FLIGHT STEERING BIAS")
        group_bias.setObjectName("HUDGroupBox")
        form_layout_bias = QFormLayout()
        form_layout_bias.setContentsMargins(15, 25, 15, 15)
        form_layout_bias.setVerticalSpacing(15)
        
        self.bias_x = QDoubleSpinBox()
        self.bias_y = QDoubleSpinBox()
        self.bias_z = QDoubleSpinBox()
        
        for spin in (self.bias_x, self.bias_y, self.bias_z):
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.1)
            spin.setValue(0.0)
            
        form_layout_bias.addRow("X-AXIS:", self.bias_x)
        form_layout_bias.addRow("Y-AXIS:", self.bias_y)
        form_layout_bias.addRow("Z-AXIS:", self.bias_z)
        group_bias.setLayout(form_layout_bias)
        
        # --- Action Button ---
        self.btn_update = QPushButton("UPDATE WEIGHTS")
        self.btn_update.setObjectName("updateBtn")
        self.btn_update.clicked.connect(self.on_update_clicked)
        # Initially disabled until a point is selected
        self.btn_update.setEnabled(False)
        self.btn_update.setToolTip("Select a Vector (Point) in the visualizer to apply biases.")
        
        # Build main layout
        layout.addWidget(group_order, stretch=1)
        layout.addWidget(group_bias)
        layout.addWidget(self.btn_update)
        
    def set_selected_point(self, point_id, coords):
        """Called when a point is selected/dragged in the visualizer."""
        self.selected_point_id = point_id
        self.last_coords = coords
        self.btn_update.setText(f"UPDATE WEIGHTS\n[ TARGET: P{point_id} ]")
        self.btn_update.setEnabled(True)
        
    def get_bias_vector(self):
        """Returns the current bias vector."""
        return (self.bias_x.value(), self.bias_y.value(), self.bias_z.value())
        
    def on_start_word_clicked(self):
        """Triggered when the Start Word neon button is clicked."""
        order_text = self.input_order.toPlainText()
        self.startWordRequested.emit(order_text)
        
    def on_update_clicked(self):
        """Triggered when the bright purple neon button is clicked."""
        if self.selected_point_id is not None:
            self.updateRequested.emit(
                self.selected_point_id, 
                self.last_coords, 
                self.get_bias_vector()
            )
