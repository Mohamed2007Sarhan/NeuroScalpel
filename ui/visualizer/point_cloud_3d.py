import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QMouseEvent
import pyqtgraph.opengl as gl

class CustomGLViewWidget(gl.GLViewWidget):
    """
    Subclassing GLViewWidget to intercept mouse events for picking and dragging points.
    """
    pointClicked = pyqtSignal(int, tuple)
    pointDragged = pyqtSignal(int, tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = None           # shape (N, 3)
        self.ids = None              # shape (N,)
        self.colors = None           # shape (N, 4)
        self.scatter_item = None
        
        self.selected_index = None
        self.is_dragging = False
        
        self.setCameraPosition(distance=30, elevation=25, azimuth=45)
        
    def set_data(self, points, ids):
        self.points = points.copy()
        self.ids = ids
        
        # Default neon cyan color with some transparency
        self.colors = np.ones((len(points), 4), dtype=np.float32)
        self.colors[:, 0] = 0.0  # R
        self.colors[:, 1] = 0.9  # G
        self.colors[:, 2] = 1.0  # B
        self.colors[:, 3] = 0.6  # Alpha
        
        if self.scatter_item:
            self.removeItem(self.scatter_item)
            
        self.scatter_item = gl.GLScatterPlotItem(
            pos=self.points, 
            color=self.colors, 
            size=6, 
            pxMode=True
        )
        self.addItem(self.scatter_item)
        
    def highlight_point(self, index):
        """Highlights the selected point by turning it glowing purple and enlarging it."""
        if self.points is None or self.scatter_item is None:
            return
            
        # Reset colors to neon cyan
        self.colors[:, 0:3] = [0.0, 0.9, 1.0]
        self.colors[:, 3] = 0.6
        sizes = np.full(len(self.points), 6.0)
        
        if index is not None and 0 <= index < len(self.points):
            self.colors[index] = [0.74, 0.0, 1.0, 1.0] # Glowing Neon Purple
            sizes[index] = 20.0
            
        self.scatter_item.setData(color=self.colors, size=sizes)
        
    def _get_proximate_point(self, screen_pos: QPoint):
        """
        Approximates a click by projecting the 3D points to 2D screen coordinates
        and finding the closest active point within a pixel threshold.
        """
        if self.points is None or len(self.points) == 0:
            return None
            
        # Proper raycasting to find nearest point in 3D based on 2D screen projection
        # This is a highly simplified mock for UI demonstration.
        import random
        return random.randint(0, len(self.points) - 1)
        
    def mousePressEvent(self, ev: QMouseEvent):
        # Intercept if clicking on/near a point (simulated here by Control key modifier to separate from rotating)
        if ev.button() == Qt.MouseButton.LeftButton:
            modifiers = ev.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                idx = self._get_proximate_point(ev.pos())
                if idx is not None:
                    self.selected_index = idx
                    self.is_dragging = True
                    self.highlight_point(idx)
                    pt = self.points[idx]
                    self.pointClicked.emit(self.ids[idx], (float(pt[0]), float(pt[1]), float(pt[2])))
                    return # Stop event from reaching camera
        
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.is_dragging and self.selected_index is not None:
            # We are currently dragging the selected point.
            # Simplified mock: moving mouse alters X/Y roughly to show feedback.
            self.points[self.selected_index][0] += 0.15
            self.points[self.selected_index][1] -= 0.15
            
            self.scatter_item.setData(pos=self.points)
            pt = self.points[self.selected_index]
            self.pointDragged.emit(self.ids[self.selected_index], (float(pt[0]), float(pt[1]), float(pt[2])))
            return
            
        super().mouseMoveEvent(ev)
        
    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            return
            
        super().mouseReleaseEvent(ev)


class PointCloud3DWidget(QWidget):
    """
    Main widget wrapper for the 3D Point Cloud.
    Contains the 3D canvas and provides public API.
    """
    pointSelected = pyqtSignal(int, tuple)
    pointMoved = pyqtSignal(int, tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.gl_widget = CustomGLViewWidget()
        
        # Connect internal signals to external signals
        self.gl_widget.pointClicked.connect(self.pointSelected.emit)
        self.gl_widget.pointDragged.connect(self.pointMoved.emit)
        
        layout.addWidget(self.gl_widget)
        
    def load_points(self, points, ids):
        self.gl_widget.set_data(points, ids)
        
    def update_point_position(self, point_id, new_coords):
        # Update programmatically (e.g., from Dashboard)
        if self.gl_widget.points is not None and self.gl_widget.ids is not None:
            idx = np.where(self.gl_widget.ids == point_id)[0]
            if len(idx) > 0:
                self.gl_widget.points[idx[0]] = new_coords
                self.gl_widget.scatter_item.setData(pos=self.gl_widget.points)
                self.gl_widget.highlight_point(idx[0])
