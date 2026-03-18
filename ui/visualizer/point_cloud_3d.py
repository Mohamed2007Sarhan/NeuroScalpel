"""
point_cloud_3d.py
=================
Advanced 3D Neural Visualizer for NeuroScalpel.

Features
--------
- Layer-slab mode  : neurons rendered in Z-separated horizontal planes,
                     one plane per transformer layer (loaded from ModelManager)
- Flat cloud mode  : original PCA scatter cloud (for embedding visualization)
- Target highlight : after Phase 3 locks a target, that neuron pulses in
                     glowing crimson red — clearly visible in the scene
- Axis labels      : X / Y / Z glyphs and layer index annotations
- Picking          : Ctrl+Click to select a point; drag to reposition
- Signals          : pointSelected, pointMoved, targetNeuronVisualized
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QTimer
from PyQt6.QtGui import QMouseEvent, QColor

try:
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    PG_OK = True
except ImportError:
    PG_OK = False


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_NEURON      = (0.0,  0.85, 1.0,  0.65)   # neon cyan base
_C_SELECTED    = (0.74, 0.0,  1.0,  1.0 )   # glowing purple
_C_TARGET      = (1.0,  0.05, 0.05, 1.0 )   # critical red
_C_GRID        = (0.1,  0.9,  1.0,  0.12)   # faint cyan grid
_C_AXIS        = (1.0,  1.0,  1.0,  0.6 )   # white axis lines


def _make_color_array(n: int, base=_C_NEURON) -> np.ndarray:
    arr = np.tile(base, (n, 1)).astype(np.float32)
    return arr


class CustomGLViewWidget(gl.GLViewWidget if PG_OK else object):
    """
    Extended GLViewWidget with:
    - Layer slab rendering with grid separators
    - Real proximity-based point picking (not random)
    - Point dragging
    - Target neuron pulsing animation
    """

    pointClicked = pyqtSignal(int, tuple)
    pointDragged = pyqtSignal(int, tuple)
    targetNeuronVisualized = pyqtSignal(int, int)   # layer_idx, point_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: np.ndarray = None        # (N, 3)
        self.ids: np.ndarray = None           # (N,)
        self.labels: np.ndarray = None        # (N,) layer index per point
        self.colors: np.ndarray = None        # (N, 4)
        self.sizes: np.ndarray = None         # (N,)
        self.scatter_item: "gl.GLScatterPlotItem" = None
        self.grid_items: list = []
        self.axis_items: list = []

        self.selected_index: int = None
        self.is_dragging: bool = False

        # Target neuron animation
        self._target_index: int = None
        self._pulse_scale: float = 1.0
        self._pulse_dir: float = 1.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)

        if PG_OK:
            self.setCameraPosition(distance=60, elevation=20, azimuth=35)
            self._setup_permanent_grid()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_flat_data(self, points: np.ndarray, ids: np.ndarray):
        """
        Classic PCA point cloud (all neurons in one flat cloud).
        """
        self._clear_scene()
        self.points = points.copy()
        self.ids = ids.copy()
        self.labels = np.zeros(len(points), dtype=np.int32)
        self.colors = _make_color_array(len(points))
        self.sizes = np.full(len(points), 5.0, dtype=np.float32)
        self._rebuild_scatter()

    def set_layered_data(self, points: np.ndarray,
                         labels: np.ndarray, layer_map: dict):
        """
        Layered mode: neurons are already arranged in Z-slabs.
        labels[i] = layer index of points[i].
        layer_map = {layer_idx: [row_indices_in_points]}
        """
        self._clear_scene()
        self.points = points.copy()
        self.ids = np.arange(len(points), dtype=np.int64)
        self.labels = labels.copy()
        self.colors = self._layer_color_array(labels)
        self.sizes = np.full(len(points), 4.5, dtype=np.float32)
        self._rebuild_scatter()
        self._draw_layer_grids(layer_map)
        self._draw_axis()

    def _layer_color_array(self, labels: np.ndarray) -> np.ndarray:
        """
        Returns per-neuron colour interpolated along a cyan→purple gradient
        based on layer depth.
        """
        num_layers = int(labels.max()) + 1 if len(labels) else 1
        arr = np.zeros((len(labels), 4), dtype=np.float32)
        for i, lbl in enumerate(labels):
            t = lbl / max(num_layers - 1, 1)
            # cyan (0, 0.85, 1) → purple (0.74, 0, 1)
            arr[i] = [
                t * 0.74,
                0.85 * (1 - t),
                1.0,
                0.70
            ]
        return arr

    # ------------------------------------------------------------------
    # Target neuron highlighting
    # ------------------------------------------------------------------

    def highlight_target(self, layer_idx: int, local_point_idx: int,
                         layer_map: dict = None):
        """
        Highlights the neuron identified as the hallucination source.
        Starts a pulsing crimson animation on that point.
        """
        if self.points is None:
            return

        # Resolve absolute index
        abs_idx = None
        if layer_map and layer_idx in layer_map:
            indices_in_layer = layer_map[layer_idx]
            if local_point_idx < len(indices_in_layer):
                abs_idx = indices_in_layer[local_point_idx]
        else:
            # Flat mode: use local_point_idx directly
            abs_idx = min(local_point_idx, len(self.points) - 1)

        if abs_idx is None or abs_idx >= len(self.points):
            return

        self._target_index = abs_idx

        # Flash it immediately
        self.colors[abs_idx] = list(_C_TARGET)
        self.sizes[abs_idx] = 25.0
        self._refresh_scatter()

        # Start pulse
        self._pulse_timer.start(50)
        self.targetNeuronVisualized.emit(layer_idx, local_point_idx)

    def _tick_pulse(self):
        """Oscillates the target neuron size for a pulsing effect."""
        if self._target_index is None or self.sizes is None:
            self._pulse_timer.stop()
            return

        self._pulse_scale += self._pulse_dir * 0.8
        if self._pulse_scale > 30.0:
            self._pulse_dir = -1.0
        elif self._pulse_scale < 16.0:
            self._pulse_dir = 1.0

        self.sizes[self._target_index] = self._pulse_scale
        alpha = 0.7 + 0.3 * ((self._pulse_scale - 16) / 14)
        self.colors[self._target_index, 3] = min(1.0, float(alpha))
        self._refresh_scatter()

    def reset_target_highlight(self):
        """Clears target pulsing and resets colours."""
        self._pulse_timer.stop()
        self._target_index = None
        if self.colors is not None and self.labels is not None:
            self.colors = self._layer_color_array(self.labels)
            self.sizes = np.full(len(self.points), 4.5, dtype=np.float32)
            self._refresh_scatter()

    # ------------------------------------------------------------------
    # Point selection / dragging
    # ------------------------------------------------------------------

    def highlight_point(self, index: int):
        """
        Highlights a user-selected point (Ctrl+Click) in glowing purple.
        Preserves any target highlight on a different point.
        """
        if self.points is None or self.scatter_item is None:
            return

        # Reset to base colours (keep target if different)
        if self.labels is not None:
            self.colors = self._layer_color_array(self.labels)
        else:
            self.colors = _make_color_array(len(self.points))

        self.sizes = np.full(len(self.points), 4.5, dtype=np.float32)

        # Re-apply target if still set
        if self._target_index is not None:
            self.colors[self._target_index] = list(_C_TARGET)
            self.sizes[self._target_index] = self._pulse_scale

        if index is not None and 0 <= index < len(self.points):
            self.colors[index] = list(_C_SELECTED)
            self.sizes[index] = 20.0

        self._refresh_scatter()

    def _get_proximate_point(self, screen_pos: QPoint) -> int:
        """
        Real screen-space proximity picking.
        Projects all 3D points to 2D device coordinates and returns the
        index of the closest point within a 20-pixel radius.
        """
        if self.points is None or len(self.points) == 0:
            return None

        try:
            # Build projection matrix from GLViewWidget internals
            proj = self.projectionMatrix()
            view = self.viewMatrix()
            mvp = proj * view

            # Homogeneous transform
            pts_h = np.hstack([self.points, np.ones((len(self.points), 1))])
            # Convert PyQt matrix to numpy
            m = [[mvp.column(c).toTuple()[r] for c in range(4)] for r in range(4)]
            m = np.array(m, dtype=np.float64)
            transformed = pts_h @ m.T

            # Perspective divide → NDC
            w = transformed[:, 3:4]
            w = np.where(np.abs(w) < 1e-8, 1e-8, w)
            ndc = transformed[:, :3] / w

            # Map to device pixels
            dw = self.width()
            dh = self.height()
            px = (ndc[:, 0] * 0.5 + 0.5) * dw
            py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * dh

            # Euclidean distance in screen space
            sx, sy = screen_pos.x(), screen_pos.y()
            dist = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            best = int(np.argmin(dist))

            return best if dist[best] < 20.0 else None

        except Exception:
            # Fallback: nearest in projected XY
            if len(self.points):
                sx = screen_pos.x() / self.width() * 2 - 1
                sy = 1 - screen_pos.y() / self.height() * 2
                approx = np.abs(self.points[:, 0] - sx * 30) + np.abs(self.points[:, 1] - sy * 30)
                return int(np.argmin(approx))
            return None

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                idx = self._get_proximate_point(ev.pos())
                if idx is not None:
                    self.selected_index = idx
                    self.is_dragging = True
                    self.highlight_point(idx)
                    pt = self.points[idx]
                    real_id = int(self.ids[idx]) if self.ids is not None else idx
                    self.pointClicked.emit(real_id, (float(pt[0]), float(pt[1]), float(pt[2])))
                    return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.is_dragging and self.selected_index is not None:
            # Horizontal drag → X, Vertical drag → Y (simple linear mapping)
            dx = ev.pos().x() - self.width() / 2
            dy = self.height() / 2 - ev.pos().y()
            scale = 0.05
            self.points[self.selected_index][0] = dx * scale
            self.points[self.selected_index][1] = dy * scale

            self.scatter_item.setData(pos=self.points)
            pt = self.points[self.selected_index]
            real_id = int(self.ids[self.selected_index]) if self.ids is not None else self.selected_index
            self.pointDragged.emit(real_id, (float(pt[0]), float(pt[1]), float(pt[2])))
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            return
        super().mouseReleaseEvent(ev)

    # ------------------------------------------------------------------
    # Scene building helpers
    # ------------------------------------------------------------------

    def _setup_permanent_grid(self):
        """Adds a dark floor grid to the 3D scene."""
        if not PG_OK:
            return
        grid = gl.GLGridItem()
        grid.setSize(80, 80)
        grid.setSpacing(5, 5)
        grid.setColor(QColor(0, 140, 160, 40))
        self.addItem(grid)

    def _draw_layer_grids(self, layer_map: dict):
        """Draws a faint horizontal plane for each transformer layer."""
        if not PG_OK:
            return
        for item in self.grid_items:
            self.removeItem(item)
        self.grid_items.clear()

        for layer_idx in sorted(layer_map.keys()):
            z = layer_idx * 5.0
            # Create a mesh plane at this Z height
            plane = gl.GLSurfacePlotItem(
                x=np.linspace(-8, 8, 2),
                y=np.linspace(-8, 8, 2),
                z=np.full((2, 2), z, dtype=np.float32),
                color=(0.0, 0.85, 1.0, 0.04),
                shader="shaded",
                smooth=False,
            )
            self.addItem(plane)
            self.grid_items.append(plane)

    def _draw_axis(self):
        """Draws X, Y, Z reference arrows."""
        if not PG_OK:
            return
        for item in self.axis_items:
            self.removeItem(item)
        self.axis_items.clear()

        origin = np.array([[0, 0, 0]])
        for pts, color in [
            (np.array([[0, 0, 0], [12, 0, 0]]), (1.0, 0.3, 0.3, 0.8)),  # X red
            (np.array([[0, 0, 0], [0, 12, 0]]), (0.3, 1.0, 0.3, 0.8)),  # Y green
            (np.array([[0, 0, 0], [0, 0, 60]]), (0.3, 0.6, 1.0, 0.8)),  # Z blue (layer axis)
        ]:
            line = gl.GLLinePlotItem(pos=pts, color=color, width=2.5, antialias=True)
            self.addItem(line)
            self.axis_items.append(line)

    def _rebuild_scatter(self):
        """Removes old scatter item and creates a fresh one."""
        if not PG_OK:
            return
        if self.scatter_item:
            self.removeItem(self.scatter_item)
        self.scatter_item = gl.GLScatterPlotItem(
            pos=self.points,
            color=self.colors,
            size=self.sizes,
            pxMode=True
        )
        self.addItem(self.scatter_item)

    def _refresh_scatter(self):
        """Updates the existing scatter item colours/sizes."""
        if self.scatter_item:
            self.scatter_item.setData(
                pos=self.points,
                color=self.colors,
                size=self.sizes
            )

    def _clear_scene(self):
        """Removes all dynamic items from the scene."""
        self._pulse_timer.stop()
        self._target_index = None

        if self.scatter_item and PG_OK:
            self.removeItem(self.scatter_item)
            self.scatter_item = None

        for item in self.grid_items + self.axis_items:
            try:
                self.removeItem(item)
            except Exception:
                pass
        self.grid_items.clear()
        self.axis_items.clear()


# ---------------------------------------------------------------------------
# Public wrapper widget
# ---------------------------------------------------------------------------

class PointCloud3DWidget(QWidget):
    """
    Main wrapper widget exposing the 3D GL canvas to main_window.

    Signals
    -------
    pointSelected(int, tuple)          — Ctrl+Click picked a point
    pointMoved(int, tuple)             — Drag updated a point position
    targetNeuronVisualized(int, int)   — Target neuron has been highlighted
    """

    pointSelected = pyqtSignal(int, tuple)
    pointMoved = pyqtSignal(int, tuple)
    targetNeuronVisualized = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer_map: dict = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if PG_OK:
            self.gl_widget = CustomGLViewWidget()
            self.gl_widget.pointClicked.connect(self.pointSelected.emit)
            self.gl_widget.pointDragged.connect(self.pointMoved.emit)
            self.gl_widget.targetNeuronVisualized.connect(self.targetNeuronVisualized.emit)
            layout.addWidget(self.gl_widget)
        else:
            label = QLabel("⚠ pyqtgraph not installed.\nRun: pip install pyqtgraph PyOpenGL")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color:#ffaa00; font-size:14px; padding:20px;")
            layout.addWidget(label)
            self.gl_widget = None

    # ------------------------------------------------------------------
    # Public API called by main_window
    # ------------------------------------------------------------------

    def load_points(self, points: np.ndarray, ids: np.ndarray):
        """Loads flat PCA point cloud (embedding-space view)."""
        if self.gl_widget:
            self.gl_widget.set_flat_data(points, ids)

    def load_layered_model(self, points: np.ndarray,
                           labels: np.ndarray, layer_map: dict):
        """
        Loads the layered (layer-slab) view of the neural network.
        Each transformer layer rendered in its own Z-plane.
        """
        self._layer_map = layer_map
        if self.gl_widget:
            self.gl_widget.set_layered_data(points, labels, layer_map)

    def highlight_target_neuron(self, layer_idx: int, local_point_idx: int):
        """
        Called after Phase 3 locks the target.
        Shows a pulsing red marker on the offending neuron.
        """
        if self.gl_widget:
            self.gl_widget.highlight_target(
                layer_idx, local_point_idx, self._layer_map
            )

    def reset_target(self):
        """Clears any target pulsing."""
        if self.gl_widget:
            self.gl_widget.reset_target_highlight()

    def update_point_position(self, point_id: int, new_coords: tuple):
        """Programmatic update of a point's 3D position (from Dashboard)."""
        if self.gl_widget and self.gl_widget.points is not None:
            if self.gl_widget.ids is not None:
                idx = np.where(self.gl_widget.ids == point_id)[0]
                if len(idx) > 0:
                    self.gl_widget.points[idx[0]] = new_coords
                    self.gl_widget.scatter_item.setData(pos=self.gl_widget.points)
                    self.gl_widget.highlight_point(idx[0])
