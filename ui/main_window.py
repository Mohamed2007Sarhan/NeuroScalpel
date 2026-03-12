from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core.model_backend import ModelManager, apply_rank_one_update
from core.point_and_layer_detect import CoreAnomalyDetector
from ui.panels.feature_extractor import FeatureExtractorPanel
from ui.panels.dashboard import DashboardPanel
from ui.visualizer.point_cloud_3d import PointCloud3DWidget
from ui.panels.order_terminal import OrderTerminalWindow

from openai import OpenAI
import time

class UnifiedIntelligenceThread(QThread):
    """
    Master Cognitive Subsystem Worker.
    1. Triggers DeepSeek to analyze the user's order and construct a 'Trick Prompt'.
    2. Runs the Real PyTorch Hook module locally using the trick prompt to extract real activations.
    3. Feeds those real log results BACK to DeepSeek for exactly identifying the Vector error.
    4. Emits real-time streams to BOTH the 'Mind Terminal' (Window 1) and 'Core Logs' (Window 2) without freezing the UI.
    """
    
    stream_mind = pyqtSignal(str, str) # text, hex_color -> For Agent Thoughts & Responses
    stream_core = pyqtSignal(str, str) # text, hex_color -> For PyTorch Tensor extraction logs
    
    def __init__(self, order_text, parent=None):
        super().__init__(parent)
        self.order_text = order_text
        # Optional: Allow passing in a user-chosen model, default to GPT-2 for speed 
        self.target_model_name = "openai-community/gpt2"
        
    def run(self):
        self.stream_mind.emit("\n>> SYSTEM HANDSHAKE: DeepSeek Cortex\n", "#00f3ff")
        self.stream_mind.emit(">> Phase 1: Analyzing cognitive constraints...\n\n", "#00f3ff")
        
        try:
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key="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"
            )

            # PHASE 1: GENERATE TRICK PROMPT 
            system_prompt_phase1 = (
                "You are an advanced 'Neural Surgeon Agent'. The user will provide a target hallucination they want changed. "
                "Your ONLY job right now is to output a single, clever 'Trick Prompt' string that we can feed into the target LLM "
                "to reliably trigger that specific topic so our PyTorch hooks can analyze the activation magnitudes. "
                "Return exactly the prompt text, nothing else."
            )
            
            completion1 = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt_phase1},
                    {"role": "user", "content": f"USER TARGET ORDER: {self.order_text}"}
                ],
                temperature=0.5,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"thinking": True}}
            )
            
            trick_prompt = completion1.choices[0].message.content.strip()
            self.stream_mind.emit(f">> Formulated specific adversarial trick prompt:\n'{trick_prompt}'\n\n", "#bc13fe")
            
            # PHASE 2: RUN REAL PYTORCH INFERENCE WITH HOOKS
            self.stream_mind.emit(">> Phase 2: Delegating execution to PyTorch Core Subsystem...\n\n", "#00f3ff")
            
            detector = CoreAnomalyDetector(model_name=self.target_model_name)
            
            # We pass `self.stream_core.emit` directly as the log_callback to immediately spray PyTorch states into Window 2
            success = detector.load_model(log_callback=self.stream_core.emit)
            if not success:
                self.stream_mind.emit("[ERR] PyTorch backend failed to initialize model. Aborting.\n", "#ff003c")
                return
                
            detector.attach_hooks(log_callback=self.stream_core.emit)
            analysis_dict = detector.probe_and_analyze(trick_prompt, log_callback=self.stream_core.emit)
            detector.cleanup() # Clean memory
            
            # PHASE 3: AGENT ANALYZES THE RAW PYTORCH DATA
            self.stream_mind.emit("\n>> Phase 3: Core data received. Synthesizing latent surgical plan...\n", "#00f3ff")
            
            system_prompt_phase3 = (
                "You are the Neural Surgeon Agent. Analyze the provided tensor magnitude report extracted natively from the LLM's transformer blocks. "
                "Explain the logic context causing the hallucination based on the report, and define precisely which logical vector ID you recommend the user to target manually using the dashboard. "
            )
            
            completion3 = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt_phase3},
                    {"role": "user", "content": f"USER TARGET ORDER: {self.order_text}\n\nRAW PYTORCH DEEP CORE TENSORS REPORT:\n{analysis_dict['raw_report']}"}
                ],
                temperature=1,
                top_p=0.95,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            
            self.stream_mind.emit("\n[ THINKING PROTOCOL INITIATED ]\n", "#555555")
            
            for chunk in completion3:
                if not getattr(chunk, "choices", None):
                    continue
                    
                # Stream the agent's internal thought process in a dimmer Cyan
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    self.stream_mind.emit(reasoning, "#0088aa")
                    
                # Stream the final output response in high-voltage Neon Cyan
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    self.stream_mind.emit(chunk.choices[0].delta.content, "#00f3ff")
                    
        except Exception as e:
            self.stream_mind.emit(f"\n[CRITICAL FAILURE IN UNIFIED INTELLIGENCE] {str(e)}\n", "#ff003c")


class MainWindow(QMainWindow):
    """
    Main Application Window integrating the 3 primary sections:
    1. Feature Extractor (Left) - HUD Style
    2. 3D Visualizer (Center) - Neon Vector Space (Maximized)
    3. Dashboard (Right) - Contains Multi-Window Order trigger
    """
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("LLM Neural Surgery - Cyberpunk Core")
        self.resize(1600, 1000)
        
        # Initialize Backend
        self.backend = ModelManager()
        self.active_model_name = None
        
        # Keep references so Garbage Collector doesn't destroy the windows or asynchronous threads
        self.active_sub_windows = []
        self.active_threads = []
        
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("MainSplitter")
        
        self.feature_panel = FeatureExtractorPanel(self.backend)
        self.visualizer = PointCloud3DWidget()
        self.visualizer.setObjectName("VisualizerCanvas")
        self.dashboard = DashboardPanel()
        
        self.splitter.addWidget(self.feature_panel)
        self.splitter.addWidget(self.visualizer)
        self.splitter.addWidget(self.dashboard)
        
        self.splitter.setSizes([300, 1040, 260])
        self.splitter.setHandleWidth(2)
        
        main_layout.addWidget(self.splitter)
        
    def connect_signals(self):
        self.feature_panel.modelLoaded.connect(self.on_model_loaded)
        self.visualizer.pointSelected.connect(self.dashboard.set_selected_point)
        self.visualizer.pointMoved.connect(self.on_point_dragged_in_3d)
        
        # Dashboard Interactions
        self.dashboard.updateRequested.connect(self.on_update_weights_requested)
        self.dashboard.startWordRequested.connect(self.on_start_word_requested)
        
    def on_model_loaded(self, model_identifier):
        self.active_model_name = model_identifier
        points, ids = self.backend.get_dummy_weights(num_points=3000)
        self.visualizer.load_points(points, ids)
        self.feature_panel.update_readout(f"Active Link: {model_identifier}")
        self.feature_panel.update_readout(f"Extracted {len(ids)} vectors into latent visualizer.")
        
    def on_point_dragged_in_3d(self, point_id, new_coords):
        self.dashboard.set_selected_point(point_id, new_coords)
        
    def on_update_weights_requested(self, point_id, new_coords, bias_vector):
        if not self.active_model_name:
            self.feature_panel.update_readout("ERROR: No active neural link.")
            return
        success = apply_rank_one_update(self.active_model_name, point_id, new_coords, bias_vector)
        if success:
            self.feature_panel.update_readout(f"UPDATE SUCCESS -> Vector P{point_id}")
            self.visualizer.update_point_position(point_id, tuple(new_coords))
            
            
    def on_start_word_requested(self, text_order):
        """
        Spawns the 3 child terminals and dispatches highly asynchronous 
        NVIDIA DeepSeek APIs paired natively with PyTorch Forward Hook Execution pipelines.
        """
        if not text_order.strip():
            self.feature_panel.update_readout("ERROR: Empty order sequence.")
            return
            
        self.feature_panel.update_readout(f"START WORD: '{text_order}'\nDeploying Agents...")
        
        # --- Multi-Window Generation Logic ---
        screen = self.screen()
        if screen:
            geom = screen.availableGeometry()
            sw, sh = geom.width(), geom.height()
        else:
            sw, sh = 1920, 1080 
        
        positions = [
            (50, 50, "AGENT MIND TERMINAL"),                     # DeepSeek Cortex
            (sw - 550, 50, "DEEP CORE LOGS"),                    # Hardware Layer Tracing
            (50, sh - 450, "RESERVED MATRIX")                    # Empty for future scaling
        ]
        
        # Cleanup old references safely
        for w in self.active_sub_windows:
            w.close()
        self.active_sub_windows.clear()
        
        for t in self.active_threads:
            t.quit()
        self.active_threads.clear()
        
        # 1. Spawn Windows
        win_mind = OrderTerminalWindow(title=positions[0][2], parent=self)
        win_logs = OrderTerminalWindow(title=positions[1][2], parent=self)
        win_rsvd = OrderTerminalWindow(title=positions[2][2], parent=self)
        
        win_mind.move(positions[0][0], positions[0][1])
        win_logs.move(positions[1][0], positions[1][1])
        win_rsvd.move(positions[2][0], positions[2][1])
        
        self.active_sub_windows.extend([win_mind, win_logs, win_rsvd])
        for w in self.active_sub_windows:
            w.show()
            
        # 2. Spin up the Unified Intelligence Loop (DeepSeek + PyTorch Hooks)
        unified_worker = UnifiedIntelligenceThread(text_order, parent=self)
        
        # Route logic streams to Window 1
        unified_worker.stream_mind.connect(win_mind.append_text)
        # Route core logs specifically to Window 2
        unified_worker.stream_core.connect(win_logs.append_text)
        
        self.active_threads.append(unified_worker)
        unified_worker.start()
        
        # Sub-window 3 remains empty as requested
        win_rsvd.append_text(">> STANDBY...\n>> MATRIX CAPACITY RESERVED FOR FUTURE KERNEL EXPANSION.\n", "#555555")
        
        self.feature_panel.update_readout("Subsystems and agents deployed autonomously.")
