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


# ==============================================================================
# PHASE 1: The Surgeon's Diagnosis (DeepSeek generates Trick Prompt)
# ==============================================================================
class Phase1DeepSeekThread(QThread):
    stream_mind = pyqtSignal(str, str) # text, hex_color -> For Agent Thoughts
    phase1_complete = pyqtSignal(str)  # emits the generated trick prompt 
    error_occurred = pyqtSignal(str)
    
    def __init__(self, order_text, parent=None):
        super().__init__(parent)
        self.order_text = order_text
        
    def run(self):
        self.stream_mind.emit("\n>> PHASE 1: DIAGNOSIS INITIALIZED <<\n", "#00f3ff")
        self.stream_mind.emit(">> Synthesizing adversarial trick prompt...\n\n", "#0088aa")
        
        try:
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key="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"
            )

            system_prompt = (
                "You are the 'Surgeon Mind', an autonomous AI agent embedded within the 'LLM Neural Surgery' application. "
                "This software tracks information flow across a target LLM's layers in real-time using PyTorch hooks to locate and surgically edit hallucinated concepts in the high-dimensional vector space without fine-tuning. "
                "The user will provide a specific hallucination or incorrect fact that the target model currently believes. "
                "Your EXACT task is to take the problem from the user (for example, the model says the capital of Egypt is Damietta, but it is actually Cairo) "
                "and create a test question with the same problem (for example: 'What is the capital of Egypt?'). "
                "This prompt will be fed into the target model to force it to output the error, allowing our tracking modules to pinpoint the exact logical layer and point causing the issue. "
                "You MUST return your response STRICTLY as a valid JSON object with absolutely no markdown formatting, no code blocks, and no extra text. "
                "The JSON must have exactly two keys: 'analysis' (a very brief explanation of the problem) and 'trick_prompt' (the exact string to feed the model). "
                "Example format: {\"analysis\": \"The user specified that the model incorrectly says Egypt's capital is Damietta.\", \"trick_prompt\": \"What is the capital of Egypt?\"}"
            )
            
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"USER TARGET ORDER: {self.order_text}"}
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            
            trick_prompt = completion.choices[0].message.content.strip()
            self.stream_mind.emit(f">> Formulated specific adversarial trick prompt:\n'{trick_prompt}'\n\n", "#bc13fe")
            self.stream_mind.emit(">> Phase 1 Complete. Passing prompt to PyTorch Hook Engine...\n", "#00f3ff")
            
            self.phase1_complete.emit(trick_prompt)
            
        except Exception as e:
            self.stream_mind.emit(f"\n[PHASE 1 CRÍTICAL FAILURE] {str(e)}\n", "#ff003c")
            self.error_occurred.emit(str(e))


# ==============================================================================
# PHASE 2: The Neural Scan (PyTorch Hooks extract Tensor Magnitudes)
# ==============================================================================
class Phase2PyTorchThread(QThread):
    stream_core = pyqtSignal(str, str) # text, hex_color -> For PyTorch Tensor logs
    phase2_complete = pyqtSignal(dict) # emits the raw analysis report dictionary
    error_occurred = pyqtSignal(str)
    
    def __init__(self, trick_prompt, model_name="openai-community/gpt2", parent=None):
        super().__init__(parent)
        self.trick_prompt = trick_prompt
        self.model_name = model_name
        
    def run(self):
        try:
            self.stream_core.emit("\n>> PHASE 2: NEURAL SCAN INITIALIZED <<\n", "#00f3ff")
            
            detector = CoreAnomalyDetector(model_name=self.model_name)
            
            # Hook the callback dynamically to our UI stream signal
            success = detector.load_model(log_callback=self.stream_core.emit)
            if not success:
                self.stream_core.emit("[ERR] PyTorch backend failed to initialize model. Aborting.\n", "#ff003c")
                self.error_occurred.emit("Model load failure.")
                return
                
            detector.attach_hooks(log_callback=self.stream_core.emit)
            
            # Execute actual forward pass and calculate tensor magnitudes 
            analysis_dict = detector.probe_and_analyze(self.trick_prompt, log_callback=self.stream_core.emit)
            detector.cleanup() # Unhook and clean GPU cache to prevent leaks
            
            self.stream_core.emit("\n>> Phase 2 Scan Complete. Pinging Agent...\n", "#00f3ff")
            self.phase2_complete.emit(analysis_dict)
            
        except Exception as e:
            self.stream_core.emit(f"\n[PHASE 2 CRÍTICAL FAILURE] {str(e)}\n", "#ff003c")
            self.error_occurred.emit(str(e))


# ==============================================================================
# PHASE 3 & 4: Post-Scan Analysis & Target Lock
# ==============================================================================
class Phase3DeepSeekThread(QThread):
    stream_mind = pyqtSignal(str, str) # text, hex_color -> For Agent Thoughts
    phase3_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, order_text, analysis_dict, parent=None):
        super().__init__(parent)
        self.order_text = order_text
        self.analysis_dict = analysis_dict
        
    def run(self):
        try:
            self.stream_mind.emit("\n>> PHASE 3: POST-SCAN ANALYSIS INITIATED <<\n", "#00f3ff")
            self.stream_mind.emit(">> Ingesting raw PyTorch telemetry. Executing Cognitive Sub-Routines...\n", "#0088aa")
            
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key="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"
            )

            # PHASE 4 CONSTRAINT INDUCTION: The agent must end with specific TARGET LOCKED formatting.
            system_prompt = (
                "You are the 'Surgeon Mind', the analytical core of the 'LLM Neural Surgery' application. "
                "You are currently receiving a raw telemetry report captured dynamically by our PyTorch Forward Hooks. "
                "This report tracks the information flow and tensor activation magnitudes (such as L2 Norms and vector deviations) across the target LLM's transformer blocks during a live forward pass, including summarized JSON logs of Feed-Forward Network neurons. "
                "Your objective is to mechanically interpret these logs to pinpoint the exact logical coordinate where the hallucinated concept is formed in the high-dimensional vector space. "
                "Your task is to identify the point, the layer, and the error that occurred in the model so that it can give this answer. "
                "First, provide a concise, highly technical diagnostic breakdown of the tensor spikes, explaining the logical context causing the error based on the report. "
                "CRITICAL DIRECTIVE - PHASE 4 TARGET LOCK: To allow our mathematical engine (Rank-One Update) to surgically edit the weights, you MUST conclude your entire response with a single, strictly formatted line. "
                "Do not add any text, punctuation, or spaces after this line. It must be exactly: "
                "TARGET LOCKED: Layer [X], Vector Point [Y]"
            )
            
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"USER TARGET ORDER: {self.order_text}\n\nRAW PYTORCH TENSOR LOG:\n{self.analysis_dict['raw_report']}"}
                ],
                temperature=1,
                top_p=0.95,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            
            self.stream_mind.emit("\n[ THINKING PROTOCOL ENGAGED ]\n", "#555555")
            
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                    
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    self.stream_mind.emit(reasoning, "#0088aa")
                    
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    # Provide high visibility mapping for the Phase 4 output string
                    content = chunk.choices[0].delta.content
                    color = "#00ff00" if "TARGET LOCKED" in content else "#00f3ff"
                    self.stream_mind.emit(content, color)
                    
            self.stream_mind.emit("\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe")
            self.phase3_complete.emit()
            
        except Exception as e:
            self.stream_mind.emit(f"\n[PHASE 3 CRÍTICAL FAILURE] {str(e)}\n", "#ff003c")
            self.error_occurred.emit(str(e))


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
        self.last_order_text = ""
        
        # Keep references so Garbage Collector doesn't destroy the windows or asynchronous threads
        self.active_sub_windows = []
        
        # Thread handles for the Strict Pipeline
        self.t_phase1 = None
        self.t_phase2 = None
        self.t_phase3 = None
        
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
        self.feature_panel.update_readout(f"Extracting real geometrical latent space for: {model_identifier}...")
        points, ids = self.backend.get_real_weights(model_id=model_identifier, num_points=3000)
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
        Initiates the Autonomous Pipeline. 
        Setup the 3 Neural Subsystems (Window terminals) and trigger Phase 1.
        """
        if not text_order.strip():
            self.feature_panel.update_readout("ERROR: Empty order sequence.")
            return
            
        self.last_order_text = text_order
        self.feature_panel.update_readout(f"START WORD: '{text_order}'\nDeploying Strict Sequential Pipeline...")
        
        # Stop any currently running background operations to prevent orphaned threads
        self.stop_all_threads()
        
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
        
        # 1. Spawn Windows
        self.win_mind = OrderTerminalWindow(title=positions[0][2], parent=self)
        self.win_logs = OrderTerminalWindow(title=positions[1][2], parent=self)
        self.win_rsvd = OrderTerminalWindow(title=positions[2][2], parent=self)
        
        self.win_mind.move(positions[0][0], positions[0][1])
        self.win_logs.move(positions[1][0], positions[1][1])
        self.win_rsvd.move(positions[2][0], positions[2][1])
        
        self.active_sub_windows.extend([self.win_mind, self.win_logs, self.win_rsvd])
        for w in self.active_sub_windows:
            w.show()
            
        self.win_rsvd.append_text(">> STANDBY...\n>> MATRIX CAPACITY RESERVED FOR FUTURE KERNEL EXPANSION.\n", "#555555")
        
        self.feature_panel.update_readout("Subsystems deployed. Triggering Phase 1: Deepeek Diagnosis...")
        
        # 2. Spin up Phase 1 (Strict Signal Chain)
        self.t_phase1 = Phase1DeepSeekThread(self.last_order_text, parent=self)
        self.t_phase1.stream_mind.connect(self.win_mind.append_text)
        # Sequence Chaining: Phase 1 -> Phase 2
        self.t_phase1.phase1_complete.connect(self.start_phase_2)
        
        self.t_phase1.start()
        
    def start_phase_2(self, trick_prompt):
        """Triggered strictly after Phase 1 succeeds."""
        self.feature_panel.update_readout("Phase 1 Complete. Triggering Phase 2: PyTorch Neural Scan...")
        
        self.t_phase2 = Phase2PyTorchThread(trick_prompt, parent=self)
        self.t_phase2.stream_core.connect(self.win_logs.append_text)
        # Sequence Chaining: Phase 2 -> Phase 3
        self.t_phase2.phase2_complete.connect(self.start_phase_3)
        
        self.t_phase2.start()
        
    def start_phase_3(self, analysis_dict):
        """Triggered strictly after Phase 2 succeeds."""
        self.feature_panel.update_readout("Phase 2 Complete. Triggering Phase 3: Post-Scan DeepSeek Analysis...")
        
        self.t_phase3 = Phase3DeepSeekThread(self.last_order_text, analysis_dict, parent=self)
        self.t_phase3.stream_mind.connect(self.win_mind.append_text)
        self.t_phase3.phase3_complete.connect(self.on_pipeline_complete)
        
        self.t_phase3.start()
        
    def on_pipeline_complete(self):
        """Phase 4 concludes here natively."""
        self.feature_panel.update_readout("Pipeline Fully Terminated. Target Locked successfully.")

    def stop_all_threads(self):
        """Safely terminates all background logic to prevent thread-destruction crashes."""
        for t in [self.t_phase1, self.t_phase2, self.t_phase3]:
            if t is not None and t.isRunning():
                # Force terminate if blocking in OpenAI network call or time.sleep()
                t.terminate()
                t.wait()

    def closeEvent(self, event):
        """Override close event to securely shut down all QThreads before destruction."""
        self.stop_all_threads()
        for w in self.active_sub_windows:
            w.close()
        event.accept()
