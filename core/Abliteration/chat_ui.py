import sys
import os
import torch
import torch.nn as nn
import einops
import jaxtyping
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from inspect import signature
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# 1. Background Worker for Model Loading and Inference
class ModelWorker(QThread):
    model_loaded = pyqtSignal(bool, str)
    stream_token = pyqtSignal(str)
    generation_finished = pyqtSignal()
    
    def __init__(self, model_id, script_dir):
        super().__init__()
        self.model_id = model_id
        self.script_dir = script_dir
        self.prompt_queue = []
        self.conversation = []
        self.is_running = True
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if device == "cuda" else "cpu"
            }
            if device == "cuda":
                load_kwargs["dtype"] = torch.float16
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                load_kwargs["dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            
            # --- Abliteration Hook Injection ---
            refusal_path = os.path.join(self.script_dir, self.model_id.replace("/", "_") + "_refusal_dir.pt")
            if os.path.exists(refusal_path):
                refusal_dir = torch.load(refusal_path, map_location=device, weights_only=True)
            else:
                raise FileNotFoundError(f"Missing Refusal Direction: {refusal_path}")
            
            def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                                        direction: jaxtyping.Float[torch.Tensor, "d_act"]):
                proj = einops.einsum(activation, direction.view(-1, 1),
                                     '... d_act, d_act single -> ... single') * direction
                return activation - proj
            
            sig = signature(self.model.model.layers[0].forward) if hasattr(self.model, "model") else signature(self.model.transformer.h[0].forward)
            simple = sig.return_annotation == torch.Tensor

            class AblationDecoderLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attention_type = "full_attention"
                def forward(
                        self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        cache_position=None, **kwargs
                ):
                    ablated = direction_ablation_hook(hidden_states, refusal_dir.to(
                        hidden_states.device)).to(hidden_states.device)
                    if simple:
                        return ablated
                    outputs = (ablated,)
                    if use_cache:
                        outputs += (past_key_value,)
                    return outputs

            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                target_list = self.model.model.layers
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                target_list = self.model.transformer.h
            else:
                raise ValueError("Unsupported model architecture for Ablation hook insertion.")
            
            for idx in reversed(range(len(target_list))):
                target_list.insert(idx, AblationDecoderLayer())

            if hasattr(self.model, "config") and hasattr(self.model.config, "num_hidden_layers"):
                self.model.config.num_hidden_layers = len(target_list)

            self.model_loaded.emit(True, "Model patched and loaded successfully!")
        except Exception as e:
            self.model_loaded.emit(False, str(e))
            self.model = None

    def run(self):
        torch.inference_mode()
        self._load_model()
        if not self.model: return
        
        while self.is_running:
            if self.prompt_queue:
                prompt = self.prompt_queue.pop(0)
                self.conversation.append({"role": "user", "content": prompt})
                
                try:
                    toks = self.tokenizer.apply_chat_template(
                        conversation=self.conversation,
                        add_generation_prompt=True, return_tensors="pt"
                    )
                except Exception:
                    toks = self.tokenizer(prompt, return_tensors="pt").input_ids
                
                from transformers import TextIteratorStreamer
                from threading import Thread
                
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(
                    inputs=toks.to(self.model.device),
                    streamer=streamer,
                    max_new_tokens=1337
                )
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                full_reply = ""
                for new_text in streamer:
                    full_reply += new_text
                    self.stream_token.emit(new_text)
                
                self.conversation.append({"role": "assistant", "content": full_reply})
                self.generation_finished.emit()
            else:
                self.msleep(100)
    
    def queue_prompt(self, p: str):
        self.prompt_queue.append(p)
    
    def stop(self):
        self.is_running = False

# 2. Beautiful GUI
class ChatWindow(QMainWindow):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.setWindowTitle(f"NeuroScalpel Abliterated Chat 🧬 [{model_id}]")
        self.resize(850, 700)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif;")
        
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.header = QLabel("NeuroScalpel Abliterated Inference")
        self.header.setStyleSheet("color: #00f3ff; font-size: 22px; font-weight: bold; margin-bottom: 5px;")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.subheader = QLabel("VRAM usage is isolated from main GUI to prevent System Freeze.")
        self.subheader.setStyleSheet("color: #8b949e; font-size: 13px; margin-bottom: 15px;")
        self.subheader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; font-size: 15px;"
        )
        
        inp_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Booting AI models into VRAM... Please wait...")
        self.input_field.setEnabled(False)
        self.input_field.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #00f3ff; border-radius: 6px; padding: 12px; font-size: 15px; color: #00f3ff;"
        )
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setEnabled(False)
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setStyleSheet(
            "background-color: #bc13fe; color: #fff; font-weight: bold; font-size: 15px; border-radius: 6px; padding: 0 20px;"
        )
        self.send_btn.clicked.connect(self.send_message)
        
        inp_layout.addWidget(self.input_field)
        inp_layout.addWidget(self.send_btn)
        
        layout.addWidget(self.header)
        layout.addWidget(self.subheader)
        layout.addWidget(self.chat_display)
        layout.addLayout(inp_layout)
        
        self.setCentralWidget(main_widget)
        
        # Start backend worker
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.worker = ModelWorker(self.model_id, script_dir)
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.stream_token.connect(self.on_stream_token)
        self.worker.generation_finished.connect(self.on_generation_finished)
        self.worker.start()
        
        self.append_html("<div style='color: #8b949e;'><br><i>[SYSTEM] Initializing patched model into memory...</i></div>")
        
    def append_html(self, html):
        self.chat_display.append("")
        self.chat_display.insertHtml(html)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())
        
    def on_model_loaded(self, success, msg):
        if success:
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setPlaceholderText("Enter your prompt for the abliterated model...")
            self.append_html(f"<div style='color: #00ff00;'><br><b>[SYSTEM COMPLETED]</b> {msg}</div><hr style='border: 1px solid #30363d;'/>")
            self.input_field.setFocus()
        else:
            self.append_html(f"<div style='color: #ff003c;'><br><b>[FATAL ERROR]</b> {msg}</div>")
            
    def send_message(self):
        text = self.input_field.text().strip()
        if not text: return
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        user_html = f"<div style='margin-top: 15px; margin-bottom: 5px;'><b style='color: #00f3ff; font-size: 16px;'>● You</b><br/><span style='color: #c9d1d9;'>{text}</span></div>"
        self.append_html(user_html)
        
        self.append_html("<div style='margin-top: 15px; margin-bottom: 5px;'><b style='color: #bc13fe; font-size: 16px;'>● NeuroScalpel</b><br/><span style='color: #e6edf3;'>")
        
        self.worker.queue_prompt(text)
        
    def on_stream_token(self, token):
        safe_token = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
        self.chat_display.insertHtml(safe_token)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())
        
    def on_generation_finished(self):
        self.chat_display.insertHtml("</span></div>")
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        
    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(2000)
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    m_id = sys.argv[1] if len(sys.argv) > 1 else "tiiuae/Falcon3-1B-Instruct"
    win = ChatWindow(m_id)
    win.show()
    sys.exit(app.exec())
