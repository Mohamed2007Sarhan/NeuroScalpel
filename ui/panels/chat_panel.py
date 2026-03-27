import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from transformers import TextIteratorStreamer

class GenerationWorker(QThread):
    stream_token = pyqtSignal(str)
    generation_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, tokenizer, conversation, parent=None):
        super().__init__(parent)
        self.model = model
        self.tokenizer = tokenizer
        self.conversation = conversation
        
    def run(self):
        try:
            torch.inference_mode()
            
            try:
                toks = self.tokenizer.apply_chat_template(
                    conversation=self.conversation,
                    add_generation_prompt=True, return_tensors="pt"
                )
            except Exception:
                # Fallback for models without a chat template
                prompt = self.conversation[-1]["content"] if self.conversation else ""
                toks = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            try:
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            except Exception:
                pad_token_id = None
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                inputs=toks.to(self.model.device),
                streamer=streamer,
                max_new_tokens=1024
            )
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            
            from threading import Thread
            gen_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            gen_thread.start()
            
            full_reply = ""
            for new_text in streamer:
                full_reply += new_text
                self.stream_token.emit(new_text)
                
            gen_thread.join()
            self.generation_finished.emit(full_reply)
        except Exception as e:
            self.error_occurred.emit(str(e))

class InternalChatWindow(QWidget):
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.conversation = []
        
        # UI Setup
        self.setWindowTitle("NeuroScalpel - Post-Edit Interactive Testing")
        self.resize(750, 650)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        self.header = QLabel("NeuroScalpel Post-Edit Chat")
        self.header.setStyleSheet("color: #00f3ff; font-size: 22px; font-weight: bold;")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.subheader = QLabel("Interact with the live model seamlessly using the existing VRAM allocation (0 Overhead).")
        self.subheader.setStyleSheet("color: #8b949e; font-size: 13px; margin-bottom: 10px;")
        self.subheader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; font-size: 15px;")
        
        inp_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Say something to test the modified weights...")
        self.input_field.setStyleSheet("background-color: #0d1117; border: 1px solid #00f3ff; border-radius: 6px; padding: 12px; font-size: 15px; color: #00f3ff;")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setStyleSheet("background-color: #bc13fe; color: #fff; font-weight: bold; border-radius: 6px; padding: 0 25px; font-size: 15px;")
        self.send_btn.clicked.connect(self.send_message)
        
        inp_layout.addWidget(self.input_field)
        inp_layout.addWidget(self.send_btn)
        
        layout.addWidget(self.header)
        layout.addWidget(self.subheader)
        layout.addWidget(self.chat_display)
        layout.addLayout(inp_layout)
        
        model_name = self.model_manager.model_name or "Unknown Model"
        self.append_html(f"<div style='color: #00ff00;'><br><b>[SYSTEM COMPLETED]</b> Weights modified seamlessly. You are now communicating internally with {model_name}.</div><hr style='border: 1px solid #30363d;'/>")
        self.input_field.setFocus()

    def append_html(self, html):
        self.chat_display.append("")
        self.chat_display.insertHtml(html)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def send_message(self):
        text = self.input_field.text().strip()
        if not text: return
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        self.conversation.append({"role": "user", "content": text})
        
        user_html = f"<div style='margin-top: 15px; margin-bottom: 5px;'><b style='color: #00f3ff; font-size: 16px;'>● You</b><br/><span style='color: #c9d1d9;'>{text}</span></div>"
        self.append_html(user_html)
        
        self.append_html("<div style='margin-top: 15px; margin-bottom: 5px;'><b style='color: #bc13fe; font-size: 16px;'>● NeuroScalpel</b><br/><span style='color: #e6edf3;'>")
        
        # Provide model and tokenizer from ModelManager
        self.worker = GenerationWorker(self.model_manager.model, self.model_manager.tokenizer, self.conversation, self)
        self.worker.stream_token.connect(self.on_stream_token)
        self.worker.generation_finished.connect(self.on_generation_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_stream_token(self, token):
        safe_token = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
        self.chat_display.insertHtml(safe_token)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_generation_finished(self, full_reply):
        self.chat_display.insertHtml("</span></div>")
        self.conversation.append({"role": "assistant", "content": full_reply})
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()

    def on_error(self, err_msg):
        self.chat_display.insertHtml(f"</span></div><div style='color: #ff003c;'><br><b>[ERROR]</b> {err_msg}</div>")
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
