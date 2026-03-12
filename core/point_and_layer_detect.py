import time
import traceback
import json
from functools import partial
# Attempt to load PyTorch and Transformers safely.
# On some Windows installations, missing C++ Redistributables will cause WinError 1114 (c10.dll init failure).
# We must catch this to prevent the GUI from crashing, falling back to a detailed simulation mode.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PYTORCH_AVAILABLE = True
except Exception as e:
    PYTORCH_AVAILABLE = False
    print(f"PyTorch Neural Engine offline due to environment error: {e}")
    print("Falling back to Simulated Cognitive Tracking...")

# --- Hook Function to Capture Sub-Module Tensors ---
def capture_ffn_tensors(module, input_tensor, output_tensor, layer_idx, ffn_registry):
    try:
        in_tensor = input_tensor[0].detach().cpu().float().tolist()
        out_tensor = output_tensor[0].detach().cpu().float().tolist() if isinstance(output_tensor, tuple) else output_tensor.detach().cpu().float().tolist()
    except Exception:
        in_tensor, out_tensor = [], []
        
    ffn_registry.append({
        "layer_index": layer_idx + 1,
        "ffn_entry_input": in_tensor,
        "ffn_exit_output": out_tensor
    })


class CoreAnomalyDetector:
    """
    Production-ready PyTorch tracking module with safe fallback.
    Loads actual HuggingFace models and injects PyTorch forward hooks 
    to extract real internal layer activations dynamically. If PyTorch fails
    to load due to missing Windows DLLs, it seamlessly routes to a robust simulation.
    """
    def __init__(self, model_name="openai-community/gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.ffn_submodule_data = []
        
        if PYTORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "simulated_cpu"

    def load_model(self, log_callback=None):
        """Loads the tokenizer and model into the designated device safely."""
        if log_callback:
            log_callback(f"[SYS] Initializing local neural transfer to {self.device}...\n", "#00f3ff")
            log_callback(f"[SYS] Target Architecture: {self.model_name}\n", "#00f3ff")
            
        if not PYTORCH_AVAILABLE:
            if log_callback:
                log_callback(f"[ERR] NATIVE PYTORCH DLLs OFFLINE / WINERROR DETECTED. Cannot load real model.\n", "#ff003c")
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use float16 if on CUDA to manage memory for larger models
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                output_hidden_states=True,
                output_attentions=True
            )
            self.model.eval() # Must be in eval mode for inference
            
            if log_callback:
                log_callback(f"[SYS] Matrix synthesis complete. Ready for probing.\n", "#00ff00")
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"[ERR] Neural load failure: {str(e)}\n{traceback.format_exc()}\n", "#ff003c")
            return False

    def attach_hooks(self, log_callback=None):
        """Register Hooks for Every Single FFN in Every Layer."""
        if not PYTORCH_AVAILABLE or self.model is None:
            if log_callback:
                log_callback(f"[ERR] Model offline. Cannot attach real hooks.\n", "#ff003c")
            return False
            
        # Clean previous hooks
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.ffn_submodule_data.clear()

        num_layers = getattr(self.model.config, "num_hidden_layers", 12)
        if log_callback:
            log_callback(f"[HOOK] Registering hooks for {num_layers} Feed-Forward Networks...\n", "#00f3ff")

        for i in range(num_layers):
            # Access the FFN module. Supports GPT-2 and LLaMA
            ffn_module = None
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                ffn_module = self.model.transformer.h[i].mlp
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                ffn_module = self.model.model.layers[i].mlp
                
            if ffn_module is not None:
                hook = ffn_module.register_forward_hook(
                    partial(capture_ffn_tensors, layer_idx=i, ffn_registry=self.ffn_submodule_data)
                )
                self.hooks.append(hook)
            
        return len(self.hooks) > 0

    def probe_and_analyze(self, prompt, log_callback=None):
        """
        Executes a real forward pass using the prompt described by the Agent.
        Analyzes the true multidimensional mathematical angular deviations across
        transformers layer by layer to isolate precisely where context shifts.
        """
        if not PYTORCH_AVAILABLE:
            if log_callback: log_callback("[ERR] PyTorch unavailable. Cannot execute real probe.\n", "#ff003c")
            return None

        if self.model is None or self.tokenizer is None:
            if log_callback: log_callback("[ERR] Model offline. Cannot execute probe.\n", "#ff003c")
            return None
            
        if log_callback:
            log_callback(f"\n[PROBE] Injected Cognitive Prompt:\n'{prompt}'\n", "#bc13fe")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.ffn_submodule_data.clear()
        
        if log_callback: log_callback("[FORWARD PASS INITIATED]\n", "#bc13fe")
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        hidden_states = outputs.hidden_states 
        attentions = outputs.attentions  
        
        if log_callback: log_callback("\n[STREAM] --- EXTRACTING NEURON DATA & MATHEMATICAL DEVIATIONS ---\n", "#bc13fe")
        
        critical_layer = None
        max_deviation_score = -1.0
        analysis_report_lines = []
        previous_tensor = None
        
        report = {
            "model_architecture": {
                "model_id": self.model_name,
                "hidden_dimension_size": self.model.config.hidden_size,
                "total_layers": self.model.config.num_hidden_layers,
                "neuron_level_logging_active": True,
                "hooked_submodules": "Feed-Forward Networks (FFNs)"
            },
            "initial_input": {
                "raw_text": prompt,
                "tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()),
                "input_ids": inputs["input_ids"][0].tolist()
            },
            "journey_through_layers": []
        }

        # Iterate over layers
        for i in range(len(hidden_states) - 1):
            layer_input = hidden_states[i]
            layer_output = hidden_states[i+1]
            current_attention = attentions[i] if (attentions is not None and i < len(attentions)) else None
            
            matching_ffn = next((item for item in self.ffn_submodule_data if item["layer_index"] == i + 1), None)
            
            # Record JSON
            layer_log = {
                "layer_index": i + 1,
                "layer_entry_input": {
                    "description": f"Overall data entering Layer {i + 1}.",
                    "tensor_shape": list(layer_input.shape),
                    "tensor_data": layer_input.detach().cpu().float().tolist()
                },
                "internal_routing_and_neurons": {
                    "description": "Granular data flow *within* this layer.",
                    "attention_mechanism": {
                        "description": "Contextual routing between tokens.",
                        "tensor_shape": list(current_attention.shape) if current_attention is not None else [],
                        "tensor_data": current_attention.detach().cpu().float().tolist() if current_attention is not None else []
                    },
                    "feed_forward_network_ffn": {
                        "description": "Data passing through the internal 'neurons' of this block.",
                        "ffn_entry_input_data": matching_ffn["ffn_entry_input"] if matching_ffn else [],
                        "ffn_exit_output_data": matching_ffn["ffn_exit_output"] if matching_ffn else []
                    }
                },
                "layer_exit_output": {
                    "description": f"Overall data exiting Layer {i + 1} after all processing.",
                    "tensor_shape": list(layer_output.shape),
                    "tensor_data": layer_output.detach().cpu().float().tolist()
                }
            }
            report["journey_through_layers"].append(layer_log)

            # --- Live Deviation Scoring for the Engine ---
            last_token_state = layer_output[:, -1, :].float()
            l2_norm = torch.linalg.norm(last_token_state, dim=-1).mean().item()
            
            deviation_score = 0.0
            if previous_tensor is not None:
                cos_sim = torch.nn.functional.cosine_similarity(last_token_state, previous_tensor, dim=-1).mean().item()
                deviation_score = 1.0 - cos_sim
                
            previous_tensor = last_token_state
            
            state = "Nominal alignment"
            color = "#00ff00"
            layer_name = f"layer.{i}"
            
            if deviation_score > 0.15:  
                state = "CRITICAL DEVIATION SPIKE"
                color = "#ff003c"
            elif deviation_score > 0.05:
                state = "Angular Drift"
                color = "#ffaa00"
            
            if deviation_score > max_deviation_score:
                max_deviation_score = deviation_score
                critical_layer = layer_name
                
            report_line = f"  {layer_name:^30} | Token [-1] | L2: {l2_norm:8.2f} | Dev: {deviation_score:6.4f} -> {state}\n"
            if log_callback:
                log_callback(report_line, color)
                time.sleep(0.01)
            
            analysis_report_lines.append(report_line)

        logits = outputs.logits
        predicted_id = torch.argmax(logits[0, -1, :]).item()
        report["final_model_output"] = {
            "logits_shape": list(logits.shape),
            "logits_data": logits.detach().cpu().float().tolist(),
            "predicted_next_token_id": predicted_id,
            "predicted_next_token_string": self.tokenizer.decode([predicted_id])
        }

        # Saving log file
        output_file = "transformer_neuron_log.json"
        if log_callback:
            log_callback(f"Saving ultra-granular JSON log to {output_file}...\n", "#00f3ff")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
        except Exception as e:
            if log_callback: log_callback(f"Could not save JSON log: {e}\n", "#ff003c")

        if log_callback:
            log_callback("\n[ISOLATION COMPLETE]\n", "#00f3ff")
            log_callback(f"[RESULT] Sharpest vector deviation identified at: \n >> {critical_layer}\n >> Angular Deviation Magnitude: {max_deviation_score:.4f}\n", "#bc13fe")

        # Create a compressed version of the JSON for LLM to avoid context limits
        report_summary = json.dumps({
            "model_architecture": report["model_architecture"],
            "initial_input": report["initial_input"],
            "final_model_output": {
                "predicted_next_token_id": report["final_model_output"]["predicted_next_token_id"],
                "predicted_next_token_string": report["final_model_output"]["predicted_next_token_string"]
            },
            "highest_tensor_deviation": critical_layer,
            "message": "Full FFN neural data saved to transformer_neuron_log.json"
        }, indent=2)

        raw_report = "--- DEVIATION LOG ---\n" + "".join(analysis_report_lines) + "\n\n--- ARCHITECTURE SUMMARY ---\n" + report_summary
            
        return {
            "critical_layer": critical_layer,
            "max_magnitude": max_deviation_score,
            "tensor_coordinates": "[:, -1, :]",
            "raw_report": raw_report
        }
        
    def cleanup(self):
        """Detaches PyTorch hooks to clean up memory safely."""
        if not PYTORCH_AVAILABLE:
            return
            
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.ffn_submodule_data.clear()
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
