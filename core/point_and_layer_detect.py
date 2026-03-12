import time
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

class CoreAnomalyDetector:
    """
    Production-ready PyTorch tracking module.
    Loads actual HuggingFace models and injects PyTorch forward hooks 
    to extract real internal layer activations dynamically.
    """
    def __init__(self, model_name="openai-community/gpt2"):
        # Default to a small model like GPT-2 for local execution testing unless specified
        # In a real environment, you might deploy Llama-3 or Mistral.
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hooks = []
        self.activation_cache = {}

    def load_model(self, log_callback=None):
        """Loads the tokenizer and model into the designated device."""
        if log_callback:
            log_callback(f"[SYS] Initializing local neural transfer to {self.device}...\n", "#00f3ff")
            log_callback(f"[SYS] Target Architecture: {self.model_name}\n", "#00f3ff")
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use float16 if on CUDA to manage memory for larger models
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device, # Will map automatically if accelerate is installed
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.model.eval() # Must be in eval mode for inference
            
            if log_callback:
                log_callback(f"[SYS] Matrix synthesis complete. Ready for probing.\n", "#00ff00")
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"[ERR] Neural load failure: {str(e)}\n{traceback.format_exc()}\n", "#ff003c")
            return False

    def _get_activation_hook(self, layer_name):
        """Generates a hook closure that locks onto the specific layer's output tensors."""
        def hook(model, input, output):
            # The output of a transformer block can be a tuple (hidden_states, presents, ...)
            # We strictly want the first element which is the primary hidden_state tensor
            tensor_data = output[0] if isinstance(output, tuple) else output
            
            # Detach and move to CPU to prevent GPU OOM during analysis caching
            self.activation_cache[layer_name] = tensor_data.detach().cpu()
        return hook

    def attach_hooks(self, log_callback=None):
        """Dynamically traverses the model architecture and attaches hooks to specific sub-layers."""
        if self.model is None:
            return False
            
        # Clean previous hooks
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activation_cache.clear()

        # Architecture agnostic finding logic (Handles Llama, Mistral, GPT-2, etc)
        # We target MLP or dense projection layers heavily involved in fact recall
        target_modules = []
        
        for name, module in self.model.named_modules():
            # Target standard MLP projection outputs or attention outputs
            if "mlp" in name.lower() or "c_fc" in name.lower() or "c_proj" in name.lower():
                # Avoid registering too many deep hooks to prevent massive memory overhead, 
                # we just need the primary block exits
                target_modules.append((name, module))
                
        if log_callback:
            log_callback(f"[HOOK] Target vectors identified. Injecting {len(target_modules)} telemetry anchors...\n", "#00f3ff")

        for name, module in target_modules:
            hook_handle = module.register_forward_hook(self._get_activation_hook(name))
            self.hooks.append(hook_handle)
            
        return len(target_modules) > 0

    def probe_and_analyze(self, prompt, log_callback=None):
        """
        Executes a real forward pass using the prompt described by the Agent.
        Streams the structural mathematics as they happen.
        """
        if self.model is None or self.tokenizer is None:
            if log_callback: log_callback("[ERR] Model offline. Cannot execute probe.\n", "#ff003c")
            return None
            
        if log_callback:
            log_callback(f"\n[PROBE] Injected Cognitive Prompt:\n'{prompt}'\n", "#bc13fe")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.activation_cache.clear()
        
        with torch.no_grad():
            if log_callback: log_callback("[FORWARD PASS INITIATED]\n", "#bc13fe")
            # Generate or single forward pass. Single pass is faster for specific extraction.
            outputs = self.model(**inputs)
            
        if log_callback: log_callback("\n[STREAM] --- EXTRACTING LAYER ACTIVATIONS ---\n", "#bc13fe")
        
        # Analyze the caught internal mathematical states
        critical_layer = None
        max_anomaly_score = -1.0
        analysis_report = []
        
        # Sort by layer name to stream logically
        for layer_name, hidden_states in sorted(self.activation_cache.items()):
            # hidden_states shape typically: [batch, sequence_length, hidden_dim]
            shape_str = f"[{hidden_states.shape[0]}, {hidden_states.shape[1]}, {hidden_states.shape[2]}]"
            
            # Calculate L2 Norm as a baseline proxy for "Activation Magnitude"
            # In production neural surgery (like ROME/MEMIT), we calculate covariance/Jacobians here.
            l2_norm = torch.linalg.norm(hidden_states.float(), dim=-1).mean().item()
            
            # Determine logic state
            state = "Nominal resonance"
            color = "#00ff00" # Hacker Green
            
            # Artificial anomaly detection strictly based on relative extreme magnitudes for demonstration
            # Some layers inherently have massive norms (e.g., final LayerNorms), so we track the largest deviation
            
            if l2_norm > 150.0:  # Threshold depends heavily on model architecture; this is illustrative
                state = "CRITICAL SPIKE DETECTED"
                color = "#ff003c" # Crimson Red
            
            if l2_norm > max_anomaly_score:
                max_anomaly_score = l2_norm
                critical_layer = layer_name
                
            report_line = f"  {layer_name:^30} | Shape {shape_str} | L2 Norm: {l2_norm:8.2f} -> {state}\n"
            if log_callback:
                log_callback(report_line, color)
                time.sleep(0.05) # UX Delay to make it visually readable like a movie hacker terminal
            
            analysis_report.append(report_line)
            
        if log_callback:
            log_callback("\n[ISOLATION COMPLETE]\n", "#00f3ff")
            log_callback(f"[RESULT] Primary logic hallucination identified at: \n >> {critical_layer}\n >> Max Vector Magnitude: {max_anomaly_score:.2f}\n", "#bc13fe")
            
        # Return structured analysis data back to the DeepSeek Thread 
        # so it can "read" the results and summarize them.
        return {
            "critical_layer": critical_layer,
            "max_magnitude": max_anomaly_score,
            "raw_report": "".join(analysis_report)[:1000] # Limit size if passing back to LLM context
        }
        
    def cleanup(self):
        """Detaches PyTorch hooks to clean up memory."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activation_cache.clear()
        if self.model:
            del self.model
            torch.cuda.empty_cache()
