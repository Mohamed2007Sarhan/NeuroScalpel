import random
import sys
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from tqdm import tqdm

torch.inference_mode()

if len(sys.argv) > 1:
    MODEL_ID = sys.argv[1]
else:
    MODEL_ID = "tiiuae/Falcon3-1B-Instruct"

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

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# settings:
instructions = 32

if hasattr(model, "model") and hasattr(model.model, "layers"):
    layer_count = len(model.model.layers)
elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    layer_count = len(model.transformer.h)
else:
    raise ValueError("Unsupported model architecture for layer extraction.")

layer_idx = int(layer_count * 0.6)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "harmful.txt"), "r") as f:
    harmful = f.readlines()

with open(os.path.join(script_dir, "harmless.txt"), "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

def get_toks(insn):
    try:
        return tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        # Fallback for models without chat templates like gpt2
        return tokenizer(insn, return_tensors="pt").input_ids

harmful_toks = [get_toks(insn) for insn in harmful_instructions]
harmless_toks = [get_toks(insn) for insn in harmless_instructions]

max_its = instructions*2
bar = tqdm(total=max_its)


def generate(toks):
    bar.update(n=1)
    return model.generate(toks.to(model.device),
                          use_cache=False,
                          max_new_tokens=1,
                          return_dict_in_generate=True,
                          output_hidden_states=True)


harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()

harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

print(harmful_hidden)

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

print(harmful_mean)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print(refusal_dir)

refusal_path = os.path.join(script_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
torch.save(refusal_dir, refusal_path)
print(f"Saved to {refusal_path}")
