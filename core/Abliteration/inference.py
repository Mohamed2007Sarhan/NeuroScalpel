import os
import sys

import einops
import jaxtyping
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from inspect import signature

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

script_dir = os.path.dirname(os.path.abspath(__file__))
refusal_path = os.path.join(script_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
refusal_dir = torch.load(refusal_path)


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1),
                         '... d_act, d_act single -> ... single') * direction
    return activation - proj


# Some model developers thought it was stupid to pass a tuple of tuple of tuples around (rightfully so), but unfortunately now we have a divide
sig = signature(model.model.layers[0].forward)
simple = sig.return_annotation == torch.Tensor


class AblationDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_type = "full_attention"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        assert not output_attentions

        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(
            hidden_states.device)).to(hidden_states.device)

        if simple:
            return ablated

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


if hasattr(model, "model") and hasattr(model.model, "layers"):
    target_list = model.model.layers
elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    target_list = model.transformer.h
else:
    raise ValueError("Unsupported model architecture for Ablation hook insertion.")

for idx in reversed(range(len(target_list))):
    target_list.insert(idx, AblationDecoderLayer())

# bruh
if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
    model.config.num_hidden_layers = len(target_list)

conversation = []

streamer = TextStreamer(tokenizer)

print(f"Chat with {MODEL_ID}:")
while True:
    prompt = input()
    conversation.append({"role": "user", "content": prompt})
    try:
        toks = tokenizer.apply_chat_template(conversation=conversation,
                                             add_generation_prompt=True, return_tensors="pt")
    except Exception:
        toks = tokenizer(prompt, return_tensors="pt").input_ids

    gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=1337)

    decoded = tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": "".join(decoded)})
