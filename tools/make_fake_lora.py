import json
import os
import sys

import torch
from safetensors.torch import save_file

out = "/home/lipton/lora_test/adapter"
os.makedirs(out, exist_ok=True)
base = "huihui-ai/Huihui-Qwen3.5-2B-abliterated"
r = 16
hidden = 2048
inter = 6144
tensors = {}
for layer in (0, 1):
    for mod, in_dim, out_dim in (
        ("q_proj", hidden, hidden),
        ("v_proj", hidden, 512),
        ("o_proj", 512, hidden),
        ("gate_proj", hidden, inter),
        ("down_proj", inter, hidden),
    ):
        pre = f"base_model.model.model.layers.{layer}.self_attn.{mod}" if mod in ("q_proj", "v_proj", "o_proj") else f"base_model.model.model.layers.{layer}.mlp.{mod}"
        tensors[pre + ".lora_A.weight"] = torch.zeros(r, in_dim, dtype=torch.float16)
        tensors[pre + ".lora_B.weight"] = torch.zeros(out_dim, r, dtype=torch.float16)
save_file(tensors, os.path.join(out, "adapter_model.safetensors"))
cfg = {
    "base_model_name_or_path": base,
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": r,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj"],
    "inference_mode": False,
}
json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
print("fake adapter", len(tensors), "tensors")
