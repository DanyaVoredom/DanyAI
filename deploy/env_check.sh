#!/usr/bin/env bash
cd /home/lipton/DanyAI || exit 1
echo "--train--"
grep -aoE "step [0-9]+/[0-9]+ loss [0-9.]+|eval step [0-9]+ val [0-9.]+|@@CKPT [^ ]+" runs/lora14_a1.txt 2>/dev/null | tail -4
echo "--llama--"
if [ -d /home/lipton/llama.cpp ]; then echo llama-ok; else echo no-llama; fi
echo "--py--"
/home/lipton/venv/bin/python - <<'EOF'
try:
    import gguf
    print("gguf", getattr(gguf, "__version__", "ok"))
except Exception as e:
    print("no gguf", type(e).__name__)
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("no torch", type(e).__name__)
try:
    import peft
    print("peft", peft.__version__)
except Exception as e:
    print("no peft", type(e).__name__)
EOF
echo "--disk--"
df -h / | tail -1
