#!/usr/bin/env bash
set -e
W=/home/lipton/lora_work
mkdir -p "$W"
ZIP=""
for p in "$W"/danyai_lora*.zip /home/lipton/lora_in/*.zip /home/lipton/DanyAI/runs/lora14_a1/danyai_runs/*.zip; do
  [ -f "$p" ] || continue
  if [ -z "$ZIP" ] || [ "$p" -nt "$ZIP" ]; then ZIP="$p"; fi
done
if [ -z "$ZIP" ]; then
  echo "no zip found"
  exit 1
fi
echo "using $ZIP $(stat -c %s "$ZIP") bytes"
rm -rf "$W/adapter"
/home/lipton/venv/bin/python - "$ZIP" <<'EOF'
import sys, zipfile
zp = sys.argv[1]
with zipfile.ZipFile(zp) as z:
    names = z.namelist()
    print("entries", len(names))
    print(names[:8])
    z.extractall("/home/lipton/lora_work/adapter")
EOF
ls -la "$W/adapter" | head -8
/home/lipton/venv/bin/python /home/lipton/llama.cpp/convert_lora_to_gguf.py "$W/adapter" \
  --outfile "$W/danyai-lora.gguf" --outtype f16
ls -la "$W/danyai-lora.gguf"
