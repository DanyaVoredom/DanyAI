#!/usr/bin/env bash
set -e
URL="$1"
TAG="${2:-lora-v1}"
WORK=/home/lipton/lora_work
mkdir -p "$WORK"
cd "$WORK"
echo "downloading $URL"
curl -sL --retry 3 -o adapter.zip "$URL"
ls -la adapter.zip
rm -rf adapter
/home/lipton/venv/bin/python - <<'EOF'
import zipfile
with zipfile.ZipFile("/home/lipton/lora_work/adapter.zip") as z:
    names = z.namelist()
    print("files", len(names), names[:6])
    z.extractall("/home/lipton/lora_work/adapter")
EOF
ls -la adapter | head -10
/home/lipton/venv/bin/python /home/lipton/llama.cpp/convert_lora_to_gguf.py /home/lipton/lora_work/adapter \
  --outfile /home/lipton/lora_work/danyai-lora.gguf --outtype f16
ls -la danyai-lora.gguf
if gh release view "$TAG" -R DanyaVoredom/DanyAI >/dev/null 2>&1; then
  gh release upload "$TAG" danyai-lora.gguf -R DanyaVoredom/DanyAI --clobber
else
  gh release create "$TAG" danyai-lora.gguf -R DanyaVoredom/DanyAI --title "DanyAI LoRA $TAG" --notes "LoRA adapter for Huihui-Qwen3.5-2B-abliterated"
fi
gh release view "$TAG" -R DanyaVoredom/DanyAI --json url,assets
