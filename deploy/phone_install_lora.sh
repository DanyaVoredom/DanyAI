#!/data/data/com.termux/files/usr/bin/bash
set -e
MODELS="$HOME/models_gguf"
mkdir -p "$MODELS"
BASE="$MODELS/qwen35-2b-abliterated-q4km.gguf"
LORA="$MODELS/danyai-lora.gguf"
BASE_URL="https://huggingface.co/interimlabs/InterimLabs-Huihui-Qwen3.5-2B-abliterated-GGUF-Q4_K_M/resolve/main/model.gguf"
LORA_URL="https://github.com/DanyaVoredom/DanyAI/releases/download/lora-v1/danyai-lora.gguf"

if [ ! -s "$BASE" ]; then
  echo "downloading base"
  curl -L --retry 3 -o "$BASE.part" "$BASE_URL"
  mv "$BASE.part" "$BASE"
fi
ls -la "$BASE"

echo "downloading lora"
curl -L --retry 3 -o "$LORA.part" "$LORA_URL"
mv "$LORA.part" "$LORA"
ls -la "$LORA"

RUN=/data/data/com.termux/files/usr/var/service/danyai-llama/run
sed -i "s#models_gguf/qwen35-2b-q4km.gguf#models_gguf/qwen35-2b-abliterated-q4km.gguf#" "$RUN" 2>/dev/null || true
grep -n "abliterated\|lora" "$RUN" | head -4
sv restart danyai-llama
sleep 20
curl -s -m 5 http://127.0.0.1:8081/health
echo
