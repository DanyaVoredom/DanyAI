#!/data/data/com.termux/files/usr/bin/bash
set -eu

MODELS="$HOME/models_gguf"
mkdir -p "$MODELS"
BASE="$MODELS/qwen35-2b-abliterated-q4km.gguf"
LORA="$MODELS/danyai-lora.gguf"
BASE_MIN=1260000000
LORA_MIN=10000000
BASE_URL="https://huggingface.co/interimlabs/InterimLabs-Huihui-Qwen3.5-2B-abliterated-GGUF-Q4_K_M/resolve/main/model.gguf"
LORA_URL="https://github.com/DanyaVoredom/DanyAI/releases/download/lora-v1/danyai-lora.gguf"
SVDIR="$PREFIX/var/service"
RUN="$SVDIR/danyai-llama/run"

gguf_ok() {
  [ -s "$1" ] && [ "$(head -c 4 "$1")" = "GGUF" ]
}

fetch() {
  url=$1
  out=$2
  min=$3
  if [ -s "$out" ] && [ "$(wc -c < "$out")" -ge "$min" ] && gguf_ok "$out"; then
    echo "cached $out $(wc -c < "$out")"
    return 0
  fi
  rm -f "$out" "$out.part"
  echo "downloading $out"
  curl -fL --retry 5 --retry-delay 3 --retry-all-errors -C - -o "$out.part" "$url"
  size=$(wc -c < "$out.part")
  if [ "$size" -lt "$min" ] || [ "$(head -c 4 "$out.part")" != "GGUF" ]; then
    echo "bad download $out $size"
    rm -f "$out.part"
    return 1
  fi
  mv "$out.part" "$out"
  echo "saved $out $size"
}

avail_kb=$(df -k "$HOME" | awk 'NR==2{print $4}')
if [ "${avail_kb:-0}" -lt 1700000 ]; then
  echo "low disk: ${avail_kb}K free"
  exit 1
fi

fetch "$BASE_URL" "$BASE" "$BASE_SIZE"
[ "$(wc -c < "$BASE")" -eq "$BASE_SIZE" ] || { echo "base size mismatch $(wc -c < "$BASE")"; exit 1; }

fetch "$LORA_URL" "$LORA" "$LORA_MIN"
sha=$(sha256sum "$LORA" | cut -d' ' -f1)
echo "lora sha256 $sha"

if [ -f "$RUN" ]; then
  cp "$RUN" "$RUN.bak.$(date +%s)"
  sed -i 's#qwen35-2b-q4km\.gguf#qwen35-2b-abliterated-q4km.gguf#g' "$RUN"
  if ! grep -q -- '--lora' "$RUN"; then
    sed -i 's#-m "\$MODEL"#-m "$MODEL" --lora "$HOME/models_gguf/danyai-lora.gguf"#' "$RUN"
  fi
  if ! grep -q -- '--lora' "$RUN"; then
    echo "run script has no --lora, patching literal model flag"
    sed -i 's#-m "[^"]*qwen35-2b-abliterated-q4km\.gguf"#& --lora "$HOME/models_gguf/danyai-lora.gguf"#' "$RUN"
  fi
  grep -n -- '--lora\|abliterated' "$RUN" || { echo "patch failed"; exit 1; }
  sv restart danyai-llama
else
  echo "no run script at $RUN"
  exit 1
fi

for i in $(seq 1 90); do
  if curl -sf -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1; then
    echo "health ok after ${i}s"
    break
  fi
  sleep 1
done

curl -s -m 5 http://127.0.0.1:8081/health || true
echo
curl -s -m 5 http://127.0.0.1:8081/v1/models | head -c 400 || true
echo
sv status danyai-llama || true
tail -c 4000 "$HOME/DanyAI/logs/danyai-llama/current" 2>/dev/null | grep -i "lora\|abliterated" | tail -5 || true

try=$(curl -s -m 60 -X POST http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"danyai-qwen","messages":[{"role":"user","content":"reply with one word: ok"}],"max_tokens":16,"temperature":0}')
echo "smoke $try"
