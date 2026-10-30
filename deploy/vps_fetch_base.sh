#!/usr/bin/env bash
set -e
M=/home/lipton/models
mkdir -p "$M"
URL="https://huggingface.co/interimlabs/InterimLabs-Huihui-Qwen3.5-2B-abliterated-GGUF-Q4_K_M/resolve/main/model.gguf"
if [ ! -s "$M/abliterated-q4km.gguf" ]; then
  curl -L --retry 3 -o "$M/abliterated-q4km.gguf.part" "$URL"
  mv "$M/abliterated-q4km.gguf.part" "$M/abliterated-q4km.gguf"
fi
ls -la "$M"
