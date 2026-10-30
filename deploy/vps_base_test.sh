#!/usr/bin/env bash
cd /home/lipton/llama.cpp || exit 1
BIN=/home/lipton/llama.cpp/build/bin/llama-cli
M=/home/lipton/models/abliterated-q4km.gguf
$BIN -m "$M" -p "Привет. Кто ты? Ответь одним предложением." -n 48 -t 8 --temp 0.3 \
  -no-cnv 2>/dev/null | tail -6
