#!/usr/bin/env bash
cd /home/lipton/llama.cpp || exit 1
BIN=build/bin/llama-cli
M=/home/lipton/models/abliterated-q4km.gguf
$BIN -m "$M" -p "Привет. Кто ты? Ответь одним предложением." -n 48 -t 8 --temp 0.3 2>&1 | tail -25
