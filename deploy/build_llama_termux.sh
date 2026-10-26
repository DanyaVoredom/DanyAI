#!/data/data/com.termux/files/usr/bin/bash
set -eu
export PATH=/data/data/com.termux/files/usr/bin:$PATH

SRC=/data/data/com.termux/files/usr/etc/apt/sources.list
if grep -q gnlug.org "$SRC" 2>/dev/null; then
  sed -i 's#https\?://[^ ]*termux-main#https://packages.termux.dev/apt/termux-main#' "$SRC"
fi

apt update -y
apt install -y cmake

cd "$HOME"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_OPENMP=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j3 --target llama-server llama-cli

mkdir -p "$HOME/models_gguf"
echo "готово: $HOME/llama.cpp/build/bin/llama-server"
