#!/usr/bin/env bash
cd /home/lipton || exit 1
LOG=/home/lipton/prep.log
: > "$LOG"
{
  echo "start $(date -Is)"
  command -v cmake || echo no-cmake
  command -v g++ || echo no-gxx
  if [ ! -d /home/lipton/llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp /home/lipton/llama.cpp 2>&1 | tail -3
  fi
  /home/lipton/venv/bin/pip -q install -U gguf safetensors numpy 2>&1 | tail -3
  echo "deps done"
  cd /home/lipton/llama.cpp || exit 1
  cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -5
  cmake --build build -j"$(nproc)" 2>&1 | tail -8
  echo "build rc=$?"
  ls -la build/bin/llama-cli 2>/dev/null
  echo "end $(date -Is)"
} >> "$LOG" 2>&1
echo started
