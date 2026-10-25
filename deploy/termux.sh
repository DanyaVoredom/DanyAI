#!/data/data/com.termux/files/usr/bin/bash
set -e

pkg install -y python python-numpy git curl

cd "$HOME"
if [ ! -d DanyAI ]; then
  git clone https://github.com/DanyaVoredom/DanyAI.git
fi
cd DanyAI

mkdir -p models
if [ ! -f models/model.npz ]; then
  echo "put checkpoint into models/model.npz and tokenizer into models/tokenizer.json"
  exit 1
fi
if [ ! -f models/tokenizer.json ]; then
  cp data/tokenizer.json models/tokenizer.json 2>/dev/null || true
fi

termux-wake-lock || true
exec python -m danyai.server --ckpt models/model.npz --tok models/tokenizer.json \
  --host 127.0.0.1 --port 8080 --api-key "${DANYAI_API_KEY:-}"
