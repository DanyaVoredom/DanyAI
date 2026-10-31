#!/data/data/com.termux/files/usr/bin/bash
# Регистрирует runit-сервисы DanyAI (сервер, веб-чат, обучение) и автозапуск при загрузке.
set -eu

PREFIX=${PREFIX:-/data/data/com.termux/files/usr}
SVDIR="$PREFIX/var/service"
HOME_DIR="$HOME/DanyAI"
STEPS=${STEPS:-12000}
WORKERS=${WORKERS:-4}

export SVDIR
mkdir -p "$HOME_DIR/logs"

write_service() {
  name=$1
  body=$2
  dir="$SVDIR/$name"
  mkdir -p "$dir/log" "$HOME_DIR/logs/$name"
  cat > "$dir/run" <<EOF
#!/data/data/com.termux/files/usr/bin/sh
$body
EOF
  chmod +x "$dir/run"
  cat > "$dir/log/run" <<EOF
#!/data/data/com.termux/files/usr/bin/sh
exec svlogd -tt "$HOME_DIR/logs/$name"
EOF
  chmod +x "$dir/log/run"
}

llama_body='sleep 10
cd "$HOME/DanyAI" || exit 1
BIN="$HOME/llama.cpp/build/bin/llama-server"
[ -x "$BIN" ] || { echo "llama-server не собран"; exec sleep 3600; }
MODEL="$HOME/models_gguf/qwen35-2b-q4km.gguf"
[ -f "$MODEL" ] || { echo "нет $MODEL"; exec sleep 3600; }
LORA=""
[ -f "$HOME/models_gguf/danyai-lora.gguf" ] && LORA="--lora $HOME/models_gguf/danyai-lora.gguf"
TS=""
taskset -c 4-7 true 2>/dev/null && TS="taskset -c 4-7"
timeout 5 termux-wake-lock 2>/dev/null || true
exec $TS "$BIN" -m "$MODEL" $LORA --host 0.0.0.0 --port 8081 -c 4096 -np 1 -t 4 -tb 4 --flash-attn on --cache-reuse 256 --jinja --chat-template-kwargs "{\"enable_thinking\": false}" 2>&1'

litellm_body='sleep 20
cd "$HOME/DanyAI" || exit 1
export LD_PRELOAD=/data/data/com.termux/files/usr/lib/libpython3.14.so
BIN=/data/data/com.termux/files/usr/bin/litellm
[ -x "$BIN" ] || { echo "litellm не установлен"; exec sleep 3600; }
timeout 5 termux-wake-lock 2>/dev/null || true
exec "$BIN" --config configs/litellm.yaml --host 0.0.0.0 --port 4000 2>&1'

server_body='sleep 5
cd "$HOME/DanyAI" || exit 1
timeout 5 termux-wake-lock 2>/dev/null || true
export OPENBLAS_NUM_THREADS=2
exec python -m danyai.server --ckpt models/model.npz --tok models/tokenizer.json --host 0.0.0.0 --port 8080 2>&1'

webchat_body='sleep 5
cd "$HOME/DanyAI" || exit 1
timeout 5 termux-wake-lock 2>/dev/null || true
export OPENBLAS_NUM_THREADS=2
exec python -m danyai.webchat --ckpt models/model.npz --tok models/tokenizer.json --host 0.0.0.0 --port 8090 2>&1'

webchat_qwen_body='sleep 8
cd "$HOME/DanyAI" || exit 1
timeout 5 termux-wake-lock 2>/dev/null || true
exec python -m danyai.webchat --ckpt models/model.npz --tok models/tokenizer.json --host 0.0.0.0 --port 8091 --api-base http://127.0.0.1:8081/v1 --api-model danyai-qwen 2>&1'

train_body="sleep 60
cd \"\$HOME/DanyAI\" || exit 1
timeout 5 termux-wake-lock 2>/dev/null || true
mkdir -p runs/phone
if python -c \"import numpy as np,sys; z=np.load('runs/phone/ckpt.npz'); sys.exit(0 if int(z['step']) >= $STEPS else 1)\" 2>/dev/null; then
  echo \"обучение завершено, $STEPS шагов\"
  exec sleep 3600
fi
if [ -f runs/phone/ckpt.npz ]; then
  exec python -m danyai.train --config configs/phone.json --data data_phone --out runs/phone --steps $STEPS --workers $WORKERS --resume runs/phone/ckpt.npz
fi
exec python -m danyai.train --config configs/phone.json --data data_phone --out runs/phone --steps $STEPS --workers $WORKERS"

write_service danyai-llama "$llama_body"
write_service danyai-litellm "$litellm_body"
write_service danyai-server "$server_body"
write_service danyai-webchat "$webchat_body"
write_service danyai-webchat-qwen "$webchat_qwen_body"
write_service danyai-train "$train_body"

mkdir -p "$HOME/.termux/boot"
cat > "$HOME/.termux/boot/danyai" <<EOF
#!/data/data/com.termux/files/usr/bin/sh
timeout 5 termux-wake-lock 2>/dev/null || true
pgrep -f "runsvdir $SVDIR" >/dev/null || exec runsvdir "$SVDIR"
EOF
chmod +x "$HOME/.termux/boot/danyai"

for name in danyai-llama danyai-litellm danyai-server danyai-webchat danyai-webchat-qwen danyai-train; do
  sv up "$name" 2>/dev/null || true
done

sleep 8
for name in danyai-llama danyai-litellm danyai-server danyai-webchat danyai-webchat-qwen danyai-train; do
  printf "%s: " "$name"
  sv status "$name" 2>/dev/null || echo "нет статуса"
done
curl -s --max-time 5 http://127.0.0.1:8080/health || true
echo
