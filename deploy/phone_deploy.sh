#!/data/data/com.termux/files/usr/bin/bash
# Atomic model swap on the phone: models/*.new -> models/* and restart services.
set -eu
cd ~/DanyAI

for f in model.npz tokenizer.json meta.json; do
  if [ ! -f "models/$f.new" ]; then
    echo "нет models/$f.new"
    exit 1
  fi
done

stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p models/backup-$stamp
for f in model.npz tokenizer.json meta.json; do
  [ -f "models/$f" ] && cp "models/$f" "models/backup-$stamp/$f"
  mv "models/$f.new" "models/$f"
done

pkill -f 'danyai.webchat' 2>/dev/null || true
pkill -f 'danyai.server' 2>/dev/null || true
sleep 2
nohup python -m danyai.server --ckpt models/model.npz --tok models/tokenizer.json --host 0.0.0.0 --port 8080 > server.log 2>&1 &
nohup python -m danyai.webchat --ckpt models/model.npz --tok models/tokenizer.json --host 0.0.0.0 --port 8090 > webchat.log 2>&1 &
sleep 8
curl -s http://127.0.0.1:8080/health
echo
echo "backup models/backup-$stamp"
