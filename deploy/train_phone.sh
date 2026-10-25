#!/data/data/com.termux/files/usr/bin/bash
# Offline training on Android/Termux. Resumes from runs/phone/ckpt.npz.
set -eu
cd ~/DanyAI

DATA_DIR=${DATA_DIR:-data}
OUT=${OUT:-runs/phone}
STEPS=${STEPS:-12000}
LOG=${LOG:-$OUT/train.log}

termux-wake-lock 2>/dev/null || true
mkdir -p "$OUT"

if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/tokenizer.json" ]; then
  echo "нет $DATA_DIR/train.bin или tokenizer.json"
  echo "положи файлы из релиза data-v2: train.bin, val.bin, tokenizer.json, meta.json"
  exit 1
fi

RESUME=""
if [ -f "$OUT/ckpt.npz" ]; then
  RESUME="--resume $OUT/ckpt.npz"
fi

nohup python -m danyai.train \
  --config configs/phone.json \
  --data "$DATA_DIR" \
  --out "$OUT" \
  --steps "$STEPS" \
  $RESUME >> "$LOG" 2>&1 &

echo "pid $!"
echo "лог: $LOG"
echo "tail -f $LOG"
