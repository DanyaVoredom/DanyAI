#!/usr/bin/env bash
export PATH="/home/lipton/bin:/usr/local/bin:/usr/bin:/bin"
export DISPLAY=:99
unset COLAB_CDP_ENDPOINT
cd /home/lipton/DanyAI || exit 1
mkdir -p runs

if ! pgrep -f "Xvfb :99" >/dev/null; then
  nohup Xvfb :99 -screen 0 1600x1200x24 >/dev/null 2>&1 &
  sleep 2
fi

LOG=runs/lora14.log
: > "$LOG"
RESUME=""
for attempt in 1 2 3; do
  DUMP="runs/lora14_a${attempt}.txt"
  echo "=== attempt $attempt resume='$RESUME' $(date -Is)" >> "$LOG"
  /home/lipton/venv/bin/python tools/colab_cli.py --trainer lora \
    --model huihui-ai/Huihui-Qwen3.5-2B-abliterated \
    --data-release data-v2 --data data2 \
    --profile /home/lipton/DanyAI/colab-profile \
    --out "runs/lora14_a${attempt}" --runs /content/danyai_runs \
    --timeout 14400 --connect-wait 1800 \
    --resume-url "$RESUME" \
    --dump "$DUMP" >> "$LOG" 2>&1
  rc=$?
  echo "=== attempt $attempt rc=$rc $(date -Is)" >> "$LOG"
  last=$(grep -o "@@CKPT [^ ]*" "$DUMP" 2>/dev/null | tail -1 | cut -d' ' -f2)
  if [ -z "$last" ]; then
    last=$(grep -o "@@CKPT [^ ]*" "$LOG" 2>/dev/null | tail -1 | cut -d' ' -f2)
  fi
  case "$last" in
    scp:*) RESUME="file:///home/lipton/lora_in/${last#scp:}" ;;
    http:*) RESUME="file:///home/lipton/lora_in/${last#http:}" ;;
    *) RESUME="$last" ;;
  esac
  if [ "$rc" = "0" ]; then
    echo "=== done at attempt $attempt" >> "$LOG"
    break
  fi
  if [ -n "$last" ]; then
    RESUME="$last"
    echo "=== resume from $RESUME" >> "$LOG"
  else
    echo "=== no snapshot, retry fresh after pause" >> "$LOG"
    sleep 600
  fi
done
echo "=== finished $(date -Is)" >> "$LOG"
