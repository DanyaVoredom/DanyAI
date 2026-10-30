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

LOG=runs/lora15.log
echo "=== start $(date -Is)" >> "$LOG"
RESUME=""
i=0
while [ "$i" -lt 12 ]; do
  i=$((i + 1))
  DUMP="runs/lora15_a${i}.txt"
  echo "=== attempt $i resume='$RESUME' $(date -Is)" >> "$LOG"
  /home/lipton/venv/bin/python tools/colab_cli.py --trainer lora \
    --model huihui-ai/Huihui-Qwen3.5-2B-abliterated \
    --data-release data-v2 --data data2 \
    --profile /home/lipton/DanyAI/colab-profile \
    --out "runs/lora15_a${i}" --runs /content/danyai_runs \
    --timeout 14400 --connect-wait 1800 \
    --resume-url "$RESUME" \
    --dump "$DUMP" >> "$LOG" 2>&1
  rc=$?
  echo "=== attempt $i rc=$rc $(date -Is)" >> "$LOG"
  if [ "$rc" = "0" ]; then
    echo "=== done at attempt $i" >> "$LOG"
    break
  fi
  last=$(grep -o "@@CKPT [^ ]*" "$DUMP" 2>/dev/null | tail -1 | cut -d' ' -f2)
  if [ -z "$last" ]; then
    last=$(grep -o "@@CKPT [^ ]*" "$LOG" 2>/dev/null | tail -1 | cut -d' ' -f2)
  fi
  if [ -n "$last" ]; then
    case "$last" in
      scp:*) RESUME="file:///home/lipton/lora_in/${last#scp:}" ;;
      http:*) RESUME="file:///home/lipton/lora_in/${last#http:}" ;;
      *) RESUME="$last" ;;
    esac
    echo "=== resume from $RESUME" >> "$LOG"
  fi
  if [ "$rc" = "6" ]; then
    echo "=== gpu quota, backoff 2h" >> "$LOG"
    sleep 7200
  elif [ "$rc" = "4" ]; then
    echo "=== session limit, backoff 1h" >> "$LOG"
    sleep 3600
  else
    sleep 600
  fi
done
echo "=== finished $(date -Is)" >> "$LOG"
