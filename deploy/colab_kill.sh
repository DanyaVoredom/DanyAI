#!/usr/bin/env bash
for p in $(pgrep -f "colab_lora_run|tools/colab_cli" 2>/dev/null); do
  [ "$p" = "$$" ] && continue
  kill -9 "$p" 2>/dev/null || true
done
for p in $(pgrep -x chrome 2>/dev/null) $(pgrep -x chrome_crashpad 2>/dev/null); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 3
echo "killed"
pgrep -af "colab_lora_run|tools/colab_cli" | head -3 || true
pgrep -c -x chrome || echo chrome-none
