#!/usr/bin/env bash
for p in $(pgrep -f "colab_lora_run|tools/colab_cli|google-chrome" 2>/dev/null); do
  [ "$p" = "$$" ] && continue
  kill -9 "$p" 2>/dev/null || true
done
sleep 2
echo "killed"
pgrep -af "colab_lora_run|tools/colab_cli" | head -3 || true
