#!/usr/bin/env bash
export PATH="/home/lipton/bin:/usr/local/bin:/usr/bin:/bin"
W=/home/lipton/lora_work
STATE=/home/lipton/lora_publish.state
mkdir -p "$W"
cd /home/lipton || exit 1
LAST=""
[ -f "$STATE" ] && LAST=$(cat "$STATE")
while true; do
  ZIP=""
  for p in /home/lipton/lora_in/*.zip /home/lipton/DanyAI/runs/lora15_a*/danyai_runs/*.zip /home/lipton/DanyAI/runs/lora15_a*/lora.zip; do
    [ -f "$p" ] || continue
    if [ -z "$ZIP" ] || [ "$p" -nt "$ZIP" ]; then ZIP="$p"; fi
  done
  if [ -n "$ZIP" ] && [ "$ZIP" != "$LAST" ]; then
    echo "=== new zip $ZIP $(date -Is)" >> /home/lipton/lora_publish.log
    bash /home/lipton/DanyAI/deploy/lora_convert.sh >> /home/lipton/lora_publish.log 2>&1
    if [ -f "$W/danyai-lora.gguf" ]; then
      TAG=lora-v1
      if gh release view "$TAG" -R DanyaVoredom/DanyAI >/dev/null 2>&1; then
        gh release upload "$TAG" "$W/danyai-lora.gguf" -R DanyaVoredom/DanyAI --clobber >> /home/lipton/lora_publish.log 2>&1
      else
        gh release create "$TAG" "$W/danyai-lora.gguf" -R DanyaVoredom/DanyAI --title "DanyAI LoRA $TAG" --notes "LoRA adapter for Huihui-Qwen3.5-2B-abliterated" >> /home/lipton/lora_publish.log 2>&1
      fi
      cp -f "$W/danyai-lora.gguf" /home/lipton/models/danyai-lora.gguf 2>/dev/null || true
      echo "published $TAG $(date -Is)" >> /home/lipton/lora_publish.log
      echo "$ZIP" > "$STATE"
      LAST="$ZIP"
    else
      echo "convert failed for $ZIP" >> /home/lipton/lora_publish.log
    fi
  fi
  sleep 300
done
