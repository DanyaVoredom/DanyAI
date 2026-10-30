#!/usr/bin/env bash
date
echo "--timers--"
systemctl list-timers --all --no-pager 2>/dev/null | grep -iE "lora|modal" || echo none
echo "--units--"
for u in lora12 lora12-later lora13-retry lora13-retry2 modal-lora13; do
  printf "%s: %s %s\n" "$u" "$(systemctl is-active "$u" 2>/dev/null)" "$(systemctl show -p Result --value "$u" 2>/dev/null)"
done
echo "--logs--"
for u in lora13-retry lora13-retry2 lora12; do
  echo "[$u]"
  journalctl -u "$u" --no-pager -n 15 2>/dev/null | tail -15
done
echo "--files--"
ls -la /home/lipton/DanyAI/runs 2>/dev/null | tail -10
find /home/lipton -maxdepth 4 \( -iname "*adapter*" -o -iname "*lora*.zip" -o -iname "*lora*.gguf" \) -newermt "-5 days" 2>/dev/null | head -20
echo "--home--"
ls -la /home/lipton | tail -15
free -m | sed -n 2p
df -h / | tail -1
