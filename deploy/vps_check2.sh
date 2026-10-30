#!/usr/bin/env bash
echo "--lora12_output tail--"
tail -40 /home/lipton/DanyAI/runs/lora12_output.txt 2>/dev/null
echo "--lora12 dir--"
ls -la /home/lipton/DanyAI/runs/lora12 2>/dev/null
echo "--archives--"
find /home/lipton /tmp -maxdepth 4 -newermt "-4 days" \( -iname "*.gguf" -o -iname "*.safetensors" -o -iname "*.zip" -o -iname "*adapter*" \) 2>/dev/null | grep -v node_modules | head -30
echo "--colab procs--"
pgrep -af "colab|chrome" | head -10
echo "--timers all--"
systemctl list-timers --all --no-pager 2>/dev/null | head -10
