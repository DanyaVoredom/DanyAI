#!/usr/bin/env bash
cd /home/lipton/DanyAI || exit 1
setsid nohup bash /home/lipton/DanyAI/deploy/colab_lora_run.sh > /home/lipton/DanyAI/runs/wrapper14.log 2>&1 < /dev/null &
sleep 4
echo "started"
pgrep -af "colab_lora_run" | head -2
tail -2 /home/lipton/DanyAI/runs/lora14.log 2>/dev/null
