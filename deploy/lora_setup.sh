#!/usr/bin/env bash
cd /home/lipton/DanyAI || exit 1
sudo systemctl stop danyai-lora-run 2>/dev/null
bash deploy/colab_kill.sh
bash deploy/lora_unit_install.sh
sudo systemctl start danyai-lora-run
sleep 6
systemctl is-active danyai-lora-run
tail -3 runs/lora15.log 2>/dev/null
