#!/usr/bin/env bash
set -e
cat > /tmp/danyai-lora-run.service <<'EOF'
[Unit]
Description=DanyAI LoRA training on Colab
After=network-online.target
Wants=network-online.target

[Service]
User=lipton
Group=lipton
Environment=HOME=/home/lipton
WorkingDirectory=/home/lipton/DanyAI
ExecStart=/home/lipton/DanyAI/deploy/colab_lora_run.sh
Restart=no
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 644 /tmp/danyai-lora-run.service /etc/systemd/system/danyai-lora-run.service
sudo systemctl daemon-reload
sudo systemctl enable danyai-lora-run >/dev/null 2>&1 || true
echo "unit installed"
