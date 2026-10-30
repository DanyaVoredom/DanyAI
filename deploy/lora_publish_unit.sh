#!/usr/bin/env bash
set -e
cat > /tmp/danyai-lora-publish.service <<'EOF'
[Unit]
Description=DanyAI LoRA convert and publish watcher
After=network-online.target
Wants=network-online.target

[Service]
User=lipton
Group=lipton
Environment=HOME=/home/lipton
WorkingDirectory=/home/lipton/DanyAI
ExecStart=/home/lipton/DanyAI/deploy/lora_publish_watch.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 644 /tmp/danyai-lora-publish.service /etc/systemd/system/danyai-lora-publish.service
sudo systemctl daemon-reload
sudo systemctl enable danyai-lora-publish >/dev/null 2>&1 || true
sudo systemctl restart danyai-lora-publish
sleep 3
systemctl is-active danyai-lora-publish
