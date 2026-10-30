#!/usr/bin/env bash
set -e
mkdir -p /home/lipton/bin
if [ ! -x /home/lipton/bin/cloudflared ]; then
  curl -sL -o /home/lipton/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /home/lipton/bin/cloudflared
fi
cat > /tmp/cloudflared-ssh.service <<'EOF'
[Unit]
Description=cloudflared quick tunnel ssh
After=network-online.target
Wants=network-online.target

[Service]
User=lipton
ExecStart=/home/lipton/bin/cloudflared tunnel --url tcp://localhost:22 --no-autoupdate --logfile /home/lipton/cf_ssh.log
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 644 /tmp/cloudflared-ssh.service /etc/systemd/system/cloudflared-ssh.service
sudo systemctl daemon-reload
sudo systemctl enable cloudflared-ssh >/dev/null 2>&1 || true
sudo systemctl restart cloudflared-ssh
sleep 12
grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /home/lipton/cf_ssh.log | head -1
systemctl is-active cloudflared-ssh
