#!/usr/bin/env bash
echo "--cf log--"
tail -15 /home/lipton/cf_ssh.log
echo "--sshd--"
sudo ss -ltnp 2>/dev/null | grep -E ":22 |:8022 " || echo no-sshd
sudo systemctl is-active ssh sshd 2>/dev/null
echo "--local ssh test--"
timeout 15 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8 lipton@127.0.0.1 "echo local-ssh-ok" 2>&1 | tail -3
echo "--journal--"
sudo journalctl -u cloudflared-ssh -n 10 --no-pager 2>/dev/null | tail -10
