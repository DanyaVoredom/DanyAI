#!/usr/bin/env bash
cd /home/lipton/DanyAI || exit 1
for p in $(pgrep -f lora_receiver 2>/dev/null); do
  [ "$p" = "$$" ] && continue
  kill "$p" 2>/dev/null || true
done
sleep 1
sudo ufw allow 18099/tcp >/dev/null 2>&1 || true
sudo iptables -C INPUT -p tcp --dport 18099 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT 2>/dev/null || true
setsid nohup /home/lipton/venv/bin/python tools/lora_receiver.py 18099 > /home/lipton/lora_recv.log 2>&1 < /dev/null &
sleep 2
curl -s -m 5 http://127.0.0.1:18099/ && echo recv-ok
sudo ss -ltnp 2>/dev/null | grep 18099 | head -1

