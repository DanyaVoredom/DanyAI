#!/usr/bin/env bash
set -e
H=/home/lipton
mkdir -p "$H/lora_in" "$H/.ssh"
chmod 700 "$H/.ssh"
if [ ! -f "$H/.ssh/colab_push" ]; then
  ssh-keygen -t ed25519 -f "$H/.ssh/colab_push" -N "" -C colab-push -q
fi
PUB=$(cat "$H/.ssh/colab_push.pub")
ENTRY="restrict,command=\"cd /home/lipton/lora_in && scp -t .\" $PUB"
touch "$H/.ssh/authorized_keys"
chmod 600 "$H/.ssh/authorized_keys"
grep -q "colab-push" "$H/.ssh/authorized_keys" || echo "$ENTRY" >> "$H/.ssh/authorized_keys"
echo "--- test ---"
echo hello > /tmp/t.txt
scp -q -o StrictHostKeyChecking=no -o BatchMode=yes -i "$H/.ssh/colab_push" /tmp/t.txt lipton@127.0.0.1:test.txt && echo scp-ok || echo scp-fail
ls -la "$H/lora_in" | tail -3
