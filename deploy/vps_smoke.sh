#!/usr/bin/env bash
cd /home/lipton/llama.cpp || exit 1
BIN=build/bin/llama-server
M=${1:-/home/lipton/models/abliterated-q4km.gguf}
LORA=${2:-}
PORT=8181
ARGS="-m $M --host 127.0.0.1 --port $PORT -c 2048 -t 8 -np 1 --jinja"
[ -n "$LORA" ] && ARGS="$ARGS --lora $LORA"
$BIN $ARGS > /tmp/srv.log 2>&1 &
PID=$!
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://127.0.0.1:$PORT/health 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 2
done
echo "ready=$code"
curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Привет. Кто ты? Ответь одним предложением."}],"max_tokens":64,"temperature":0.3,"chat_template_kwargs":{"enable_thinking":false}}' \
  | /home/lipton/venv/bin/python -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][:400])"
kill $PID 2>/dev/null
wait $PID 2>/dev/null
