import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.112:4000/v1"
KEY = sys.argv[2] if len(sys.argv) > 2 else "sk-danyai-local"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "danyai-qwen"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def call(messages, tools=None, max_tokens=256):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    rq = urllib.request.Request(
        BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + KEY,
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(rq, timeout=600) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data, time.time() - t0


def show(tag, data, dt):
    msg = data["choices"][0]["message"]
    tools = msg.get("tool_calls") or []
    print("==", tag, "%.1fs" % dt, "usage", data.get("usage"))
    if tools:
        for t in tools:
            fn = t.get("function", {})
            print("tool_call", fn.get("name"), fn.get("arguments"))
    print("content", (msg.get("content") or "")[:400].replace("\n", " "))


cases = [
    ("ru-plain", [{"role": "user", "content": "Привет. Кто ты и что умеешь? Ответь кратко."}], None),
    ("en-plain", [{"role": "user", "content": "Hi. Who are you and what can you do? Keep it short."}], None),
    ("ru-tool", [{"role": "user", "content": "Какая сейчас погода в Париже?"}], TOOLS),
    ("en-tool", [{"role": "user", "content": "List the files in /tmp please."}], TOOLS),
    (
        "ru-system",
        [
            {"role": "system", "content": "Ты ассистент DanyAI. Отвечай по-русски."},
            {"role": "user", "content": "Сколько будет 17*23?"},
        ],
        None,
    ),
]

fail = 0
for tag, msgs, tools in cases:
    try:
        data, dt = call(msgs, tools)
        show(tag, data, dt)
        if tools and not (data["choices"][0]["message"].get("tool_calls")):
            fail += 1
            print("missing tool call for", tag)
    except urllib.error.HTTPError as e:
        fail += 1
        print("==", tag, "http", e.code, e.read().decode("utf-8", "replace")[:300])
    except Exception as e:
        fail += 1
        print("==", tag, "error", type(e).__name__, str(e)[:200])

print("failures", fail)
sys.exit(1 if fail else 0)
