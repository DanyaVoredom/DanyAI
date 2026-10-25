import argparse
import json
import os
import time
import urllib.parse
import urllib.request

ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "system": "system", "tool": "tool"}


def rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    return json.loads(urllib.request.urlopen(url, timeout=60).read())


def convert(row):
    conv = row.get("conversations") or []
    msgs = []
    for m in conv:
        role = ROLES.get(str(m.get("from", "")).strip().lower())
        value = str(m.get("value") or "").strip()
        if not role or not value:
            continue
        msgs.append({"role": role, "content": value})
    if len(msgs) < 2:
        return None
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    tools = row.get("tools")
    if tools and not any(m["role"] == "system" for m in msgs):
        try:
            parsed = json.loads(tools) if isinstance(tools, str) else tools
            if parsed:
                msgs.insert(0, {"role": "system", "content": "Available tools:\n" + json.dumps(parsed, ensure_ascii=False)})
        except Exception:
            pass
    return {"messages": msgs, "source": row.get("category") or "tools"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=9000)
    ap.add_argument("--configs", default="func_calling,func_calling_singleturn,json_mode_agentic")
    ap.add_argument("--dataset", default="NousResearch/hermes-function-calling-v1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "tools.jsonl")
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
            offset = 0
            while written < args.limit:
                got = None
                for attempt in range(4):
                    try:
                        got = rows(args.dataset, config, "train", offset, 100)
                        break
                    except Exception as e:
                        print(f"{config} offset {offset} failed ({attempt + 1}): {e}", flush=True)
                        time.sleep(2 + attempt * 3)
                if not got:
                    break
                batch = got.get("rows") or []
                if not batch:
                    break
                for item in batch:
                    row = convert(item.get("row") or {})
                    if not row:
                        continue
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.limit:
                        break
                print(f"{config} offset {offset} total {written}", flush=True)
                offset += len(batch)
                if len(batch) < 100:
                    break
    print("tools rows", written)


if __name__ == "__main__":
    main()
