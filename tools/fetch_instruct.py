import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset, config, split, offset, length=100, tries=6):
    url = (ENDPOINT + "?dataset=" + urllib.parse.quote(dataset, safe="")
           + "&config=" + urllib.parse.quote(config, safe="")
           + "&split=" + urllib.parse.quote(split, safe="")
           + f"&offset={offset}&length={length}")
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "danyai"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 502, 503, None):
                raise
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return {"rows": []}


def iter_pairs(dataset, config, split, limit, mapper):
    offset = 0
    produced = 0
    while produced < limit:
        data = fetch_rows(dataset, config, split, offset)
        rows = data.get("rows") or []
        if not rows:
            break
        for item in rows:
            pair = mapper(item.get("row") or {})
            if not pair:
                continue
            user, assistant = pair
            user = user.strip()
            assistant = assistant.strip()
            if len(user) < 2 or len(assistant) < 2:
                continue
            yield {"user": user, "assistant": assistant}
            produced += 1
            if produced >= limit:
                break
        offset += len(rows)
        time.sleep(0.4)


def map_alpaca(row):
    inst = row.get("instruction") or ""
    extra = row.get("input") or ""
    user = inst if not extra else inst + "\n" + extra
    return user, row.get("output") or ""


def map_dolly(row):
    inst = row.get("instruction") or ""
    ctx = row.get("context") or ""
    user = inst if not ctx else inst + "\n" + ctx
    return user, row.get("response") or ""


def map_saiga(row):
    if row.get("is_bad_by_regex"):
        return None
    msgs = row.get("messages") or []
    user = ""
    last = ""
    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            user = content
        elif role == "assistant" and user:
            last = content
    if not user or not last:
        return None
    return user, last


def map_den4ik(row):
    dlg = row.get("dialogue") or []
    if len(dlg) < 2:
        return None
    return dlg[0], dlg[1]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--en-limit", type=int, default=24000)
    ap.add_argument("--ru-limit", type=int, default=24000)
    args = ap.parse_args()

    en_a = args.en_limit * 3 // 4
    en_d = args.en_limit - en_a
    ru_s = args.ru_limit * 3 // 4
    ru_d = args.ru_limit - ru_s

    en = list(iter_pairs("tatsu-lab/alpaca", "default", "train", en_a, map_alpaca))
    en += list(iter_pairs("databricks/databricks-dolly-15k", "default", "train", en_d, map_dolly))
    ru = list(iter_pairs("IlyaGusev/saiga_scored", "default", "train", ru_s, map_saiga))
    ru += list(iter_pairs("Den4ikAI/russian_instructions", "default", "train", ru_d, map_den4ik))

    n_en = write_jsonl(os.path.join(args.out, "instruct_en.jsonl"), en)
    n_ru = write_jsonl(os.path.join(args.out, "instruct_ru.jsonl"), ru)
    print(f"instruct en {n_en} ru {n_ru}", flush=True)


if __name__ == "__main__":
    main()
