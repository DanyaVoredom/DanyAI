import argparse
import base64
import pathlib
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="лог с блоками @@BEGIN name len ... @@END name")
    ap.add_argument("--name", required=True, help="имя блока, например __RUNS__/lora.zip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = pathlib.Path(args.log).read_text(encoding="utf-8", errors="replace")
    begin = "@@BEGIN " + args.name
    end = "@@END " + args.name
    start = text.find(begin)
    if start < 0:
        raise SystemExit(f"блок {args.name} не найден")
    stop = text.find(end, start)
    if stop < 0:
        raise SystemExit(f"блок {args.name} не закрыт")

    head = text[start:start + 200].splitlines()[0]
    m = re.search(r"(\d+)$", head)
    body = text[text.find("\n", start) + 1:stop]
    chunks = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("@@")]
    data = base64.b64decode("".join(chunks))
    pathlib.Path(args.out).write_bytes(data)
    size = len(data)
    expect = int(m.group(1)) if m else -1
    print(f"{args.out}: {size} байт, ожидалось base64 {expect}")
    if expect > 0 and size * 4 // 3 + 4 < expect:
        raise SystemExit("данные короче ожидаемого")


if __name__ == "__main__":
    main()
