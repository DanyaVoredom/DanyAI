import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from danyai.sample import load_ckpt
from danyai.tokenizer import BPETokenizer

PROMPTS = [
    "<|user|>Кто ты?<|assistant|>",
    "<|user|>Расскажи короткую историю про кота.<|assistant|>",
    "<|user|>What is the capital of France?<|assistant|>",
    "<|user|>Привет, как дела?<|assistant|>",
]


def quality(text):
    cyr = [c for c in text if "\u0400" <= c <= "\u04ff"]
    if len(cyr) < 10:
        return {"cyr": len(cyr), "lower": None, "mojibake": False}
    low = sum(1 for c in cyr if c.islower()) / len(cyr)
    return {"cyr": len(cyr), "lower": round(low, 3), "mojibake": low < 0.55}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--rep", type=float, default=1.15)
    args = ap.parse_args()

    engine, cfg = load_ckpt(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    bad = 0
    for prompt in PROMPTS:
        out = engine.generate(tok, prompt, args.tokens, args.temp, 40, 0.9, args.rep,
                              0.0, 0.0, 1, stop=["<|end|>", "<|user|>"])
        text = out[len(prompt):].strip()
        q = quality(text)
        bad += 1 if q["mojibake"] else 0
        print(json.dumps({"prompt": prompt, "answer": text[:200], **q}, ensure_ascii=False))
    print(json.dumps({"mojibake_answers": bad, "total": len(PROMPTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
