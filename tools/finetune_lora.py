import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_targets(model):
    names = [n for n, _ in model.named_modules()]
    want = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    got = sorted({w for w in want if any(n.endswith(w) for n in names)})
    return got


class ChatData(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            user = str(r.get("user", ""))
            assistant = str(r.get("assistant", ""))
            if not user.strip() or not assistant.strip():
                continue
            msgs = [{"role": "user", "content": user}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full_msgs = msgs + [{"role": "assistant", "content": assistant}]
            full = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) > max_len:
                f_ids = f_ids[:max_len]
            if len(f_ids) < 4:
                continue
            cut = min(len(p_ids), len(f_ids) - 1)
            labels = list(f_ids)
            for i in range(cut):
                labels[i] = -100
            if all(x == -100 for x in labels):
                continue
            self.items.append((f_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        lab[i, :len(y)] = torch.tensor(y, dtype=torch.long)
    return ids, lab


def lr_at(step, steps, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="runs/lora")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-share", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--fallback", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--grad-ckpt", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    dtype = torch.float16 if cuda else torch.float32

    def load_model(name):
        import transformers
        errors = []
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True)
            except Exception as e:
                errors.append(f"{cls_name}: {type(e).__name__}: {str(e)[:160]}")
        raise RuntimeError("; ".join(errors))

    try:
        model = load_model(args.model)
    except Exception as e:
        print("load failed:", e, flush=True)
        if args.fallback:
            print("falling back to", args.fallback, flush=True)
            args.model = args.fallback
            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token
            model = load_model(args.model)
        else:
            raise
    model.to(device)
    model.config.use_cache = False

    targets = find_targets(model)
    print("lora targets", targets, flush=True)
    if not targets:
        raise SystemExit("no lora target modules found")
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lcfg)
    if args.grad_ckpt and cuda:
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except Exception as e:
            print("grad ckpt disabled:", e, flush=True)
    model.print_trainable_parameters()

    rows = load_jsonl(args.data)
    random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[:args.limit]
    n_val = max(1, int(len(rows) * args.eval_share))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    t0 = time.time()
    train = ChatData(train_rows, tok, args.max_len)
    val = ChatData(val_rows, tok, args.max_len)
    print(f"train {len(train)} val {len(val)} prepare {time.time() - t0:.0f}s", flush=True)
    if not train:
        raise SystemExit("no usable examples")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    use_amp = cuda
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    steps_per_epoch = max(1, len(train) // (args.batch * args.grad_accum))
    total = max(1, int(steps_per_epoch * args.epochs))
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    pos = 0
    step = 0
    seen = 0
    t0 = time.time()
    model.train()

    while step < total:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            batch = []
            for _ in range(args.batch):
                if pos >= len(order):
                    rng.shuffle(order)
                    pos = 0
                batch.append(train[order[pos]])
                pos += 1
            ids, lab = collate(batch, tok.pad_token_id)
            ids = ids.to(device)
            lab = lab.to(device)
            with autocast:
                out = model(input_ids=ids, labels=lab)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            seen += int((lab >= 0).sum())
        step += 1
        lr = lr_at(step, total, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0 or step == total:
            print(f"step {step}/{total} loss {loss.item() * args.grad_accum:.4f} lr {lr:.2e} "
                  f"gn {gn:.2f} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total:
            model.eval()
            tot = 0.0
            cnt = 0
            with torch.no_grad():
                for i in range(0, len(val), args.batch):
                    ids, lab = collate(val[i:i + args.batch], tok.pad_token_id)
                    ids = ids.to(device)
                    lab = lab.to(device)
                    with autocast:
                        l = model(input_ids=ids, labels=lab).loss
                    n = int((lab >= 0).sum())
                    tot += float(l) * n
                    cnt += n
            model.train()
            print(f"eval step {step} val {tot / max(1, cnt):.4f}", flush=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            print(f"saved adapter {args.out}", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"done {time.time() - t0:.0f}s steps {total} tokens {seen}", flush=True)

    if args.merge:
        merged = args.out + "-merged"
        model.eval()
        base = model.merge_and_unload()
        base.save_pretrained(merged, safe_serialization=True)
        tok.save_pretrained(merged)
        print("merged", merged, flush=True)


if __name__ == "__main__":
    main()
