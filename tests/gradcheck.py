import argparse
import json
import sys
import numpy as np

import danyai.tensor as tt
from danyai.tensor import T, mul, matmul, gelu, layernorm, softmax, embedding, cross_entropy, backward


def numgrad(f, x, eps=1e-6, n=6):
    flat = x.data.reshape(-1)
    out = np.zeros(min(flat.size, n))
    for k in range(out.size):
        old = flat[k]
        flat[k] = old + eps
        fp = float(f().data)
        flat[k] = old - eps
        fm = float(f().data)
        flat[k] = old
        out[k] = (fp - fm) / (2 * eps)
    return out


def zero(*ts):
    for t in ts:
        t.grad = None


def check(name, a, b, tol=1e-6):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    scale = max(np.max(np.abs(b)), 1e-9)
    err = np.max(np.abs(a - b)) / scale
    print(f"{name:22s} {'ok' if err < tol else 'FAIL':4s} rel={err:.2e}")
    return err < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/nano.json")
    ap.add_argument("--seq", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=97)
    args = ap.parse_args()

    tt.DTYPE = np.float64
    rng = np.random.default_rng(0)
    ok = True

    x = T(rng.normal(size=(4, 5)), requires_grad=True)
    y = T(rng.normal(size=(5, 3)), requires_grad=True)
    zero(x, y)
    backward(matmul(x, y))
    ok &= check("matmul dx", x.grad.reshape(-1)[:6],
                numgrad(lambda: T(float((x.data @ y.data).sum())), x))

    a = T(rng.normal(size=(2, 3)), requires_grad=True)
    b = T(rng.normal(size=(2, 3)), requires_grad=True)
    zero(a, b)
    backward(mul(a, b))
    ok &= check("mul db", b.grad.reshape(-1)[:6],
                numgrad(lambda: T(float((a.data * b.data).sum())), b))

    h = T(rng.normal(size=(3, 16)), requires_grad=True)
    g = T(rng.normal(size=(16,)), requires_grad=True)
    bb = T(rng.normal(size=(16,)), requires_grad=True)

    zero(h, g, bb)
    backward(layernorm(h, g, bb))

    def fln():
        m = h.data.mean(-1, keepdims=True)
        v = h.data.var(-1, keepdims=True)
        return T(float((((h.data - m) / np.sqrt(v + 1e-5)) * g.data + bb.data).sum()))

    ok &= check("layernorm dh", h.grad.reshape(-1)[:6], numgrad(fln, h))

    r = rng.normal(size=(3, 16))
    zero(h)
    rs = T(r)
    backward(mul(softmax(h), rs))

    def fsm():
        z = h.data - h.data.max(-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(-1, keepdims=True)
        return T(float((p * r).sum()))

    ok &= check("softmax dh", h.grad.reshape(-1)[:6], numgrad(fsm, h))

    zero(h)
    backward(gelu(h))

    def fgelu():
        z = h.data
        c = np.sqrt(2.0 / np.pi)
        return T(float((0.5 * z * (1 + np.tanh(c * (z + 0.044715 * z ** 3)))).sum()))

    ok &= check("gelu dh", h.grad.reshape(-1)[:6], numgrad(fgelu, h))

    w = T(rng.normal(size=(20, 8)), requires_grad=True)
    idx = rng.integers(0, 20, size=(2, 5))
    zero(w)
    backward(embedding(w, idx))
    ok &= check("embedding dw", w.grad.reshape(-1)[:6],
                numgrad(lambda: T(float(w.data[idx].sum())), w))

    with open(args.model, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["vocab_size"] = args.vocab
    cfg["block_size"] = max(args.seq, 8)
    cfg["n_layer"] = min(cfg["n_layer"], 2)
    from danyai.model import GPT
    model = GPT(cfg)

    xb = rng.integers(0, args.vocab, size=(args.batch, args.seq))
    yb = rng.integers(0, args.vocab, size=(args.batch, args.seq))

    def floss():
        return T(float(cross_entropy(model.forward(xb), yb).data))

    backward(model.loss(xb, yb))

    for name in ["wte", "wpe", "h0.wq", "h0.wk", "h0.wv", "h0.wo", "h1.w1", "h1.w2", "h0.b1", "h0.ln1.g", "lnf.b"]:
        p = model.params[name]
        ok &= check("gpt " + name, p.grad.reshape(-1)[:4], numgrad(floss, p, n=4), tol=1e-5)

    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()