import numpy as np
from .backend import xp, to_numpy, scatter_add

DTYPE = xp.float32


def _acc(a, b):
    if a is None:
        return b
    a += b
    return a


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class T:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, requires_grad=False, parents=(), backward=None):
        self.data = xp.asarray(data, dtype=DTYPE)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = parents
        self._backward = backward

    @property
    def shape(self):
        return self.data.shape


def _const(x):
    return x if isinstance(x, T) else T(x)


def add(a, b):
    a, b = _const(a), _const(b)
    out = T(a.data + b.data, a.requires_grad or b.requires_grad, (a, b))

    def bw():
        if a.requires_grad:
            a.grad = _acc(a.grad, _unbroadcast(out.grad, a.data.shape))
        if b.requires_grad:
            b.grad = _acc(b.grad, _unbroadcast(out.grad, b.data.shape))

    out._backward = bw
    return out


def mul(a, b):
    a, b = _const(a), _const(b)
    out = T(a.data * b.data, a.requires_grad or b.requires_grad, (a, b))

    def bw():
        if a.requires_grad:
            a.grad = _acc(a.grad, _unbroadcast(out.grad * b.data, a.data.shape))
        if b.requires_grad:
            b.grad = _acc(b.grad, _unbroadcast(out.grad * a.data, b.data.shape))

    out._backward = bw
    return out


def matmul(a, b, tb=False):
    bb = b.data.swapaxes(-1, -2) if tb else b.data
    out = T(a.data @ bb, a.requires_grad or b.requires_grad, (a, b))

    def bw():
        g = out.grad
        if a.requires_grad:
            ga = g @ bb.swapaxes(-1, -2)
            a.grad = _acc(a.grad, _unbroadcast(ga, a.data.shape))
        if b.requires_grad:
            gb = a.data.swapaxes(-1, -2) @ g
            if tb:
                gb = gb.swapaxes(-1, -2)
            b.grad = _acc(b.grad, _unbroadcast(gb, b.data.shape))

    out._backward = bw
    return out


def reshape(t, shape):
    out = T(t.data.reshape(shape), t.requires_grad, (t,))

    def bw():
        t.grad = _acc(t.grad, out.grad.reshape(t.data.shape))

    out._backward = bw
    return out


def transpose(t, axes):
    out = T(t.data.transpose(axes), t.requires_grad, (t,))
    inv = tuple(int(i) for i in to_numpy(xp.argsort(xp.asarray(axes))))

    def bw():
        t.grad = _acc(t.grad, out.grad.transpose(inv))

    out._backward = bw
    return out


_GELU_C = xp.float32(xp.sqrt(2.0 / 3.141592653589793))
_GELU_A = xp.float32(0.044715)
_GELU_3A = xp.float32(3.0 * 0.044715)
_HALF = xp.float32(0.5)
_ONE = xp.float32(1.0)


def gelu(t):
    x = t.data
    x2 = x * x
    u = _GELU_C * (x + _GELU_A * x2 * x)
    th = xp.tanh(u)
    out = T(_HALF * x * (_ONE + th), t.requires_grad, (t,))

    def bw():
        sech2 = _ONE - th * th
        du = _GELU_C * (_ONE + _GELU_3A * x2)
        d = _HALF * (_ONE + th) + _HALF * x * sech2 * du
        d *= out.grad
        t.grad = _acc(t.grad, d)

    out._backward = bw
    return out


def silu(t):
    x = t.data
    s = 1.0 / (1.0 + xp.exp(-x))
    out = T(x * s, t.requires_grad, (t,))

    def bw():
        d = s + x * s * (1.0 - s)
        d *= out.grad
        t.grad = _acc(t.grad, d)

    out._backward = bw
    return out


def split_last(t, k):
    a = T(t.data[..., :k], t.requires_grad, (t,))
    b = T(t.data[..., k:], t.requires_grad, (t,))

    def bw_a():
        g = xp.zeros_like(t.data)
        g[..., :k] = a.grad
        t.grad = _acc(t.grad, g)

    def bw_b():
        g = xp.zeros_like(t.data)
        g[..., k:] = b.grad
        t.grad = _acc(t.grad, g)

    a._backward = bw_a
    b._backward = bw_b
    return a, b


def concat_last(a, b):
    out = T(xp.concatenate([a.data, b.data], axis=-1),
            a.requires_grad or b.requires_grad, (a, b))
    k = a.data.shape[-1]

    def bw():
        if a.requires_grad:
            a.grad = _acc(a.grad, out.grad[..., :k])
        if b.requires_grad:
            b.grad = _acc(b.grad, out.grad[..., k:])

    out._backward = bw
    return out


def repeat_heads(t, rep):
    if rep == 1:
        return t
    out = T(xp.repeat(t.data, rep, axis=1), t.requires_grad, (t,))

    def bw():
        g = out.grad
        shape = g.shape
        g = g.reshape(shape[0], shape[1] // rep, rep, shape[2], shape[3]).sum(axis=2)
        t.grad = _acc(t.grad, g)

    out._backward = bw
    return out


def rmsnorm(x, g, eps=1e-5):
    xd = x.data
    rstd = xp.float32(1.0) / xp.sqrt((xd * xd).mean(-1, keepdims=True) + eps)
    xhat = xd * rstd
    out = T(xhat * g.data, x.requires_grad or g.requires_grad, (x, g))
    axes = tuple(range(xd.ndim - 1))

    def bw():
        d = out.grad
        if g.requires_grad:
            g.grad = _acc(g.grad, (d * xhat).sum(axis=axes))
        if x.requires_grad:
            dxhat = d * g.data
            dxhat -= xhat * (dxhat * xhat).mean(-1, keepdims=True)
            dxhat *= rstd
            x.grad = _acc(x.grad, dxhat)

    out._backward = bw
    return out


def layernorm(x, g, b, eps=1e-5):
    xd = x.data
    mu = xd.mean(-1, keepdims=True)
    xc = xd - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = xp.float32(1.0) / xp.sqrt(var + eps)
    xhat = xc * rstd
    out = T(xhat * g.data + b.data, x.requires_grad or g.requires_grad or b.requires_grad, (x, g, b))
    axes = tuple(range(xd.ndim - 1))

    def bw():
        d = out.grad
        if g.requires_grad:
            g.grad = _acc(g.grad, (d * xhat).sum(axis=axes))
        if b.requires_grad:
            b.grad = _acc(b.grad, d.sum(axis=axes))
        if x.requires_grad:
            dxhat = d * g.data
            dxhat -= dxhat.mean(-1, keepdims=True)
            dxhat -= xhat * (dxhat * xhat).mean(-1, keepdims=True)
            dxhat *= rstd
            x.grad = _acc(x.grad, dxhat)

    out._backward = bw
    return out


def rope(t, cos, sin):
    x = t.data
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = T(xp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1),
            t.requires_grad, (t,))

    def bw():
        g = out.grad
        g1 = g[..., :half]
        g2 = g[..., half:]
        d = xp.concatenate([g1 * cos + g2 * sin, -g1 * sin + g2 * cos], axis=-1)
        t.grad = _acc(t.grad, d)

    out._backward = bw
    return out


def softmax(x):
    z = x.data - x.data.max(-1, keepdims=True)
    e = xp.exp(z)
    p = e / e.sum(-1, keepdims=True)
    out = T(p, x.requires_grad, (x,))

    def bw():
        g = out.grad
        s = (g * p).sum(-1, keepdims=True)
        x.grad = _acc(x.grad, p * (g - s))

    out._backward = bw
    return out


def embedding(w, idx):
    idx = xp.asarray(idx)
    out = T(w.data[idx], w.requires_grad, (w,))

    def bw():
        g = xp.zeros_like(w.data)
        scatter_add(g, xp.asarray(idx).ravel(), out.grad.reshape(-1, w.data.shape[-1]))
        w.grad = _acc(w.grad, g)

    out._backward = bw
    return out


def cross_entropy(logits, targets):
    targets = xp.asarray(targets)
    x = logits.data
    z = x - x.max(-1, keepdims=True)
    e = xp.exp(z)
    p = e / e.sum(-1, keepdims=True)
    v = p.shape[-1]
    flat = p.reshape(-1, v)
    y = targets.ravel()
    rows = xp.arange(y.size)
    keep = y >= 0
    n = int(keep.sum())
    if n:
        loss = -xp.log(flat[rows[keep], y[keep]] + 1e-9).mean()
    else:
        loss = xp.float32(0.0)
    out = T(loss, logits.requires_grad, (logits,))

    def bw():
        d = flat.copy()
        if n:
            d[rows[keep], y[keep]] -= 1.0
            d *= xp.float32(1.0 / n)
        else:
            d[...] = 0
        logits.grad = _acc(logits.grad, d.reshape(logits.data.shape))

    out._backward = bw
    return out


def backward(root):
    topo, seen = [], set()

    def visit(t):
        if id(t) in seen:
            return
        seen.add(id(t))
        for p in t._parents:
            visit(p)
        topo.append(t)

    visit(root)
    root.grad = xp.ones_like(root.data)
    for t in reversed(topo):
        if t._backward is not None and t.grad is not None:
            t._backward()