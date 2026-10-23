import numpy as np

DTYPE = np.float32


def _acc(a, b):
    return b.copy() if a is None else a + b


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class T:
    def __init__(self, data, requires_grad=False, parents=(), backward=None):
        self.data = np.asarray(data, dtype=DTYPE)
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
    bb = np.swapaxes(b.data, -1, -2) if tb else b.data
    out = T(a.data @ bb, a.requires_grad or b.requires_grad, (a, b))

    def bw():
        if a.requires_grad:
            a.grad = _acc(a.grad, out.grad @ np.swapaxes(bb, -1, -2))
        if b.requires_grad:
            if tb:
                if b.data.ndim == 2:
                    gb = out.grad.reshape(-1, out.grad.shape[-1]).T @ a.data.reshape(-1, a.data.shape[-1])
                else:
                    gb = np.swapaxes(out.grad, -1, -2) @ a.data
            else:
                if b.data.ndim == 2:
                    gb = a.data.reshape(-1, a.data.shape[-1]).T @ out.grad.reshape(-1, out.grad.shape[-1])
                else:
                    gb = np.swapaxes(a.data, -1, -2) @ out.grad
            b.grad = _acc(b.grad, gb)

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
    inv = tuple(int(i) for i in np.argsort(axes))

    def bw():
        t.grad = _acc(t.grad, out.grad.transpose(inv))

    out._backward = bw
    return out


def gelu(t):
    c = np.float32(np.sqrt(2.0 / np.pi))
    x = t.data
    inner = c * (x + np.float32(0.044715) * x ** 3)
    th = np.tanh(inner)
    out = T(0.5 * x * (1.0 + th), t.requires_grad, (t,))

    def bw():
        dinner = c * (1.0 + np.float32(3 * 0.044715) * x ** 2)
        d = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th ** 2) * dinner
        t.grad = _acc(t.grad, out.grad * d)

    out._backward = bw
    return out


def layernorm(x, g, b, eps=1e-5):
    mu = x.data.mean(-1, keepdims=True)
    var = x.data.var(-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x.data - mu) / std
    out = T(xhat * g.data + b.data, x.requires_grad or g.requires_grad or b.requires_grad, (x, g, b))

    def bw():
        d = out.grad
        if g.requires_grad:
            g.grad = _acc(g.grad, (d * xhat).sum(axis=tuple(range(d.ndim - 1))))
        if b.requires_grad:
            b.grad = _acc(b.grad, d.sum(axis=tuple(range(d.ndim - 1))))
        if x.requires_grad:
            dxhat = d * g.data
            dx = (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
            x.grad = _acc(x.grad, dx)

    out._backward = bw
    return out


def softmax(x):
    z = x.data - x.data.max(-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(-1, keepdims=True)
    out = T(p, x.requires_grad, (x,))

    def bw():
        g = out.grad
        x.grad = _acc(x.grad, p * (g - (g * p).sum(-1, keepdims=True)))

    out._backward = bw
    return out


def embedding(w, idx):
    idx = np.asarray(idx)
    out = T(w.data[idx], w.requires_grad, (w,))

    def bw():
        g = np.zeros_like(w.data)
        np.add.at(g, idx.ravel(), out.grad.reshape(-1, w.data.shape[-1]))
        w.grad = _acc(w.grad, g)

    out._backward = bw
    return out


def cross_entropy(logits, targets):
    targets = np.asarray(targets)
    x = logits.data
    z = x - x.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1, keepdims=True))
    logp = z - lse
    n = targets.size
    flat = logp.reshape(-1, logp.shape[-1])
    loss = -flat[np.arange(n), targets.ravel()].mean()
    out = T(loss, logits.requires_grad, (logits,))

    def bw():
        p = np.exp(logp)
        p = p.reshape(-1, p.shape[-1]).copy()
        p[np.arange(n), targets.ravel()] -= 1.0
        logits.grad = _acc(logits.grad, (p / n).reshape(logits.data.shape))

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
    root.grad = np.ones_like(root.data)
    for t in reversed(topo):
        if t._backward is not None and t.grad is not None:
            t._backward()