import numpy as np

DTYPE = np.float32


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
    inv = tuple(int(i) for i in np.argsort(axes))

    def bw():
        t.grad = _acc(t.grad, out.grad.transpose(inv))

    out._backward = bw
    return out


_GELU_C = np.float32(np.sqrt(2.0 / np.pi))
_GELU_A = np.float32(0.044715)
_GELU_3A = np.float32(3.0 * 0.044715)
_HALF = np.float32(0.5)
_ONE = np.float32(1.0)


def gelu(t):
    x = t.data
    x2 = x * x
    u = _GELU_C * (x + _GELU_A * x2 * x)
    th = np.tanh(u)
    out = T(_HALF * x * (_ONE + th), t.requires_grad, (t,))

    def bw():
        sech2 = _ONE - th * th
        du = _GELU_C * (_ONE + _GELU_3A * x2)
        d = _HALF * (_ONE + th) + _HALF * x * sech2 * du
        d *= out.grad
        t.grad = _acc(t.grad, d)

    out._backward = bw
    return out


def layernorm(x, g, b, eps=1e-5):
    xd = x.data
    mu = xd.mean(-1, keepdims=True)
    xc = xd - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = np.float32(1.0) / np.sqrt(var + eps)
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


def softmax(x):
    z = x.data - x.data.max(-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(-1, keepdims=True)
    out = T(p, x.requires_grad, (x,))

    def bw():
        g = out.grad
        s = (g * p).sum(-1, keepdims=True)
        x.grad = _acc(x.grad, p * (g - s))

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
    e = np.exp(z)
    p = e / e.sum(-1, keepdims=True)
    v = p.shape[-1]
    flat = p.reshape(-1, v)
    n = targets.size
    y = targets.ravel()
    rows = np.arange(n)
    loss = -np.log(flat[rows, y] + 1e-9).mean()
    out = T(loss, logits.requires_grad, (logits,))

    def bw():
        d = flat
        d[rows, y] -= 1.0
        d *= np.float32(1.0 / n)
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
    root.grad = np.ones_like(root.data)
    for t in reversed(topo):
        if t._backward is not None and t.grad is not None:
            t._backward()