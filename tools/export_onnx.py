import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def build(cfg, params, out, opset=18):
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    T = int(cfg["block_size"])
    V = int(cfg["vocab_size"])
    D = int(cfg["n_embd"])
    H = int(cfg["n_head"])
    hd = D // H
    L = int(cfg["n_layer"])
    norm = cfg.get("norm", "ln")
    ffn = cfg.get("ffn", "gelu")
    rope = bool(cfg.get("rope", False)) and hd % 2 == 0
    kv = int(cfg.get("n_kv_head", H))
    if kv < 1 or H % kv or D % H:
        kv = H
    base = float(cfg.get("rope_base", 10000.0))

    nodes, inits = [], []

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))

    def i64(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))

    def node(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))

    init("wte", params["wte"])
    i64("ax", np.array([-1], dtype=np.int64))
    init("eps", np.array([1e-5], dtype=np.float32))
    node("Transpose", ["wte"], ["wte_t"], perm=[1, 0])

    if rope:
        node("Gather", ["wte", "ids"], ["x"], axis=0)
        x = "x"
    else:
        init("wpe", params["wpe"])
        i64("pos", np.arange(T, dtype=np.int64)[None, :])
        node("Gather", ["wte", "ids"], ["xe"], axis=0)
        node("Gather", ["wpe", "pos"], ["pe"], axis=0)
        node("Add", ["xe", "pe"], ["x"])
        x = "x"

    mask = np.triu(np.full((1, 1, T, T), -1e9, dtype=np.float32), 1)
    init("mask", mask)
    init("scale", np.array([1.0 / np.sqrt(hd)], dtype=np.float32))

    if rope:
        half = hd // 2
        ang = (np.arange(T, dtype=np.float64)[:, None]
               * (base ** (-np.arange(half, dtype=np.float64) * 2.0 / hd))[None, :])
        init("cos", np.cos(ang).astype(np.float32)[None, None])
        init("sin", np.sin(ang).astype(np.float32)[None, None])
        i64("split", np.array([half, half], dtype=np.int64))

    def norm_block(src, prefix, out_name):
        g = prefix + ".g"
        init(g, params[g])
        if norm == "rms":
            node("Mul", [src, src], [out_name + "_sq"])
            node("ReduceMean", [out_name + "_sq", "ax"], [out_name + "_ms"], keepdims=1)
            node("Add", [out_name + "_ms", "eps"], [out_name + "_ms2"])
            node("Sqrt", [out_name + "_ms2"], [out_name + "_rs"])
            node("Div", [src, out_name + "_rs"], [out_name + "_xh"])
            node("Mul", [out_name + "_xh", g], [out_name])
            return out_name
        b = prefix + ".b"
        init(b, params[b])
        node("ReduceMean", [src, "ax"], [out_name + "_mu"], keepdims=1)
        node("Sub", [src, out_name + "_mu"], [out_name + "_xc"])
        node("Mul", [out_name + "_xc", out_name + "_xc"], [out_name + "_sq"])
        node("ReduceMean", [out_name + "_sq", "ax"], [out_name + "_var"], keepdims=1)
        node("Add", [out_name + "_var", "eps"], [out_name + "_v2"])
        node("Sqrt", [out_name + "_v2"], [out_name + "_rs"])
        node("Div", [out_name + "_xc", out_name + "_rs"], [out_name + "_xh"])
        node("Mul", [out_name + "_xh", g], [out_name + "_m"])
        node("Add", [out_name + "_m", b], [out_name])
        return out_name

    def gelu(src, dst):
        c = np.float32(np.sqrt(2.0 / np.pi))
        a = np.float32(0.044715)
        init(dst + "_c", np.array([c], dtype=np.float32))
        init(dst + "_a", np.array([a], dtype=np.float32))
        init(dst + "_h", np.array([0.5], dtype=np.float32))
        init(dst + "_o", np.array([1.0], dtype=np.float32))
        init(dst + "_3", np.array([3.0], dtype=np.float32))
        node("Pow", [src, dst + "_3"], [dst + "_x3"])
        node("Mul", [dst + "_x3", dst + "_a"], [dst + "_ax3"])
        node("Add", [src, dst + "_ax3"], [dst + "_s"])
        node("Mul", [dst + "_s", dst + "_c"], [dst + "_u"])
        node("Tanh", [dst + "_u"], [dst + "_t"])
        node("Add", [dst + "_t", dst + "_o"], [dst + "_p"])
        node("Mul", [src, dst + "_p"], [dst + "_m"])
        node("Mul", [dst + "_m", dst + "_h"], [dst])

    for i in range(L):
        p = f"h{i}"
        n1 = norm_block(x, p + ".ln1", f"n1_{i}")
        for w in (".wq", ".wk", ".wv", ".wo"):
            init(p + w, params[p + w])
        node("MatMul", [n1, p + ".wq"], [f"q_{i}"])
        node("MatMul", [n1, p + ".wk"], [f"k_{i}"])
        node("MatMul", [n1, p + ".wv"], [f"v_{i}"])
        i64(f"sh_q{i}", np.array([1, T, H, hd], dtype=np.int64))
        i64(f"sh_k{i}", np.array([1, T, kv, hd], dtype=np.int64))
        node("Reshape", [f"q_{i}", f"sh_q{i}"], [f"qr_{i}"])
        node("Transpose", [f"qr_{i}"], [f"qt_{i}"], perm=[0, 2, 1, 3])
        node("Reshape", [f"k_{i}", f"sh_k{i}"], [f"kr_{i}"])
        node("Transpose", [f"kr_{i}"], [f"kt_{i}"], perm=[0, 2, 1, 3])
        node("Reshape", [f"v_{i}", f"sh_k{i}"], [f"vr_{i}"])
        node("Transpose", [f"vr_{i}"], [f"vt_{i}"], perm=[0, 2, 1, 3])
        if rope:
            for tag in ("q", "k"):
                pfx = f"ro{tag}{i}_"
                node("Split", [f"{tag}t_{i}", "split"], [pfx + "a", pfx + "b"], axis=-1)
                node("Mul", [pfx + "a", "cos"], [pfx + "c"])
                node("Mul", [pfx + "b", "sin"], [pfx + "d"])
                node("Sub", [pfx + "c", pfx + "d"], [pfx + "e"])
                node("Mul", [pfx + "a", "sin"], [pfx + "f"])
                node("Mul", [pfx + "b", "cos"], [pfx + "g"])
                node("Add", [pfx + "f", pfx + "g"], [pfx + "h"])
                node("Concat", [pfx + "e", pfx + "h"], [f"{tag}rope_{i}"], axis=-1)
            qs, ks = f"qrope_{i}", f"krope_{i}"
        else:
            qs, ks = f"qt_{i}", f"kt_{i}"
        if kv != H:
            i64(f"gidx{i}", np.repeat(np.arange(kv, dtype=np.int64), H // kv))
            node("Gather", [ks, f"gidx{i}"], [f"kf_{i}"], axis=1)
            node("Gather", [f"vt_{i}", f"gidx{i}"], [f"vf_{i}"], axis=1)
        else:
            node("Identity", [ks], [f"kf_{i}"])
            node("Identity", [f"vt_{i}"], [f"vf_{i}"])
        node("Transpose", [f"kf_{i}"], [f"ktt_{i}"], perm=[0, 1, 3, 2])
        node("MatMul", [qs, f"ktt_{i}"], [f"att_{i}"])
        node("Mul", [f"att_{i}", "scale"], [f"atts_{i}"])
        node("Add", [f"atts_{i}", "mask"], [f"attm_{i}"])
        node("Softmax", [f"attm_{i}"], [f"prob_{i}"], axis=-1)
        node("MatMul", [f"prob_{i}", f"vf_{i}"], [f"ctx_{i}"])
        i64(f"csh{i}", np.array([1, T, D], dtype=np.int64))
        node("Transpose", [f"ctx_{i}"], [f"ct_{i}"], perm=[0, 2, 1, 3])
        node("Reshape", [f"ct_{i}", f"csh{i}"], [f"cf_{i}"])
        node("MatMul", [f"cf_{i}", p + ".wo"], [f"ao_{i}"])
        node("Add", [x, f"ao_{i}"], [f"x1_{i}"])

        n2 = norm_block(f"x1_{i}", p + ".ln2", f"n2_{i}")
        init(p + ".w1", params[p + ".w1"])
        init(p + ".w2", params[p + ".w2"])
        init(p + ".b2", params[p + ".b2"])
        if ffn == "swiglu":
            init(p + ".w3", params[p + ".w3"])
            node("MatMul", [n2, p + ".w1"], [f"g1_{i}"])
            node("MatMul", [n2, p + ".w3"], [f"g3_{i}"])
            node("Sigmoid", [f"g1_{i}"], [f"sg_{i}"])
            node("Mul", [f"g1_{i}", f"sg_{i}"], [f"sw_{i}"])
            node("Mul", [f"sw_{i}", f"g3_{i}"], [f"swg_{i}"])
            node("MatMul", [f"swg_{i}", p + ".w2"], [f"mm_{i}"])
        else:
            init(p + ".b1", params[p + ".b1"])
            node("MatMul", [n2, p + ".w1"], [f"g1_{i}"])
            node("Add", [f"g1_{i}", p + ".b1"], [f"g1b_{i}"])
            gelu(f"g1b_{i}", f"gel_{i}")
            node("MatMul", [f"gel_{i}", p + ".w2"], [f"mm_{i}"])
        node("Add", [f"mm_{i}", p + ".b2"], [f"ml_{i}"])
        node("Add", [f"x1_{i}", f"ml_{i}"], [f"x2_{i}"])
        x = f"x2_{i}"

    nf = norm_block(x, "lnf", "fn")
    node("MatMul", [nf, "wte_t"], ["logits"])

    graph = helper.make_graph(
        nodes, "danyai",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, T])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, T, V])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out)


def main():
    try:
        import onnx  # noqa: F401
    except Exception:
        raise SystemExit("onnx not installed: pip install onnx")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/model.onnx")
    args = ap.parse_args()

    z = np.load(args.ckpt, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    qcfg = {}
    if "quant" in z.files:
        qcfg = json.loads(str(z["quant"]))
    params = {}
    for k in z.files:
        if k in ("cfg", "quant", "step", "best"):
            continue
        arr = z[k]
        if k in qcfg:
            params[k] = arr.astype(np.float32) * np.float32(qcfg[k])
        else:
            params[k] = arr.astype(np.float32)
    build(cfg, params, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()