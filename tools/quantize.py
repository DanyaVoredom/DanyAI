import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def quantize(params, bits=8):
    qcfg = {}
    out = {}
    qmax = (1 << (bits - 1)) - 1
    for k, v in params.items():
        v = np.asarray(v, dtype=np.float32)
        if v.ndim < 2:
            out[k] = v
            continue
        m = float(np.max(np.abs(v)))
        if m <= 0:
            out[k] = v
            continue
        s = m / qmax
        out[k] = np.clip(np.round(v / s), -qmax, qmax).astype(np.int8)
        qcfg[k] = float(s)
    return out, qcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    z = np.load(args.ckpt, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    params = {k: z[k] for k in z.files if k not in ("cfg", "quant", "step", "best")}
    qparams, qcfg = quantize(params, args.bits)
    np.savez(args.out, cfg=np.array(json.dumps(cfg)),
             quant=np.array(json.dumps(qcfg)), **qparams)
    print(f"quantized {len(qcfg)} tensors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()