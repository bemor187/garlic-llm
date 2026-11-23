"""GPTQ rounding for Nemotron 3.5 ASR's 4-bit matrices, on the q4i grid of convert_nemotron.py (blocks of 32 weights,
one fp16 scale each, value (q - 8) * d). Rounding each weight to the nearest step loses too much in Japanese and
Chinese; GPTQ (Frantar et al., arXiv:2210.17323) rounds a matrix column by column and moves each column's error onto
the columns not yet rounded, so that the matrix's outputs on real inputs change as little as possible
(H = X^T X of its inputs).

Layer by layer: the calibration frames go through the layers already rounded. Rounded here: the subsampling linear,
the 24 layers' matrices (qkv, o, both feed-forwards, both pointwise convs), the prompt projector's audio matrix and
the fused output matrix. The decoder and the joint head stay as the converter makes them.

Calibration: 20 sentences of FLEURS' dev split (google/fleurs, CC-BY 4.0) in each language Garlic dictates with
this model; the test split is left for measuring. --fetch downloads them (each archive is read only up to its
20th sentence; about 90 MB are kept).
Needs PyTorch (the CPU build is enough).

Run: python gptq_nemotron.py --fetch     -> models/fleurs_dev/
     python gptq_nemotron.py             -> models/nemotron-gptq.npz (about 15 min on a laptop CPU)
     python convert_nemotron.py --rounded
"""

import csv
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

import nemotron_ref as N
from parakeet_ref import conv2d, mel_filters, read_wav
from q4 import dequantize_q4

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "models" / "fleurs_dev"
OUT = ROOT / "models" / "nemotron-gptq.npz"
FLEURS = {"ar_eg": "ar", "ja_jp": "ja-JP", "ko_kr": "ko-KR", "cmn_hans_cn": "zh-CN", "tr_tr": "tr-TR", "vi_vn": "vi-VN"}
FACTORS = (1.0, 0.95, 0.9, 0.85, 0.8)
FP16_MIN = 6.2e-5


def fetch(out, per_lang=20):
    """the first per_lang distinct sentences of each language's dev split -> out/<fleurs lang>/*.wav + meta.json"""
    base = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
    for lang in FLEURS:
        d = Path(out) / lang
        d.mkdir(parents=True, exist_ok=True)
        tsv = urllib.request.urlopen(f"{base}/{lang}/dev.tsv", timeout=120).read().decode("utf-8")
        rows = {r[1]: dict(id=int(r[0]), text=r[3]) for r in csv.reader(io.StringIO(tsv), delimiter="\t",
                                                                          quoting=csv.QUOTE_NONE)}
        seen, meta = set(), []
        with urllib.request.urlopen(f"{base}/{lang}/audio/dev.tar.gz", timeout=120) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for m in tar:
                    name = m.name.rsplit("/", 1)[-1]
                    if not m.isfile() or name not in rows or rows[name]["id"] in seen:
                        continue
                    seen.add(rows[name]["id"])
                    file = f"{len(meta):02d}_{rows[name]['id']}.wav"
                    (d / file).write_bytes(tar.extractfile(m).read())
                    meta.append(dict(file=file, **rows[name]))
                    if len(meta) >= per_lang:
                        break
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(lang, len(meta), flush=True)


def block_scale(b):
    """[rows, 32] -> fp32 value of the fp16 scale with the least squared error among FACTORS (as q4.quantize_q4_mse)"""
    import torch
    peak = b.gather(1, b.abs().argmax(1, keepdim=True))
    best = best_err = None
    for f in FACTORS:
        d = peak / -8.0 * f
        d = torch.where(d.abs() < FP16_MIN, torch.where(d < 0, -FP16_MIN, FP16_MIN), d).half().float()
        q = torch.clamp(torch.round(b / d) + 8, 0, 15)
        err = (((q - 8) * d - b) ** 2).sum(1, keepdim=True)
        if best is None:
            best, best_err = d, err
        else:
            m = err < best_err
            best, best_err = torch.where(m, d, best), torch.where(m, err, best_err)
    return best[:, 0]


def gptq(W, H, block=128):
    """W [rows, cols] fp32, H [cols, cols] -> (packed uint8 [rows, cols / 2], d16 [rows, cols / 32]) as q4.py"""
    import torch
    W = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).clone()
    H = torch.from_numpy(H.astype(np.float32)).clone()
    rows, cols = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    for damp in (0.01, 0.03, 0.1, 0.3, 1.0):
        try:
            L = torch.linalg.cholesky(H + damp * torch.mean(torch.diag(H)) * torch.eye(cols))
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except Exception:   # noqa: BLE001 - not positive definite: more damping
            continue
    Q = torch.zeros(rows, cols, dtype=torch.uint8)
    D = torch.zeros(rows, cols // 32)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        d = None
        for i in range(i2 - i1):
            if i % 32 == 0:
                d = block_scale(W1[:, i:i + 32])
                D[:, (i1 + i) // 32] = d
            w = W1[:, i]
            q = torch.clamp(torch.round(w / d) + 8, 0, 15)
            Q[:, i1 + i] = q.to(torch.uint8)
            e = (w - (q - 8) * d) / Hi[i, i]
            W1[:, i:] -= e.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    q = Q.numpy().reshape(rows, cols // 32, 32)
    return (q[..., :16] | (q[..., 16:] << 4)).reshape(rows, cols // 2), D.half().numpy()


def hessian(xs):
    import torch
    X = torch.from_numpy(np.concatenate(xs).astype(np.float32))
    return (X.T @ X).double().numpy() * (2.0 / len(X))


def sub_flat(feat, w):
    """the subsampling up to its linear: [T3, 17 * 256] in the file's column order (frequency-major)"""
    S = "encoder.subsampling."
    x = np.maximum(N.causal_conv(feat[None], w[S + "conv_in.weight"], w[S + "conv_in.bias"], len(feat)), 0)
    for l in (0, 1):
        p = f"{S}layers.{l}."
        x = N.causal_conv(x, w[p + "depthwise_conv.weight"], w[p + "depthwise_conv.bias"], x.shape[1], groups=256)
        x = np.maximum(conv2d(x, w[p + "pointwise_conv.weight"], w[p + "pointwise_conv.bias"], 1, 0), 0)
    return x.transpose(1, 2, 0).reshape(x.shape[1], -1)


def main(src, out):
    w32 = N.prepare(N.load_weights())
    wq = dict(w32)
    fb = mel_filters()
    utts = []
    for lang, prompt in sorted(FLEURS.items()):
        for m in json.loads((src / lang / "meta.json").read_text(encoding="utf-8")):
            utts.append((N.features(read_wav(src / lang / m["file"]), fb), N.PROMPTS[prompt]))
    res = {}
    t0 = time.time()
    S = "encoder.subsampling."
    flats = [sub_flat(f, w32) for f, _ in utts]
    print(f"{len(utts)} sentences, {sum(len(x) for x in flats)} frames", flush=True)
    lin = w32[S + "linear.weight"].reshape(1024, 256, 17).transpose(0, 2, 1).reshape(1024, 4352)
    res["sub_lin"] = gptq(lin, hessian(flats))
    Wl = dequantize_q4(*res["sub_lin"])                             # file column order, as flats
    wq[S + "linear.weight"] = Wl.reshape(1024, 17, 256).transpose(0, 2, 1).reshape(1024, 4352)
    xs = [x @ Wl.T + w32[S + "linear.bias"] for x in flats]
    for i in range(24):
        p = f"encoder.layers.{i}."
        taps = {}
        for x in xs:
            N.encoder_layer(x, wq, i, lambda name, X: taps.setdefault(name, []).append(X))
        a, c = p + "self_attn.", p + "conv."
        mats = {   # file tensor: (its input, the checkpoint matrices stacked in it)
            f"{i}.feed_forward1.1": ("ff1.1", [p + "feed_forward1.linear1.weight"]),
            f"{i}.feed_forward1.2": ("ff1.2", [p + "feed_forward1.linear2.weight"]),
            f"{i}.wqkv": ("qkv", [a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"]),
            f"{i}.wo": ("o", [a + "o_proj.weight"]),
            f"{i}.pw1": ("pw1", [c + "pointwise_conv1.weight"]),
            f"{i}.pw2": ("pw2", [c + "pointwise_conv2.weight"]),
            f"{i}.feed_forward2.1": ("ff2.1", [p + "feed_forward2.linear1.weight"]),
            f"{i}.feed_forward2.2": ("ff2.2", [p + "feed_forward2.linear2.weight"]),
        }
        for key, (tap, names) in mats.items():
            W = np.concatenate([w32[n][:, :, 0] if w32[n].ndim == 3 else w32[n] for n in names])
            res[key] = gptq(W, hessian(taps[tap]))
            Wd, r0 = dequantize_q4(*res[key]), 0
            for n in names:
                rows = w32[n].shape[0]
                wq[n] = Wd[r0:r0 + rows][:, :, None] if w32[n].ndim == 3 else Wd[r0:r0 + rows]
                r0 += rows
        xs = [N.encoder_layer(x, wq, i) for x in xs]
        print(f"layer {i}: {time.time() - t0:.0f} s", flush=True)
    res["prompt_w"] = gptq(w32["prompt_projector.w_audio"], hessian(xs))
    wq["prompt_projector.w_audio"] = dequantize_q4(*res["prompt_w"])
    hs = [np.maximum(x @ wq["prompt_projector.w_audio"].T + w32["prompt_projector.bias_by_prompt"][pr], 0)
          for x, (_, pr) in zip(xs, utts)]
    res["out_w"] = gptq(w32["fused_out.weight"], hessian(hs))
    np.savez(out, **{f"{k}|packed": v[0] for k, v in res.items()}, **{f"{k}|d16": v[1] for k, v in res.items()})
    print(f"wrote {out}: {len(res)} matrices in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch(CALIB)
    else:
        main(CALIB, OUT)
