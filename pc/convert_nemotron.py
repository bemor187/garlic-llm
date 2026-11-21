"""Convert Nemotron 3.5 ASR (models/nemotron-3.5-asr-streaming-0.6b/model.safetensors) to the PS4 format of
convert_parakeet.py: one file, tensors on 256-byte boundaries, index in models/<name>.json.

  q4i   all big matrices (shaders/parakeet.cl: mm_q4i), rows padded to a multiple of 64. The encoder's and the
        prompt projector's come already rounded by gptq_nemotron.py (--rounded): rounded to nearest, Japanese and
        Chinese lose about half again as many characters. The rest are rounded to nearest, each block's scale the
        one with the least squared error (q4.quantize_q4_mse).
  q8i   the joint head (shaders/nemotron.cl: mm_q8i): in 4 bits alone it raises the word errors by half
  f16   per layer, the positional rows of the 83 relative distances the chunked attention uses (nemotron_ref.py:
        rel_table), so linear_pos is not needed on the console; the prompt projector's bias for each of the 128
        language prompts
  f32   norms, biases, bias_u / bias_v, depthwise kernels, the small subsampling convolutions (pointwise ones
        transposed: Wt[c][o]), the mel filter bank

The file stays smaller than Parakeet's (the same memory on the console).

Run: python convert_nemotron.py --rounded   -> models/nemotron-3.5-asr-q4i.bin + .json (--rounded: with
                                             models/nemotron-gptq.npz, or the file named after it)
     python convert_nemotron.py --assets   -> chat/assets/asr_nm.idx + asr_nm_vocab.bin (from the .json)
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

import nemotron_ref as N
from parakeet_ref import mel_filters
from q4 import dequantize_q4, interleave_q4, quantize_q4_mse

ROOT = Path(__file__).resolve().parent.parent
NAME = "nemotron-3.5-asr-q4i"
VOCAB_PAD = 13120                  # 13088 rounded up to a multiple of 64


def quantize_q8(w):
    """w [rows, cols] -> (int8 [rows, cols], fp16 scales [rows, cols / 32]): q = round(w / d), d = max |w| / 127"""
    rows, cols = w.shape
    b = w.astype(np.float32).reshape(rows, cols // 32, 32)
    d16 = (np.abs(b).max(axis=2) / 127.0).astype(np.float16)
    d = d16.astype(np.float32)
    q = np.clip(np.rint(b / np.where(d == 0, 1.0, d)[..., None]), -127, 127).astype(np.int8)
    return q.reshape(rows, cols), d16


def interleave_q8(q, d16):
    """q8i: as q4i (q4.interleave_q4), but 32 bytes per row and block"""
    rows, cols = q.shape
    nb = cols // 32
    nib = q.reshape(rows // 64, 64, nb, 32).transpose(0, 2, 1, 3)
    sc = d16.reshape(rows // 64, 64, nb).transpose(0, 2, 1)
    return np.ascontiguousarray(nib).tobytes(), np.ascontiguousarray(sc).tobytes()


def main(rounded=None):
    """rounded: an .npz of matrices already on the q4i grid ("<tensor>|packed", "<tensor>|d16", as gptq_nemotron.py
    writes them), used instead of rounding those to nearest"""
    w = N.prepare(N.load_weights())
    pre = np.load(rounded) if rounded else {}
    out = open(ROOT / "models" / f"{NAME}.bin", "wb")
    index, offset = {}, 0

    def put(key, data, **meta):
        nonlocal offset
        pad = (-offset) % 256
        out.write(bytes(pad))
        offset += pad
        out.write(data)
        index[key] = dict(offset=offset, size=len(data), **meta)
        offset += len(data)

    def f32(key, a):
        put(key, np.ascontiguousarray(a, dtype=np.float32).tobytes(), n=int(np.size(a)))

    def f16(key, a):
        a = np.asarray(a)
        put(key, np.ascontiguousarray(a, dtype=np.float16).tobytes(), n=int(a.size), fmt="f16",
            rows=a.shape[0], cols=a.shape[1])

    def q4i(key, m, bits=4):
        m = np.ascontiguousarray(m, dtype=np.float32)
        rows = m.shape[0]
        if rows % 64:
            m = np.concatenate([m, np.zeros((64 - rows % 64, m.shape[1]), np.float32)])
        if bits == 8:
            nib, sc = interleave_q8(*quantize_q8(m))
        elif f"{key}|packed" in pre:
            assert pre[f"{key}|packed"].shape == (m.shape[0], m.shape[1] // 2), key
            nib, sc = interleave_q4(pre[f"{key}|packed"], pre[f"{key}|d16"])
        else:
            nib, sc = interleave_q4(*quantize_q4_mse(m))
        put(key, nib + sc, rows=m.shape[0], cols=m.shape[1], scales=len(nib), fmt=f"q{bits}i", true_rows=rows)

    S = "encoder.subsampling."
    f32("mel", mel_filters())                                          # [128][257]
    f32("sub0_w", w[S + "conv_in.weight"].reshape(256, 9))
    f32("sub0_b", w[S + "conv_in.bias"])
    for k in (1, 2):
        p = f"{S}layers.{k - 1}."
        f32(f"sub{k}_dw", w[p + "depthwise_conv.weight"].reshape(256, 9))
        f32(f"sub{k}_dwb", w[p + "depthwise_conv.bias"])
        f32(f"sub{k}_pwt", w[p + "pointwise_conv.weight"].reshape(256, 256).T)
        f32(f"sub{k}_pwb", w[p + "pointwise_conv.bias"])
    # linear 4352 -> 1024 over (channel, freq) flattened channel-major; the GPU has [freq][channel]
    lin = w[S + "linear.weight"].reshape(1024, 256, 17).transpose(0, 2, 1).reshape(1024, 4352)
    q4i("sub_lin", lin)
    f32("sub_lin_b", w[S + "linear.bias"])
    for i in range(24):
        p = f"encoder.layers.{i}."
        for n in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            f32(f"{i}.{n}.w", w[p + n + ".weight"])
            f32(f"{i}.{n}.b", w[p + n + ".bias"])
        for ff in ("feed_forward1", "feed_forward2"):
            q4i(f"{i}.{ff}.1", w[p + ff + ".linear1.weight"])
            q4i(f"{i}.{ff}.2", w[p + ff + ".linear2.weight"])
        a = p + "self_attn."
        q4i(f"{i}.wqkv", np.concatenate([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]]))
        q4i(f"{i}.wo", w[a + "o_proj.weight"])
        f32(f"{i}.bias_u", w[a + "bias_u"])
        f32(f"{i}.bias_v", w[a + "bias_v"])
        f16(f"{i}.rel", w[a + "rel_table"])
        c = p + "conv."
        q4i(f"{i}.pw1", w[c + "pointwise_conv1.weight"][:, :, 0])
        q4i(f"{i}.pw2", w[c + "pointwise_conv2.weight"][:, :, 0])
        f32(f"{i}.dw", w[c + "depthwise_conv.weight"][:, 0, :])
        f32(f"{i}.cn.w", w[c + "norm.weight"])
        f32(f"{i}.cn.b", w[c + "norm.bias"])
        print(f"layer {i}, {offset / 2**20:.0f} MiB", end="\r")
    q4i("prompt_w", w["prompt_projector.w_audio"])
    f16("prompt_b", w["prompt_projector.bias_by_prompt"])             # [128][2048]
    q4i("out_w", w["fused_out.weight"])
    f32("out_b", w["fused_out.bias"])
    q4i("dec_emb", w["decoder.embedding.weight"])
    for l in range(2):
        q4i(f"lstm{l}", np.concatenate([w[f"decoder.lstm.weight_ih_l{l}"], w[f"decoder.lstm.weight_hh_l{l}"]], axis=1))
        f32(f"lstm{l}_b", w[f"decoder.lstm.bias_ih_l{l}"] + w[f"decoder.lstm.bias_hh_l{l}"])
    q4i("dec_proj", w["decoder.decoder_projector.weight"])
    f32("dec_proj_b", w["decoder.decoder_projector.bias"])
    q4i("joint", w["joint.head.weight"], bits=8)
    jb = np.zeros(index["joint"]["rows"], np.float32)
    jb[:len(w["joint.head.bias"])] = w["joint.head.bias"]
    f32("joint_b", jb)
    out.close()
    meta = dict(size=offset, tensors=index, blank=N.BLANK, vocab=N.VOCAB, chunk=N.CHUNK, left_chunks=N.LEFT_CHUNKS,
                d_min=N.D_MIN, n_dist=N.N_DIST)
    (ROOT / "models" / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote models/{NAME}.bin: {offset / 2**20:.1f} MiB, {len(index)} tensors")


def weights_from_file(name=NAME):
    """The weight dict of nemotron_ref (after prepare), rebuilt from the converted file: what the PS4 computes with."""
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    raw = open(ROOT / "models" / f"{name}.bin", "rb").read()
    t = meta["tensors"]

    def get(k):
        e = t[k]
        b = raw[e["offset"]:e["offset"] + e["size"]]
        if e.get("fmt") == "f16":
            return np.frombuffer(b, np.float16).astype(np.float32).reshape(e["rows"], e["cols"])
        if e.get("fmt") not in ("q4i", "q8i"):
            return np.frombuffer(b, np.float32).copy()
        rows, cols = e["rows"], e["cols"]
        nb = cols // 32
        if e["fmt"] == "q8i":
            q = np.frombuffer(b[:e["scales"]], np.int8).reshape(rows // 64, nb, 64, 32).transpose(0, 2, 1, 3)
            sc = np.frombuffer(b[e["scales"]:], np.float16).reshape(rows // 64, nb, 64).transpose(0, 2, 1)
            m = q.reshape(rows, nb, 32).astype(np.float32) * sc.reshape(rows, nb).astype(np.float32)[..., None]
            return m.reshape(rows, cols)[:e["true_rows"]]
        nib = np.frombuffer(b[:e["scales"]], np.uint8).reshape(rows // 64, nb, 64, 16).transpose(0, 2, 1, 3)
        sc = np.frombuffer(b[e["scales"]:], np.float16).reshape(rows // 64, nb, 64).transpose(0, 2, 1)
        return dequantize_q4(nib.reshape(rows, cols // 2), sc.reshape(rows, nb))[:e["true_rows"]]

    w = N.prepare(N.load_weights())          # fp32 originals, then overwritten with the PS4's values
    S = "encoder.subsampling."
    w[S + "linear.weight"] = get("sub_lin").reshape(1024, 17, 256).transpose(0, 2, 1).reshape(1024, 4352)
    for i in range(24):
        p = f"encoder.layers.{i}."
        for ff in ("feed_forward1", "feed_forward2"):
            w[p + ff + ".linear1.weight"] = get(f"{i}.{ff}.1")
            w[p + ff + ".linear2.weight"] = get(f"{i}.{ff}.2")
        a = p + "self_attn."
        qkv = get(f"{i}.wqkv")
        w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"] = qkv[:1024], qkv[1024:2048], qkv[2048:]
        w[a + "o_proj.weight"] = get(f"{i}.wo")
        w[a + "rel_table"] = get(f"{i}.rel")
        c = p + "conv."
        w[c + "pointwise_conv1.weight"] = get(f"{i}.pw1")[:, :, None]
        w[c + "pointwise_conv2.weight"] = get(f"{i}.pw2")[:, :, None]
    w["prompt_projector.w_audio"] = get("prompt_w")
    w["prompt_projector.bias_by_prompt"] = get("prompt_b")
    w["fused_out.weight"] = get("out_w")
    w["decoder.embedding.weight"] = get("dec_emb")
    for l in range(2):
        m = get(f"lstm{l}")
        w[f"decoder.lstm.weight_ih_l{l}"], w[f"decoder.lstm.weight_hh_l{l}"] = m[:, :640], m[:, 640:]
    w["decoder.decoder_projector.weight"] = get("dec_proj")
    w["joint.head.weight"] = get("joint")
    return w


def export_app_assets(out_dir=ROOT / "chat" / "assets"):
    """asr_nm.idx: "NASR", u32 count, char[128] weights path, u64 file size, then per tensor <32sQQIIQ (name, offset,
    size, rows, cols, scales offset), as asr.idx. asr_nm_vocab.bin: "PVOC", u32 count (13088), then per token
    <H length + UTF-8 piece (U+2581 already a space; special tokens and the blank empty)."""
    out_dir = Path(out_dir)
    meta = json.loads((ROOT / "models" / f"{NAME}.json").read_text())
    recs = b"".join(struct.pack("<32sQQIIQ", k.encode(), e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0)) for k, e in meta["tensors"].items())
    head = struct.pack("<4sI128sQ", b"NASR", len(meta["tensors"]), f"/data/{NAME}.bin".encode(), meta["size"])
    (out_dir / "asr_nm.idx").write_bytes(head + recs)
    tj = json.loads((N.MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    special = {a["id"] for a in tj["added_tokens"] if a.get("special", True)}
    pieces = [b""] * N.VOCAB
    for p, i in tj["model"]["vocab"].items():
        if i < N.BLANK and i not in special:
            pieces[i] = p.replace("▁", " ").encode()
    blob = struct.pack("<4sI", b"PVOC", N.VOCAB) + b"".join(struct.pack("<H", len(p)) + p for p in pieces)
    (out_dir / "asr_nm_vocab.bin").write_bytes(blob)
    print(f"asr_nm.idx: {len(meta['tensors'])} tensors; asr_nm_vocab.bin: {len(blob)} bytes, {len(special)} special")


if __name__ == "__main__":
    if "--assets" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--assets"]
        export_app_assets(*rest)
    else:
        k = sys.argv.index("--rounded") if "--rounded" in sys.argv else -1
        main(None if k < 0 else sys.argv[k + 1] if k + 1 < len(sys.argv) else ROOT / "models" / "nemotron-gptq.npz")
