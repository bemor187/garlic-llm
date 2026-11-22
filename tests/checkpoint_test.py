"""Tests for pc/checkpoint.py (the converters' check of a Qwen3.5-9B checkpoint) on small local checkpoints made here:
their shards hold the headers of every tensor the converters read, and data only for a few small test tensors. No
network. Run: python tests/checkpoint_test.py"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pc"))
from checkpoint import (FLOATS, LAYER_TYPES, SHAPE, STOCK, Checkpoint, CheckpointError, check,  # noqa: E402
                        expected)

fails = checks = 0


def ok(cond, what):
    global fails, checks
    checks += 1
    if not cond:
        fails += 1
        print("FAIL:", what)


def config(text_only=False, **change):
    c = dict(SHAPE, rms_norm_eps=FLOATS["rms_norm_eps"], layer_types=list(LAYER_TYPES))
    rope = dict(rope_theta=FLOATS["rope_theta"], partial_rotary_factor=FLOATS["partial_rotary_factor"])
    if text_only:   # a config of the text model alone, RoPE at the top
        return dict(c, **rope, tie_word_embeddings=False, **change)
    return {"tie_word_embeddings": False, "text_config": dict(dict(c, rope_parameters=rope), **change)}


SMALL = {"test.bf16": ("BF16", np.array([[1.5, -2.0, 0.25]], np.float32)),
         "test.f16": ("F16", np.array([3.0, -0.5], np.float32)),
         "test.f32": ("F32", np.array([[1e-3], [7.0]], np.float32))}


def make(folder, prefix="model.language_model.", mtp=True, text_only=False, shapes=None, dtypes=None, index=True,
         **change):
    """A checkpoint in folder: two shards, every tensor of expected() (with shapes / dtypes overriding some)."""
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config(text_only, **change)))
    names = dict(expected(prefix, mtp))
    names.update(shapes or {})
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    data = b""
    for name, (dtype, x) in SMALL.items():   # real data for these
        raw = (x.view(np.uint32) >> 16).astype(np.uint16).tobytes() if dtype == "BF16" else \
            x.astype(np.float16 if dtype == "F16" else np.float32).tobytes()
        shards["model-00001-of-00002.safetensors"][name] = dict(dtype=dtype, shape=list(x.shape),
                                                                data_offsets=[len(data), len(data) + len(raw)])
        data += raw
    end = len(data)
    for i, (name, shape) in enumerate(sorted(names.items())):   # headers only: their data is never read
        size = int(np.prod(shape)) * 2
        shard = list(shards)[i % 2]
        shards[shard][name] = dict(dtype=(dtypes or {}).get(name, "BF16"), shape=shape, data_offsets=[end, end + size])
        end += size
    weight_map = {}
    for shard, h in shards.items():
        head = json.dumps(h).encode()
        (folder / shard).write_bytes(struct.pack("<Q", len(head)) + head + (data if "00001" in shard else b""))
        weight_map.update({n: shard for n in h})
    if index:
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return folder


def refused(folder, *, mtp=False, says):
    try:
        check(Checkpoint(folder), mtp=mtp)
    except CheckpointError as e:
        ok(says in str(e), f"{folder.name}: the message names {says!r}: {e}")
        return
    ok(False, f"{folder.name}: accepted")


def main():
    with tempfile.TemporaryDirectory(prefix="checkpoint_test_") as t:
        run(Path(t))
    print(f"{checks} checks, {fails} failed")
    return fails != 0


def run(tmp):
    # Qwen3.5-9B's own layout, with its MTP head
    ck = Checkpoint(make(tmp / "stock"))
    check(ck)
    check(ck, mtp=True)
    ok(ck.prefix == "model.language_model." and ck.has_mtp, "stock layout: prefix and MTP head")
    for name, (dtype, x) in SMALL.items():
        got = ck.tensor(name)
        ok(got.dtype == np.float32 and got.shape == x.shape and np.allclose(got, x, rtol=1e-3), f"read {dtype}")
    # the text model alone (model.* names, config without text_config), without an MTP head
    ck = Checkpoint(make(tmp / "text_only", prefix="model.", mtp=False, text_only=True))
    check(ck)
    ok(ck.prefix == "model." and not ck.has_mtp, "text-only layout: prefix, no MTP head")
    refused(tmp / "text_only", mtp=True, says="no tensor mtp.")
    # what is refused, and the message says why
    p = "model.language_model.layers.3.self_attn.k_proj.weight"
    refused(make(tmp / "shape", shapes={p: [2048, 4096]}), says=p)
    refused(make(tmp / "config", hidden_size=5120), says="hidden_size")
    refused(make(tmp / "layers", layer_types=["full_attention"] * 32), says="layer_types")
    refused(make(tmp / "rope", rope_parameters=dict(rope_theta=1e6, partial_rotary_factor=0.25)), says="rope_theta")
    refused(make(tmp / "dtype", dtypes={"lm_head.weight": "I8"}), says="dtype I8")
    try:
        Checkpoint(make(tmp / "no_index", index=False))
        ok(False, "a checkpoint without model.safetensors.index.json is accepted")
    except CheckpointError as e:
        ok("model.safetensors.index.json" in str(e), f"no index: {e}")
    for bad in ("not a repository", "a/b/c"):
        try:
            Checkpoint(bad)
            ok(False, f"{bad!r} is taken as a repository")
        except CheckpointError:
            ok(True, "")
    ok(STOCK == "Qwen/Qwen3.5-9B", "the built-in model's repository")


if __name__ == "__main__":
    sys.exit(main())
