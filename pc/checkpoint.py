"""A Qwen3.5-9B checkpoint for the converters (pc/modal_qwen35_9b.py, pc/convert_9b_q4s.py): a Hugging Face repository,
read by byte ranges (only the parts asked for are downloaded), or a local folder. It must be laid out as
Qwen/Qwen3.5-9B: config.json and sharded safetensors with model.safetensors.index.json.

check() compares its configuration and the names and shapes of every tensor the app uses with Qwen3.5-9B's, before
anything is converted: the app's engine, kernels and model.idx are made for exactly those shapes, so a fine-tune of
Qwen3.5-9B converts and other models do not.

  python checkpoint.py <owner/name or folder>     -> says whether the converters take it, and if not, why

"owner/name@revision" reads a branch, tag or commit other than main. A private or gated repository needs a Hugging
Face token in the HF_TOKEN environment variable.
"""
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

STOCK = "Qwen/Qwen3.5-9B"
# Qwen3.5-9B's text model (config.json "text_config"): what the app is built for
SHAPE = dict(num_hidden_layers=32, hidden_size=4096, intermediate_size=12288, num_attention_heads=16,
             num_key_value_heads=4, head_dim=256, vocab_size=248320, linear_num_key_heads=16,
             linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)
FLOATS = dict(rms_norm_eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25)
LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 8
DTYPES = ("BF16", "F16", "F32")
MTP = "mtp."   # the MTP head's tensors (Qwen3.5's multi-token prediction layer)


class CheckpointError(Exception):
    pass


def expected(prefix, mtp=False):
    """{tensor name: shape} of every tensor the converters read (the MTP head's with mtp)."""
    c = SHAPE
    H, I, V, hd = c["hidden_size"], c["intermediate_size"], c["vocab_size"], c["head_dim"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    nkh, nvh, dk, dv = (c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                        c["linear_value_head_dim"])

    def attention(p):   # a full-attention layer: q_proj holds the query and its output gate, per head
        return {p + "self_attn.q_proj.weight": [2 * nh * hd, H], p + "self_attn.k_proj.weight": [nkv * hd, H],
                p + "self_attn.v_proj.weight": [nkv * hd, H], p + "self_attn.o_proj.weight": [H, nh * hd],
                p + "self_attn.q_norm.weight": [hd], p + "self_attn.k_norm.weight": [hd]}

    def mlp_norms(p):
        return {p + "input_layernorm.weight": [H], p + "post_attention_layernorm.weight": [H],
                p + "mlp.gate_proj.weight": [I, H], p + "mlp.up_proj.weight": [I, H], p + "mlp.down_proj.weight": [H, I]}

    t = {prefix + "embed_tokens.weight": [V, H], prefix + "norm.weight": [H], "lm_head.weight": [V, H]}
    conv = 2 * nkh * dk + nvh * dv
    for i, kind in enumerate(LAYER_TYPES):
        p = f"{prefix}layers.{i}."
        t.update(mlp_norms(p))
        if kind == "full_attention":
            t.update(attention(p))
        else:
            a = p + "linear_attn."
            t.update({a + "in_proj_qkv.weight": [conv, H], a + "in_proj_z.weight": [nvh * dv, H],
                      a + "in_proj_b.weight": [nvh, H], a + "in_proj_a.weight": [nvh, H],
                      a + "conv1d.weight": [conv, 1, c["linear_conv_kernel_dim"]], a + "A_log": [nvh],
                      a + "dt_bias": [nvh], a + "norm.weight": [dv], a + "out_proj.weight": [H, nvh * dv]})
    if mtp:
        t.update({MTP + "pre_fc_norm_embedding.weight": [H], MTP + "pre_fc_norm_hidden.weight": [H],
                  MTP + "fc.weight": [H, 2 * H], MTP + "norm.weight": [H]})
        t.update(mlp_norms(MTP + "layers.0."))
        t.update(attention(MTP + "layers.0."))
    return t


class _NoAuthAcrossHosts(urllib.request.HTTPRedirectHandler):
    """Follows redirects (Hugging Face sends file contents from another host) without taking the token along."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_NoAuthAcrossHosts)


class Checkpoint:
    def __init__(self, src):
        self.src = str(src)
        self.local = Path(self.src).is_dir()
        if not self.local:
            repo, _, rev = self.src.partition("@")
            if repo.count("/") != 1 or not all(repo.split("/")):
                raise CheckpointError(f"{self.src}: not a folder, nor a Hugging Face repository such as {STOCK}")
            self.repo, self.rev = repo, rev or "main"
        self.headers = {}
        self.config = self.json("config.json")
        self.text_config = self.config.get("text_config", self.config)
        try:
            self.weight_map = self.json("model.safetensors.index.json")["weight_map"]
        except CheckpointError as e:
            raise CheckpointError(f"{e}\nThe converters take sharded safetensors with model.safetensors.index.json, as "
                                  f"{STOCK} has; save the model that way (save_pretrained does) and try again.")
        # the text model's tensors: model.language_model.* as in Qwen3.5-9B, or model.* when only the text model
        # was saved
        self.prefix = next((p for p in ("model.language_model.", "model.") if p + "embed_tokens.weight" in
                            self.weight_map), None)
        if self.prefix is None:
            raise CheckpointError(f"{self.src}: no model.language_model.embed_tokens.weight (nor "
                                  "model.embed_tokens.weight) in model.safetensors.index.json: not a Qwen3.5-9B")
        self.has_mtp = all(n in self.weight_map for n in expected(self.prefix, mtp=True) if n.startswith(MTP))

    # ---- reading ----

    def _url(self, name):
        return f"https://huggingface.co/{self.repo}/resolve/{urllib.parse.quote(self.rev, safe='')}/{name}"

    def _get(self, name, start=None, end=None):
        """Bytes [start, end) of a file (the whole file without start), with a few retries."""
        if self.local:
            p = Path(self.src) / name
            if not p.is_file():
                raise CheckpointError(f"{self.src}: no {name}")
            with open(p, "rb") as f:
                if start is None:
                    return f.read()
                f.seek(start)
                data = f.read(end - start)
            if len(data) != end - start:
                raise CheckpointError(f"{p}: shorter than its header says")
            return data
        headers = {"User-Agent": "garlic-llm-converter"}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        if start is not None:
            headers["Range"] = f"bytes={start}-{end - 1}"
        for attempt in range(5):
            try:
                with _opener.open(urllib.request.Request(self._url(name), headers=headers), timeout=600) as r:
                    data = r.read()
                if start is None or len(data) == end - start:
                    return data
                err = f"{len(data)} bytes of {end - start}"
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    why = {404: "not found"}.get(e.code, "no access: a private or gated repository needs HF_TOKEN")
                    raise CheckpointError(f"{self.repo}: {name}: HTTP {e.code} ({why})")
                err = str(e)
            except (urllib.error.URLError, OSError) as e:   # (a long download may drop: retry)
                err = str(e)
            print(f"  retry {name}: {err}", flush=True)
            time.sleep(2 * (attempt + 1))
        raise CheckpointError(f"{self.repo}: cannot read {name}: {err}")

    def json(self, name):
        try:
            return json.loads(self._get(name))
        except ValueError:
            raise CheckpointError(f"{self.src}: {name} is not JSON")

    def header(self, shard):
        """(where the shard's data starts, its safetensors header)."""
        if shard not in self.headers:
            n = struct.unpack("<Q", self._get(shard, 0, 8))[0]
            if not 2 <= n <= 1 << 28:
                raise CheckpointError(f"{self.src}: {shard} is not a safetensors file")
            self.headers[shard] = (8 + n, json.loads(self._get(shard, 8, 8 + n)))
        return self.headers[shard]

    def info(self, name):
        """(dtype, shape) of a tensor."""
        if name not in self.weight_map:
            raise CheckpointError(f"{self.src}: no tensor {name}")
        e = self.header(self.weight_map[name])[1].get(name)
        if e is None:
            raise CheckpointError(f"{self.src}: {self.weight_map[name]} has no {name}, though the index says so")
        return e["dtype"], e["shape"]

    def tensor(self, name):
        """A tensor as float32 (from BF16, F16 or F32)."""
        shard = self.weight_map[name]
        base, h = self.header(shard)
        e = h[name]
        a, b = e["data_offsets"]
        raw = self._get(shard, base + a, base + b)
        if e["dtype"] == "BF16":
            x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif e["dtype"] == "F16":
            x = np.frombuffer(raw, np.float16).astype(np.float32)
        elif e["dtype"] == "F32":
            x = np.frombuffer(raw, np.float32).copy()
        else:
            raise CheckpointError(f"{self.src}: {name} is {e['dtype']}; the converters read {', '.join(DTYPES)}")
        return x.reshape(e["shape"])


def check(ck, mtp=False):
    """Raises CheckpointError, with every difference found (the first ones), unless the checkpoint has Qwen3.5-9B's
    configuration and the names, shapes and dtypes of its tensors (the MTP head's too with mtp)."""
    problems = []
    c = ck.text_config
    for k, v in SHAPE.items():
        if c.get(k) != v:
            problems.append(f"config.json: {k} is {c.get(k)}, Qwen3.5-9B has {v}")
    rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
    got = dict(rms_norm_eps=c.get("rms_norm_eps"), rope_theta=rope.get("rope_theta", c.get("rope_theta")),
               partial_rotary_factor=rope.get("partial_rotary_factor", c.get("partial_rotary_factor")))
    for k, v in FLOATS.items():
        if not isinstance(got[k], (int, float)) or abs(got[k] - v) > 1e-6 * v:
            problems.append(f"config.json: {k} is {got[k]}, Qwen3.5-9B has {v:g}")
    if c.get("layer_types") != LAYER_TYPES:
        problems.append("config.json: layer_types differ from Qwen3.5-9B's (three linear-attention layers, then one "
                        "full-attention layer, eight times)")
    if c.get("tie_word_embeddings", ck.config.get("tie_word_embeddings")):
        problems.append("config.json: tie_word_embeddings is true; Qwen3.5-9B has its own lm_head")
    for name, shape in expected(ck.prefix, mtp).items():
        if len(problems) >= 12:
            break
        if name not in ck.weight_map:
            problems.append(f"no tensor {name}")
            continue
        dtype, got_shape = ck.info(name)
        if got_shape != shape:
            problems.append(f"{name}: shape {got_shape}, Qwen3.5-9B has {shape}")
        elif dtype not in DTYPES:
            problems.append(f"{name}: dtype {dtype}; the converters read {', '.join(DTYPES)}")
    if problems:
        raise CheckpointError(f"{ck.src} does not have Qwen3.5-9B's shapes:\n  " + "\n  ".join(problems) +
                              "\nThe app runs Qwen3.5-9B and fine-tunes of it with the same shapes, tensor names "
                              "and file layout; other models do not convert.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    try:
        ck = Checkpoint(args[0])
        check(ck)
        if ck.has_mtp:
            check(ck, mtp=True)
    except CheckpointError as e:
        sys.exit(str(e))
    dtypes = sorted({ck.info(n)[0] for n in ("lm_head.weight", ck.prefix + "embed_tokens.weight")})
    print(f"{ck.src}: Qwen3.5-9B's shapes ({len(expected(ck.prefix))} tensors checked, {'/'.join(dtypes)}, text "
          f"tensors under {ck.prefix}*). MTP head: {'yes' if ck.has_mtp else 'no (the app uses the built-in one)'}.")


if __name__ == "__main__":
    main()
