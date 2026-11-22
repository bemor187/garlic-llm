"""Qwen3.5-9B for the PS4, on Modal: conversion and reference run (see modal_qwen35.py for the 4B).

Mixed precision so it fits the ~4.4 GiB of GPU memory an app gets:
  MLP gate/up       q3i  (pc/q3.py: 3 bits + fp16 scale per 32, error-minimising scale), interleaved
  other matrices    q4i  (pc/q4.py, interleaved), including the untied output head (lm_head)
  input embedding   q4 rows kept OUT of GPU memory: the app reads the row of each token from the file

File: [GPU part: final_norm, layers, lm_head ...][embedding rows], index in <name>.json:
  gpu_size  bytes from the start that go to GPU memory
  embed     fmt "q4row": row r at offset + r * rowbytes, 2048 bytes of nibbles (as q4) then 128 fp16 scales
Per layer the same tensors as the 4B file (wgu fmt "q3i").

Run from pc/:
  python -m modal run modal_qwen35_9b.py::convert
  python -m modal run modal_qwen35_9b.py::reference --prompt "..." --n 30

Another model with Qwen3.5-9B's shapes (a fine-tune of it), from a public Hugging Face repository:
  python -m modal run modal_qwen35_9b.py::convert --src <owner>/<model> [--name <name>]
The checkpoint is checked on this PC first (pc/checkpoint.py: Qwen3.5-9B's configuration, tensor names and shapes,
sharded safetensors with model.safetensors.index.json), so a model that cannot convert stops before Modal runs.
The result stays in the Modal volume garlic-hf-cache as out/<name>-mix.bin and .json (<name>: qwen3.5-9b for
Qwen/Qwen3.5-9B, else the repository's name); the command prints how to download it for pc/convert_9b_q4s.py, which
makes the folder the app loads. --src and --name also work with ::reference.
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
STOCK = "Qwen/Qwen3.5-9B"


def default_name(src):
    """qwen3.5-9b for Qwen/Qwen3.5-9B, else the repository's name (owner/name@revision -> name)."""
    return "qwen3.5-9b" if src == STOCK else src.split("@")[0].rstrip("/").split("/")[-1]


def split_src(src):
    repo, _, rev = src.partition("@")
    return repo, rev or None


image =(modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.20.1", "safetensors", "numpy>=2", "huggingface_hub",
                      "accelerate", "jinja2")
         .env({"HF_HOME": "/cache/huggingface"})
         .add_local_file(HERE / "q4.py", "/root/q4.py")
         .add_local_file(HERE / "q3.py", "/root/q3.py"))
cache = modal.Volume.from_name("garlic-hf-cache", create_if_missing=True)
app = modal.App("ps4-qwen35-9b", image=image)


@app.function(cpu=16, memory=65536, timeout=7200, volumes={"/cache": cache})
def convert_remote(src: str = STOCK, prefix: str = "model.language_model.", name: str = "qwen3.5-9b"):
    """src checked by pc/checkpoint.py on the PC; prefix: where its text model's tensors are (as check found)."""
    import json
    import sys
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    sys.path.insert(0, "/root")
    from q3 import interleave_q3, quantize_q3
    from q4 import interleave_q4, quantize_q4

    repo, rev = split_src(src)
    d = Path(snapshot_download(repo, revision=rev,
                               allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt"]))
    cfg = json.loads((d / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    NAME = f"{name}-mix"
    files = {}

    def get(name):
        f = wmap[name]
        if f not in files:
            files[f] = safe_open(str(d / f), "pt")
        return files[f].get_tensor(name).float().numpy()

    out_path = Path("/cache/out")
    out_path.mkdir(exist_ok=True)
    out = open(out_path / f"{NAME}.bin", "wb")
    index, offset = {}, 0

    def put(key, data, **meta):
        nonlocal offset
        pad = (-offset) % 256
        out.write(bytes(pad))
        offset += pad
        out.write(data)
        index[key] = dict(offset=offset, size=len(data), **meta)
        offset += len(data)

    def put_f32(key, a):
        put(key, np.ascontiguousarray(a, dtype=np.float32).tobytes(), n=int(np.size(a)))

    def put_q4i(key, mat):
        packed, d16 = quantize_q4(np.ascontiguousarray(mat, dtype=np.float32))
        nib, sc = interleave_q4(packed, d16)
        put(key, nib + sc, rows=mat.shape[0], cols=mat.shape[1], scales=len(nib), fmt="q4i")

    def put_q3i(key, mat):
        q, d16 = quantize_q3(np.ascontiguousarray(mat, dtype=np.float32))
        words, sc = interleave_q3(q, d16)
        put(key, words + sc, rows=mat.shape[0], cols=mat.shape[1], scales=len(words), fmt="q3i")

    P = prefix
    put_f32("final_norm", 1.0 + get(P + "norm.weight"))
    nh, hd = cfg["num_attention_heads"], cfg["head_dim"]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"{P}layers.{i}."
        put_f32(f"{i}.attn_norm", 1.0 + get(p + "input_layernorm.weight"))
        put_f32(f"{i}.mlp_norm", 1.0 + get(p + "post_attention_layernorm.weight"))
        if kind == "full_attention":
            a = p + "self_attn."
            put_f32(f"{i}.q_norm", 1.0 + get(a + "q_norm.weight"))
            put_f32(f"{i}.k_norm", 1.0 + get(a + "k_norm.weight"))
            qg = get(a + "q_proj.weight").reshape(nh, 2, hd, -1)          # per head: [query | gate]
            q, gate = qg[:, 0].reshape(nh * hd, -1), qg[:, 1].reshape(nh * hd, -1)
            put_q4i(f"{i}.wqkv", np.concatenate([q, get(a + "k_proj.weight"), get(a + "v_proj.weight"), gate]))
            put_q4i(f"{i}.wo", get(a + "o_proj.weight"))
        else:
            a = p + "linear_attn."
            put_q4i(f"{i}.w_in", np.concatenate([get(a + "in_proj_qkv.weight"), get(a + "in_proj_z.weight"),
                                                  get(a + "in_proj_b.weight"), get(a + "in_proj_a.weight")]))
            put_f32(f"{i}.conv", get(a + "conv1d.weight").reshape(-1, cfg["linear_conv_kernel_dim"]))
            put_f32(f"{i}.A", -np.exp(get(a + "A_log")))
            put_f32(f"{i}.dt_bias", get(a + "dt_bias"))
            put_f32(f"{i}.gnorm", get(a + "norm.weight"))
            put_q4i(f"{i}.wout", get(a + "out_proj.weight"))
        put_q3i(f"{i}.wgu", np.concatenate([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]))
        put_q4i(f"{i}.wdown", get(p + "mlp.down_proj.weight"))
        print(f"layer {i} ({kind}) done, {offset / 2**20:.0f} MiB", flush=True)
    put_q4i("lm_head", get("lm_head.weight"))
    gpu_size = (offset + 4095) // 4096 * 4096
    out.write(bytes(gpu_size - offset))
    offset = gpu_size
    # input embedding, one contiguous record per row: nibbles then fp16 scales
    emb = get(P + "embed_tokens.weight")
    rows, cols = emb.shape
    rowbytes = cols // 2 + cols // 32 * 2
    start = offset
    for r0 in range(0, rows, 8192):
        packed, d16 = quantize_q4(emb[r0:r0 + 8192])
        out.write(np.concatenate([packed, d16.view(np.uint8)], axis=1).tobytes())
    offset = start + rows * rowbytes
    index["embed"] = dict(offset=start, size=rows * rowbytes, rows=rows, cols=cols, scales=cols // 2,
                          rowbytes=rowbytes, fmt="q4row")
    out.close()
    (out_path / f"{NAME}.json").write_text(json.dumps(dict(config=cfg, size=offset, gpu_size=gpu_size,
                                                           tensors=index, checkpoint=src), indent=1))
    cache.commit()
    return gpu_size, offset


def quantized_like_ps4(model):
    """Put exactly the PS4's weights into a transformers Qwen3.5-9B (in place)."""
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, "/root")
    from q3 import dequantize_q3, quantize_q3
    from q4 import dequantize_q4, quantize_q4
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model

    def q4(t):
        return torch.from_numpy(dequantize_q4(*quantize_q4(t.detach().float().cpu().numpy())))

    def q3(t):
        return torch.from_numpy(dequantize_q3(*quantize_q3(t.detach().float().cpu().numpy())))

    with torch.no_grad():
        for name, mod in lm.named_modules():
            if isinstance(mod, torch.nn.Linear):
                gu = ".mlp.gate_proj" in name or ".mlp.up_proj" in name
                mod.weight.copy_((q3 if gu else q4)(mod.weight).to(mod.weight.dtype))
        lm.embed_tokens.weight.copy_(q4(lm.embed_tokens.weight).to(lm.embed_tokens.weight.dtype))
        head = model.get_output_embeddings()
        head.weight.copy_(q4(head.weight).to(head.weight.dtype))


@app.function(gpu=["A100-80GB", "H100"], timeout=3600, volumes={"/cache": cache})
def reference_remote(ids: list, n_new: int, src: str = STOCK) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    repo, rev = split_src(src)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    quantized_like_ps4(model)
    model = model.to("cuda")
    with torch.no_grad():
        x = torch.tensor([ids], device="cuda")
        logits = model(input_ids=x).logits[0, -1].float().cpu().numpy()
        out = model.generate(input_ids=x, max_new_tokens=n_new, do_sample=False)[0, len(ids):].tolist()
    return dict(logits=logits.tobytes(), out=out)


@app.function(timeout=600, volumes={"/cache": cache})
def chat_ids(messages: list, thinking: bool = False, src: str = STOCK) -> list:
    from transformers import AutoTokenizer
    repo, rev = split_src(src)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    r = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
    r = r["input_ids"] if hasattr(r, "keys") else r
    return [int(v) for v in (r[0] if r and isinstance(r[0], list) else r)]


def checked(src):
    """pc/checkpoint.py's check, on this PC: where the text model's tensors are, or the program ends saying what
    differs from Qwen3.5-9B."""
    import sys
    sys.path.insert(0, str(HERE))
    from checkpoint import Checkpoint, CheckpointError, check
    try:
        ck = Checkpoint(src)
        check(ck)
    except CheckpointError as e:
        sys.exit(str(e))
    return ck.prefix


@app.local_entrypoint()
def convert(src: str = STOCK, name: str = ""):
    name = name or default_name(src)
    prefix = checked(src)
    print(f"{src}: Qwen3.5-9B's shapes; converting into out/{name}-mix.bin on Modal", flush=True)
    gpu_size, total = convert_remote.remote(src, prefix, name)
    print(f"converted: GPU part {gpu_size / 2**20:.1f} MiB, file {total / 2**20:.1f} MiB (out/{name}-mix.bin)")
    models = HERE.parent / "models"
    models.mkdir(exist_ok=True)
    print(f'Download it for convert_9b_q4s.py:\n  modal volume get garlic-hf-cache out/{name}-mix.bin "{models}"\n'
          f'  modal volume get garlic-hf-cache out/{name}-mix.json "{models}"')


@app.local_entrypoint()
def reference(prompt: str = "¿Quién es Paolo Guerrero y dónde nació?", n: int = 30, src: str = STOCK, name: str = ""):
    import numpy as np
    name = name or default_name(src)
    ids = chat_ids.remote([{"role": "user", "content": prompt}], False, src)
    r = reference_remote.remote(ids, n, src)
    np.savez(HERE.parent / "build" / f"ref_{name}-mix.npz", ids=np.array(ids),
             logits=np.frombuffer(r["logits"], np.float32), out=np.array(r["out"]))
    print("prompt", len(ids), "tokens ->", r["out"])
