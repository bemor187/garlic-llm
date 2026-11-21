"""Nemotron 3.5 ASR (nvidia/nemotron-3.5-asr-streaming-0.6b, OpenMDW-1.1) in plain NumPy, written step by step as the
PS4 will run it, and checked against transformers' Nemotron3_5AsrForRNNT. The same FastConformer family as Parakeet
(parakeet_ref.py); what differs:

  features   as Parakeet's, without the per-mel normalisation
  encoder    cache-aware: every convolution is causal. Subsampling convs pad time and frequency by (2 before, 1 after),
             so 128 mels -> 65 -> 33 -> 17 bins, and the linear takes 17 x 256 = 4352 inputs. Attention is limited to
             chunks of CHUNK frames: frame i sees the frames of its own chunk and of the LEFT_CHUNKS chunks before it,
             so the relative distance i - j only runs from -(CHUNK - 1) to LEFT_CHUNKS * CHUNK + CHUNK - 1. The
             positional term therefore needs N_DIST rows of linear_pos(sinusoids), computed once per layer.
             Conv module: pointwise 2048, GLU, causal depthwise 9 (8 frames of left context), LayerNorm, SiLU,
             pointwise. Run over the whole sentence at once, the chunked mask gives what streaming chunk by chunk
             gives.
  language   a one-hot of 128 prompts is appended to each encoder frame: relu(W1 [x | onehot] + b1) is
             relu(W1[:, :1024] x + (b1 + W1[:, 1024 + prompt])), then linear_2 and the encoder projector (fused here
             into one 640 x 2048 matrix)
  decoder    RNN-T (no durations): embedding 640, 2-layer LSTM 640, projector; joint relu(enc + dec) -> 13088 logits
             (13087 = blank). Greedy: a blank moves to the next frame, a token stays (at most 10 per frame).

The chunk of 14 frames (1.12 s: 13 frames of look-ahead) is the most accurate setting of the model card.

Run: python nemotron_ref.py [wav] [language]   (compares with transformers on the CPU)
"""

import sys
from pathlib import Path

import numpy as np

from parakeet_ref import conv2d, layer_norm, mel_filters, read_wav, rel_pos_emb, silu

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "nemotron-3.5-asr-streaming-0.6b"
HF_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
BLANK, VOCAB = 13087, 13088
LOOKAHEAD = 13                     # right context in encoder frames: chunks of 14 frames (1.12 s)
CHUNK = LOOKAHEAD + 1
LEFT_CHUNKS = 56 // CHUNK          # left context of 56 frames (sliding window 57)
D_MIN, D_MAX = -(CHUNK - 1), LEFT_CHUNKS * CHUNK + CHUNK - 1
N_DIST = D_MAX - D_MIN + 1         # 83 relative distances
# prompt ids (processor_config.json) of the languages Garlic dictates with this model (its zh-TW prompt, 5, is not
# trained: Mandarin is zh-CN)
PROMPTS = {"ar": 7, "ja-JP": 10, "ko-KR": 14, "zh-CN": 4, "tr-TR": 18, "vi-VN": 33, "es": 3, "en": 0,
           "auto": 101}


def load_weights(model_dir=MODEL):
    from safetensors.numpy import load_file
    return load_file(str(model_dir / "model.safetensors"))


def features(audio, fb=None):
    """audio float32 [L] at 16 kHz -> log-mel [T = L // 160, 128] (not normalised)."""
    fb = mel_filters() if fb is None else fb
    x = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]]).astype(np.float32)
    n_fft, hop, win = 512, 160, 400
    window = np.zeros(n_fft)
    window[(n_fft - win) // 2:(n_fft - win) // 2 + win] = np.hanning(win)
    xp = np.pad(x.astype(np.float64), (n_fft // 2, n_fft // 2))
    T = len(audio) // hop
    frames = np.stack([xp[t * hop:t * hop + n_fft] * window for t in range(T)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(spec @ fb.T.astype(np.float64) + 2.0 ** -24).astype(np.float32)


def sub_len(n):
    """frames out of one causal subsampling conv (kernel 3, stride 2, padding 2 + 1) for n valid frames in"""
    return n // 2 + 1


def causal_conv(x, w, b, valid, groups=1):
    """x [C, T, F] with T = valid frames (zeros after them): causal 3x3 stride-2 conv -> [O, sub_len(T), F']"""
    C, T, F = x.shape
    To = sub_len(valid)
    xp = np.zeros((C, 2 * To + 1, F + 3), np.float32)       # 2 zero rows / columns before, 1 after (and the rest)
    xp[:, 2:2 + T, 2:2 + F] = x
    return conv2d(xp, w, b, 2, 0, groups)[:, :To]


def subsampling(feat, w):
    S = "encoder.subsampling."
    T = len(feat)
    x = causal_conv(feat[None], w[S + "conv_in.weight"], w[S + "conv_in.bias"], T)
    x = np.maximum(x, 0)
    for l in (0, 1):
        p = f"{S}layers.{l}."
        x = causal_conv(x, w[p + "depthwise_conv.weight"], w[p + "depthwise_conv.bias"], x.shape[1], groups=256)
        x = conv2d(x, w[p + "pointwise_conv.weight"], w[p + "pointwise_conv.bias"], 1, 0)
        x = np.maximum(x, 0)
    C, T3, F = x.shape                                          # [256, T3, 17]
    x = x.transpose(1, 0, 2).reshape(T3, C * F)                 # channel-major, as transformers
    return x @ w[S + "linear.weight"].T + w[S + "linear.bias"]


def rel_table(w_pos):
    """linear_pos of the sinusoids for distances D_MIN .. D_MAX: [N_DIST, 1024] (row k: distance D_MIN + k)."""
    T = max(-D_MIN, D_MAX) + 1
    pe = rel_pos_emb(T)                                         # row p: distance T - 1 - p
    d = np.arange(D_MIN, D_MAX + 1)
    return pe[T - 1 - d] @ w_pos.T


def chunk_mask(T):
    c = np.arange(T) // CHUNK
    diff = c[:, None] - c[None, :]
    return (diff >= 0) & (diff <= LEFT_CHUNKS)


def encoder_layer(x, w, i, tap=None):
    """Conformer block i on x [T, 1024]. tap(name, X), if given, sees the input X of each matrix product."""
    tap = tap or (lambda name, X: None)
    T = len(x)
    H, D = 8, 128
    mask = chunk_mask(T)
    didx = np.clip(np.arange(T)[:, None] - np.arange(T)[None, :] - D_MIN, 0, N_DIST - 1)
    p = f"encoder.layers.{i}."
    L = lambda n: (w[p + n + ".weight"], w[p + n + ".bias"])
    h = layer_norm(x, *L("norm_feed_forward1"))
    tap("ff1.1", h)
    h = silu(h @ w[p + "feed_forward1.linear1.weight"].T)
    tap("ff1.2", h)
    x = x + 0.5 * (h @ w[p + "feed_forward1.linear2.weight"].T)
    h = layer_norm(x, *L("norm_self_att"))
    tap("qkv", h)
    a = p + "self_attn."
    q = (h @ w[a + "q_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
    k = (h @ w[a + "k_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
    v = (h @ w[a + "v_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
    R = w[a + "rel_table"].reshape(N_DIST, H, D).transpose(1, 0, 2)                             # [H, N_DIST, D]
    qu = q + w[a + "bias_u"][:, None, :]
    qv = q + w[a + "bias_v"][:, None, :]
    bd = np.take_along_axis(qv @ R.transpose(0, 2, 1), np.broadcast_to(didx, (H, T, T)), axis=2)
    s = (qu @ k.transpose(0, 2, 1) + bd) / np.sqrt(D)
    s = np.where(mask, s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = (s @ v).transpose(1, 0, 2).reshape(T, H * D)
    tap("o", o)
    x = x + o @ w[a + "o_proj.weight"].T
    h = layer_norm(x, *L("norm_conv"))
    tap("pw1", h)
    c = p + "conv."
    h = h @ w[c + "pointwise_conv1.weight"][:, :, 0].T              # [T, 2048]
    h = h[:, :1024] / (1 + np.exp(-h[:, 1024:]))                   # GLU
    dw = w[c + "depthwise_conv.weight"][:, 0, :]                   # [1024, 9]
    hp = np.pad(h, ((8, 0), (0, 0)))                               # causal: 8 frames of left context
    h = sum(hp[j:j + T] * dw[:, j] for j in range(9))
    h = silu(layer_norm(h, w[c + "norm.weight"], w[c + "norm.bias"]))
    tap("pw2", h)
    x = x + h @ w[c + "pointwise_conv2.weight"][:, :, 0].T
    h = layer_norm(x, *L("norm_feed_forward2"))
    tap("ff2.1", h)
    h = silu(h @ w[p + "feed_forward2.linear1.weight"].T)
    tap("ff2.2", h)
    x = x + 0.5 * (h @ w[p + "feed_forward2.linear2.weight"].T)
    return layer_norm(x, *L("norm_out"))


def encoder(feat, w):
    x = subsampling(feat, w)
    for i in range(24):
        x = encoder_layer(x, w, i)
    return x.astype(np.float32)


def fused_out(w):
    """linear_2 of the prompt projector followed by the encoder projector, as one 640 x 2048 matrix + bias"""
    W2, b2 = w["prompt_projector.linear_2.weight"], w["prompt_projector.linear_2.bias"]
    Wp, bp = w["encoder_projector.weight"], w["encoder_projector.bias"]
    return (Wp.astype(np.float64) @ W2).astype(np.float32), (Wp.astype(np.float64) @ b2 + bp).astype(np.float32)


def project(enc, w, prompt):
    """encoder frames [T, 1024] + language prompt -> joint inputs [T, 640]"""
    h = np.maximum(enc @ w["prompt_projector.w_audio"].T + w["prompt_projector.bias_by_prompt"][prompt], 0)
    return h @ w["fused_out.weight"].T + w["fused_out.bias"]


class Decoder:
    """Prediction network + joint, one step at a time (as the PS4 runs it)."""

    def __init__(self, w):
        self.w = w
        self.h = [np.zeros(640, np.float32) for _ in range(2)]
        self.c = [np.zeros(640, np.float32) for _ in range(2)]
        self.out = None

    def step(self, token):
        w = self.w
        x = w["decoder.embedding.weight"][token]
        for l in range(2):
            g = w[f"decoder.lstm.weight_ih_l{l}"] @ x + w[f"decoder.lstm.bias_ih_l{l}"] + \
                w[f"decoder.lstm.weight_hh_l{l}"] @ self.h[l] + w[f"decoder.lstm.bias_hh_l{l}"]
            i, f, gg, o = np.split(g, 4)
            sig = lambda z: 1 / (1 + np.exp(-z))
            self.c[l] = sig(f) * self.c[l] + sig(i) * np.tanh(gg)
            self.h[l] = sig(o) * np.tanh(self.c[l])
            x = self.h[l]
        self.out = w["decoder.decoder_projector.weight"] @ x + w["decoder.decoder_projector.bias"]

    def joint(self, enc_t):
        w = self.w
        return w["joint.head.weight"] @ np.maximum(enc_t + self.out, 0) + w["joint.head.bias"]


def rnnt_greedy(proj, w, max_symbols=10):
    dec = Decoder(w)
    dec.step(BLANK)
    out, t, at_frame = [], 0, 0
    while t < len(proj):
        tok = int(np.argmax(dec.joint(proj[t])))
        if tok == BLANK:
            t, at_frame = t + 1, 0
            continue
        out.append(tok)
        dec.step(tok)
        at_frame += 1
        if at_frame >= max_symbols:
            t, at_frame = t + 1, 0
    return out


def prepare(w):
    """the derived tensors the PS4 uses, added to w: per layer the positional rows of the N_DIST distances
    (rel_table), the audio half of the prompt projector's first matrix (w_audio [2048, 1024]), its bias plus the
    one-hot column of every prompt (bias_by_prompt [128, 2048]), and the fused output matrix"""
    for i in range(24):
        a = f"encoder.layers.{i}.self_attn."
        w[a + "rel_table"] = rel_table(w[a + "relative_k_proj.weight"].astype(np.float64)).astype(np.float32)
    W1, b1 = w["prompt_projector.linear_1.weight"], w["prompt_projector.linear_1.bias"]
    w["prompt_projector.w_audio"] = np.ascontiguousarray(W1[:, :1024])
    w["prompt_projector.bias_by_prompt"] = np.ascontiguousarray(b1[None, :] + W1[:, 1024:].T)
    w["fused_out.weight"], w["fused_out.bias"] = fused_out(w)
    return w


def transcribe(audio, w, prompt, fb=None):
    enc = encoder(features(audio, fb), w)
    return rnnt_greedy(project(enc, w, prompt), w)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL / "tokenizer.json"))


def hf_model():
    import torch
    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
    src = str(MODEL) if (MODEL / "config.json").exists() else HF_ID
    proc = AutoProcessor.from_pretrained(src)
    proc.set_num_lookahead_tokens(LOOKAHEAD)
    model = Nemotron3_5AsrForRNNT.from_pretrained(src, torch_dtype=torch.float32).eval()
    return proc, model


def hf_transcribe(proc, model, audio, language):
    import torch
    inputs = proc(audio, sampling_rate=16000, language=language, return_tensors="pt")
    with torch.no_grad():
        g = model.generate(**inputs, return_dict_in_generate=True)
    seq = [t for t in g.sequences[0].tolist() if t != BLANK]
    return seq, inputs


def main():
    import torch
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "test_en.wav")
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    audio = read_wav(path)
    w = prepare(load_weights())
    tok = tokenizer()
    feat = features(audio)
    enc = encoder(feat, w)
    toks = rnnt_greedy(project(enc, w, PROMPTS[lang]), w)
    print("NumPy:       ", repr(tok.decode(toks)))
    proc, model = hf_model()
    seq, inputs = hf_transcribe(proc, model, audio, lang)
    T = len(audio) // 160
    f_ref = inputs.input_features[0, :T].numpy()
    print(f"features: {feat.shape} vs {tuple(inputs.input_features.shape)}, max abs diff {np.abs(feat - f_ref).max():.2e}")
    with torch.no_grad():
        e_ref = model.get_audio_features(**{k: v for k, v in inputs.items() if k != "num_lookahead_tokens"},
                                         num_lookahead_tokens=LOOKAHEAD).last_hidden_state[0].numpy()
    n = min(len(enc), len(e_ref))
    print(f"encoder: {enc.shape} vs {e_ref.shape}, rel diff {np.abs(enc[:n] - e_ref[:n]).max() / np.abs(e_ref).max():.2e}")
    print("transformers:", repr(tok.decode(seq)), "same tokens" if seq == toks else "DIFFERENT tokens")


if __name__ == "__main__":
    main()
