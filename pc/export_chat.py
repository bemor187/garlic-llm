"""Write the data files the standalone chat app reads from /app0/assets (see chat/*.c for the readers):

  kernels.bin    compiled GPU code of shaders/chat.cl (qwen3.cl for a Qwen3 model) + one record per kernel (entry,
                 RSRC1/2, user SGPRs)
  model.idx      model shape + where every tensor sits in the weights file (which stays in /data on the PS4)
  tokenizer.bin  byte-level BPE: raw bytes of every token id, merge table in rank order, byte -> id map
  font.bin       text font: Selawik and Noto Sans (+ Noto Sans Arabic, Arimo, DejaVu Sans and the Noto symbol
                 fonts) + Twemoji (CC-BY 4.0) incl. flags and sequences + Chinese, Japanese and Korean (Noto Sans SC,
                 JP, TC, KR) at 4 bits a pixel
  font_bold.bin / font_italic.bin  bold / italic (markdown), Latin, Cyrillic + Arabic (bold)
  font_small.bin text at 24 px (hints, chips);  font_big.bin  titles (semibold at 40 px + keyboard emoji); both
                 with the Chinese, Japanese and Korean characters of the interface's own text
  mono.bin       code (Inconsolata, Cascadia Mono for Cyrillic)

The font files are read from pc/fonts_libres and the emoji from third_party/twemoji: both are downloaded, not part of
the repository (CREDITS.md lists them).

All little endian. Run: python export_chat.py [model name] [hf model dir]
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

from gpu import CLANG, SHADERS, Kernel, elf_symbols

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "chat" / "assets"
MAX_SEQ = 2048
EXTRA_CHARS = "‘’‚“”„–—…•◦▪€™←↑→↓↕·"
TEXT_FACE = "segoeui.ttf"
TEXT_RANGES = [(32, 127), (160, 0x250), (0x400, 0x460), (0x2010, 0x2028)]   # Latin, Cyrillic, punctuation
VAR_JP, VAR_TC, UI_VAR = 0x200000, 0x400000, 0x800000   # see chat/ui.c


def export_kernels(shader="qwen3"):
    src, obj = SHADERS / f"{shader}.cl", SHADERS / f"{shader}.o"
    subprocess.run(CLANG + ["-c", str(src), "-o", str(obj)], check=True)
    syms, sections = elf_symbols(obj.read_bytes())
    text = sections[".text"]
    recs = []
    for sym, (sec, value, _size) in sorted(syms.items()):
        if sym.endswith(".kd"):
            name = sym[:-3]
            entry = syms[name][1]
            assert entry % 256 == 0, name
            k = Kernel(name, entry, sections[sec][value: value + 64])
            recs.append(struct.pack("<32sIIIII", name.encode(), entry, k.rsrc1, k.rsrc2, k.user_sgprs, k.kernarg_sgpr))
    data = struct.pack("<4sII", b"PKRN", len(recs), len(text)) + b"".join(recs) + text
    (OUT / "kernels.bin").write_bytes(data)
    print(f"kernels.bin: {len(recs)} kernels from {shader}.cl, {len(text)} bytes of code")


def export_model_index(name):
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    c, t = meta["config"], meta["tensors"]
    if c.get("model_type") == "qwen3_5_text":
        return export_model_index_qwen35(name, meta)
    order = ["embed", "final_norm"]
    for i in range(c["num_hidden_layers"]):
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", "q_norm", "k_norm", "wqkv", "wo", "wgu", "wdown")]
    path = f"/data/{name}.bin".encode()
    head = struct.pack("<4s10I2f128sQI", b"PMDL", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"], MAX_SEQ,
                       151645, 151643, c["rms_norm_eps"], c["rope_theta"], path, meta["size"], len(order))
    recs = []
    for k in order:
        e = t[k]
        fmt = {"q4": 2, "q8": 1}.get(e.get("fmt"), 0)
        recs.append(struct.pack("<QQIIQI", e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0), fmt))
    (OUT / "model.idx").write_bytes(head + b"".join(recs))
    print(f"model.idx: {len(order)} tensors, weights at {path.decode()} ({meta['size'] / 2**20:.0f} MiB)")


QWEN35_MAX_SEQ = 8192   # MAX_SEQ in shaders/qwen35.cl


def export_model_index_qwen35(name, meta):
    """PMQ5: the PMDL header (eos = <|im_end|>, eos2 = <|endoftext|>) + "<8I" (linear key heads, value heads,
    key / value head dims, RoPE dims, mask of full-attention layers, <|im_start|>, 0); then per tensor "<QQIIQI" in
    this order: embed, final_norm, then per layer attn_norm, mlp_norm, and either (full attention) q_norm, k_norm,
    wqkv, wo or (Gated DeltaNet) w_in, conv, A, dt_bias, gnorm, wout; then wgu, wdown."""
    c, t = meta["config"], meta["tensors"]
    order, mask = ["embed", "final_norm"], 0
    for i, kind in enumerate(c["layer_types"]):
        full = kind == "full_attention"
        mask |= full << i
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if full else ("w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    flags = 0
    if "lm_head" in t:                       # untied output head (9B): last record
        order.append("lm_head")
        flags |= 1
    if t["embed"].get("fmt") == "q4row":     # input embedding rows stay in the file, past gpu_size
        flags |= 2
    rp = c["rope_parameters"]
    rope_dim = int(c["head_dim"] * rp.get("partial_rotary_factor", 1.0))
    path = f"/data/{name}.bin".encode()
    head = struct.pack("<4s10I2f128sQI", b"PMQ5", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"],
                       QWEN35_MAX_SEQ, 248046, 248044, c["rms_norm_eps"], rp["rope_theta"], path,
                       meta.get("gpu_size", meta["size"]), len(order))
    head += struct.pack("<8I", c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                        c["linear_value_head_dim"], rope_dim, mask, 248045, flags)
    recs = []
    for k in order:
        e = t[k]
        fmt = {"q4": 2, "q8": 1, "q4i": 3, "q4row": 4, "q3i": 5}.get(e.get("fmt"), 0)
        recs.append(struct.pack("<QQIIQI", e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0), fmt))
    (OUT / "model.idx").write_bytes(head + b"".join(recs))
    print(f"model.idx (Qwen3.5, flags {flags}): {len(order)} tensors, full attention mask {mask:#010x}, weights at "
          f"{path.decode()} ({meta['size'] / 2**20:.0f} MiB)")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export_tokenizer(model_dir):
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    dec = bytes_to_unicode()
    n = max(max(vocab.values()), max(a["id"] for a in tj["added_tokens"])) + 1
    raw = [b""] * n
    for tok, i in vocab.items():
        raw[i] = bytes(dec[ch] for ch in tok)
    special = [0] * n
    for a in tj["added_tokens"]:
        raw[a["id"]] = a["content"].encode()
        special[a["id"]] = 1
    byte_id = [vocab[next(ch for ch, b in dec.items() if b == x)] for x in range(256)]
    mrec = []
    for m in merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        mrec.append(struct.pack("<III", vocab[a], vocab[b], vocab[a + b]))
    out = struct.pack("<4sII", b"PTOK", n, len(mrec)) + struct.pack("<256I", *byte_id)
    out += b"".join(struct.pack("<BH", special[i], len(r)) + r for i, r in enumerate(raw))
    out += b"".join(mrec)
    (OUT / "tokenizer.bin").write_bytes(out)
    print(f"tokenizer.bin: {n} tokens, {len(mrec)} merges")


# Glyph sources, in order of preference: the text faces, the symbol faces for arrows / math / shapes, Twemoji
# (third_party/twemoji, CC-BY 4.0) for emoji, including flags, keycaps and joined sequences. A symbol Twemoji lacks
# comes from the symbol faces, in one colour.
FONTS = "C:/Windows/Fonts/"
TWEMOJI = ROOT / "third_party" / "twemoji" / "assets" / "72x72"
SKIP = set(range(0x200B, 0x2010)) | set(range(0xFE00, 0xFE10)) | set(range(0x1F3FB, 0x1F400)) | {0x20E3}
SEQ_BASE = 0x110000                    # records for multi-codepoint emoji use codepoints past Unicode
MONO_RANGES = [(32, 127), (160, 384), (0x400, 0x460), (0x2010, 0x2028), (0x2190, 0x2200), (0x2500, 0x25A0)]

def arabic_codepoints():
    """The Arabic block and the contextual forms chat/ui.c draws (pc/gen_arabic_forms.py), not the other
    presentation forms (ligatures it never asks for)."""
    from gen_arabic_forms import forms
    table, lam_alef = forms()
    return frozenset(set(range(0x0600, 0x0700)) | {f for row in table.values() for f in row if f} |
                     {f for pair in lam_alef.values() for f in pair if f})


ARABIC = arabic_codepoints()

# Two sets of faces, by role; within a role the first face that has a character draws it. A face is
# (file, weight of a variable font or None, scale[, highest codepoint it may draw, or the set of codepoints it may
# draw]). "windows" are the Windows faces
# the app first had (not ours to ship); "free" (pc/fonts_libres, SIL OFL 1.1 except DejaVu's own free licence) look
# as close to them as we found: Selawik is Microsoft's open stand-in for Segoe UI with the same widths, but Latin
# only, so Noto Sans (scaled to Segoe UI's x-height) draws Cyrillic, Greek and the rest, and Noto Sans Italic the
# italics (Selawik has none). With "free" every font keeps the line height and baseline of the face it replaces
# (LINE below), so the layout does not move. Arabic: Noto Sans Arabic, at the scale of the Latin Noto Sans it
# was drawn with. Chinese, Japanese and Korean: Noto Sans SC, JP, TC and KR (the same design in each region's
# forms).
FREE = ROOT / "pc" / "fonts_libres"
NOTO_X = 0.5 / 0.536                   # Segoe UI's x-height / Noto Sans's
NOTO_AR = f"{FREE}/NotoSansArabic/NotoSansArabic[wdth,wght].ttf"
FACE_SETS = {
    "windows": {
        "text": [(FONTS + "segoeui.ttf", None, 1)], "bold": [(FONTS + "segoeuib.ttf", None, 1)],
        "italic": [(FONTS + "segoeuii.ttf", None, 1)], "semibold": [(FONTS + "seguisb.ttf", None, 1)],
        "arial": [(FONTS + "arial.ttf", None, 1)], "symbols": [(FONTS + "seguisym.ttf", None, 1)],
        "emoji": [(FONTS + "seguiemj.ttf", None, 1)], "mono": [(FONTS + "consola.ttf", None, 1)],
        "zh": [(FONTS + "msyh.ttc", None, 1)], "ko": [(FONTS + "malgun.ttf", None, 1)],
        "ja": [(FONTS + "YuGothR.ttc", None, 1)], "tc": [(FONTS + "msjh.ttc", None, 1)],
    },
    "free": {
        "text": [(f"{FREE}/Selawik/selawk.ttf", None, 1), (f"{FREE}/NotoSans/NotoSans[wdth,wght].ttf", 400, NOTO_X),
                 (NOTO_AR, 400, NOTO_X, ARABIC)],
        "bold": [(f"{FREE}/Selawik/selawkb.ttf", None, 1), (f"{FREE}/NotoSans/NotoSans[wdth,wght].ttf", 700, NOTO_X),
                 (NOTO_AR, 700, NOTO_X, ARABIC)],
        "italic": [(f"{FREE}/NotoSans/NotoSans-Italic[wdth,wght].ttf", 400, NOTO_X)],
        "semibold": [(f"{FREE}/Selawik/selawksb.ttf", None, 1),
                     (f"{FREE}/NotoSans/NotoSans[wdth,wght].ttf", 600, NOTO_X), (NOTO_AR, 600, NOTO_X, ARABIC)],
        "arial": [(f"{FREE}/Arimo/Arimo[wght].ttf", 400, 1, 0x1FFF)],   # letters only: its arrows / math are narrow
        # DejaVu Sans (scaled to Segoe UI's x-height) has the arrows, math and shapes closest to Segoe UI Symbol's
        "symbols": [(f"{FREE}/DejaVu/DejaVuSans.ttf", None, 0.914),
                    (f"{FREE}/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf", None, 1),
                    (f"{FREE}/NotoSansSymbols/NotoSansSymbols[wght].ttf", 400, 1),
                    (f"{FREE}/NotoSansMath/NotoSansMath-Regular.ttf", None, 1),
                    (f"{FREE}/NotoMusic/NotoMusic-Regular.ttf", None, 1)],
        "emoji": [],                   # Twemoji only; the symbols Twemoji lacks come from "symbols", in one color
        # Inconsolata was drawn after Consolas; scaled to Consolas's width. Cascadia Mono (light, same width) adds
        # the Cyrillic Inconsolata lacks.
        "mono": [(f"{FREE}/Inconsolata/Inconsolata[wdth,wght].ttf", 400, 28 / 26),   # 14 px a letter at 26, as
                 (f"{FREE}/CascadiaMono/CascadiaMono[wght].ttf", 350, 24 / 26)],      # Consolas

        "zh": [(f"{FREE}/NotoSansSC/NotoSansSC[wght].ttf", 400, 1)],
        "ko": [(f"{FREE}/NotoSansKR/NotoSansKR[wght].ttf", 400, 1)],
        "ja": [(f"{FREE}/NotoSansJP/NotoSansJP[wght].ttf", 400, 1)],
        "tc": [(f"{FREE}/NotoSansTC/NotoSansTC[wght].ttf", 400, 1)],
    },
}
LINE = {"mono": (1521, 527, 2048)}     # (ascender, descender, units per em) of Consolas; the rest: Segoe UI's
SEGOE_LINE = (2210, 514, 2048)
face_set = "free"                      # --fonts=free | windows (the Windows ones: only to compare)
font_out = OUT                         # --out=DIR: where the font files go


def load_face(spec, size):
    """A PIL font for (file, weight, scale) at size px (scaled); variable fonts set to the weight."""
    from PIL import ImageFont
    path, weight, scale = spec[:3]
    f = ImageFont.truetype(path, size if scale == 1 else max(1, int(round(size * scale))), index=0)
    if weight is not None:
        names = [a.get("name", b"") for a in f.get_variation_axes()]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        f.set_variation_by_axes([weight if n.lower() in ("weight", "wght") else a["default"]
                                 for n, a in zip(names, f.get_variation_axes())])
    return f


def allowed(spec, cp):
    """May face spec draw cp (its optional fourth element: the highest codepoint, or a set of them)?"""
    if len(spec) < 4:
        return True
    return cp in spec[3] if isinstance(spec[3], frozenset) else cp <= spec[3]


def font_cmaps(specs):
    from fontTools.ttLib import TTFont
    return {s: {cp for cp in TTFont(s[0], lazy=True, fontNumber=0).getBestCmap() if allowed(s, cp)} for s in specs}


def charset(encoding, rows, cols):
    """The characters of a two-byte encoding, rows x cols."""
    out = set()
    for b1 in rows:
        for b2 in cols:
            try:
                out.add(ord(bytes([b1, b2]).decode(encoding)))
            except UnicodeDecodeError:
                pass
    return out


def cjk_variant_cp(cp):
    """Characters that may have forms of their own per language (cjk_variant_cp in chat/ui.c)."""
    return (0x2E80 <= cp <= 0x9FFF or 0xAC00 <= cp <= 0xD7A3 or 0xF900 <= cp <= 0xFAFF or 0xFE30 <= cp <= 0xFE4F or
            0xFF00 <= cp <= 0xFFEF or 0x1100 <= cp <= 0x11FF or 0x20000 <= cp <= 0x3FFFF)


CJK_PUNCT = set(range(0x3000, 0x3040)) | set(range(0xFF01, 0xFF5F))   # CJK punctuation, full-width forms


def cjk_codepoints():
    """Common East Asian characters at their own codepoints: the 6,763 hanzi of GB2312, kana, CJK punctuation and
    full-width forms (in Simplified Chinese forms), the 2,350 hangul syllables of KS X 1001 and the jamo."""
    yahei = charset("gb2312", range(0xB0, 0xF8), range(0xA1, 0xFF))
    yahei |= CJK_PUNCT | set(range(0x3040, 0x3100))
    hangul = charset("euc-kr", range(0xB0, 0xC9), range(0xA1, 0xFF))
    hangul = {cp for cp in hangul if 0xAC00 <= cp <= 0xD7A3} | set(range(0x3131, 0x318F))   # + the lone jamo
    return {cp: "zh" for cp in yahei} | {cp: "ko" for cp in hangul}


def cjk_variants():
    """{key: (role, codepoint)}: the Japanese forms of JIS X 0208 (its 6,355 kanji, kana, punctuation and full-width
    forms) at VAR_JP + codepoint, and the Traditional Chinese forms of Big5's 5,401 common hanzi, its bopomofo and
    punctuation at VAR_TC + codepoint. export_font keeps only those that differ from the glyph at the codepoint
    itself (or that it lacks)."""
    big5_cols = list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF))
    jis = charset("euc_jp", range(0xA1, 0xF5), range(0xA1, 0xFF)) | CJK_PUNCT | set(range(0x3040, 0x3100))
    big5 = charset("big5", range(0xA1, 0xA4), big5_cols) | charset("big5", range(0xA4, 0xC7), big5_cols) | CJK_PUNCT
    out = {VAR_JP + cp: ("ja", cp) for cp in jis if cjk_variant_cp(cp)}
    out |= {VAR_TC + cp: ("tc", cp) for cp in big5 if cjk_variant_cp(cp)}
    return out


UI_TEXT = ROOT / "chat" / "ui_strings.inc"
# LANG_* order (chat/ui.h); the languages whose interface text gets glyphs of its own at 24 and 40 px, with the
# role of its face and the codepoint offset of its forms
LANGS = ["es", "en", "pt", "ru", "fr", "de", "ar", "ja", "zh-Hans", "zh-Hant", "ko"]
UI_CJK = {"ja": ("ja", VAR_JP), "zh-Hans": ("zh", 0), "zh-Hant": ("tc", VAR_TC), "ko": ("ko", 0)}
# the strings drawn as titles (font_big); text drawn in the chat, toasts and what only the model reads are left out
# of the hint size (font_small)
BIG_STRINGS = {"S_LISTENING", "S_TRANSCRIBING", "S_ERROR_TITLE", "S_PY_DONE", "S_PY_ERROR", "S_PY_STOPPED",
               "S_PY_STARTING"}
NOT_SMALL = {"S_SYSTEM_PROMPT", "S_PY_FIX_MSG", "S_PY_FIX_ASK", "S_CONTEXT_FULL", "S_API_DONE", "S_TAGLINE",
             "S_PLACEHOLDER", "S_NO_MIC", "S_NO_SPEECH", "S_ASR_OFF", "S_HOLD_LONGER", "S_THINK_ON_TOAST",
             "S_THINK_OFF_TOAST", "S_API_ON", "S_API_OFF", "S_API_FAILED", "S_API_BUSY", "S_VOICE_NO_TTS",
             "S_VOICE_LOADING", "S_PY_INPUT", "S_LISTENING_SHORT", "S_THINKING"} | BIG_STRINGS


def c_group(text, i):
    """The brace group of C initializers starting at text[i] == "{": a list of elements, each a string (adjacent
    literals joined) or a nested list. Comments are skipped."""
    assert text[i] == "{"
    i += 1
    out, cur, pending = [], None, False
    while True:
        c = text[i]
        if text.startswith("//", i):
            i = text.index("\n", i)
        elif c == '"':
            j, lit = i + 1, []
            while text[j] != '"':
                if text[j] == "\\":
                    e = text[j + 1]
                    if e == "u":
                        lit.append(chr(int(text[j + 2:j + 6], 16)))
                        j += 6
                        continue
                    lit.append({"n": "\n", "t": "\t"}.get(e, e))
                    j += 2
                    continue
                lit.append(text[j])
                j += 1
            cur = (cur or "") + "".join(lit)
            pending, i = True, j + 1
        elif c == "{":
            cur, i = c_group(text, i)
            pending = True
        elif c == ",":
            out.append(cur)
            cur, pending, i = None, False, i + 1
        elif c == "}":
            if pending:
                out.append(cur)
            return out, i + 1
        else:
            i += 1


def ui_strings():
    """{StrId name: [the string in each language]} and [the emoji page titles of each language] from
    chat/ui_strings.inc, and {language: [names]} of the Python key names (chat/assets/py/_gx.py)."""
    import ast
    import re
    text = UI_TEXT.read_text(encoding="utf-8")
    table = {}
    start = text.index("strings[N_STRINGS][N_LANGS] = {")
    end = text.index("\n};", start)
    for m in re.finditer(r"\[(S_\w+)\] = \{", text[start:end]):
        table[m.group(1)] = c_group(text, start + m.end() - 1)[0]
    titles = c_group(text, text.index("{", text.index("emoji_titles[N_LANGS]")))[0]
    gx = (ROOT / "chat" / "assets" / "py" / "_gx.py").read_text(encoding="utf-8")
    names = ast.literal_eval(re.search(r"_NAMES = (\{.*?\}\})", gx, re.S).group(1))
    return table, titles, {lang: list(v.values()) for lang, v in names.items()}


def ui_cjk_chars(big):
    """{key: (role, codepoint)}: the Chinese, Japanese and Korean characters of the interface's own text, each at
    UI_VAR + its language's offset + codepoint. big: those of the titles (font_big), else those of the hints."""
    table, titles, py_names = ui_strings()
    out = {}
    for li, lang in enumerate(LANGS):
        if lang not in UI_CJK:
            continue
        role, off = UI_CJK[lang]
        texts = [v[li] for k, v in table.items() if (k in BIG_STRINGS) == big and (big or k not in NOT_SMALL)]
        if not big:
            texts += list(titles[li]) + py_names.get(lang, [])
        for cp in sorted({ord(c) for t in texts for c in t if cjk_variant_cp(ord(c))}):
            out.setdefault(UI_VAR + off + cp, (role, cp))
    return out


def pack4(data):
    """8-bit coverage -> 4 bits a pixel, two a byte (the first in the low half), as glyph_alpha in chat/ui.c reads
    them."""
    q = [(v * 15 + 127) // 255 for v in data]
    if len(q) % 2:
        q.append(0)
    return bytes(q[k] | q[k + 1] << 4 for k in range(0, len(q), 2))


def twemoji_files():
    """{codepoint tuple without FE0F: png path}"""
    out = {}
    for p in sorted(TWEMOJI.glob("*.png")):
        cps = tuple(int(x, 16) for x in p.stem.split("-") if x != "fe0f")
        out.setdefault(cps, p)
    return out


def export_font(fname, size, full=True, role="text", emoji=(), emoji_size=None, ui_cjk=None):
    """PFN2: header <4sHHII (magic, line height, ascent, glyph count, bitmap bytes); per glyph <IHHhhHBBI
    (codepoint, width, height, left, top, advance, kind 0 = alpha / 1 = RGBA / 2 = 4-bit alpha, pad, bitmap
    offset), sorted by codepoint; then the bitmaps (1 or 4 bytes per pixel, or 4 bits); then, if any, the emoji
    sequences: <4sI ("SEQS", count) and per sequence <BBHI (length, 0, 0, glyph index) followed by its codepoints
    (u32, FE0F removed).
    role: the faces of FACE_SETS[face_set] that draw it ("text", "bold", "italic", "semibold", "mono").
    full: the text font, with every script; Chinese, Japanese and Korean at 4 bits a pixel (identical bitmaps
    stored once), the Japanese and Traditional Chinese forms at VAR_JP / VAR_TC + codepoint when they differ.
    ui_cjk: the weight of the interface's Chinese, Japanese and Korean characters (ui_cjk_chars), or None."""
    from PIL import Image, ImageDraw
    fs = FACE_SETS[face_set]
    cjk_roles = ("zh", "ko", "ja", "tc")
    if ui_cjk:                                         # the CJK faces at this font's weight
        fs = dict(fs, **{r: [(s[0], ui_cjk, *s[2:]) for s in fs[r]] for r in cjk_roles})
    roles = ("emoji", "text", "arial", "symbols") + cjk_roles if full else (role,) + (cjk_roles if ui_cjk else ())
    specs = list(dict.fromkeys(s for r in roles + (role, "arial") for s in fs[r]))
    cmaps = font_cmaps(specs)
    faces = {s: load_face(s, size) for s in specs}
    base = fs[role][0]
    if face_set == "windows":
        ascent, descent = faces[base].getmetrics()
    else:                                              # the replaced face's line, so nothing on screen moves
        a, d, upm = LINE.get(role, SEGOE_LINE)
        ascent, descent = -(-size * a // upm), -(-size * d // upm)
    emoji_px = emoji_size or int(round(size * 0.95))
    tw = twemoji_files() if full else {}
    plan, seqs = {}, []
    real, four = {}, set()                             # key -> the codepoint it draws; keys drawn at 4 bits

    def first(chain, cp):
        return next((s for s in chain if cp in cmaps[s]), None)
    if full:
        for cps, path in tw.items():
            if len(cps) == 1 and cps[0] >= 0x2300 and cps[0] not in SKIP:
                plan[cps[0]] = path
        for s in fs["emoji"]:
            for cp in sorted(cmaps[s]):
                if cp >= 0x2600 and cp not in SKIP and cp not in plan:
                    plan[cp] = s
        for s in fs["text"] + fs["arial"] + fs["symbols"]:   # first face that has the character wins
            for cp in cmaps[s]:
                if cp >= 32 and cp not in SKIP and cp not in plan and not (0xD800 <= cp < 0xE000):
                    plan[cp] = s
        for cp, r in cjk_codepoints().items():         # East Asian text the fonts above lack
            s = first(fs[r], cp)
            if cp not in plan and s:
                plan[cp] = s
                four.add(cp)
        for key, (r, cp) in cjk_variants().items():   # their Japanese and Traditional Chinese forms
            s = first(fs[r], cp)
            if s:
                plan[key], real[key] = s, cp
                four.add(key)
        for k, (cps, path) in enumerate(sorted((c, p) for c, p in tw.items() if len(c) > 1)):
            plan[SEQ_BASE + k] = path
            seqs.append(cps)
    else:
        cps_wanted = [cp for lo, hi in (MONO_RANGES if role == "mono" else TEXT_RANGES)
                      for cp in range(lo, hi)] + [ord(c) for c in EXTRA_CHARS] + [0xFFFD]
        if role != "mono":
            cps_wanted += sorted(ARABIC - set(cps_wanted))
        for cp in cps_wanted:
            s = first(fs[role], cp)
            if s or cp == 0xFFFD:
                plan[cp] = s or base
        if emoji:                                       # extra Twemoji (the keyboard's emoji, drawn big)
            tw = twemoji_files()
            for e in emoji:
                cps = tuple(ord(c) for c in e if ord(c) != 0xFE0F)
                if len(cps) == 1:
                    plan[cps[0]] = tw[cps]
                else:
                    plan[SEQ_BASE + len(seqs)] = tw[cps]
                    seqs.append(cps)
        for key, (r, cp) in (ui_cjk_chars(role == "semibold").items() if ui_cjk else ()):
            s = first(fs[r], cp)
            if s:
                plan[key], real[key] = s, cp
    plan.setdefault(0xFFFD, fs["arial"][0] if full else base)
    recs, bitmaps = [], bytearray()
    index, plain4, shared = {}, {}, {}                 # plain4: the 4-bit glyphs at their own codepoints
    for cp in sorted(plan):
        src = plan[cp]
        if not isinstance(src, tuple):                 # a Twemoji PNG
            index[cp] = len(recs)
            img = Image.open(src).convert("RGBA").resize((emoji_px, emoji_px), Image.LANCZOS)
            top = ascent - emoji_px + int(round(size * 0.12)) if not emoji_size else (ascent + descent - emoji_px) // 2
            recs.append(struct.pack("<IHHhhHBBI", cp, emoji_px, emoji_px, 1, top, emoji_px + 2, 1, 0, len(bitmaps)))
            bitmaps += img.tobytes()
            continue
        ch, f = chr(real.get(cp, cp)), faces[src]
        color = src in fs["emoji"]
        dy = ascent - f.getmetrics()[0]                # line the font up on the base font's baseline
        left, top, right, bottom = f.getbbox(ch)
        w, h = max(0, right - left), max(0, bottom - top)
        adv = int(round(f.getlength(ch)))
        data = b""
        if w and h:
            img = Image.new("RGBA" if color else "L", (w, h), (0, 0, 0, 0) if color else 0)
            ImageDraw.Draw(img).text((-left, -top), ch, font=f, fill=(255, 255, 255, 255) if color else 255,
                                     embedded_color=color)
            data = img.tobytes()
            if not any(data):
                data, w, h = b"", 0, 0
        kind = 1 if color and data else 0
        if cp in four and data:
            data, kind = pack4(data), 2
        glyph = (w, h, left, top + dy, adv, data)
        if cp in four and cp < VAR_JP:
            plain4[cp] = glyph
        elif cp in real and cp < UI_VAR and plain4.get(real[cp]) == glyph:
            continue                                   # the same form as at the codepoint itself
        index[cp] = len(recs)
        off = len(bitmaps)
        if kind == 2:                                  # identical 4-bit bitmaps (a Japanese and a Traditional
            off = shared.setdefault(data, off)         # form alike) are stored once
        recs.append(struct.pack("<IHHhhHBBI", cp, w, h, left, top + dy, adv, kind, 0, off))
        if off == len(bitmaps):
            bitmaps += data
    head = struct.pack("<4sHHII", b"PFN2", ascent + descent, ascent, len(recs), len(bitmaps))
    tail = b""
    if seqs:
        tail = struct.pack("<4sI", b"SEQS", len(seqs))
        for k, cps in enumerate(seqs):
            tail += struct.pack("<BBHI", len(cps), 0, 0, index[SEQ_BASE + k]) + struct.pack(f"<{len(cps)}I", *cps)
    (font_out / fname).write_bytes(head + b"".join(recs) + bytes(bitmaps) + tail)
    n_tw = sum(1 for v in plan.values() if not isinstance(v, tuple))
    n_var = sum(1 for k in index if VAR_JP <= k < UI_VAR)
    n_ui = sum(1 for k in index if k >= UI_VAR)
    print(f"{fname}: {len(recs)} glyphs at {size}px ({n_tw} Twemoji incl. {len(seqs)} sequences, {n_var} Japanese / "
          f"Traditional Chinese forms, {n_ui} interface CJK), "
          f"{(len(head) + 20 * len(recs) + len(bitmaps) + len(tail)) / 2**20:.2f} MiB")


def main():
    global face_set, font_out
    argv = [a for a in sys.argv if not a.startswith("--")]
    name = argv[1] if len(argv) > 1 else "qwen3.5-4b-q4"
    model_dir = ROOT / "models" / (argv[2] if len(argv) > 2 else "Qwen3.5-4B")
    for a in sys.argv:
        if a.startswith("--fonts="):
            face_set = a.split("=", 1)[1]
        elif a.startswith("--out="):
            font_out = Path(a.split("=", 1)[1])
    OUT.mkdir(parents=True, exist_ok=True)
    font_out.mkdir(parents=True, exist_ok=True)
    if "--fonts-only" not in sys.argv:
        export_kernels("chat" if name.startswith("qwen3.5") else "qwen3")   # chat.cl = qwen35 + parakeet
        export_model_index(name)
        export_tokenizer(model_dir)
    if "--no-fonts" in sys.argv:
        return
    print(f"fonts: {face_set} -> {font_out}")
    export_font("font.bin", 30, full=True)
    export_font("font_bold.bin", 30, full=False, role="bold")
    export_font("font_italic.bin", 30, full=False, role="italic")
    export_font("font_small.bin", 24, full=False, role="text", ui_cjk=400)
    from gen_emoji_pages import keyboard_emoji
    export_font("font_big.bin", 40, full=False, role="semibold", emoji=keyboard_emoji(), emoji_size=25,   # keys
                ui_cjk=600)                                                                           # ~25 px
    export_font("mono.bin", 26, full=False, role="mono")


if __name__ == "__main__":
    main()
