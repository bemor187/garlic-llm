"""Japanese kanji readings for the voice: chat/assets/ja_kana.bin, from the dictionary of pykakasi 2.3.0 (the version
Chatterbox Multilingual pins). The official tokenizer turns kanji into hiragana with pykakasi before its BPE; the
app does the same in C (chat/ja_kana.c): at each kanji, the longest dictionary word that starts there, read with
the first reading pykakasi would pick, or the one in FIXES.

Dictionary: pykakasi's kanwadict4 (the KAKASI dictionary, made from SKK's, plus part of UniDic) and itaijidict4
(variant kanji -> their usual form). pykakasi and its dictionary are under GPL-3.0-or-later (KAKASI: GPL-2.0-or-
later; UniDic: GPL-2.0 / LGPL-2.1 / BSD-3-Clause, the part in pykakasi relicensed as GPL-3.0-or-later), so the
file can ship with this GPL-3.0 app (see CREDITS.md).

Words are looked up in the text as written (composed): the official tokenizer decomposes the text (NFKD) before
pykakasi, so a word whose kana ending is voiced never matches there (食べる, 上がる, 近づく are read くべる, うえがる,
きんづく); in composed text they do. Words with a variant kanji are dropped (the text is converted to the usual
forms first), as are words with ASCII letters.

ja_kana.bin:
  "PJKD", u32 n_variants, u32 n_first, u32 n_words
  n_variants x <II   variant codepoint -> usual codepoint (sorted)
  n_first + 1 x <II  first character of a word (sorted; last = 0xFFFFFFFF), offset of its block in the data
  the blocks: the words that start with that character, sorted by the rest of the word; per word
    u8 shared   bytes of the rest's code shared with the word before (in the same block)
    u8 n        bytes that follow, the rest of the rest's code
    u8 y        bits 0-5: bytes of the reading's code; bits 6-7: characters at the end of the word that also
                end the reading (okurigana: 行く -> い + く)
    the reading's code
  code of a character: hiragana and katakana U+3041..U+30FF are one byte (codepoint - 0x3040, 0x01..0xBF); any
  other character is its 3-byte UTF-8 (lead byte 0xE0..0xEF).

Run: python pc/export_ja_kana.py   (needs pykakasi==2.3.0: pip install pykakasi==2.3.0)
"""
import pickle
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "chat" / "assets" / "ja_kana.bin"


# pykakasi's first reading of a few very common words is not the usual one: 今日は is read as the greeting
# こんにちは, 人 as にん, 日 as にち, 月 as がつ, 日本 as にっぽん, 飯 (ご飯) as めし. These are read the usual way.
FIXES = {"今日": "きょう", "今日は": "きょうは", "日本": "にほん", "人": "ひと", "日": "ひ", "月": "つき", "飯": "はん",
         "表": "ひょう", "開け": "あけ", "入っ": "はいっ", "売上が": "うりあげが"}


def k2h(s):   # pykakasi's katakana -> hiragana for the readings (IConv._k2h)
    return "".join(chr(ord(c) - 0x60) if 0x30A0 < ord(c) < 0x30F7 else c for c in s)


def code(s):
    out = b""
    for c in s:
        o = ord(c)
        if 0x3041 <= o <= 0x30FF:
            out += bytes([o - 0x3040])
        else:
            assert 0x800 <= o <= 0xFFFF, hex(o)
            out += c.encode()
        assert len(out) < 256
    return out


def main():
    import pykakasi
    from importlib.metadata import version
    assert version("pykakasi") == "2.3.0", version("pykakasi")
    data = Path(pykakasi.__file__).parent / "data"
    kanwa = pickle.loads((data / "kanwadict4.db").read_bytes())
    itaiji = pickle.loads((data / "itaijidict4.db").read_bytes())
    # variants: only character -> character (the variation selectors pykakasi deletes are dropped by the app)
    var = sorted((k, ord(v)) for k, v in itaiji.items() if v and len(v) == 1)
    changed = {k for k, v in var if k != v}
    words, dropped = {}, 0
    for first, table in kanwa.items():
        for key, readings in table.items():
            y = next((y for y, cond in readings if cond is None), None)   # pykakasi: the first reading that applies
            if y is None:
                continue
            if any(ord(c) in changed for c in key) or any(ord(c) < 0x3000 for c in key):
                dropped += 1
                continue
            words[key] = k2h(FIXES.get(key, y))
    blocks = {}
    for key, y in words.items():
        blocks.setdefault(ord(key[0]), []).append((code(key[1:]), key, y))
    data_blob, index = b"", []
    for first in sorted(blocks):
        index.append((first, len(data_blob)))
        prev = b""
        for rest, key, y in sorted(blocks[first]):
            tail = 0   # okurigana: kana at the end of the word that the reading ends with too
            while tail < 3 and tail < len(key) - 1 and 0x3041 <= ord(key[-1 - tail]) <= 0x309F and \
                    y.endswith(key[len(key) - 1 - tail:]) and len(y) > tail + 1:
                tail += 1
            yc = code(y[:len(y) - tail])
            assert len(yc) < 64, (key, y)
            shared = 0
            while shared < min(len(prev), len(rest), 255) and prev[shared] == rest[shared]:
                shared += 1
            data_blob += bytes([shared, len(rest) - shared]) + rest[shared:] + bytes([len(yc) | tail << 6]) + yc
            prev = rest
    index.append((0xFFFFFFFF, len(data_blob)))
    blob = struct.pack("<4sIII", b"PJKD", len(var), len(index) - 1, len(words))
    blob += b"".join(struct.pack("<II", a, b) for a, b in var)
    blob += b"".join(struct.pack("<II", a, b) for a, b in index) + data_blob
    OUT.write_bytes(blob)
    big = max(index[i + 1][1] - index[i][1] for i in range(len(index) - 1))
    assert all(k in words for k in FIXES), [k for k in FIXES if k not in words]
    print(f"ja_kana.bin: {len(words)} words ({dropped} that can never match dropped), {len(index) - 1} first "
          f"characters, {len(var)} variants, largest block {big} bytes, {len(blob)} bytes")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
