"""Writes chat/arabic_forms.h: the contextual forms of the Arabic letters (Arabic, Persian, Urdu...), from the
compatibility decompositions of Unicode's Arabic Presentation Forms-A and -B. The app joins letters by drawing
these forms (chat/ui.c, arabic_shape); pc/export_chat.py puts their glyphs in the fonts.

Run: python gen_arabic_forms.py
"""

import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMS = ("isolated", "final", "initial", "medial")
ALEFS = (0x0622, 0x0623, 0x0625, 0x0627)   # alef with madda, hamza above, hamza below, plain


def forms():
    """{letter: [isolated, final, initial, medial]} (0 = no such form) and {alef: (isolated, final)} for lam-alef."""
    table, lam_alef = {}, {}
    for cp in list(range(0xFB50, 0xFE00)) + list(range(0xFE70, 0xFF00)):
        d = unicodedata.decomposition(chr(cp)).split()
        if len(d) < 2 or d[0].strip("<>") not in FORMS:
            continue
        kind, base = d[0].strip("<>"), [int(x, 16) for x in d[1:]]
        if len(base) == 1 and 0x0620 <= base[0] < 0x0700:
            table.setdefault(base[0], [0, 0, 0, 0])
            if not table[base[0]][FORMS.index(kind)]:
                table[base[0]][FORMS.index(kind)] = cp
        elif len(base) == 2 and base[0] == 0x0644 and base[1] in ALEFS and kind in ("isolated", "final"):
            lam_alef.setdefault(base[1], [0, 0])[FORMS.index(kind)] = cp
    return table, lam_alef


def main():
    table, lam_alef = forms()
    out = ["// Written by pc/gen_arabic_forms.py: the contextual forms of the Arabic letters (Unicode's presentation",
           "// forms). {letter, isolated, final, initial, medial}, 0 = none; a letter with an initial form joins on",
           "// both sides, one with only a final form joins only to the letter before it.",
           "static const uint16_t ar_forms[][5] = {"]
    rows = [f"{{0x{cp:04X}, " + ", ".join(f"0x{f:04X}" if f else "0" for f in table[cp]) + "}"
            for cp in sorted(table)]
    for i in range(0, len(rows), 4):
        out.append("    " + ", ".join(rows[i:i + 4]) + ",")
    out.append("};")
    out.append("// lam + alef: {alef, isolated ligature, final ligature}")
    out.append("static const uint16_t ar_lam_alef[][3] = {" +
               ", ".join(f"{{0x{a:04X}, 0x{i:04X}, 0x{f:04X}}}" for a, (i, f) in sorted(lam_alef.items())) + "};")
    (ROOT / "chat" / "arabic_forms.h").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"arabic_forms.h: {len(table)} letters, {len(lam_alef)} lam-alef ligatures")


if __name__ == "__main__":
    main()
