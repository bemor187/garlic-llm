// See ja_kana.h. The steps follow pykakasi 2.3.0 (Kakasi.convert, JConv.convert) as Chatterbox uses it
// (hiragana_normalize in its tokenizer.py): characters of the kana, Latin, Greek, Cyrillic and symbol ranges are
// left as they are; at a kanji (U+3400..U+DFFF) the longest dictionary word wins; codepoints U+F000..U+FFFD and
// above U+FFFF are dropped. Besides the composed text and the fixed readings (ja_kana.h), two differences, both on
// text that a reply hardly has: a kanji the dictionary does not know is dropped alone (pykakasi also drops the
// character after it), and variation selectors are dropped before the lookup.
#include "ja_kana.h"

#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <stdio.h>   // (the PC tests)
static int read_at(const JaKana* j, uint64_t off, void* buf, size_t n) {
    FILE* f = fopen(j->path, "rb");
    int ok = f && !fseek(f, (long)off, SEEK_SET) && fread(buf, 1, n, f) == n;
    if (f) fclose(f);
    return ok ? 0 : -1;
}
#else
#include <fcntl.h>
#include <unistd.h>
static int read_at(const JaKana* j, uint64_t off, void* buf, size_t n) {
    return pread(j->fd, buf, n, (off_t)off) == (ssize_t)n ? 0 : -1;
}
#endif

int jakana_open(JaKana* j, const char* path) {
    memset(j, 0, sizeof(*j));
    snprintf(j->path, sizeof(j->path), "%s", path);
#ifndef _WIN32
    j->fd = open(path, O_RDONLY);
    if (j->fd < 0) return -1;
#endif
    uint32_t h[4];
    if (read_at(j, 0, h, 16) || memcmp(h, "PJKD", 4)) goto fail;
    j->n_var = h[1], j->n_first = h[2], j->n_words = h[3];
    j->var = malloc((size_t)j->n_var * 8);
    j->first = malloc(((size_t)j->n_first + 1) * 8);
    if (!j->var || !j->first || read_at(j, 16, j->var, (size_t)j->n_var * 8) ||
        read_at(j, 16 + (uint64_t)j->n_var * 8, j->first, ((size_t)j->n_first + 1) * 8))
        goto fail;
    j->data = 16 + (uint64_t)j->n_var * 8 + ((uint64_t)j->n_first + 1) * 8;
    j->ready = 1;
    return 0;
fail:
#ifndef _WIN32
    close(j->fd);
#endif
    free(j->var), free(j->first);
    memset(j, 0, sizeof(*j));
    return -1;
}

static uint32_t usual(const JaKana* j, uint32_t c) {   // variant kanji -> its usual form (itaijidict)
    int lo = 0, hi = (int)j->n_var - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (j->var[2 * m] == c) return j->var[2 * m + 1];
        if (j->var[2 * m] < c) lo = m + 1;
        else hi = m - 1;
    }
    return c;
}

static int put_code(uint32_t c, uint8_t* o) {   // a character in the dictionary's code; 0 if no word can have it
    if (c >= 0x3041 && c <= 0x30FF) return o[0] = (uint8_t)(c - 0x3040), 1;
    if (c < 0x800 || c > 0xFFFF) return 0;
    o[0] = (uint8_t)(0xE0 | c >> 12), o[1] = (uint8_t)(0x80 | (c >> 6 & 63)), o[2] = (uint8_t)(0x80 | (c & 63));
    return 3;
}

static int get_code(const uint8_t* p, uint32_t* c) {
    if (p[0] < 0xC0) return *c = 0x3040u + p[0], 1;
    *c = (uint32_t)(p[0] & 15) << 12 | (uint32_t)(p[1] & 63) << 6 | (p[2] & 63);
    return 3;
}

// The longest word at t[0, m): its length in characters (0: none), its reading in yomi (composed hiragana)
static int lookup(const JaKana* j, const uint32_t* t, int m, uint32_t* yomi, int* ny) {
    uint32_t c0 = usual(j, t[0]);
    int lo = 0, hi = (int)j->n_first - 1, b = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (j->first[2 * mid] == c0) {
            b = mid;
            break;
        }
        if (j->first[2 * mid] < c0) lo = mid + 1;
        else hi = mid - 1;
    }
    if (b < 0) return 0;
    uint32_t off = j->first[2 * b + 1], size = j->first[2 * b + 3] - off;
    static uint8_t* buf;
    static uint32_t cap;
    if (size > cap) {
        uint8_t* nb = realloc(buf, size);
        if (!nb) return 0;
        buf = nb, cap = size;
    }
    if (read_at(j, j->data + off, buf, size)) return 0;
    uint8_t te[64];   // the text after the first character, in the dictionary's code (words have up to 15 characters)
    int tend[20], nt = 0, tl = 0;
    for (int k = 1; k < m && nt < 16 && tl + 3 <= (int)sizeof(te); k++) {
        int l = put_code(usual(j, t[k]), te + tl);
        if (!l) break;
        tl += l, tend[nt++] = tl;
    }
    uint8_t rest[256];
    int best = -1;
    const uint8_t* by = NULL;
    int byl = 0, btail = 0;
    for (uint32_t p = 0; p + 3 <= size;) {
        int shared = buf[p], n = buf[p + 1];
        memcpy(rest + shared, buf + p + 2, n);
        int rl = shared + n;
        p += 2 + n;
        int yl = buf[p] & 63, tail = buf[p] >> 6;
        const uint8_t* yc = buf + p + 1;
        p += 1 + yl;
        if (rl > tl || memcmp(rest, te, rl)) continue;
        int chars = 0;
        while (chars < nt && tend[chars] < rl) chars++;
        if (rl && (chars >= nt || tend[chars] != rl)) continue;
        if (rl) chars++;
        if (chars > best) best = chars, by = yc, byl = yl, btail = tail;
    }
    if (best < 0) return 0;
    int len = 1 + best, k = 0;
    for (int q = 0; q < byl;) q += get_code(by + q, &yomi[k++]);
    for (int q = len - btail; q < len; q++) yomi[k++] = t[q];
    *ny = k;
    return len;
}

static int passes(uint32_t c) {   // pykakasi's kana, Latin, symbol (its ranges, as written), long-vowel and end marks
    return (c >= 0x20 && c < 0x7F) || (c >= 0xA1 && c <= 0xFF) || (c >= 0x391 && c <= 0x30A1) ||
           (c > 0x30A0 && c < 0x30FD) || c == 0x2015 || c == 0x2212 || (c >= 0x3030 && c <= 0x303F) ||
           (c >= 0xFF01 && c <= 0xFF1A) || c == 0xFF1C || c == 0xFF1E || c == 0xFF1F || (c >= 0xFF20 && c <= 0xFF5E) ||
           (c > 0xFF65 && c < 0xFF9F) || (c >= 0x1B150 && c <= 0x1B152) || (c >= 0x1B164 && c <= 0x1B167);
}

int jakana_convert(const JaKana* j, const uint32_t* in, int n, uint32_t* out, int max) {
    int o = 0;
    uint32_t yomi[96];
    for (int i = 0; i < n && o < max;) {
        uint32_t c = in[i];
        // variation selectors, and the zero-width space that keeps a word apart (st_speak: 約 + a number's reading)
        if ((c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xE0100 && c <= 0xE01EF) || c == 0x200B) {
            i++;
            continue;
        }
        if (!passes(c) && c >= 0x3400 && c < 0xE000) {
            int ny = 0, len = j->ready ? lookup(j, in + i, n - i, yomi, &ny) : 0;
            int kanji = 0;   // Chatterbox reads the word only if it has a kanji of U+4E00..U+9FFF
            for (int q = i; q < i + (len ? len : 1); q++) kanji |= in[q] >= 0x4E00 && in[q] <= 0x9FFF;
            if (!len) {   // unknown: pykakasi has no reading for it, and Chatterbox writes the empty reading
                if (!kanji) out[o++] = c;
                i++;
                continue;
            }
            if (!kanji) {
                for (int q = i; q < i + len && o < max; q++) out[o++] = in[q];
            } else {
                // は, へ (not after the prefixes ご, お: ご飯 is ごはん)
                if ((yomi[0] == 0x306F || yomi[0] == 0x3078) && o < max && !(o && (out[o - 1] == 0x3054 || out[o - 1] == 0x304A)))
                    out[o++] = ' ';
                for (int q = 0; q < ny && o < max; q++) out[o++] = yomi[q];
            }
            i += len;
            continue;
        }
        if (!passes(c) && ((c >= 0xF000 && c <= 0xFFFD) || (c >= 0x10000 && c <= 0x10FFFD))) {   // dropped
            i++;
            continue;
        }
        out[o++] = c;
        i++;
    }
    return o;
}
