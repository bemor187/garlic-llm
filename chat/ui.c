// Garlic LLM user interface (see ui.h). Software rendering into `back` (1920 x 1080, 0x80RRGGBB): a dark,
// blue theme; anti-aliased shapes from signed distances; bitmap fonts from pc/export_chat.py; a small markdown
// renderer (headings, lists, bold, italics, code, quotes, tables); the chat, the keyboard and the voice screens.
#include "ui.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32   // (this file also builds for Windows, which uses malloc instead)
#include <orbis/libkernel.h>
#endif

// ---- theme ---------------------------------------------------------------------------------------------------
#define C_BG_TOP 0x0B1C3A
#define C_BG_BOT 0x040A18
#define C_ACCENT 0x0070D1
#define C_ACCENT_HI 0x3FA9F5
#define C_TEXT 0xE8EFFA
#define C_TEXT_DIM 0xA3B5D2
#define C_TEXT_MUTED 0x6A80A3
#define C_SURFACE 0x0E1D3A
#define C_SURFACE_HI 0x15284B
#define C_LINE 0x1C3358
#define C_LINE_HI 0x2A4672
#define C_USER 0x0B5DC6
#define C_PANEL 0x0A1631
#define C_KEY 0x152A4E
#define C_KEY_SPECIAL 0x0F1F3E
#define C_GREEN 0x3DDC97
#define C_RED 0xFF6B76
#define C_CROSS 0x8AB4FF
#define C_CIRCLE 0xFF7A85
#define C_TRIANGLE 0x45D6A0
#define C_SQUARE 0xF59ADB

#define HEADER_H 100
#define FOOTER_H 84
#define COL_W 1440
#define COL_X ((SCREEN_W - COL_W) / 2)
#define AI_INDENT 84
#define USER_MAX_W 1000
#define BUB_PAD_X 28
#define BUB_PAD_Y 16
#define MSG_GAP 40

int ui_lang = LANG_ES;
const char* ui_error = "";

// ---- strings ---------------------------------------------------------------------------------------------------

#include "ui_strings.inc"

const char* tr(StrId id) { return strings[id][ui_lang] ? strings[id][ui_lang] : strings[id][LANG_EN]; }

// OrbisSystemParamLanguage (orbis/_types/sys_service.h): 0 Japanese, 1 English (US), 2 French, 3 Spanish, 4 German,
// 7 Portuguese (Portugal), 8 Russian, 9 Korean, 10 Chinese (Traditional), 11 Chinese (Simplified), 17 Portuguese
// (Brazil), 18 English (UK), 20 Spanish (Latin America), 21 Arabic, 22 French (Canada). The rest: English.
int lang_from_system(int l) {
    switch (l) {
    case 0: return LANG_JA;
    case 2: case 22: return LANG_FR;
    case 3: case 20: return LANG_ES;
    case 4: return LANG_DE;
    case 7: case 17: return LANG_PT;
    case 8: return LANG_RU;
    case 9: return LANG_KO;
    case 10: return LANG_ZH_HANT;
    case 11: return LANG_ZH_HANS;
    case 21: return LANG_AR;
    default: return LANG_EN;
    }
}

int lang_rtl(int lang) { return lang == LANG_AR; }

// ---- pixels and shapes -------------------------------------------------------------------------------------------

static int clip_x0, clip_y0, clip_x1 = SCREEN_W, clip_y1 = SCREEN_H;
static int text_alpha = 255;   // multiplies text and shapes drawn by the fading toast

static void set_clip(int x0, int y0, int x1, int y1) {
    clip_x0 = x0 < 0 ? 0 : x0, clip_y0 = y0 < 0 ? 0 : y0;
    clip_x1 = x1 > SCREEN_W ? SCREEN_W : x1, clip_y1 = y1 > SCREEN_H ? SCREEN_H : y1;
}
static void no_clip(void) { set_clip(0, 0, SCREEN_W, SCREEN_H); }

static inline void blend(uint32_t* d, uint32_t rgb, uint32_t a) {
    if (a >= 255) {
        *d = 0x80000000 | rgb;
        return;
    }
    uint32_t s = *d, ia = 255 - a;
    uint32_t r = ((s >> 16 & 255) * ia + (rgb >> 16 & 255) * a) * 257 >> 16;
    uint32_t g = ((s >> 8 & 255) * ia + (rgb >> 8 & 255) * a) * 257 >> 16;
    uint32_t b = ((s & 255) * ia + (rgb & 255) * a) * 257 >> 16;
    *d = 0x80000000 | r << 16 | g << 8 | b;
}

static void fill_a(int x, int y, int w, int h, uint32_t rgb, int alpha) {
    int x0 = x < clip_x0 ? clip_x0 : x, y0 = y < clip_y0 ? clip_y0 : y;
    int x1 = x + w > clip_x1 ? clip_x1 : x + w, y1 = y + h > clip_y1 ? clip_y1 : y + h;
    alpha = alpha * text_alpha / 255;
    for (int j = y0; j < y1; j++) {
        uint32_t* row = back + j * SCREEN_W;
        if (alpha >= 255)
            for (int i = x0; i < x1; i++) row[i] = 0x80000000 | rgb;
        else
            for (int i = x0; i < x1; i++) blend(row + i, rgb, alpha);
    }
}
static void fill(int x, int y, int w, int h, uint32_t rgb) { fill_a(x, y, w, h, rgb, 255); }

static inline float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

// Signed distance from (px, py) to a rounded box (centre, half size, corner radius r); negative inside.
static inline float rbox_sd(float px, float py, float cx, float cy, float hw, float hh, float r) {
    float qx = fabsf(px - cx) - (hw - r), qy = fabsf(py - cy) - (hh - r);
    float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
    float in = qx > qy ? qx : qy;
    if (in > 0) in = 0;
    float out = ox == 0 ? oy : oy == 0 ? ox : sqrtf(ox * ox + oy * oy);
    return out + in - r;
}

// Rounded rectangle, filled (thick 0) or outlined; radii top-left, top-right, bottom-right, bottom-left.
static void rrect4(int x, int y, int w, int h, const int rad[4], int thick, uint32_t rgb, int alpha) {
    if (w <= 0 || h <= 0) return;
    alpha = alpha * text_alpha / 255;
    float cx = x + w * 0.5f, cy = y + h * 0.5f, hw = w * 0.5f, hh = h * 0.5f;
    int rmax = rad[0];
    for (int k = 1; k < 4; k++)
        if (rad[k] > rmax) rmax = rad[k];
    int band = (thick > rmax ? thick : rmax) + 1;
    int y0 = y < clip_y0 ? clip_y0 : y, y1 = y + h > clip_y1 ? clip_y1 : y + h;
    int xa = x < clip_x0 ? clip_x0 : x, xb = x + w > clip_x1 ? clip_x1 : x + w;
    for (int py = y0; py < y1; py++) {
        uint32_t* row = back + py * SCREEN_W;
        float fy = py + 0.5f;
        int top = fy < cy;
        int edge_row = py < y + band || py >= y + h - band;
        for (int px = xa; px < xb; px++) {
            if (!edge_row && px >= x + band && px < x + w - band) {   // the plain middle of the row
                if (thick) {
                    px = x + w - band - 1;
                    continue;
                }
                int stop = x + w - band < xb ? x + w - band : xb;
                for (; px < stop; px++) blend(row + px, rgb, alpha);
                px--;
                continue;
            }
            float fx = px + 0.5f;
            float r = (float)(fx < cx ? (top ? rad[0] : rad[3]) : (top ? rad[1] : rad[2]));
            if (r > hw) r = hw;
            if (r > hh) r = hh;
            float d = rbox_sd(fx, fy, cx, cy, hw, hh, r);
            float cov = clamp01(0.5f - d);
            if (thick) cov -= clamp01(0.5f - (d + thick));
            if (cov > 0.004f) blend(row + px, rgb, (uint32_t)(cov * alpha));
        }
    }
}
static void rrect(int x, int y, int w, int h, int r, uint32_t rgb, int alpha) {
    int rad[4] = {r, r, r, r};
    rrect4(x, y, w, h, rad, 0, rgb, alpha);
}
static void rrect_line(int x, int y, int w, int h, int r, int thick, uint32_t rgb, int alpha) {
    int rad[4] = {r, r, r, r};
    rrect4(x, y, w, h, rad, thick, rgb, alpha);
}

// Disc (thick 0) or ring of radius r around (cx, cy). Per row, only the pixels near an edge need their distance:
// further out they are not covered, and the inner span is either fully covered (a disc: cov 1) or the ring's hole
// (cov 0). The margins of 0.01 px keep those spans strictly where the per-pixel formula gives exactly 1 or 0, so the
// result is the same as computing every pixel (this keeps the voice screen's big halo discs cheap to draw).
static void circle(float cx, float cy, float r, float thick, uint32_t rgb, int alpha) {
    alpha = alpha * text_alpha / 255;
    int x0 = (int)floorf(cx - r - 1), x1 = (int)ceilf(cx + r + 1), y0 = (int)floorf(cy - r - 1), y1 = (int)ceilf(cy + r + 1);
    if (x0 < clip_x0) x0 = clip_x0;
    if (y0 < clip_y0) y0 = clip_y0;
    if (x1 > clip_x1) x1 = clip_x1;
    if (y1 > clip_y1) y1 = clip_y1;
    const float ro = r + 0.51f, ri = thick > 0 ? r - thick - 0.51f : r - 0.51f;
    for (int py = y0; py < y1; py++) {
        float dy = py + 0.5f - cy, ady = fabsf(dy);
        if (ady >= ro) continue;   // the whole row is outside (d > 0.5: cov 0)
        float ho = sqrtf(ro * ro - dy * dy);
        int o0 = (int)floorf(cx - ho - 0.5f), o1 = (int)ceilf(cx + ho - 0.5f) + 1;   // outside [o0, o1): cov 0
        if (o0 < x0) o0 = x0;
        if (o1 > x1) o1 = x1;
        int a0 = o1, a1 = o1;   // [a0, a1): the inner span
        if (ri > ady) {
            float hi = sqrtf(ri * ri - dy * dy);
            a0 = (int)ceilf(cx - hi - 0.5f), a1 = (int)floorf(cx + hi - 0.5f) + 1;
            if (a0 < o0) a0 = o0;
            if (a1 > o1) a1 = o1;
            if (a1 < a0) a1 = a0;
        }
        uint32_t* row = back + py * SCREEN_W;
        for (int px = o0; px < o1; px++) {
            if (px == a0) {
                if (thick <= 0)
                    for (; px < a1; px++) blend(row + px, rgb, (uint32_t)(1.0f * alpha));
                px = a1;
                if (px >= o1) break;
            }
            float dx = px + 0.5f - cx;
            float d = sqrtf(dx * dx + dy * dy) - r;
            float cov = clamp01(0.5f - d);
            if (thick > 0) cov -= clamp01(0.5f - (d + thick));
            if (cov > 0.004f) blend(row + px, rgb, (uint32_t)(cov * alpha));
        }
    }
}

// Line with round ends, width wd.
static void line(float x0, float y0, float x1, float y1, float wd, uint32_t rgb, int alpha) {
    alpha = alpha * text_alpha / 255;
    float r = wd * 0.5f;
    int bx0 = (int)floorf(fminf(x0, x1) - r - 1), bx1 = (int)ceilf(fmaxf(x0, x1) + r + 1);
    int by0 = (int)floorf(fminf(y0, y1) - r - 1), by1 = (int)ceilf(fmaxf(y0, y1) + r + 1);
    if (bx0 < clip_x0) bx0 = clip_x0;
    if (by0 < clip_y0) by0 = clip_y0;
    if (bx1 > clip_x1) bx1 = clip_x1;
    if (by1 > clip_y1) by1 = clip_y1;
    float vx = x1 - x0, vy = y1 - y0, len2 = vx * vx + vy * vy;
    for (int py = by0; py < by1; py++)
        for (int px = bx0; px < bx1; px++) {
            float wx = px + 0.5f - x0, wy = py + 0.5f - y0;
            float t = len2 > 0 ? (wx * vx + wy * vy) / len2 : 0;
            t = clamp01(t);
            float dx = wx - vx * t, dy = wy - vy * t;
            float cov = clamp01(r - sqrtf(dx * dx + dy * dy) + 0.5f);
            if (cov > 0.004f) blend(back + py * SCREEN_W + px, rgb, (uint32_t)(cov * alpha));
        }
}

// ---- background: a dark blue gradient with a soft glow, computed once --------------------------------------------

static uint32_t* bg;

// The four 8 MB screen copies (the frame being drawn, this background, the backdrop of modal screens and the
// keyboard's picture) come from a 32 MiB block of direct memory of their own, CPU cached (WB Onion) like malloc's, so
// they stay out of the system RAM (flexible memory), where the 9B's dictation weights leave ~20 MiB free.
// The engine then leaves 32 MiB less for the system (ui_screen_pool_mb). Without the block: malloc.
static uint8_t* scr_pool;
static int scr_tried, scr_used;
#define SCR_POOL (32u << 20)
#define SCR_ONE (((size_t)SCREEN_W * SCREEN_H * 4 + 0xFFFF) & ~(size_t)0xFFFF)

int ui_screen_pool_mb(void) {
#ifndef _WIN32
    if (!scr_tried) {
        scr_tried = 1;
        off_t off = 0;
        void* m = NULL;
        if (!sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), SCR_POOL, 0x200000, 0 /* WB Onion */, &off)) {
            if (!sceKernelMapDirectMemory(&m, SCR_POOL, 0x03 /* CPU read/write */, 0, off, 0x200000)) scr_pool = m;
            else sceKernelReleaseDirectMemory(off, SCR_POOL);
        }
    }
#endif
    return scr_pool ? (int)(SCR_POOL >> 20) : 0;
}

static int in_pool(const void* p) { return scr_pool && (const uint8_t*)p >= scr_pool && (const uint8_t*)p < scr_pool + SCR_POOL; }

uint32_t* ui_screen_buf(void) {
    ui_screen_pool_mb();
    if (scr_pool && (size_t)(scr_used + 1) * SCR_ONE <= SCR_POOL) return (uint32_t*)(scr_pool + SCR_ONE * scr_used++);
    return malloc(SCREEN_W * SCREEN_H * 4);
}

static void make_background(void) {
    bg = ui_screen_buf();
    uint32_t seed = 1;
    for (int y = 0; y < SCREEN_H; y++) {
        float t = (float)y / (SCREEN_H - 1);
        float base[3] = {(C_BG_TOP >> 16 & 255) + ((C_BG_BOT >> 16 & 255) - (C_BG_TOP >> 16 & 255)) * t,
                         (C_BG_TOP >> 8 & 255) + ((C_BG_BOT >> 8 & 255) - (C_BG_TOP >> 8 & 255)) * t,
                         (C_BG_TOP & 255) + ((C_BG_BOT & 255) - (C_BG_TOP & 255)) * t};
        for (int x = 0; x < SCREEN_W; x++) {
            float dx = (x - SCREEN_W * 0.5f) / (SCREEN_W * 0.75f), dy = (y + 200.0f) / (SCREEN_H * 1.1f);
            float g = 1.0f - (dx * dx + dy * dy);
            g = g > 0 ? g * g * 0.55f : 0;
            seed = seed * 1664525u + 1013904223u;
            float dither = ((seed >> 24) / 255.0f - 0.5f) * 1.2f;   // breaks up banding in the gradient
            int r = (int)(base[0] + g * 6 + dither), gr = (int)(base[1] + g * 34 + dither), b = (int)(base[2] + g * 80 + dither);
            r = r < 0 ? 0 : r > 255 ? 255 : r, gr = gr < 0 ? 0 : gr > 255 ? 255 : gr, b = b < 0 ? 0 : b > 255 ? 255 : b;
            if (y < HEADER_H || y >= SCREEN_H - FOOTER_H) {   // header and footer bars: darker
                r = r * 5 / 9 + 1, gr = gr * 5 / 9 + 3, b = b * 5 / 9 + 8;
                if (y == HEADER_H - 1 || y == SCREEN_H - FOOTER_H) r = C_LINE >> 16, gr = C_LINE >> 8 & 255, b = C_LINE & 255;
            }
            bg[y * SCREEN_W + x] = 0x80000000 | r << 16 | gr << 8 | b;
        }
    }
}

static void draw_background(void) { memcpy(back, bg, SCREEN_W * SCREEN_H * 4); }

// Modal screens (keyboard, voice) draw the scene behind them once, keep it here and restore it every frame.
// Little system memory may be left (the dictation weights are there), so neither copy is guaranteed:
// without one, the plain background stands in.
static uint32_t* backdrop;
static int kb_cache_sig = -1;
static uint32_t* kb_cache;
void ui_backdrop_save(void) {
    kb_cache_sig = -1;   // the keyboard's cached picture contains the old backdrop
    if (!backdrop) backdrop = ui_screen_buf();
    if (backdrop) memcpy(backdrop, back, SCREEN_W * SCREEN_H * 4);
}
void ui_backdrop_draw(void) {
    if (backdrop) memcpy(back, backdrop, SCREEN_W * SCREEN_H * 4);
    else draw_background();
}
// Gives back the two 8 MB screen copies (they come back when next needed, if there is room); those in the screen
// block are not system RAM and stay.
void ui_free_caches(void) {
    if (backdrop && !in_pool(backdrop)) free(backdrop), backdrop = NULL;
    if (kb_cache && !in_pool(kb_cache)) free(kb_cache), kb_cache = NULL;
    kb_cache_sig = -1;
}

// Darkens the area between the header and the footer by half (cheap: one shift per pixel).
void ui_dim(void) {
    for (uint32_t* p = back + HEADER_H * SCREEN_W; p < back + (SCREEN_H - FOOTER_H) * SCREEN_W; p++)
        *p = 0x80000000 | ((*p & 0xFEFEFE) >> 1) + 0x010307;
}

// Rows y0..y1 fade from the background (alpha a0) to the current picture (a1): soft edges for scrolled text.
static void fade_rows(int y0, int y1, int a0, int a1) {
    for (int y = y0; y < y1; y++) {
        int a = a0 + (a1 - a0) * (y - y0) / (y1 - y0);
        uint32_t* d = back + y * SCREEN_W;
        const uint32_t* s = bg + y * SCREEN_W;
        for (int x = 0; x < SCREEN_W; x++) blend(d + x, s[x] & 0xFFFFFF, 255 - a);
    }
}

// ---- fonts (PFN2 from pc/export_chat.py) ----------------------------------------------------------------------
// Header "<4sHHII" (magic, line height, ascent, glyph count, bitmap bytes); per glyph "<IHHhhHBBI" (codepoint,
// width, height, left, top, advance, kind 0 = alpha / 1 = RGBA / 2 = 4-bit alpha (two pixels a byte, the first in
// the low half), pad, bitmap offset), sorted by codepoint; then the bitmaps; then optionally "SEQS" + count + per
// sequence "<BBHI" (length, 0, 0, glyph index) and its codepoints: emoji made of several characters (flags,
// keycaps, joined emoji), drawn as one image.
// Emoji images are Twemoji (CC-BY 4.0, credited in assets/CREDITOS.txt).
//
// Chinese, Japanese and Korean: the Simplified Chinese forms sit at their own codepoints; the Japanese and the
// Traditional Chinese forms that differ (or that the first set lacks) at VAR_JP / VAR_TC + codepoint. Each
// language looks first in its own set (cjk_order). The hint and title fonts also hold the interface's own
// characters at UI_VAR + (VAR_JP / VAR_TC +) codepoint; font_head, the copy of font_big for markdown headings,
// stops before them, so a heading is not drawn half in one size and half in another.
#define FONT_HEADER 16
#define FONT_REC 20
#define VAR_JP 0x200000u
#define VAR_TC 0x400000u
#define UI_VAR 0x800000u

typedef struct Font {
    uint16_t line, ascent;
    uint32_t n;
    const uint8_t* recs;
    const uint8_t* bitmaps;
    uint32_t* cps;
    int fallback;
    const struct Font* next;   // font to ask when this one lacks a character
    uint32_t n_seq;
    uint32_t* seq_first;
    uint8_t* seq_len;
    uint32_t* seq_glyph;
    const uint8_t** seq_cps;
    int space;                 // advance of ' '
    int has_ui;                // holds interface characters (UI_VAR)
    int embolden;              // bold: a glyph it borrows from the regular font is drawn twice, a pixel apart
} Font;

typedef struct {
    uint16_t w, h, adv;
    int16_t bx, by;
    uint8_t kind;
    uint8_t borrowed;   // from a font further down the chain (font->next...)
    const uint8_t* bm;
} Glyph;

static Font font, font_bold, font_italic, font_small, font_big, font_head, mono;

static uint32_t utf8_next(const char** s) {
    const uint8_t* p = (const uint8_t*)*s;
    uint32_t c = *p++;
    int extra = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0;
    if (extra) c &= 0x3F >> extra;
    for (int k = 0; k < extra && (*p & 0xC0) == 0x80; k++) c = c << 6 | (*p++ & 0x3F);
    *s = (const char*)p;
    return c;
}

static int utf8_put(uint32_t c, char* o) {
    if (c < 0x80) return o[0] = (char)c, 1;
    if (c < 0x800) return o[0] = (char)(0xC0 | c >> 6), o[1] = (char)(0x80 | (c & 63)), 2;
    if (c < 0x10000)
        return o[0] = (char)(0xE0 | c >> 12), o[1] = (char)(0x80 | (c >> 6 & 63)), o[2] = (char)(0x80 | (c & 63)), 3;
    o[0] = (char)(0xF0 | c >> 18), o[1] = (char)(0x80 | (c >> 12 & 63)), o[2] = (char)(0x80 | (c >> 6 & 63));
    o[3] = (char)(0x80 | (c & 63));
    return 4;
}

static void glyph_at(const Font* f, int i, Glyph* g) {
    const uint8_t* r = f->recs + i * FONT_REC;
    uint32_t off;
    memcpy(&g->w, r + 4, 2);
    memcpy(&g->h, r + 6, 2);
    memcpy(&g->bx, r + 8, 2);
    memcpy(&g->by, r + 10, 2);
    memcpy(&g->adv, r + 12, 2);
    g->kind = r[14];
    memcpy(&off, r + 16, 4);
    g->bm = f->bitmaps + off;
}

// Coverage (0-255) of pixel k (row by row) of an alpha glyph.
static inline uint32_t glyph_alpha(const Glyph* g, int k) {
    return g->kind == 2 ? (uint32_t)(g->bm[k >> 1] >> ((k & 1) << 2) & 15) * 17 : g->bm[k];
}

static int find_cp(const Font* f, uint32_t cp) {
    int lo = 0, hi = (int)f->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (f->cps[mid] == cp) return mid;
        if (f->cps[mid] < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int cjk_variant_cp(uint32_t c) {   // may have forms of its own per language (see VAR_JP)
    return (c >= 0x2E80 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7A3) || (c >= 0xF900 && c <= 0xFAFF) ||
           (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF00 && c <= 0xFFEF) || (c >= 0x1100 && c <= 0x11FF) ||
           (c >= 0x20000 && c <= 0x3FFFF);
}

static const uint32_t* cjk_order(void) {   // where to look first: the forms of the interface's language
    static const uint32_t jp[3] = {VAR_JP, 0, VAR_TC}, tc[3] = {VAR_TC, 0, VAR_JP}, sc[3] = {0, VAR_TC, VAR_JP};
    return ui_lang == LANG_JA ? jp : ui_lang == LANG_ZH_HANT || ui_lang == LANG_KO ? tc : sc;   // (Korean hanja:
}                                                                                          // traditional forms)

static int find_glyph(const Font* f, uint32_t cp) {
    if (!cjk_variant_cp(cp)) return find_cp(f, cp);
    const uint32_t* o = cjk_order();
    for (int u = f->has_ui ? 0 : 1; u < 2; u++)
        for (int r = 0; r < 3; r++) {
            int i = find_cp(f, (u ? 0 : UI_VAR) + o[r] + cp);
            if (i >= 0) return i;
        }
    return -1;
}

static int font_load(Font* f, const char* path) {
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < FONT_HEADER || memcmp(d, "PFN2", 4)) return -1;
    memset(f, 0, sizeof(*f));
    uint32_t bitmap_bytes;
    memcpy(&f->line, d + 4, 2);
    memcpy(&f->ascent, d + 6, 2);
    memcpy(&f->n, d + 8, 4);
    memcpy(&bitmap_bytes, d + 12, 4);
    f->recs = d + FONT_HEADER;
    f->bitmaps = d + FONT_HEADER + f->n * FONT_REC;
    f->cps = malloc(f->n * 4);
    for (uint32_t i = 0; i < f->n; i++) {
        memcpy(&f->cps[i], f->recs + i * FONT_REC, 4);
        if (f->cps[i] == 0xFFFD) f->fallback = (int)i;
    }
    f->has_ui = f->n && f->cps[f->n - 1] >= UI_VAR;
    const uint8_t* p = f->bitmaps + bitmap_bytes;
    if ((size_t)(p - d) + 8 <= size && !memcmp(p, "SEQS", 4)) {
        memcpy(&f->n_seq, p + 4, 4);
        p += 8;
        f->seq_first = malloc(f->n_seq * 4);
        f->seq_len = malloc(f->n_seq);
        f->seq_glyph = malloc(f->n_seq * 4);
        f->seq_cps = malloc(f->n_seq * sizeof(*f->seq_cps));
        for (uint32_t k = 0; k < f->n_seq; k++) {
            f->seq_len[k] = p[0];
            memcpy(&f->seq_glyph[k], p + 4, 4);
            memcpy(&f->seq_first[k], p + 8, 4);
            f->seq_cps[k] = p + 12;
            p += 8 + 4 * f->seq_len[k];
        }
    }
    int sp = find_cp(f, ' ');
    Glyph g;
    if (sp >= 0) glyph_at(f, sp, &g), f->space = g.adv;
    else f->space = f->line / 4;
    return 0;
}

// Joiners, variation selectors, keycap marks, skin-tone modifiers outside a known sequence and the bidirectional
// controls draw as nothing.
static int invisible(uint32_t cp) {
    return (cp >= 0x200B && cp <= 0x200F) || (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0x1F3FB && cp <= 0x1F3FF) ||
           (cp >= 0xE0000 && cp <= 0xE007F) || cp == 0x20E3 || (cp >= 0x202A && cp <= 0x202E) ||
           (cp >= 0x2066 && cp <= 0x2069);
}

static void missing(uint32_t cp) {   // each character no font can draw is logged once
    static uint32_t seen[64];
    static int n;
    for (int i = 0; i < n; i++)
        if (seen[i] == cp) return;
    if (n < 64) seen[n++] = cp;
    say("no glyph for U+%04X", (unsigned)cp);
}

static int match_sequence(const Font* f, uint32_t cp, const char* s, const char* end, const char** after) {
    if (!f->n_seq) return -1;
    int lo = 0, hi = (int)f->n_seq;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (f->seq_first[mid] < cp) lo = mid + 1;
        else hi = mid;
    }
    int best = -1, best_len = 0;
    for (int k = lo; k < (int)f->n_seq && f->seq_first[k] == cp; k++) {
        const char* q = s;
        int ok = 1;
        for (int j = 1; j < f->seq_len[k] && ok; j++) {
            uint32_t want;
            memcpy(&want, f->seq_cps[k] + 4 * (j - 1), 4);
            uint32_t got = 0;
            do {
                if (q >= end || !*q) {
                    ok = 0;
                    break;
                }
                got = utf8_next(&q);
            } while (got == 0xFE0F);
            if (ok && got != want) ok = 0;
        }
        if (ok && f->seq_len[k] > best_len) {
            best = k, best_len = f->seq_len[k];
            *after = q;
        }
    }
    return best;
}

// ---- Arabic letters: joined by drawing each one in its contextual form (also Persian and Urdu) ------------------

#include "arabic_forms.h"

static const uint16_t* ar_letter(uint32_t cp) {   // its row of ar_forms, or NULL
    if (cp < 0x0621 || cp > 0x06FF) return NULL;
    int lo = 0, hi = (int)(sizeof(ar_forms) / sizeof(ar_forms[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (ar_forms[mid][0] == cp) return ar_forms[mid];
        if (ar_forms[mid][0] < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

// Vowel signs and other marks over or under a letter: they do not break the joining.
static int ar_mark(uint32_t c) {
    return (c >= 0x0610 && c <= 0x061A) || (c >= 0x064B && c <= 0x065F) || c == 0x0670 || (c >= 0x06D6 && c <= 0x06DC) ||
           (c >= 0x06DF && c <= 0x06E4) || c == 0x06E7 || c == 0x06E8 || (c >= 0x06EA && c <= 0x06ED);
}

// Joins on to the next letter (dual-joining letters, tatweel, zero-width joiner) / to the one before (those and
// the right-joining letters). A zero-width non-joiner joins neither way.
static int ar_joins_next(uint32_t c) {
    const uint16_t* r;
    return c == 0x0640 || c == 0x200D || ((r = ar_letter(c)) && r[3]);
}
static int ar_joins_prev(uint32_t c) {
    const uint16_t* r;
    return c == 0x0640 || c == 0x200D || ((r = ar_letter(c)) && r[2]);
}

static uint32_t cp_before(const char* begin, const char* at) {   // the character before `at`, marks skipped; 0 = none
    while (at > begin) {
        const char* p = at - 1;
        while (p > begin && ((uint8_t)*p & 0xC0) == 0x80) p--;
        const char* q = p;
        uint32_t c = utf8_next(&q);
        if (!ar_mark(c)) return c;
        at = p;
    }
    return 0;
}

static uint32_t cp_after(const char* s, const char* end) {   // the next character, marks skipped; 0 = none
    while (s < end && *s) {
        uint32_t c = utf8_next(&s);
        if (!ar_mark(c)) return c;
    }
    return 0;
}

// The form of letter cp (which started at `at`; *s is past it) among the text that began at `begin`: isolated,
// final, initial or medial; lam followed by alef becomes one ligature (*s then moves past the alef).
static uint32_t arabic_shape(uint32_t cp, const char* begin, const char* at, const char** s, const char* end) {
    const uint16_t* r = ar_letter(cp);
    if (!r) return cp;
    int before = r[2] && ar_joins_next(cp_before(begin, at));
    if (cp == 0x0644 && *s < end) {
        const char* q = *s;
        uint32_t n = utf8_next(&q);
        for (size_t k = 0; k < sizeof(ar_lam_alef) / sizeof(ar_lam_alef[0]); k++)
            if (ar_lam_alef[k][0] == n) {
                *s = q;
                return ar_lam_alef[k][before ? 2 : 1];
            }
    }
    int after = r[3] && ar_joins_prev(cp_after(*s, end));
    uint16_t f = before ? (after ? r[4] : r[2]) : (after ? r[3] : r[1]);
    return f ? f : cp;
}

// Brackets read the other way round inside right-to-left text.
static uint32_t mirror_cp(uint32_t c) {
    static const uint32_t pairs[][2] = {{'(', ')'}, {')', '('}, {'[', ']'}, {']', '['}, {'{', '}'}, {'}', '{'},
                                        {'<', '>'}, {'>', '<'}, {0xAB, 0xBB}, {0xBB, 0xAB}, {0x2039, 0x203A},
                                        {0x203A, 0x2039}, {0xFD3E, 0xFD3F}, {0xFD3F, 0xFD3E}};
    for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++)
        if (pairs[i][0] == c) return pairs[i][1];
    return c;
}

// The next glyph of s[..end) (the text started at begin: Arabic letters look back to join); rtl: drawn right to
// left (brackets mirrored). Returns 0 at the end, 2 for a mark that sits on the glyph before it, else 1.
static int next_glyph(const Font* f, const char* begin, const char** s, const char* end, Glyph* g, int rtl) {
    while (*s < end && **s) {
        const char* at = *s;
        uint32_t cp = utf8_next(s), base = cp;
        if (invisible(cp)) continue;
        if (cp == '\t') cp = ' ';
        if (cp >= 0x0621 && cp <= 0x06FF) cp = arabic_shape(cp, begin, at, s, end);
        if (rtl) cp = mirror_cp(cp);
        int mark = ar_mark(base) || (base >= 0x0300 && base <= 0x036F);
        for (int pass = 0; pass < 2; pass++, cp = base) {   // (a form no font has: the plain letter)
            for (const Font* t = f; t; t = t->next) {
                const char* after;
                int k = match_sequence(t, cp, *s, end, &after);
                if (k >= 0) {
                    *s = after;
                    glyph_at(t, (int)t->seq_glyph[k], g);
                    if (t != f) g->by += (int)f->ascent - (int)t->ascent;   // keep the baseline of the asking font
                    g->borrowed = t != f;
                    return 1;
                }
                int i = find_glyph(t, cp);
                if (i >= 0) {
                    glyph_at(t, i, g);
                    if (t != f) g->by += (int)f->ascent - (int)t->ascent;
                    g->borrowed = t != f;
                    return mark ? 2 : 1;
                }
            }
            if (cp == base) break;
        }
        missing(base);
        glyph_at(f, f->fallback, g);
        g->borrowed = 0;
        return 1;
    }
    return 0;
}

static int has_rtl(const char* s, const char* e);
static int bidi_draw(const Font* f, int x, int y, const char* s, const char* e, uint32_t rgb, int measure);

static int text_width(const Font* f, const char* s, const char* end) {
    int w = 0;
    const char* begin = s;
    Glyph g;
    while (next_glyph(f, begin, &s, end, &g, 0)) w += g.adv;
    return w;
}
static int str_width(const Font* f, const char* s) { return text_width(f, s, s + strlen(s)); }

static void blit_glyph(const Glyph* g, int gx, int gy, uint32_t rgb) {
    if (gx + g->w <= clip_x0 || gx >= clip_x1 || gy + g->h <= clip_y0 || gy >= clip_y1) return;
    for (int j = 0; j < g->h; j++) {
        int py = gy + j;
        if (py < clip_y0 || py >= clip_y1) continue;
        uint32_t* row = back + py * SCREEN_W;
        for (int i = 0; i < g->w; i++) {
            int px = gx + i;
            if (px < clip_x0 || px >= clip_x1) continue;
            if (g->kind == 1) {
                const uint8_t* c = g->bm + (j * g->w + i) * 4;
                if (c[3]) blend(row + px, (uint32_t)c[0] << 16 | c[1] << 8 | c[2], c[3] * text_alpha / 255);
            } else {
                uint32_t a = glyph_alpha(g, j * g->w + i);
                if (a) blend(row + px, rgb, a * text_alpha / 255);
            }
        }
    }
}

// Left to right, top-left at (x, y); returns the x after the text.
static void blit_font_glyph(const Font* f, const Glyph* g, int gx, int gy, uint32_t rgb) {
    blit_glyph(g, gx, gy, rgb);
    if (f->embolden && g->borrowed && g->kind != 1) blit_glyph(g, gx + 1, gy, rgb);
}

static int draw_ltr(const Font* f, int x, int y, const char* s, const char* end, uint32_t rgb) {
    Glyph g;
    const char* begin = s;
    while (next_glyph(f, begin, &s, end, &g, 0)) {
        blit_font_glyph(f, &g, x + g.bx, y + g.by, rgb);
        x += g.adv;
    }
    return x;
}

// Right to left, from the right edge x1 (Arabic): each glyph goes to the left of the one before; a mark is centred
// over its letter.
static void draw_rtl(const Font* f, int x1, int y, const char* s, const char* end, uint32_t rgb) {
    Glyph g;
    const char* begin = s;
    int x = x1, bx0 = x1, bw = 0;
    for (int k; (k = next_glyph(f, begin, &s, end, &g, 1));) {
        if (k == 2 && !g.adv) {
            blit_font_glyph(f, &g, bx0 + (bw - g.w) / 2, y + g.by, rgb);
            continue;
        }
        x -= g.adv;
        bx0 = x + g.bx, bw = g.w;
        blit_font_glyph(f, &g, bx0, y + g.by, rgb);
    }
}

// Draws s[0..end) with its top-left at (x, y); returns the x after the text. Emoji keep their own colours.
// Right-to-left text and the numbers and Latin words inside it are put in reading order first.
static int draw_text(const Font* f, int x, int y, const char* s, const char* end, uint32_t rgb) {
    if (has_rtl(s, end)) return bidi_draw(f, x, y, s, end, rgb, 0);
    return draw_ltr(f, x, y, s, end, rgb);
}
static int draw_str(const Font* f, int x, int y, const char* s, uint32_t rgb) { return draw_text(f, x, y, s, s + strlen(s), rgb); }
static void draw_center(const Font* f, int cx, int y, const char* s, uint32_t rgb) { draw_str(f, cx - str_width(f, s) / 2, y, s, rgb); }

// ---- bidirectional text ------------------------------------------------------------------------------------------
// The Unicode Bidirectional Algorithm (UAX #9) for one line, without explicit embeddings or isolates (those
// characters count as nothing): character classes, the weak types (W1-W7), the neutrals (N1-N2), the levels
// (I1-I2), trailing spaces at the paragraph's level (L1) and the reversal (L2). A paragraph's direction is that of
// most of its letters, so "Python هي لغة برمجة" reads right to left.

enum { BD_L, BD_R, BD_AL, BD_EN, BD_ES, BD_ET, BD_AN, BD_CS, BD_NSM, BD_BN, BD_B, BD_S, BD_WS, BD_ON };

static int bidi_class(uint32_t c) {
    if (c < 0x80) {
        if (((c | 32) >= 'a' && (c | 32) <= 'z')) return BD_L;
        if (c >= '0' && c <= '9') return BD_EN;
        if (c == ' ') return BD_WS;
        if (c == '\t') return BD_S;
        if (c == '\n' || c == '\r') return BD_B;
        if (c == '+' || c == '-') return BD_ES;
        if (c == '#' || c == '$' || c == '%') return BD_ET;
        if (c == ',' || c == '.' || c == '/' || c == ':') return BD_CS;
        return c < 0x20 ? BD_BN : BD_ON;
    }
    if (c < 0xC0) {
        if (c == 0xA0) return BD_CS;
        if ((c >= 0xA2 && c <= 0xA5) || c == 0xB0 || c == 0xB1) return BD_ET;
        if (c == 0xB2 || c == 0xB3 || c == 0xB9) return BD_EN;
        if (c == 0xAA || c == 0xB5 || c == 0xBA) return BD_L;
        return c == 0xAD ? BD_BN : BD_ON;
    }
    if (c == 0xD7 || c == 0xF7) return BD_ON;
    if (c >= 0x0300 && c <= 0x036F) return BD_NSM;
    if (c >= 0x0590 && c <= 0x05FF) return c >= 0x0591 && c <= 0x05BD ? BD_NSM : BD_R;   // Hebrew
    if (c >= 0x07C0 && c <= 0x085F) return BD_R;   // N'Ko, Samaritan, Mandaic
    if (c >= 0x0600 && c <= 0x08FF) {               // Arabic, Syriac, Thaana
        if ((c >= 0x0660 && c <= 0x0669) || c == 0x066B || c == 0x066C || c <= 0x0605) return BD_AN;
        if (c >= 0x06F0 && c <= 0x06F9) return BD_EN;   // (Persian digits count as European ones)
        if (c == 0x060C) return BD_CS;
        if (c == 0x066A) return BD_ET;
        if (ar_mark(c) || (c >= 0x08D3 && c <= 0x08FF)) return BD_NSM;
        return BD_AL;
    }
    if (c >= 0x2000 && c <= 0x206F) {
        if (c <= 0x200A || c == 0x2028) return BD_WS;
        if (c == 0x200E) return BD_L;
        if (c == 0x200F) return BD_R;
        if (c == 0x2029) return BD_B;
        if (c == 0x202F) return BD_CS;
        if (c >= 0x2030 && c <= 0x2034) return BD_ET;
        if (c == 0x2044) return BD_CS;
        if ((c >= 0x200B && c <= 0x200D) || (c >= 0x202A && c <= 0x202E) || c >= 0x2060) return BD_BN;
        return BD_ON;
    }
    if ((c >= 0x2070 && c <= 0x2079 && c != 0x2071) || (c >= 0x2080 && c <= 0x2089)) return BD_EN;
    if (c >= 0x20A0 && c <= 0x20CF) return BD_ET;
    if (c >= 0x2100 && c <= 0x2BFF) return BD_ON;   // letterlike, arrows, maths, technical, shapes
    if (c >= 0x2E00 && c <= 0x2E7F) return BD_ON;
    if (c == 0x3000) return BD_WS;
    if ((c >= 0x3001 && c <= 0x3004) || (c >= 0x3008 && c <= 0x3020) || c == 0x3030 || c == 0x30FB) return BD_ON;
    if (c >= 0xFB1D && c <= 0xFB4F) return BD_R;
    if (c >= 0xFB50 && c <= 0xFDFF) return BD_AL;
    if (c >= 0xFE00 && c <= 0xFE0F) return BD_NSM;
    if (c >= 0xFE70 && c <= 0xFEFE) return BD_AL;
    if (c >= 0xFF10 && c <= 0xFF19) return BD_EN;
    if ((c >= 0xFF01 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) ||
        (c >= 0xFF5B && c <= 0xFF65))
        return BD_ON;
    if (c >= 0x1F000 && c <= 0x1FAFF) return BD_ON;   // emoji
    return BD_L;
}

// Quick test: any right-to-left character in s[..e)? (UTF-8 lead bytes of U+0590-07FF, presentation forms FB1D-FEFF,
// the right-to-left mark)
static int has_rtl(const char* s, const char* e) {
    for (const uint8_t* p = (const uint8_t*)s; p < (const uint8_t*)e && *p; p++) {
        if (*p >= 0xD6 && *p <= 0xDF) return 1;
        if (*p == 0xEF && p + 1 < (const uint8_t*)e && p[1] >= 0xAC && p[1] <= 0xBB) return 1;
        if (*p == 0xE2 && p + 2 < (const uint8_t*)e && p[1] == 0x80 && p[2] == 0x8F) return 1;
    }
    return 0;
}

// The direction of a paragraph: right to left when most of its letters are (a tie: its first letter decides).
static int para_rtl(const char* s, const char* e) {
    if (!has_rtl(s, e)) return 0;
    int l = 0, r = 0, first = -1;
    while (s < e && *s) {
        int k = bidi_class(utf8_next(&s));
        if (k == BD_L) l++, first = first < 0 ? 0 : first;
        else if (k == BD_R || k == BD_AL) r++, first = first < 0 ? 1 : first;
    }
    return r > l || (r == l && first == 1);
}

static int bd_neutral(int t) { return t == BD_B || t == BD_S || t == BD_WS || t == BD_ON; }

// Levels of the n characters of classes t (changed in place) in a paragraph of direction rtl.
static void bidi_levels(uint8_t* t, int n, int rtl, uint8_t* lv) {
    static uint8_t orig[1024];
    memcpy(orig, t, n);
    const int sos = rtl ? BD_R : BD_L;
    for (int i = 0; i < n; i++)   // W1: a mark takes the class of what it sits on
        if (t[i] == BD_NSM || t[i] == BD_BN) t[i] = i ? t[i - 1] : sos;
    for (int i = 0, last = sos; i < n; i++) {   // W2: numbers after Arabic letters are Arabic numbers; W3
        if (t[i] == BD_L || t[i] == BD_R || t[i] == BD_AL) last = t[i];
        else if (t[i] == BD_EN && last == BD_AL) t[i] = BD_AN;
    }
    for (int i = 0; i < n; i++)
        if (t[i] == BD_AL) t[i] = BD_R;
    for (int i = 1; i + 1 < n; i++) {   // W4: 1,5 / 1.5 / 10-20 stay one number
        if (t[i] == BD_ES && t[i - 1] == BD_EN && t[i + 1] == BD_EN) t[i] = BD_EN;
        else if (t[i] == BD_CS && t[i - 1] == t[i + 1] && (t[i - 1] == BD_EN || t[i - 1] == BD_AN)) t[i] = t[i - 1];
    }
    for (int i = 0; i < n;) {   // W5: $ and % next to a number are part of it
        if (t[i] != BD_ET) {
            i++;
            continue;
        }
        int j = i;
        while (j < n && t[j] == BD_ET) j++;
        if ((i > 0 && t[i - 1] == BD_EN) || (j < n && t[j] == BD_EN)) memset(t + i, BD_EN, j - i);
        i = j;
    }
    for (int i = 0; i < n; i++)   // W6
        if (t[i] == BD_ES || t[i] == BD_ET || t[i] == BD_CS) t[i] = BD_ON;
    for (int i = 0, last = sos; i < n; i++) {   // W7: numbers among Latin text are Latin
        if (t[i] == BD_L || t[i] == BD_R) last = t[i];
        else if (t[i] == BD_EN && last == BD_L) t[i] = BD_L;
    }
    for (int i = 0; i < n;) {   // N1, N2: spaces and punctuation take the direction around them
        if (!bd_neutral(t[i])) {
            i++;
            continue;
        }
        int j = i;
        while (j < n && bd_neutral(t[j])) j++;
        int a = i ? (t[i - 1] == BD_L ? BD_L : BD_R) : sos, b = j < n ? (t[j] == BD_L ? BD_L : BD_R) : sos;
        memset(t + i, a == b ? a : sos, j - i);
        i = j;
    }
    for (int i = 0; i < n; i++)   // I1, I2
        lv[i] = rtl ? (t[i] == BD_R ? 1 : 2) : (t[i] == BD_L ? 0 : t[i] == BD_R ? 1 : 2);
    for (int i = n - 1; i >= 0 && (orig[i] == BD_WS || orig[i] == BD_S || orig[i] == BD_B || orig[i] == BD_BN); i--)
        lv[i] = (uint8_t)rtl;   // L1
}

// A line split into pieces of one direction, in the order they are seen (left to right). The line is a list of
// spans: text [s, e) or, with s == NULL, a gap of w pixels (the space between two layout runs).
typedef struct {
    const char* s, * e;
    int w;
} BSpan;
typedef struct {
    const char* s, * e;
    int span, rtl;
} BSeg;

#define BIDI_MAX 1024

static int bidi_split(const BSpan* sp, int n_sp, int rtl, BSeg* out, int max) {
    static uint8_t t[BIDI_MAX], lv[BIDI_MAX];
    static const char* at[BIDI_MAX + 1];
    static int16_t of[BIDI_MAX];
    int n = 0;
    for (int k = 0; k < n_sp; k++) {
        if (!sp[k].s) {
            if (n == BIDI_MAX) return -1;
            at[n] = NULL, of[n] = (int16_t)k, t[n++] = BD_WS;
            continue;
        }
        for (const char* p = sp[k].s; p < sp[k].e && *p;) {
            if (n == BIDI_MAX) return -1;
            at[n] = p, of[n] = (int16_t)k;
            t[n++] = (uint8_t)bidi_class(utf8_next(&p));
        }
    }
    bidi_levels(t, n, rtl, lv);
    // runs of one span and one level, in reading order
    static BSeg seg[BIDI_MAX];
    static uint8_t seg_lv[BIDI_MAX];
    int ns = 0;
    for (int i = 0; i < n;) {
        int j = i + 1;
        while (j < n && of[j] == of[i] && lv[j] == lv[i] && at[i]) j++;
        BSeg* g = &seg[ns];
        g->span = of[i], g->rtl = lv[i] & 1;
        g->s = at[i];
        g->e = !at[i] ? NULL : j < n && of[j] == of[i] ? at[j] : sp[of[i]].e;
        seg_lv[ns++] = lv[i];
        i = j;
    }
    if (ns > max) return -1;
    // L2: from the highest level down to the lowest odd one, reverse every run at that level or higher
    int order[BIDI_MAX], hi = 0, lo_odd = 255;
    for (int i = 0; i < ns; i++) {
        order[i] = i;
        if (seg_lv[i] > hi) hi = seg_lv[i];
        if ((seg_lv[i] & 1) && seg_lv[i] < lo_odd) lo_odd = seg_lv[i];
    }
    for (int l = hi; l >= lo_odd && l > 0; l--)
        for (int i = 0; i < ns;) {
            if (seg_lv[order[i]] < l) {
                i++;
                continue;
            }
            int j = i;
            while (j < ns && seg_lv[order[j]] >= l) j++;
            for (int a = i, b = j - 1; a < b; a++, b--) {
                int x = order[a];
                order[a] = order[b], order[b] = x;
            }
            i = j;
        }
    for (int i = 0; i < ns; i++) out[i] = seg[order[i]];
    return ns;
}

// One line of mixed-direction text from (x, y): drawn (measure 0) or only measured. Returns the x after it.
static int bidi_draw(const Font* f, int x, int y, const char* s, const char* e, uint32_t rgb, int measure) {
    static BSeg seg[BIDI_MAX];
    BSpan sp = {s, e, 0};
    int n = bidi_split(&sp, 1, para_rtl(s, e), seg, BIDI_MAX);
    if (n < 0) return measure ? x + text_width(f, s, e) : draw_ltr(f, x, y, s, e, rgb);
    for (int i = 0; i < n; i++) {
        int w = text_width(f, seg[i].s, seg[i].e);
        if (!measure) {
            if (seg[i].rtl) draw_rtl(f, x + w, y, seg[i].s, seg[i].e, rgb);
            else draw_ltr(f, x, y, seg[i].s, seg[i].e, rgb);
        }
        x += w;
    }
    return x;
}

// ---- images (PIMG from pc/make_brand.py) ----------------------------------------------------------------------

typedef struct {
    int w, h;
    const uint8_t* px;
} Image;
static Image logo_s, logo_l;

static int images_load(const char* path) {
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < 8 || memcmp(d, "PIMG", 4)) return -1;
    uint32_t n;
    memcpy(&n, d + 4, 4);
    Image* out[2] = {&logo_s, &logo_l};
    for (uint32_t i = 0; i < n && i < 2; i++) {
        uint16_t w, h;
        uint32_t off;
        memcpy(&w, d + 8 + i * 8, 2);
        memcpy(&h, d + 10 + i * 8, 2);
        memcpy(&off, d + 12 + i * 8, 4);
        out[i]->w = w, out[i]->h = h, out[i]->px = d + off;
    }
    return 0;
}

static void draw_image(const Image* im, int x, int y, int alpha) {
    if (!im->px) return;
    alpha = alpha * text_alpha / 255;
    for (int j = 0; j < im->h; j++) {
        int py = y + j;
        if (py < clip_y0 || py >= clip_y1) continue;
        for (int i = 0; i < im->w; i++) {
            int px = x + i;
            if (px < clip_x0 || px >= clip_x1) continue;
            const uint8_t* c = im->px + (j * im->w + i) * 4;
            if (c[3]) blend(back + py * SCREEN_W + px, (uint32_t)c[0] << 16 | c[1] << 8 | c[2], c[3] * alpha / 255);
        }
    }
}

int ui_init(const char* dir) {
    static const struct {
        Font* f;
        const char* name;
    } files[] = {{&font, "font.bin"},           {&font_bold, "font_bold.bin"}, {&font_italic, "font_italic.bin"},
                 {&font_small, "font_small.bin"}, {&font_big, "font_big.bin"},  {&mono, "mono.bin"}};
    char path[256];
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, files[i].name);
        if (font_load(files[i].f, path)) {
            ui_error = files[i].name;
            return -1;
        }
    }
    font_bold.next = font_italic.next = font_small.next = font_big.next = mono.next = &font;
    font_head = font_big;   // the headings' copy: without the interface's own characters (UI_VAR, at the end)
    while (font_head.n && font_head.cps[font_head.n - 1] >= UI_VAR) font_head.n--;
    font_head.has_ui = 0;
    font_bold.embolden = font_head.embolden = 1;
    snprintf(path, sizeof(path), "%s/logo.bin", dir);
    if (images_load(path)) say("logo.bin missing");
    make_background();
    return 0;
}

// ---- messages --------------------------------------------------------------------------------------------------

Msg msgs[MAX_MSGS];
int n_msgs;
static void layout_free(Layout* L);

void msg_append(Msg* m, const char* s, size_t n) {
    if (m->len + n + 1 > m->cap) {
        m->cap = (m->len + n + 1) * 2;
        m->text = realloc(m->text, m->cap);
    }
    memcpy(m->text + m->len, s, n);
    m->len += n;
    m->text[m->len] = 0;
}

Msg* msg_new(int who) {
    if (n_msgs == MAX_MSGS) {
        free(msgs[0].text);
        layout_free(msgs[0].lay);
        memmove(msgs, msgs + 1, sizeof(Msg) * (MAX_MSGS - 1));
        n_msgs--;
    }
    Msg* m = &msgs[n_msgs++];
    memset(m, 0, sizeof(*m));
    m->who = who;
    msg_append(m, "", 0);
    return m;
}

void msgs_clear(void) {
    for (int i = 0; i < n_msgs; i++) free(msgs[i].text), layout_free(msgs[i].lay);
    n_msgs = 0;
}

// ---- markdown tables -------------------------------------------------------------------------------------------
// Complete table lines ("| a | b |") become grid rows:
//   '\x01' + kind (T top, R row, M separator, B bottom) + column widths in pixels ("120,80,") + ';' + cells
// separated by '\x02'.

#define MAX_COLS 12
#define CELL_PAD 18

static int split_cells(const char* a, const char* b, const char** cs, const char** ce) {
    while (a < b && *a == ' ') a++;
    if (a < b && *a == '|') a++;
    while (b > a && (b[-1] == ' ' || b[-1] == '\r')) b--;
    if (b > a && b[-1] == '|') b--;
    int n = 0;
    while (a <= b && n < MAX_COLS) {
        const char* p = a;
        while (p < b && *p != '|') p++;
        const char* s = a, * e = p;
        while (s < e && *s == ' ') s++;
        while (e > s && e[-1] == ' ') e--;
        cs[n] = s, ce[n] = e, n++;
        if (p >= b) break;
        a = p + 1;
    }
    return n;
}

static int is_separator(const char** cs, const char** ce, int n) {
    for (int i = 0; i < n; i++) {
        if (cs[i] == ce[i]) return 0;
        for (const char* p = cs[i]; p < ce[i]; p++)
            if (*p != '-' && *p != ':' && *p != ' ') return 0;
    }
    return n > 0;
}

static int clean_cell(const char* s, const char* e, char* out, int max) {   // drops ** and ` markers
    int n = 0;
    while (s < e && n < max - 1) {
        if (s + 1 < e && s[0] == '*' && s[1] == '*') {
            s += 2;
            continue;
        }
        if (*s == '`') {
            s++;
            continue;
        }
        out[n++] = *s++;
    }
    out[n] = 0;
    return n;
}

static void put_row(char** o, char kind, const int* widths, int ncols) {
    *(*o)++ = '\x01';
    *(*o)++ = kind;
    for (int i = 0; i < ncols; i++) *o += sprintf(*o, "%d,", widths[i]);
    *(*o)++ = ';';
}

static char* format_tables(const char* text) {
    size_t len = strlen(text);
    char* out = malloc(len * 2 + 4096);
    char* o = out;
    const char* s = text;
    char cell[512];
    while (*s) {
        const char* nl = strchr(s, '\n');
        const char* t = s;
        while (*t == ' ') t++;
        if (!nl || *t != '|') {
            const char* e = nl ? nl + 1 : s + strlen(s);
            memcpy(o, s, e - s);
            o += e - s;
            s = e;
            continue;
        }
        int rows = 0, widths[MAX_COLS] = {0}, ncols = 0;
        const char* cs[MAX_COLS], * ce[MAX_COLS];
        const char* p = s;
        while (*p) {
            const char* q = p;
            while (*q == ' ') q++;
            const char* e = strchr(p, '\n');
            if (*q != '|' || !e) break;
            int n = split_cells(p, e, cs, ce);
            if (!is_separator(cs, ce, n))
                for (int i = 0; i < n; i++) {
                    int cl = clean_cell(cs[i], ce[i], cell, sizeof(cell));
                    int w = text_width(&font_bold, cell, cell + cl);
                    if (w > widths[i]) widths[i] = w;
                }
            if (n > ncols) ncols = n;
            rows++;
            p = e + 1;
        }
        if (rows < 2) {
            memcpy(o, s, nl + 1 - s);
            o += nl + 1 - s;
            s = nl + 1;
            continue;
        }
        put_row(&o, 'T', widths, ncols);
        *o++ = '\n';
        for (const char* r = s; r < p;) {
            const char* e = strchr(r, '\n');
            int n = split_cells(r, e, cs, ce);
            if (is_separator(cs, ce, n)) {
                put_row(&o, 'M', widths, ncols);
            } else {
                put_row(&o, 'R', widths, ncols);
                for (int i = 0; i < n && i < ncols; i++) {
                    if (i) *o++ = '\x02';
                    o += clean_cell(cs[i], ce[i], o, (int)(ce[i] - cs[i]) + 1);
                }
            }
            *o++ = '\n';
            r = e + 1;
        }
        put_row(&o, 'B', widths, ncols);
        *o++ = '\n';
        s = p;
    }
    *o = 0;
    return out;
}

// One row (or rule) of a table from x; a right-to-left table (rtl) starts at x_right with its first column and
// aligns its cells to the right.
static void draw_table_line(const char* a, const char* b, int x, int top, int h, int header, int rtl, int x_right) {
    char kind = a[1];
    int widths[MAX_COLS], n = 0;
    const char* p = a + 2;
    while (p < b && *p != ';' && n < MAX_COLS) {
        widths[n++] = atoi(p);
        while (p < b && *p != ',') p++;
        p++;
    }
    p++;
    int xs[MAX_COLS + 1];   // column edges, left to right on screen
    xs[0] = 0;
    for (int i = 0; i < n; i++) xs[i + 1] = xs[i] + widths[i] + 2 * CELL_PAD;
    int left = rtl ? x_right - xs[n] - 2 : x;
    if (kind != 'R') {
        fill(left, top, xs[n] + 2, 2, kind == 'M' ? C_ACCENT : C_LINE_HI);
        return;
    }
    if (header) fill(left, top, xs[n], h, 0x12274C);
    for (int i = 0; i <= n; i++) fill(left + xs[i], top, 2, h, C_LINE_HI);
    const Font* f = header ? &font_bold : &font;
    for (int i = 0; i < n && p <= b; i++) {
        const char* e = p;
        while (e < b && *e != '\x02') e++;
        int c = rtl ? n - 1 - i : i, tx = left + xs[c] + CELL_PAD;
        if (rtl) tx += widths[c] - text_width(f, p, e);
        draw_text(f, tx, top + (h - f->line) / 2, p, e, header ? 0xFFFFFF : C_TEXT);
        p = e + 1;
    }
}

// ---- text layout: markdown blocks and inline styles, word wrap ----------------------------------------------

enum { ST_BOLD = 1, ST_ITALIC = 2, ST_CODE = 4, ST_STRIKE = 8, ST_LINK = 16 };
enum { DECO_NONE, DECO_CODE_TOP, DECO_CODE, DECO_CODE_BOT, DECO_TABLE, DECO_HR };

typedef struct {
    uint32_t text, strong, italic, code, code_bg, link, head, quote, marker;
} Palette;
static const Palette PAL_AI = {C_TEXT, 0xFFFFFF, 0xD2E2F8, 0x9CD6FF, 0x10264D, 0x63BBFF, 0xFFFFFF, 0xA8BAD6, C_ACCENT_HI};
static const Palette PAL_USER = {0xFFFFFF, 0xFFFFFF, 0xE6F0FF, 0xFFFFFF, 0x0A4AA6, 0xDDEEFF, 0xFFFFFF, 0xDCE9FF, 0xCFE6FF};
static const Palette PAL_NOTE = {C_TEXT_MUTED, C_TEXT_DIM, C_TEXT_MUTED, C_TEXT_DIM, C_SURFACE, C_TEXT_DIM, C_TEXT_DIM,
                                 C_TEXT_MUTED, C_TEXT_MUTED};
static const Palette PAL_FIELD = {C_TEXT, C_TEXT, C_TEXT, C_TEXT, 0, C_TEXT, C_TEXT, C_TEXT, C_TEXT};

typedef struct {
    const Font* f;
    uint32_t color;
    int x, w;
    const char* s, * e;
    uint8_t style;
    uint8_t rtl;               // right-to-left text, drawn from x + w leftwards
} Run;

typedef struct {
    int y, h, base;            // top, height, baseline offset
    uint8_t deco, quote, header;
    uint8_t rtl;               // a right-to-left paragraph: against the right edge, quote bars on that side
    int run0, nrun;
    const char* a, * b;        // table row text
} Line;

struct Layout {
    char* src;                 // copy of the source, to see whether it changed
    size_t src_len;
    char* text;                // source with the tables converted; runs point into it
    int width, plain;
    const Palette* pal;
    Line* lines;
    int n_lines, cap_lines;
    Run* runs;
    int n_runs, cap_runs;
    int height, max_w;
    int line_start;            // first run of the line being built
    int para_rtl;              // the paragraph being laid out reads right to left
};

#define TEXT_LINE_H (font.line + 7)
#define QUOTE_INDENT 30
#define LIST_INDENT 38

static void layout_free(Layout* L) {
    if (!L) return;
    free(L->src), free(L->text), free(L->lines), free(L->runs), free(L);
}

static void add_run(Layout* L, const Font* f, uint32_t color, int x, int w, const char* s, const char* e, int style) {
    if (L->n_runs > L->line_start) {   // merge with the previous run when it is the same style and text follows on
        Run* r = &L->runs[L->n_runs - 1];
        if (r->f == f && r->color == color && r->style == style && !(style & ST_CODE) &&
            (r->e == s || (r->e + 1 == s && *r->e == ' ' && x == r->x + r->w + f->space))) {
            r->e = e;
            r->w = x + w - r->x;
            return;
        }
    }
    if (L->n_runs == L->cap_runs) {
        L->cap_runs = L->cap_runs ? L->cap_runs * 2 : 64;
        L->runs = realloc(L->runs, L->cap_runs * sizeof(Run));
    }
    Run* r = &L->runs[L->n_runs++];
    r->f = f, r->color = color, r->x = x, r->w = w, r->s = s, r->e = e, r->style = (uint8_t)style, r->rtl = 0;
}

// A line with right-to-left text in reading order: its runs are cut where the direction changes and placed left to
// right as they are seen (bidi_split). A right-to-left paragraph lies against the right edge, the indent of lists
// and quotes on that side.
static void layout_bidi(Layout* L, Line* ln) {
    int n = ln->nrun, any = ln->rtl;
    for (int k = 0; k < n && !any; k++) any = has_rtl(L->runs[ln->run0 + k].s, L->runs[ln->run0 + k].e);
    if (!any || !n || n > 256) return;
    static Run old[256];
    static BSpan sp[512];
    static BSeg seg[BIDI_MAX];
    static int run_of[512], wseg[BIDI_MAX];
    memcpy(old, L->runs + ln->run0, n * sizeof(Run));
    int ns = 0;
    for (int k = 0; k < n; k++) {
        int gap = k ? old[k].x - (old[k - 1].x + old[k - 1].w) : 0;
        if (gap > 0) run_of[ns] = -1, sp[ns++] = (BSpan){NULL, NULL, gap};
        run_of[ns] = k, sp[ns++] = (BSpan){old[k].s, old[k].e, 0};
    }
    int m = bidi_split(sp, ns, ln->rtl, seg, BIDI_MAX), total = 0;
    if (m < 0) return;
    for (int i = 0; i < m; i++) {
        wseg[i] = seg[i].s ? text_width(old[run_of[seg[i].span]].f, seg[i].s, seg[i].e) : sp[seg[i].span].w;
        total += wseg[i];
    }
    if (ln->run0 + m > L->cap_runs) {
        L->cap_runs = ln->run0 + m + 64;
        L->runs = realloc(L->runs, L->cap_runs * sizeof(Run));
    }
    int x = ln->rtl ? L->width - old[0].x - total : old[0].x;
    L->n_runs = ln->run0;
    for (int i = 0; i < m; i++) {
        if (seg[i].s) {
            Run* o = &L->runs[L->n_runs++];
            *o = old[run_of[seg[i].span]];
            o->x = x, o->w = wseg[i], o->s = seg[i].s, o->e = seg[i].e, o->rtl = (uint8_t)seg[i].rtl;
        }
        x += wseg[i];
    }
    ln->nrun = L->n_runs - ln->run0;
}

static Line* end_line(Layout* L, int h, int base, int deco, int quote) {
    if (L->n_lines == L->cap_lines) {
        L->cap_lines = L->cap_lines ? L->cap_lines * 2 : 32;
        L->lines = realloc(L->lines, L->cap_lines * sizeof(Line));
    }
    Line* ln = &L->lines[L->n_lines++];
    memset(ln, 0, sizeof(*ln));
    ln->y = L->height, ln->h = h, ln->base = base, ln->deco = (uint8_t)deco, ln->quote = (uint8_t)quote;
    ln->run0 = L->line_start, ln->nrun = L->n_runs - L->line_start;
    ln->rtl = (uint8_t)(L->para_rtl && deco == DECO_NONE);
    if (deco == DECO_NONE || deco == DECO_CODE) layout_bidi(L, ln);
    for (int i = ln->run0; i < L->n_runs; i++) {   // (a right-to-left line ends at the right edge: its width
        int w = ln->rtl ? L->width - L->runs[i].x : L->runs[i].x + L->runs[i].w;   // counts from its left end)
        if (w > L->max_w) L->max_w = w;
    }
    L->height += h;
    L->line_start = L->n_runs;
    return ln;
}

// space: a space comes before it; brk: a line may end before it without a space (where Chinese or Japanese text
// starts or ends); cjk: Chinese or Japanese text, which may break after any character.
typedef struct {
    const char* s, * e;
    uint8_t style, space, brk, cjk;
} Piece;

#define MAX_PIECES 2048
static Piece pieces[MAX_PIECES];
static int n_pieces, piece_brk, piece_cjk;

static void piece(const char* s, const char* e, int style, int* space) {
    if (e <= s || n_pieces == MAX_PIECES) return;
    pieces[n_pieces].s = s, pieces[n_pieces].e = e, pieces[n_pieces].style = (uint8_t)style;
    pieces[n_pieces].space = (uint8_t)*space;
    pieces[n_pieces].brk = (uint8_t)piece_brk, pieces[n_pieces].cjk = (uint8_t)piece_cjk;
    n_pieces++;
    *space = 0, piece_brk = 0;
}

// Chinese and Japanese characters (ideographs, kana, their punctuation, full-width forms): text without spaces,
// where a line may end after any character. (Korean puts spaces between words.)
static int cjk_break(uint32_t c) {
    return (c >= 0x2E80 && c <= 0x312F) || (c >= 0x31F0 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) ||
           (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF00 && c <= 0xFFEF) || (c >= 0x20000 && c <= 0x3FFFF);
}

// Closing punctuation, small kana and the prolonged sound mark do not start a line: they stay at the end of the one
// before, even a little past its edge.
static int no_line_start(uint32_t c) {
    static const uint16_t set[] = {')', ']', '}', '.', ',', '!', '?', ':', ';', '%', 0x2019, 0x201D, 0x2025, 0x2026,
                                   0x3001, 0x3002, 0x3005, 0x3009, 0x300B, 0x300D, 0x300F, 0x3011, 0x3015, 0x3017,
                                   0x3019, 0x3041, 0x3043, 0x3045, 0x3047, 0x3049, 0x3063, 0x3083, 0x3085, 0x3087,
                                   0x308E, 0x30A1, 0x30A3, 0x30A5, 0x30A7, 0x30A9, 0x30C3, 0x30E3, 0x30E5, 0x30E7,
                                   0x30EE, 0x30F5, 0x30F6, 0x30FB, 0x30FC, 0xFF01, 0xFF09, 0xFF0C, 0xFF0E, 0xFF1A,
                                   0xFF1B, 0xFF1F, 0xFF3D, 0xFF5D};
    for (size_t i = 0; i < sizeof(set) / sizeof(set[0]); i++)
        if (set[i] == c) return 1;
    return 0;
}

static int is_alnum(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (uint8_t)c >= 0x80; }

// Is there a closing single marker c in p..b (preceded by a non-space, not part of a double marker)?
static int has_closer(const char* p, const char* b, char c) {
    for (const char* q = p + 1; q < b; q++)
        if (*q == c && q[-1] != ' ' && q[-1] != c && (q + 1 >= b || q[1] != c)) {
            if (c == '_' && q + 1 < b && is_alnum(q[1])) continue;
            return 1;
        }
    return 0;
}

// Splits a[..b) into pieces at spaces and style markers (markers are dropped).
static void parse_inline(const char* a, const char* b, int plain) {
    n_pieces = 0, piece_brk = 0, piece_cjk = 0;
    int style = 0, space = 0;
    const char* cur = a;
    const char* p = a;
    while (p < b) {
        char c = *p;
        if (((uint8_t)c & 0xC0) != 0x80) {   // a character starts: into or out of Chinese / Japanese text?
            int k = 0;
            if ((uint8_t)c >= 0xE2) {
                const char* q = p;
                k = cjk_break(utf8_next(&q));
            }
            if (k != piece_cjk) {
                piece(cur, p, style, &space);
                cur = p, piece_cjk = k, piece_brk = 1;
            }
        }
        if (c == ' ') {
            piece(cur, p, style, &space);
            while (p < b && *p == ' ') p++;
            cur = p;
            space = 1;
            continue;
        }
        if (plain) {
            p++;
            continue;
        }
        if (style & ST_CODE) {
            if (c == '`') {
                piece(cur, p, style, &space);
                style &= ~ST_CODE;
                cur = ++p;
            } else p++;
            continue;
        }
        if (c == '\\' && p + 1 < b && (p[1] == '*' || p[1] == '_' || p[1] == '`' || p[1] == '#' || p[1] == '~' || p[1] == '|')) {
            piece(cur, p, style, &space);
            cur = p + 1;
            p += 2;
            continue;
        }
        if (c == '`' && memchr(p + 1, '`', b - p - 1)) {
            piece(cur, p, style, &space);
            style |= ST_CODE;
            cur = ++p;
            continue;
        }
        if (c == '~' && p + 1 < b && p[1] == '~') {
            piece(cur, p, style, &space);
            style ^= ST_STRIKE;
            p += 2;
            cur = p;
            continue;
        }
        if (c == '*' || c == '_') {
            int n = 1;
            while (p + n < b && p[n] == c && n < 3) n++;
            char left = p > a ? p[-1] : ' ', right = p + n < b ? p[n] : ' ';
            int can_open = right != ' ', can_close = left != ' ';
            if (c == '_') can_open = can_open && !is_alnum(left), can_close = can_close && !(right != ' ' && is_alnum(right));
            int toggled = 0, ns = style;
            if (n >= 2) {
                if ((style & ST_BOLD) && can_close) ns &= ~ST_BOLD, toggled = 1;
                else if (!(style & ST_BOLD) && can_open) ns |= ST_BOLD, toggled = 1;
            }
            if (n == 1 || n == 3) {
                if ((style & ST_ITALIC) && can_close) ns &= ~ST_ITALIC, toggled = 1;
                else if (!(style & ST_ITALIC) && can_open && has_closer(p + n - 1, b, c)) ns |= ST_ITALIC, toggled = 1;
                else if (n == 3 && toggled) n = 2;   // *** that only opens bold: keep one * as text
            }
            if (!toggled && p + n == b && !can_close) {   // a marker at the end of the text still being written
                piece(cur, p, style, &space);
                p += n;
                cur = p;
                continue;
            }
            if (toggled) {
                piece(cur, p, style, &space);
                style = ns;
                p += n;
                cur = p;
                continue;
            }
            p += n;
            continue;
        }
        if (c == '[') {   // [text](link): show the text as a link
            const char* q = memchr(p, ']', b - p);
            if (q && q + 1 < b && q[1] == '(') {
                const char* r = memchr(q, ')', b - q);
                if (r) {
                    piece(cur, p, style, &space);
                    const char* t = p + 1;
                    while (t < q) {
                        const char* w = t;
                        while (w < q && *w != ' ') w++;
                        piece(t, w, style | ST_LINK, &space);
                        while (w < q && *w == ' ') w++, space = 1;
                        t = w;
                    }
                    p = r + 1;
                    cur = p;
                    continue;
                }
            }
        }
        p++;
    }
    piece(cur, b, style, &space);
}

static const Font* style_font(int st, int heading) {
    if (st & ST_CODE) return &mono;
    if (heading) return heading <= 2 ? &font_head : &font_bold;
    if (st & ST_BOLD) return &font_bold;
    if (st & ST_ITALIC) return &font_italic;
    return &font;
}

static uint32_t style_color(const Palette* pal, int st, int heading, int quote) {
    if (st & ST_CODE) return pal->code;
    if (st & ST_LINK) return pal->link;
    if (heading) return pal->head;
    if (st & ST_BOLD) return pal->strong;
    if (st & ST_ITALIC) return pal->italic;
    return quote ? pal->quote : pal->text;
}

// Places the pieces in lines from x_first (first line) / x_cont (the others) to `right`. An optional marker
// (list bullet or number) starts the first line.
static void flow(Layout* L, int x_first, int x_cont, int right, int heading, int quote, const Run* marker) {
    const Font* base_font = heading ? style_font(0, heading) : &font;
    int lh = heading ? base_font->line + (heading <= 2 ? 14 : 10) : TEXT_LINE_H;
    int base = (lh - base_font->line) / 2 + base_font->ascent;
    int x = x_first, has = 0;
    if (marker) add_run(L, marker->f, marker->color, marker->x, marker->w, marker->s, marker->e, marker->style);
    static int pw[MAX_PIECES];
    for (int i = 0; i < n_pieces; i++) pw[i] = text_width(style_font(pieces[i].style, heading), pieces[i].s, pieces[i].e);
    for (int i = 0; i < n_pieces;) {
        int j = i + 1;
        while (j < n_pieces && !pieces[j].space && !pieces[j].brk) j++;
        int ww = 0;
        for (int k = i; k < j; k++) ww += pw[k];
        int sp = has && pieces[i].space ? base_font->space : 0;
        if (pieces[i].cjk) {   // Chinese / Japanese: the line may end after any character
            x += sp;
            for (int k = i; k < j; k++) {
                const Font* f = style_font(pieces[k].style, heading);
                uint32_t col = style_color(L->pal, pieces[k].style, heading, quote);
                for (const char* c = pieces[k].s; c < pieces[k].e;) {
                    const char* nx = c;
                    uint32_t cp = utf8_next(&nx);
                    int cw = text_width(f, c, nx);
                    if (has && x + cw > right && !no_line_start(cp)) {
                        end_line(L, lh, base, DECO_NONE, quote);
                        x = x_cont, has = 0;
                    }
                    add_run(L, f, col, x, cw, c, nx, pieces[k].style);
                    x += cw, has = 1, c = nx;
                }
            }
            i = j;
            continue;
        }
        int glued = has && !pieces[i].space && pieces[i].brk;   // right after Chinese / Japanese text
        const char* q = pieces[i].s;
        if (has && x + sp + ww > right && !(glued && no_line_start(utf8_next(&q)))) {
            end_line(L, lh, base, DECO_NONE, quote);
            x = x_cont, has = 0, sp = 0;
        }
        x += sp;
        for (int k = i; k < j; k++) {
            const Font* f = style_font(pieces[k].style, heading);
            uint32_t col = style_color(L->pal, pieces[k].style, heading, quote);
            if (ww <= right - x_cont || x + pw[k] <= right) {
                add_run(L, f, col, x, pw[k], pieces[k].s, pieces[k].e, pieces[k].style);
                x += pw[k];
                has = 1;
                continue;
            }
            for (const char* c = pieces[k].s; c < pieces[k].e;) {   // a word longer than the line: cut it
                const char* nx = c;
                utf8_next(&nx);
                int cw = text_width(f, c, nx);
                if (has && x + cw > right) {
                    end_line(L, lh, base, DECO_NONE, quote);
                    x = x_cont, has = 0;
                }
                add_run(L, f, col, x, cw, c, nx, pieces[k].style);
                x += cw, has = 1, c = nx;
            }
        }
        i = j;
    }
    if (has || marker || !n_pieces) end_line(L, lh, base, DECO_NONE, quote);
}

static int is_fence(const char* t, const char* e) { return e - t >= 3 && !strncmp(t, "```", 3); }

static int is_rule(const char* t, const char* e) {
    while (e > t && (e[-1] == ' ' || e[-1] == '\r')) e--;
    if (e - t < 3 || (*t != '-' && *t != '*' && *t != '_')) return 0;
    int n = 0;
    for (const char* p = t; p < e; p++) {
        if (*p == *t) n++;
        else if (*p != ' ') return 0;
    }
    return n >= 3;
}

static void layout_build(Layout* L, const char* src, size_t len, int width, const Palette* pal, int plain) {
    free(L->src), free(L->text);
    L->src = malloc(len + 1);
    memcpy(L->src, src, len + 1);
    L->src_len = len, L->width = width, L->pal = pal, L->plain = plain;
    L->text = plain ? strdup(src) : format_tables(src);
    L->n_lines = L->n_runs = L->height = L->max_w = L->line_start = 0;
    int in_code = 0, blank = 0, header = 0, table_rtl = 0;
    const int code_pad = 24;
    for (const char* s = L->text; *s;) {
        const char* nl = strchr(s, '\n');
        const char* e = nl ? nl : s + strlen(s);
        const char* t = s;
        int spaces = 0;
        while (t < e && (*t == ' ' || *t == '\t')) spaces += *t == '\t' ? 4 : 1, t++;
        L->para_rtl = 0;
        if (plain) {
            L->para_rtl = para_rtl(s, e);
            parse_inline(s, e, 1);
            flow(L, 0, 0, width, 0, 0, NULL);
        } else if (is_fence(t, e)) {
            if (!in_code) {
                const char* lang = t + 3;
                const char* le = e;
                while (le > lang && (le[-1] == ' ' || le[-1] == '\r')) le--;
                if (le > lang) add_run(L, &font_small, C_TEXT_MUTED, code_pad, text_width(&font_small, lang, le), lang, le, 0);
                end_line(L, le > lang ? font_small.line + 14 : 16, 10 + font_small.ascent, DECO_CODE_TOP, 0);
            } else {
                end_line(L, 16, 0, DECO_CODE_BOT, 0);
            }
            in_code = !in_code;
            blank = 0;
        } else if (in_code) {   // code keeps its spaces; long lines are cut at the panel's edge
            const char* c = s;
            int x = code_pad, lh = mono.line + 4;
            const char* run_s = c;
            int run_x = x;
            while (c < e) {
                const char* nx = c;
                utf8_next(&nx);
                int cw = text_width(&mono, c, nx);
                if (x + cw > width - code_pad && c > run_s) {
                    add_run(L, &mono, 0xC7E2FF, run_x, x - run_x, run_s, c, 0);
                    end_line(L, lh, 2 + mono.ascent, DECO_CODE, 0);
                    run_s = c, x = run_x = code_pad + 2 * mono.space;
                }
                x += cw, c = nx;
            }
            if (e > run_s) add_run(L, &mono, 0xC7E2FF, run_x, x - run_x, run_s, e, 0);
            end_line(L, lh, 2 + mono.ascent, DECO_CODE, 0);
        } else if (*s == '\x01') {   // table row
            if (s[1] == 'T') {   // a new table: right to left if most of its text is
                header = 1;
                const char* end = strstr(s, "\x01" "B");
                table_rtl = para_rtl(s, end ? end : s + strlen(s));
            }
            if (s[1] == 'M') header = 0;
            Line* ln = end_line(L, s[1] == 'R' ? font.line + 16 : 2, 0, DECO_TABLE, 0);
            ln->a = s, ln->b = e, ln->header = (uint8_t)(header && s[1] == 'R'), ln->rtl = (uint8_t)table_rtl;
            if (s[1] == 'B') L->height += 10;
            blank = 0;
        } else if (t == e || (e - t == 1 && *t == '\r')) {   // blank line: paragraph gap
            if (!blank && L->n_lines) L->height += 14;
            blank = 1;
        } else if (is_rule(t, e)) {
            end_line(L, 28, 0, DECO_HR, 0);
            blank = 0;
        } else {
            blank = 0;
            int quote = 0;
            while (t < e && *t == '>') {
                quote++, t++;
                while (t < e && *t == ' ') t++;
            }
            int heading = 0;
            if (*t == '#') {
                const char* h = t;
                while (h < e && *h == '#' && heading < 6) h++, heading++;
                if (h < e && *h == ' ') {
                    t = h + 1;
                    if (L->n_lines) L->height += 10;
                } else heading = 0;
            }
            int x0 = quote * QUOTE_INDENT;
            Run marker = {0};
            int has_marker = 0;
            if (!heading && t + 1 < e && (*t == '-' || *t == '*' || *t == '+') && t[1] == ' ') {
                int level = (spaces + 1) / 3 > 3 ? 3 : (spaces + 1) / 3;   // 2-4 spaces: second level
                static const char* dots[] = {"•", "◦", "▪", "▪"};
                const char* d = dots[level];
                marker.f = &font, marker.color = L->pal->marker, marker.s = d, marker.e = d + strlen(d);
                marker.x = x0 + level * LIST_INDENT + 6, marker.w = text_width(&font, marker.s, marker.e);
                x0 += level * LIST_INDENT + LIST_INDENT;
                t += 2;
                has_marker = 1;
            } else if (!heading && *t >= '0' && *t <= '9') {
                const char* d = t;
                while (d < e && *d >= '0' && *d <= '9' && d - t < 4) d++;
                if (d + 1 < e && (*d == '.' || *d == ')') && d[1] == ' ') {
                    int level = (spaces + 1) / 3 > 3 ? 3 : (spaces + 1) / 3;   // 2-4 spaces: second level
                    marker.f = &font_bold, marker.color = L->pal->marker, marker.s = t, marker.e = d + 1;
                    marker.w = text_width(&font_bold, t, d + 1);
                    int col = x0 + level * LIST_INDENT;
                    int tx = col + (marker.w + 12 > LIST_INDENT ? marker.w + 12 : LIST_INDENT);
                    marker.x = tx - 10 - marker.w;
                    x0 = tx;
                    t = d + 2;
                    has_marker = 1;
                }
            }
            L->para_rtl = para_rtl(t, e);
            parse_inline(t, e, 0);
            flow(L, x0, x0, width, heading, quote, has_marker ? &marker : NULL);
        }
        s = nl ? nl + 1 : e;
    }
    if (in_code) end_line(L, 16, 0, DECO_CODE_BOT, 0);   // a code block still being written
}

// The layout of text at this width, rebuilt only when something changed.
static Layout* layout_get(Layout** pl, const char* text, size_t len, int width, const Palette* pal, int plain) {
    Layout* L = *pl;
    if (!L) L = *pl = calloc(1, sizeof(Layout));
    if (!L->src || L->src_len != len || L->width != width || L->pal != pal || L->plain != plain || memcmp(L->src, text, len))
        layout_build(L, text, len, width, pal, plain);
    return L;
}

// Draws the layout with its top-left at (ox, oy); right-to-left lines end at the right edge of a box box_w wide
// (L->width: the whole width; L->max_w: as wide as the text, as in a speech bubble).
static void draw_layout_w(const Layout* L, int ox, int oy, int box_w) {
    for (int i = 0; i < L->n_lines; i++) {
        const Line* ln = &L->lines[i];
        int ly = oy + ln->y;
        if (ly > clip_y1) break;
        if (ln->deco == DECO_CODE_TOP) {   // the whole code block's panel, even if its top is scrolled away
            int j = i;
            while (j + 1 < L->n_lines && L->lines[j].deco != DECO_CODE_BOT) j++;
            int bottom = oy + L->lines[j].y + L->lines[j].h;
            rrect(ox, ly, L->width, bottom - ly, 14, 0x061127, 255);
            rrect_line(ox, ly, L->width, bottom - ly, 14, 1, C_LINE, 255);
        }
        if (ly + ln->h < clip_y0) continue;
        if (ln->deco == DECO_HR) fill(ox, ly + ln->h / 2, L->width, 2, C_LINE_HI);
        if (ln->deco == DECO_TABLE) draw_table_line(ln->a, ln->b, ox, ly, ln->h, ln->header, ln->rtl, ox + box_w);
        int lx = ln->rtl ? ox + box_w - L->width : ox;
        for (int q = 0; q < ln->quote; q++)
            rrect(ln->rtl ? ox + box_w - q * QUOTE_INDENT - 8 : ox + q * QUOTE_INDENT + 4, ly, 4, ln->h, 2, C_ACCENT, 255);
        for (int k = 0; k < ln->nrun; k++) {
            const Run* r = &L->runs[ln->run0 + k];
            int ty = ly + ln->base - r->f->ascent, rx = lx + r->x;
            if (r->style & ST_CODE) rrect(rx - 5, ly + ln->base - mono.ascent - 4, r->w + 10, mono.line + 6, 8,
                                          L->pal->code_bg, 255);
            if (r->rtl) draw_rtl(r->f, rx + r->w, ty, r->s, r->e, r->color);
            else draw_ltr(r->f, rx, ty, r->s, r->e, r->color);
            if (r->style & ST_STRIKE) fill(rx, ly + ln->base - r->f->ascent / 3, r->w, 2, r->color);
            if (r->style & ST_LINK) fill(rx, ly + ln->base + 3, r->w, 1, r->color);
        }
    }
}
static void draw_layout(const Layout* L, int ox, int oy) { draw_layout_w(L, ox, oy, L->width); }

// Where the text ends (for the caret): the end of the last line (its left end if it reads right to left: *rtl),
// and that line's top and height.
static void layout_end(const Layout* L, int* x, int* y, int* h, int* rtl) {
    *x = 0, *y = 0, *h = TEXT_LINE_H, *rtl = 0;
    if (!L->n_lines) return;
    const Line* ln = &L->lines[L->n_lines - 1];
    *y = ln->y, *h = ln->h, *rtl = ln->rtl;
    if (ln->rtl) *x = L->width;
    for (int k = 0; k < ln->nrun; k++) {
        const Run* r = &L->runs[ln->run0 + k];
        if (ln->rtl ? r->x < *x : r->x + r->w > *x) *x = ln->rtl ? r->x : r->x + r->w;
    }
}

// ---- buttons and footer --------------------------------------------------------------------------------------

// Draws a controller button with its left edge at x, centred on cy; returns its width.
static int draw_btn(int btn, int x, int cy) {
    const int d = 34;
    float cx = x + d * 0.5f;
    if (btn <= BTN_SQUARE || btn == BTN_UPDOWN) {
        circle(cx, cy, d * 0.5f, 0, 0x1A2D52, 255);
        circle(cx, cy, d * 0.5f, 1.2f, 0x34548A, 255);
        switch (btn) {
        case BTN_CROSS:
            line(cx - 6.5f, cy - 6.5f, cx + 6.5f, cy + 6.5f, 3.2f, C_CROSS, 255);
            line(cx - 6.5f, cy + 6.5f, cx + 6.5f, cy - 6.5f, 3.2f, C_CROSS, 255);
            break;
        case BTN_CIRCLE: circle(cx, cy, 8.5f, 3.0f, C_CIRCLE, 255); break;
        case BTN_TRIANGLE:
            line(cx, cy - 8, cx + 8.5f, cy + 6, 2.8f, C_TRIANGLE, 255);
            line(cx + 8.5f, cy + 6, cx - 8.5f, cy + 6, 2.8f, C_TRIANGLE, 255);
            line(cx - 8.5f, cy + 6, cx, cy - 8, 2.8f, C_TRIANGLE, 255);
            break;
        case BTN_SQUARE: rrect_line((int)cx - 8, cy - 8, 16, 16, 2, 3, C_SQUARE, 255); break;
        default:   // up / down
            line(cx - 5, cy - 3, cx, cy - 8, 2.6f, 0xDCE6F5, 255);
            line(cx, cy - 8, cx + 5, cy - 3, 2.6f, 0xDCE6F5, 255);
            line(cx - 5, cy + 3, cx, cy + 8, 2.6f, 0xDCE6F5, 255);
            line(cx, cy + 8, cx + 5, cy + 3, 2.6f, 0xDCE6F5, 255);
        }
        return d;
    }
    static const char* names[] = {[BTN_L1] = "L1", [BTN_R1] = "R1", [BTN_L2] = "L2", [BTN_R2] = "R2", [BTN_OPTIONS] = "OPTIONS",
                                  [BTN_R3] = "R3"};
    const char* s = names[btn];
    int w = str_width(&font_small, s) + 20, h = 30;
    int rad[4] = {8, 8, 8, 8};
    if (btn == BTN_L2 || btn == BTN_R2) rad[0] = rad[1] = 14, rad[2] = rad[3] = 6;
    rrect4(x, cy - h / 2, w, h, rad, 0, 0x1A2D52, 255);
    rrect4(x, cy - h / 2, w, h, rad, 1, 0x34548A, 255);
    draw_str(&font_small, x + 10, cy - font_small.line / 2, s, 0xDCE6F5);
    return w;
}

static int hint_width(const Hint* h) {
    static const int glyph_w[] = {34, 34, 34, 34, 0, 0, 0, 0, 34, 0, 0};
    int g = glyph_w[h->btn];
    if (!g) {
        static const char* names[] = {[BTN_L1] = "L1", [BTN_R1] = "R1", [BTN_L2] = "L2", [BTN_R2] = "R2", [BTN_OPTIONS] = "OPTIONS",
                                      [BTN_R3] = "R3"};
        g = str_width(&font_small, names[h->btn]) + 20;
    }
    return g + 10 + str_width(&font_small, tr(h->label));
}

static int footer_extra;   // (chat screen) room kept on the right for the L2 / R2 buttons drawn after the footer

void draw_footer(const Hint* hints, int n, const char* right) {
    const int y = SCREEN_H - FOOTER_H, cy = y + FOOTER_H / 2;
    memcpy(back + y * SCREEN_W, bg + y * SCREEN_W, FOOTER_H * SCREEN_W * 4);   // a clean footer bar
    int right_w = right ? str_width(&font_small, right) + 40 : 0;
    int avail = SCREEN_W - 2 * 72 - right_w - footer_extra, total = 0;
    for (int i = 0; i < n; i++) total += hint_width(&hints[i]);
    int gap = n > 1 ? (avail - total) / (n - 1) : 0;
    if (gap > 44) gap = 44;
    if (gap < 14) gap = 14;
    int x = 72;
    for (int i = 0; i < n; i++) {
        int w = draw_btn(hints[i].btn, x, cy);
        draw_str(&font_small, x + w + 10, cy - font_small.line / 2, tr(hints[i].label), C_TEXT_DIM);
        x += hint_width(&hints[i]) + gap;
    }
    if (right) draw_str(&font_small, SCREEN_W - 72 - str_width(&font_small, right), cy - font_small.line / 2, right, C_TEXT_MUTED);
}

// ---- header ----------------------------------------------------------------------------------------------------

static void fmt_k(char* out, int n) {   // 1234 -> "1.2k"
    if (n < 1000) sprintf(out, "%d", n);
    else if (n < 9950) sprintf(out, "%d.%dk", (n + 50) / 1000, (n + 50) % 1000 / 100);
    else sprintf(out, "%dk", n / 1000);
}

// A rounded label at the right edge x_right; returns its left edge.
static int chip(int x_right, int cy, const char* text, uint32_t color, uint32_t dot, float dot_alpha, int bar_frac_x1000) {
    int tw = str_width(&font_small, text);
    int w = tw + 36 + (dot ? 20 : 0) + (bar_frac_x1000 >= 0 ? 96 : 0), h = 44, x = x_right - w;
    rrect(x, cy - h / 2, w, h, h / 2, 0x0D1D3B, 230);
    rrect_line(x, cy - h / 2, w, h, h / 2, 1, C_LINE_HI, 255);
    int tx = x + 18;
    if (dot) {
        circle(tx + 5, cy, 5, 0, dot, (int)(255 * dot_alpha));
        tx += 20;
    }
    draw_str(&font_small, tx, cy - font_small.line / 2, text, color);
    if (bar_frac_x1000 >= 0) {
        int bx = tx + tw + 14;
        rrect(bx, cy - 3, 78, 6, 3, 0x1B3156, 255);
        int fw = 78 * bar_frac_x1000 / 1000;
        if (fw > 0) rrect(bx, cy - 3, fw < 6 ? 6 : fw, 6, 3, bar_frac_x1000 > 850 ? 0xF2A65A : C_ACCENT_HI, 255);
    }
    return x;
}

static void draw_header(const ChatView* v) {
    draw_image(&logo_s, 64, (HEADER_H - logo_s.h) / 2 - 2, 255);
    int tx = 64 + (logo_s.w ? logo_s.w + 14 : 0);
    tx = draw_str(&font_big, tx, (HEADER_H - font_big.line) / 2, "Garlic", 0xFFFFFF);
    draw_str(&font_big, tx + font_big.space, (HEADER_H - font_big.line) / 2, "LLM", C_ACCENT_HI);
    int cy = HEADER_H / 2, x = SCREEN_W - 64;
    char a[24], b[24], ctx[96];
    fmt_k(a, v->pos), fmt_k(b, v->max_seq);
    snprintf(ctx, sizeof(ctx), "%s  %s / %s", tr(S_CONTEXT), a, b);
    int frac = v->max_seq ? (int)(1000.0 * v->pos / v->max_seq) : 0;
    x = chip(x, cy, ctx, C_TEXT_DIM, 0, 0, frac) - 14;
    float pulse = v->generating || v->reading ? 0.55f + 0.45f * sinf((float)v->time * 6.0f) : 1.0f;
    x = chip(x, cy, v->model ? v->model : "", 0xFFFFFF, v->generating || v->reading ? C_ACCENT_HI : C_GREEN, pulse, -1) - 14;
    if (v->think_on) {   // the next question is thought first
        char t[48];
        snprintf(t, sizeof(t), "%s · %d", tr(S_THINK_CHIP), v->think_cap);
        x = chip(x, cy, t, C_ACCENT_HI, C_ACCENT_HI, 1.0f, -1) - 14;
    }
    if (v->api)   // the API is on: where it listens (the dot pulses while it answers a request)
        x = chip(x, cy, v->api, 0xFFFFFF, v->api_busy ? C_ACCENT_HI : C_GREEN, v->api_busy ? pulse : 1.0f, -1) - 14;
    if (v->generating && v->tok_per_s > 0) {
        char s[48];
        snprintf(s, sizeof(s), "%.1f tok/s", v->tok_per_s);
        chip(x, cy, s, C_ACCENT_HI, 0, 0, -1);
    } else if (v->reading) {
        chip(x, cy, tr(S_READING), C_TEXT_DIM, 0, 0, -1);
    }
}

// ---- chat ------------------------------------------------------------------------------------------------------

static int msg_layout_width(int who) { return who == MSG_USER ? USER_MAX_W - 2 * BUB_PAD_X : who == MSG_AI ? COL_W - AI_INDENT : COL_W - 240; }
static const Palette* msg_palette(int who) { return who == MSG_USER ? &PAL_USER : who == MSG_AI ? &PAL_AI : &PAL_NOTE; }

// ---- thinking: the reasoning's last lines while the model thinks; then a short line over the answer ----------------

#define THINK_LINES 4
#define THINK_HEAD_H 50
#define THINK_PAD 16
#define THINK_TAIL 900   // bytes of the reasoning laid out (from the start of a line): its last lines are shown

static int think_panel_h(void) { return THINK_HEAD_H + 2 * THINK_PAD + THINK_LINES * TEXT_LINE_H; }
static int think_label_h(void) { return font_small.line + 14; }

static void draw_thinking(const ChatView* v, int x, int y, int w) {
    static Layout* lay;
    static char tail[THINK_TAIL + 3200];
    int cy = y + 24;   // level with the logo next to the reply
    int r = lang_rtl(ui_lang);   // (right to left: the label on the right, the counts on the left)
    float a = 0.5f + 0.5f * sinf((float)v->time * 6.0f);
    const char* label = tr(S_THINKING);
    circle(r ? x + w - 8 : x + 8, cy, 7, 0, C_ACCENT_HI, (int)(110 + 145 * a));
    draw_str(&font, r ? x + w - 28 - str_width(&font, label) : x + 28, cy - font.line / 2, label, C_ACCENT_HI);
    char s[64];
    snprintf(s, sizeof(s), "%d / %d  ·  %d s", v->think_n, v->think_cap, (int)v->think_s);
    draw_str(&font_small, r ? x : x + w - str_width(&font_small, s), cy - font_small.line / 2, s, C_TEXT_MUTED);
    int by = y + THINK_HEAD_H - 4, n = v->think_n < v->think_cap ? v->think_n : v->think_cap;
    rrect(x, by, w, 4, 2, 0x1B3156, 255);
    int fw = v->think_cap > 0 ? (int)((long long)w * n / v->think_cap) : 0;
    if (fw > 0) rrect(r ? x + w - (fw < 4 ? 4 : fw) : x, by, fw < 4 ? 4 : fw, 4, 2, C_ACCENT_HI, 255);
    int py = y + THINK_HEAD_H + 8, ph = 2 * THINK_PAD + THINK_LINES * TEXT_LINE_H - 8;
    rrect(x, py, w, ph, 18, C_SURFACE, 210);
    rrect_line(x, py, w, ph, 18, 1, C_LINE, 255);
    rrect(x + 14, py + 14, 4, ph - 28, 2, C_ACCENT, 170);
    const char* t = v->think_text;
    size_t len = v->think_len;
    if (!t || !len) return;
    size_t from = 0;   // the start of the line THINK_TAIL bytes from the end (or of a character, in a very long one)
    if (len > THINK_TAIL) {
        from = len - THINK_TAIL;
        while (from > 0 && t[from - 1] != '\n' && len - from < sizeof(tail) - 8) from--;
        if (from > 0 && t[from - 1] != '\n') {
            from = len - THINK_TAIL;
            while (from < len && ((uint8_t)t[from] & 0xC0) == 0x80) from++;
        }
    }
    int fences = 0;   // inside a code block there: open one, so that code stays code
    for (size_t i = 0; i + 2 < from; i++)
        if (t[i] == '`' && t[i + 1] == '`' && t[i + 2] == '`') fences++, i += 2;
    size_t k = 0;
    if (fences & 1) memcpy(tail, "```\n", 4), k = 4;
    memcpy(tail + k, t + from, len - from);
    k += len - from;
    tail[k] = 0;   // (layout_build reads up to the NUL; the buffer may still hold the end of a longer, earlier text)
    int tx = x + 34, tw = w - 34 - THINK_PAD;
    Layout* L = layout_get(&lay, tail, k, tw, &PAL_NOTE, 0);
    int cx0 = clip_x0, cy0 = clip_y0, cx1 = clip_x1, cy1 = clip_y1;
    int top = py + 10, bottom = py + ph - 10;
    // the top rows of the text fade into the panel (the line half shown there): the bare panel is kept first
    enum { FADE = 46 };
    static uint32_t band[FADE * SCREEN_W];
    int fx0 = tx - 4 > cx0 ? tx - 4 : cx0, fx1 = x + w - 4 < cx1 ? x + w - 4 : cx1;
    for (int r = 0; r < FADE && fx1 > fx0; r++)
        if (top + r >= cy0 && top + r < cy1) memcpy(band + r * SCREEN_W, back + (top + r) * SCREEN_W + fx0, (fx1 - fx0) * 4u);
    set_clip(cx0, top > cy0 ? top : cy0, cx1, bottom < cy1 ? bottom : cy1);
    draw_layout(L, tx, py + ph - THINK_PAD + 4 - L->height);
    set_clip(cx0, cy0, cx1, cy1);
    for (int r = 0; r < FADE && fx1 > fx0; r++) {
        if (top + r < cy0 || top + r >= cy1) continue;
        uint32_t a = 255 - 255u * r / FADE, *d = back + (top + r) * SCREEN_W + fx0;
        const uint32_t* b0 = band + r * SCREEN_W;
        for (int i = 0; i < fx1 - fx0; i++) blend(d + i, b0[i] & 0xFFFFFF, a);
    }
}

static void draw_thought(const Msg* m, int x, int y, int w) {   // (right to left: on the right of the reply)
    char s[96];
    int n = snprintf(s, sizeof(s), tr(S_THOUGHT), (int)(m->think_s + 0.5f), m->think_n);
    if (m->think_cut && n > 0 && n < (int)sizeof(s)) snprintf(s + n, sizeof(s) - n, " · %s", tr(S_THINK_CUT));
    int r = lang_rtl(ui_lang);
    circle(r ? x + w - 6 : x + 6, y + font_small.line / 2, 5, 0, C_ACCENT, 255);
    draw_str(&font_small, r ? x + w - 20 - str_width(&font_small, s) : x + 20, y, s, C_TEXT_MUTED);
}

static int msg_height(Msg* m, int typing) {
    Layout* L = layout_get(&m->lay, m->text, m->len, msg_layout_width(m->who), msg_palette(m->who), m->who == MSG_USER);
    if (m->who == MSG_USER) return L->height + 2 * BUB_PAD_Y;
    if (m->who == MSG_NOTE) return L->height + 8;
    int h = L->height + (m->think_n ? think_label_h() : 0);
    if (typing && h < 56) h = 56;
    return h < 56 ? 56 : h;
}

static void draw_empty(const ChatView* v) {
    int cx = SCREEN_W / 2, y = 250;
    circle(cx, y + logo_l.h / 2, 120, 0, C_ACCENT, 34);
    circle(cx, y + logo_l.h / 2, 92, 0, C_ACCENT_HI, 26);
    draw_image(&logo_l, cx - logo_l.w / 2, y, 255);
    y += logo_l.h + 26;
    int w1 = str_width(&font_big, "Garlic"), w2 = str_width(&font_big, "LLM");
    int x = cx - (w1 + font_big.space + w2) / 2;
    x = draw_str(&font_big, x, y, "Garlic", 0xFFFFFF);
    draw_str(&font_big, x + font_big.space, y, "LLM", C_ACCENT_HI);
    y += font_big.line + 14;
    char line_[160];
    snprintf(line_, sizeof(line_), tr(S_TAGLINE), v->model ? v->model : "");
    draw_center(&font, cx, y, line_, C_TEXT);
    y += font.line + 6;
    draw_center(&font_small, cx, y, tr(S_EMPTY_HINT), C_TEXT_MUTED);
    y += font_small.line + 44;
    // two action cards
    const Hint acts[2] = {{BTN_CROSS, S_WRITE}, {BTN_SQUARE, S_TALK_HOLD}};
    int ws[2], total = 0;
    for (int i = 0; i < 2; i++) ws[i] = 34 + 14 + str_width(&font, tr(acts[i].label)) + 56, total += ws[i];
    total += 28;
    x = cx - total / 2;
    for (int i = 0; i < 2; i++) {
        rrect(x, y, ws[i], 76, 20, C_SURFACE, 255);
        rrect_line(x, y, ws[i], 76, 20, 1, C_LINE_HI, 255);
        int bw = draw_btn(acts[i].btn, x + 28, y + 38);
        draw_str(&font, x + 28 + bw + 14, y + 38 - font.line / 2, tr(acts[i].label), 0xFFFFFF);
        x += ws[i] + 28;
    }
}

void draw_toast(const char* text, float alpha, int y) {
    if (!text || alpha <= 0) return;
    if (y <= 0) y = SCREEN_H - FOOTER_H - 36;
    text_alpha = (int)(255 * alpha);
    int tw = str_width(&font, text), w = tw + 64, h = 64, x = (SCREEN_W - w) / 2;
    rrect(x, y - h, w, h, h / 2, 0x16335F, 245);
    rrect_line(x, y - h, w, h, h / 2, 1, C_ACCENT_HI, 140);
    draw_str(&font, x + 32, y - h + (h - font.line) / 2, text, 0xFFFFFF);
    text_alpha = 255;
}

void draw_chat(ChatView* v) {
    draw_background();
    const int top = HEADER_H, bottom = SCREEN_H - FOOTER_H;
    const int area_top = top + 28, area_bottom = bottom - 24;
    if (!n_msgs) {
        draw_empty(v);
    } else {
        int typing_last = v->generating || v->reading;
        int total = 0;
        static int heights[MAX_MSGS];
        for (int i = 0; i < n_msgs; i++) {
            heights[i] = msg_height(&msgs[i], typing_last && i == n_msgs - 1);
            if (v->thinking && i == n_msgs - 1 && msgs[i].who == MSG_AI) heights[i] = think_panel_h();
            total += heights[i] + (i ? MSG_GAP : 0);
        }
        int view = area_bottom - area_top, max_scroll = total > view ? total - view : 0;
        if (v->scroll > max_scroll) v->scroll = max_scroll;
        if (v->scroll < 0) v->scroll = 0;
        int y = area_top + (total <= view ? 0 : view - total + v->scroll);
        set_clip(0, top + 1, SCREEN_W, bottom);
        for (int i = 0; i < n_msgs; i++) {
            Msg* m = &msgs[i];
            int h = heights[i];
            if (y + h >= top && y <= bottom) {
                Layout* L = m->lay;
                if (m->who == MSG_USER) {
                    int bw = L->max_w + 2 * BUB_PAD_X;
                    if (bw < 96) bw = 96;
                    int bx = COL_X + COL_W - bw;
                    int rad[4] = {26, 26, 8, 26};
                    rrect4(bx, y, bw, h, rad, 0, C_USER, 255);
                    draw_layout_w(L, bx + BUB_PAD_X, y + BUB_PAD_Y, L->max_w);
                } else if (m->who == MSG_NOTE) {
                    int nx = (SCREEN_W - L->max_w) / 2;
                    fill(COL_X + 120, y + h / 2, nx - COL_X - 150, 1, C_LINE);
                    fill(nx + L->max_w + 30, y + h / 2, COL_X + COL_W - 120 - nx - L->max_w - 30, 1, C_LINE);
                    draw_layout_w(L, nx, y + 4, L->max_w);
                } else {
                    circle(COL_X + 28, y + 28, 28, 0, 0x10244A, 255);
                    circle(COL_X + 28, y + 28, 28, 1.2f, C_LINE_HI, 255);
                    if (logo_s.px) draw_image(&logo_s, COL_X + 28 - logo_s.w / 2, y + 28 - logo_s.h / 2, 255);
                    int tx = COL_X + AI_INDENT, ty = y + 4;
                    if (v->thinking && i == n_msgs - 1) {   // still reasoning: its last lines instead
                        draw_thinking(v, tx, ty, COL_W - AI_INDENT);
                        y += h + MSG_GAP;
                        continue;
                    }
                    if (m->think_n) draw_thought(m, tx, ty, COL_W - AI_INDENT), ty += think_label_h();
                    draw_layout(L, tx, ty);
                    if (typing_last && i == n_msgs - 1) {
                        if (!L->n_lines || !L->n_runs) {   // nothing written yet: three bouncing dots
                            for (int k = 0; k < 3; k++) {
                                float ph = (float)v->time * 5.0f - k * 0.7f;
                                float a = 0.35f + 0.65f * (0.5f + 0.5f * sinf(ph));
                                circle(tx + 10 + k * 26, ty + 24 - 5 * (0.5f + 0.5f * sinf(ph)), 7, 0, C_ACCENT_HI, (int)(255 * a));
                            }
                        } else {
                            int ex, ey, eh, er;
                            layout_end(L, &ex, &ey, &eh, &er);
                            float a = 0.5f + 0.5f * sinf((float)v->time * 7.0f);
                            circle(tx + ex + (er ? -14 : 14), ty + ey + eh / 2, 7, 0, C_ACCENT_HI, (int)(90 + 165 * a));
                        }
                    }
                }
            }
            y += h + MSG_GAP;
        }
        no_clip();
        fade_rows(top + 1, top + 30, 0, 255);
        fade_rows(bottom - 30, bottom, 255, 0);
        if (max_scroll > 0) {   // scroll bar
            int track = area_bottom - area_top, x = COL_X + COL_W + 48;
            int th = track * view / total;
            if (th < 40) th = 40;
            int ty = area_top + (track - th) * (max_scroll - v->scroll) / max_scroll;
            rrect(x, area_top, 6, track, 3, 0x12244A, 255);
            rrect(x, ty, 6, th, 3, C_ACCENT_HI, 200);
        }
    }
    draw_header(v);
    Hint h[8];
    int n = 0;
    if (v->generating) h[n++] = (Hint){BTN_CIRCLE, S_STOP};
    else h[n++] = (Hint){BTN_CROSS, S_WRITE}, h[n++] = (Hint){BTN_SQUARE, S_TALK_HOLD};
    h[n++] = (Hint){BTN_TRIANGLE, S_NEW_CHAT};
    if (n_msgs) h[n++] = (Hint){BTN_UPDOWN, S_SCROLL};
    if (!v->generating) h[n++] = (Hint){BTN_L1, S_VOICE_MODE};
    if (v->can_run && !v->generating) h[n++] = (Hint){BTN_R1, S_PY_RUN};
    char right[64];
    snprintf(right, sizeof(right), "%s %.1f", tr(S_TEMPERATURE), v->temperature);
    if (v->think_avail) {   // R3: thinking. The footer leaves room for L2 / R2; without enough (Portuguese and
        footer_extra = 140;   // Russian with R1 shown) the scroll hint goes, then this one
        if (!v->generating) h[n++] = (Hint){BTN_R3, v->think_on ? S_THINK_ON_HINT : S_THINK_OFF_HINT};
        int avail = SCREEN_W - 2 * 72 - str_width(&font_small, right) - 40 - footer_extra, total = -14;
        for (int i = 0; i < n; i++) total += hint_width(&h[i]) + 14;
        for (int i = 0; i < n && total > avail; i++)
            if (h[i].btn == BTN_UPDOWN) {
                total -= hint_width(&h[i]) + 14;
                memmove(&h[i], &h[i + 1], (n - i - 1) * sizeof(Hint));
                n--;
            }
        if (total > avail && h[n - 1].btn == BTN_R3) n--;
    }
    draw_footer(h, n, right);
    footer_extra = 0;
    int rx = SCREEN_W - 72 - str_width(&font_small, right) - 18;   // L2 / R2 next to the temperature
    int w2 = draw_btn(BTN_R2, rx - 50, SCREEN_H - FOOTER_H / 2);
    draw_btn(BTN_L2, rx - 50 - w2 - 8, SCREEN_H - FOOTER_H / 2);
    draw_toast(v->toast, v->toast_alpha, 0);
}

// ---- loading and error screens --------------------------------------------------------------------------------

static void brand_block(int y) {
    int cx = SCREEN_W / 2;
    circle(cx, y + logo_l.h / 2, 130, 0, C_ACCENT, 30);
    circle(cx, y + logo_l.h / 2, 98, 0, C_ACCENT_HI, 22);
    draw_image(&logo_l, cx - logo_l.w / 2, y, 255);
    y += logo_l.h + 28;
    int w1 = str_width(&font_big, "Garlic"), w2 = str_width(&font_big, "LLM");
    int x = cx - (w1 + font_big.space + w2) / 2;
    x = draw_str(&font_big, x, y, "Garlic", 0xFFFFFF);
    draw_str(&font_big, x + font_big.space, y, "LLM", C_ACCENT_HI);
}

void draw_loading(float frac, const char* what) {
    draw_background();
    brand_block(270);
    int cx = SCREEN_W / 2, y = 600, w = 560;
    rrect(cx - w / 2, y, w, 10, 5, 0x12264A, 255);
    int fw = (int)(w * clamp01(frac));
    if (fw > 0) {
        rrect(cx - w / 2, y, fw < 10 ? 10 : fw, 10, 5, C_ACCENT, 255);
        rrect(cx - w / 2 + (fw > 40 ? fw - 40 : 0), y, fw > 40 ? 40 : fw, 10, 5, C_ACCENT_HI, 200);
    }
    char line_[160];
    snprintf(line_, sizeof(line_), "%s…  %d %%", what, (int)(clamp01(frac) * 100));
    draw_center(&font_small, cx, y + 30, line_, C_TEXT_DIM);
    draw_center(&font_small, cx, SCREEN_H - 90, "Qwen3.5  ·  Parakeet  ·  Nemotron  ·  Chatterbox", C_TEXT_MUTED);
}

void draw_error(const char* what) {
    draw_background();
    brand_block(220);
    int cx = SCREEN_W / 2, y = 560;
    draw_center(&font_big, cx, y, tr(S_ERROR_TITLE), C_RED);
    static Layout* L;
    Layout* l = layout_get(&L, what, strlen(what), 1200, &PAL_AI, 1);
    draw_layout_w(l, cx - l->max_w / 2, y + font_big.line + 20, l->max_w);
    draw_center(&font_small, cx, y + font_big.line + 40 + l->height, tr(S_ERROR_HINT), C_TEXT_MUTED);
}

// ---- the models folder: which weights the model starts with (main.c model_choose) --------------------------------

#define MODEL_ROW_W 920
#define MODEL_ROW_H 96
#define MODEL_ROW_GAP 14
#define MODEL_ROWS 4   // rows on the screen at once; the list scrolls with the selection

void draw_models(const char* const* names, const char* const* notes, int n, int sel) {
    draw_background();
    brand_block(150);
    int cx = SCREEN_W / 2, y = 150 + logo_l.h + 28 + font_big.line + 30;
    draw_center(&font, cx, y, tr(S_MODEL_TITLE), C_TEXT_DIM);
    y += font.line + 26;
    int top = sel < MODEL_ROWS ? 0 : sel - MODEL_ROWS + 1, x = cx - MODEL_ROW_W / 2;
    for (int i = top; i < n && i < top + MODEL_ROWS; i++, y += MODEL_ROW_H + MODEL_ROW_GAP) {
        int on = i == sel, cy = y + MODEL_ROW_H / 2;
        rrect(x, y, MODEL_ROW_W, MODEL_ROW_H, 24, on ? C_SURFACE_HI : C_SURFACE, 255);
        rrect_line(x, y, MODEL_ROW_W, MODEL_ROW_H, 24, on ? 2 : 1, on ? C_ACCENT_HI : C_LINE_HI, 255);
        circle(x + 48, cy, 13, 2.0f, on ? C_ACCENT_HI : C_LINE_HI, 255);
        if (on) circle(x + 48, cy, 7, 0, C_ACCENT_HI, 255);
        set_clip(x + 84, y, x + MODEL_ROW_W - 28, y + MODEL_ROW_H);
        int ty = y + (MODEL_ROW_H - font.line - font_small.line) / 2;
        draw_str(&font, x + 84, ty, names[i], on ? 0xFFFFFF : C_TEXT);
        draw_str(&font_small, x + 84, ty + font.line, notes[i], on ? C_TEXT_DIM : C_TEXT_MUTED);
        no_clip();
    }
    char where[24];
    snprintf(where, sizeof(where), "%d / %d", sel + 1, n);
    const Hint h[] = {{BTN_UPDOWN, S_MODEL_SELECT}, {BTN_CROSS, S_MODEL_START}};
    draw_footer(h, 2, n > MODEL_ROWS ? where : NULL);
}

void draw_model_bad(const char* name, const char* detail) {
    draw_background();
    brand_block(180);
    int cx = SCREEN_W / 2, y = 180 + logo_l.h + 28 + font_big.line + 56;
    draw_center(&font_big, cx, y, tr(S_MODEL_BAD_TITLE), 0xF2A65A);
    y += font_big.line + 14;
    draw_center(&font, cx, y, name, 0xFFFFFF);
    y += font.line + 22;
    static Layout* L;
    Layout* l = layout_get(&L, tr(S_MODEL_BAD), strlen(tr(S_MODEL_BAD)), 1200, &PAL_AI, 1);
    draw_layout_w(l, cx - l->max_w / 2, y, l->max_w);
    y += l->height + 16;
    if (detail && *detail) {
        draw_center(&font_small, cx, y, detail, C_TEXT_MUTED);
        y += font_small.line + 10;
    }
    draw_center(&font_small, cx, y, tr(S_MODEL_FALLBACK), C_TEXT_DIM);
    const Hint h[] = {{BTN_CROSS, S_MODEL_CONTINUE}};
    draw_footer(h, 1, NULL);
}

// ---- on-screen keyboard ----------------------------------------------------------------------------------------

#define KB_ROWS 6
#define KB_COLS 12
// Special keys: ⇧ shift, ⌫ delete, ␣ space, ⏎ send, [emoji], [lang] (next layout); dead keys (accents) are
// listed per layout in kb_dead; the kana keys ゛ ゜ 小 change the kana before them. A key repeated in neighbouring
// columns is one wide key.
static const char* kb_layouts[N_KB_LAYOUTS][KB_ROWS][KB_COLS] = {
    [KB_ES] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "¿", "?"},
               {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "´", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", "ñ", "¡", "!"},
               {"⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", ":", "-"},
               {";", "_", "+", "=", "*", "/", "%", "&", "<", ">", "(", ")"},
               {"[lang]", "\"", "'", "@", "#", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_EN] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "'", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", ":", ";", "\""},
               {"⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", "-", "/"},
               {"_", "+", "=", "*", "%", "&", "<", ">", "(", ")", "[", "]"},
               {"[lang]", "@", "#", "$", "~", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_PT] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "´", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", "ç", "~", "^"},
               {"⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", ":", "-"},
               {";", "_", "+", "=", "*", "/", "%", "&", "(", ")", "\"", "'"},
               {"[lang]", "@", "#", "`", "$", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_RU] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"й", "ц", "у", "к", "е", "н", "г", "ш", "щ", "з", "х", "⌫"},
               {"ф", "ы", "в", "а", "п", "р", "о", "л", "д", "ж", "э", "ъ"},
               {"⇧", "я", "ч", "с", "м", "и", "т", "ь", "б", "ю", "ё", ","},
               {".", ":", ";", "-", "_", "+", "=", "*", "/", "(", ")", "\""},
               {"[lang]", "@", "#", "%", "'", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_FR] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},   // AZERTY
               {"a", "z", "e", "r", "t", "y", "u", "i", "o", "p", "^", "⌫"},
               {"q", "s", "d", "f", "g", "h", "j", "k", "l", "m", "ù", "¨"},
               {"⇧", "w", "x", "c", "v", "b", "n", "é", "è", "à", "ç", "œ"},
               {",", ".", ";", ":", "-", "_", "+", "=", "*", "/", "(", ")"},
               {"[lang]", "'", "\"", "@", "€", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_DE] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "ß", "?"},   // QWERTZ
               {"q", "w", "e", "r", "t", "z", "u", "i", "o", "p", "ü", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", "ö", "ä", "!"},
               {"⇧", "y", "x", "c", "v", "b", "n", "m", ",", ".", "-", ":"},
               {";", "_", "+", "=", "*", "/", "%", "&", "(", ")", "\"", "'"},
               {"[lang]", "@", "#", "€", "~", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    // Arabic (the PC layout's letter rows), with the hamza forms of alef and the Persian letters on the fifth row
    [KB_AR] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "؟", "!"},
               {"ض", "ص", "ث", "ق", "ف", "غ", "ع", "ه", "خ", "ح", "ج", "⌫"},
               {"ش", "س", "ي", "ب", "ل", "ا", "ت", "ن", "م", "ك", "ط", "د"},
               {"ئ", "ء", "ؤ", "ر", "لا", "ى", "ة", "و", "ز", "ظ", "ذ", "،"},
               {"أ", "إ", "آ", "پ", "چ", "ژ", "گ", "ک", "ی", ".", ":", "-"},
               {"[lang]", "(", ")", "@", "%", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    // kana in the order of the syllabary: one column per consonant, one row per vowel
    [KB_JA] = {{"あ", "か", "さ", "た", "な", "は", "ま", "や", "ら", "わ", "゛", "⌫"},
               {"い", "き", "し", "ち", "に", "ひ", "み", "「", "り", "を", "゜", "小"},
               {"う", "く", "す", "つ", "ぬ", "ふ", "む", "ゆ", "る", "ん", "ー", "、"},
               {"え", "け", "せ", "て", "ね", "へ", "め", "」", "れ", "？", "！", "。"},
               {"お", "こ", "そ", "と", "の", "ほ", "も", "よ", "ろ", "⇧", "・", "〜"},
               {"[lang]", "（", "）", "：", "…", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    // two-set hangul (Shift: the double consonants and ㅒ ㅖ)
    [KB_KO] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"ㅂ", "ㅈ", "ㄷ", "ㄱ", "ㅅ", "ㅛ", "ㅕ", "ㅑ", "ㅐ", "ㅔ", ".", "⌫"},
               {"ㅁ", "ㄴ", "ㅇ", "ㄹ", "ㅎ", "ㅗ", "ㅓ", "ㅏ", "ㅣ", ",", ";", ":"},
               {"⇧", "ㅋ", "ㅌ", "ㅊ", "ㅍ", "ㅠ", "ㅜ", "ㅡ", "-", "'", "\"", "~"},
               {"_", "+", "=", "*", "/", "%", "&", "<", ">", "(", ")", "@"},
               {"[lang]", "#", "$", "^", "…", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
};
static const char* kb_names[N_KB_LAYOUTS] = {"ES", "EN", "PT", "RU", "FR", "DE", "AR", "JA", "KO"};
static const char* kb_abc[N_KB_LAYOUTS] = {"ABC", "ABC", "ABC", "АБВ", "ABC", "ABC", "أبج", "あいう", "가나다"};
static const char* kb_dead[N_KB_LAYOUTS][4] = {[KB_ES] = {"´"}, [KB_PT] = {"´", "~", "^", "`"}, [KB_FR] = {"^", "¨"}};

static int is_dead(int layout, const char* key) {
    for (int i = 0; i < 4 && kb_dead[layout][i]; i++)
        if (!strcmp(kb_dead[layout][i], key)) return 1;
    return 0;
}

static int kana_key(int layout, const char* key) {   // ゛ ゜ 小: 1, 2, 3
    if (layout != KB_JA) return 0;
    return !strcmp(key, "゛") ? 1 : !strcmp(key, "゜") ? 2 : !strcmp(key, "小") ? 3 : 0;
}

int kb_layout_for(int lang) {
    static const int map[N_LANGS] = {KB_ES, KB_EN, KB_PT, KB_RU, KB_FR, KB_DE, KB_AR, KB_JA, KB_EN, KB_EN, KB_KO};
    return lang >= 0 && lang < N_LANGS ? map[lang] : KB_EN;
}

#include "emoji_pages.h"
static const char* kb_emoji_bar[KB_COLS] = {"[abc]", "[abc]", "[t0]", "[t1]", "[t2]", "[t3]", "[t4]",
                                            "[t5]", "[t6]", "⌫", "⏎", "⏎"};
_Static_assert(EMOJI_ROWS == KB_ROWS - 1 && EMOJI_COLS == KB_COLS && EMOJI_PAGES == 7, "emoji layout");

static const char* kb_key(const Kb* kb, int row, int col) {
    if (kb->page == 0) return kb_layouts[kb->layout][row][col];
    if (row < EMOJI_ROWS) return emoji_keys[kb->page - 1][row * EMOJI_COLS + col];
    return kb_emoji_bar[col];
}

static int kb_span(const Kb* kb, int row, int col) {
    int span = 1;
    while (col + span < KB_COLS && !strcmp(kb_key(kb, row, col + span), kb_key(kb, row, col))) span++;
    return span;
}

void kb_init(Kb* kb, int layout) {
    memset(kb, 0, sizeof(*kb));
    kb->layout = kb->home = layout, kb->row = 1, kb->last_emoji_page = 1;
}

// The language key: the layout it opened with, English, then the others.
static int kb_next_layout(const Kb* kb) {
    int order[N_KB_LAYOUTS], n = 0;
    order[n++] = kb->home;
    if (kb->home != KB_EN) order[n++] = KB_EN;
    for (int l = 0; l < N_KB_LAYOUTS; l++)
        if (l != kb->home && l != KB_EN) order[n++] = l;
    for (int i = 0; i < n; i++)
        if (order[i] == kb->layout) return order[(i + 1) % n];
    return kb->home;
}

void kb_move(Kb* kb, int dx, int dy) {
    if (dy) kb->row = (kb->row + KB_ROWS + dy) % KB_ROWS;
    if (dx < 0) {
        while (kb->col > 0 && !strcmp(kb_key(kb, kb->row, kb->col - 1), kb_key(kb, kb->row, kb->col))) kb->col--;
        kb->col = (kb->col + KB_COLS - 1) % KB_COLS;
    }
    if (dx > 0) {
        while (kb->col < KB_COLS - 1 && !strcmp(kb_key(kb, kb->row, kb->col + 1), kb_key(kb, kb->row, kb->col))) kb->col++;
        kb->col = (kb->col + 1) % KB_COLS;
    }
}

void kb_toggle_emoji(Kb* kb) { kb->page = kb->page ? 0 : kb->last_emoji_page; }

void kb_append(char* out, int max, const char* s) {
    size_t n = strlen(out), m = strlen(s);
    if (n + m < (size_t)max) memcpy(out + n, s, m + 1);
}

void kb_backspace(char* out) {
    size_t n = strlen(out);
    while (n > 0) {
        n--;
        if (((uint8_t)out[n] & 0xC0) != 0x80) break;
    }
    out[n] = 0;
}

static uint32_t upper_cp(uint32_t c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c == 0xFF) return 0x178;
    if (c == 0x153) return 0x152;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    if (c >= 0x3041 && c <= 0x3096) return c + 0x60;   // hiragana -> katakana
    static const uint16_t ko[][2] = {{0x3142, 0x3143}, {0x3148, 0x3149}, {0x3137, 0x3138}, {0x3131, 0x3132},
                                     {0x3145, 0x3146}, {0x3150, 0x3152}, {0x3154, 0x3156}};   // ㅂ->ㅃ ... ㅔ->ㅖ
    for (size_t i = 0; i < sizeof(ko) / sizeof(ko[0]); i++)
        if (ko[i][0] == c) return ko[i][1];
    return c;
}

// A key's text given shift and a pending accent ("´" + "a" = "á").
static void kb_compose(const char* key, int shift, const char* dead, char* out, int max) {
    static const char* table[][3] = {{"´", "a", "á"}, {"´", "e", "é"}, {"´", "i", "í"}, {"´", "o", "ó"}, {"´", "u", "ú"},
                                     {"~", "a", "ã"}, {"~", "o", "õ"}, {"~", "n", "ñ"}, {"^", "a", "â"}, {"^", "e", "ê"},
                                     {"^", "o", "ô"}, {"^", "i", "î"}, {"^", "u", "û"}, {"`", "a", "à"}, {"`", "e", "è"},
                                     {"`", "u", "ù"}, {"¨", "a", "ä"}, {"¨", "e", "ë"}, {"¨", "i", "ï"}, {"¨", "o", "ö"},
                                     {"¨", "u", "ü"}, {"¨", "y", "ÿ"}};
    const char* base = key;
    if (dead)
        for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
            if (!strcmp(table[i][0], dead) && !strcmp(table[i][1], key)) base = table[i][2];
    int n = 0;
    for (const char* p = base; *p && n < max - 5;) {
        uint32_t c = utf8_next(&p);
        n += utf8_put(shift ? upper_cp(c) : c, out + n);
    }
    out[n] = 0;
}

// The last character of out: its codepoint and where it starts (0 if out is empty).
static uint32_t last_cp(const char* out, size_t* at) {
    size_t i = strlen(out);
    if (!i) return *at = 0, 0;
    while (i > 0 && ((uint8_t)out[--i] & 0xC0) == 0x80) {}
    const char* p = out + i;
    *at = i;
    return utf8_next(&p);
}

// Puts cps[0..n) in place of out's text from byte at, if it fits.
static void replace_tail(char* out, int max, size_t at, const uint32_t* cps, int n) {
    char s[16];
    int k = 0;
    for (int i = 0; i < n; i++) k += utf8_put(cps[i], s + k);
    if (at + k < (size_t)max) memcpy(out + at, s, k), out[at + k] = 0;
}

// ゛ ゜ 小 (kind 1, 2, 3) on the kana before them: か -> が, は -> ぱ, つ -> っ, and back. 0 = does not apply.
static uint32_t kana_change(uint32_t c, int kind) {
    int kata = c >= 0x30A1 && c <= 0x30F6;
    uint32_t h = kata ? c - 0x60 : c, r = 0;
    if (kind == 1) {
        if (h >= 0x304B && h <= 0x3062) r = (h - 0x304B) % 2 ? h - 1 : h + 1;   // か..ぢ
        else if (h >= 0x3064 && h <= 0x3069) r = (h - 0x3064) % 2 ? h - 1 : h + 1;   // つ..ど
        else if (h >= 0x306F && h <= 0x307D) r = (h - 0x306F) % 3 == 1 ? h - 1 : h - (h - 0x306F) % 3 + 1;   // は..ぽ
        else if (h == 0x3046) r = 0x3094;
        else if (h == 0x3094) r = 0x3046;
    } else if (kind == 2) {
        if (h >= 0x306F && h <= 0x307D) r = (h - 0x306F) % 3 == 2 ? h - 2 : h - (h - 0x306F) % 3 + 2;
    } else {
        static const uint16_t big[] = {0x3042, 0x3044, 0x3046, 0x3048, 0x304A, 0x3064, 0x3084, 0x3086, 0x3088, 0x308F};
        for (size_t i = 0; i < sizeof(big) / sizeof(big[0]) && !r; i++)
            r = h == big[i] ? h - 1u : h == big[i] - 1u ? h + 1u : 0;
    }
    return r ? (kata ? r + 0x60 : r) : 0;
}

// Two-set hangul: a jamo joins the syllable before it. Consonants are compatibility jamo U+3131-314E, vowels
// U+314F-3163 (in the order of the syllables' vowels); a syllable is 0xAC00 + (initial * 21 + vowel) * 28 + final.
static const int8_t ko_initial[30] = {0, 1, -1, 2, -1, -1, 3, 4, 5, -1, -1, -1, -1, -1, -1,
                                      -1, 6, 7, 8, -1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
static const int8_t ko_final[30] = {1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 17, 0, 18, 19, 20, 21, 22, 0, 23, 24, 25, 26, 27};
static const uint16_t ko_final_jamo[28] = {0, 0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136, 0x3137, 0x3139, 0x313A,
                                           0x313B, 0x313C, 0x313D, 0x313E, 0x313F, 0x3140, 0x3141, 0x3142, 0x3144,
                                           0x3145, 0x3146, 0x3147, 0x3148, 0x314A, 0x314B, 0x314C, 0x314D, 0x314E};
static const uint16_t ko_double_final[][3] = {{1, 0x3145, 3},   {4, 0x3148, 5},   {4, 0x314E, 6},  {8, 0x3131, 9},
                                              {8, 0x3141, 10},  {8, 0x3142, 11},  {8, 0x3145, 12}, {8, 0x314C, 13},
                                              {8, 0x314D, 14},  {8, 0x314E, 15},  {17, 0x3145, 18}};   // ㄳ ㄵ ㄶ ㄺ ...
static const uint8_t ko_double_vowel[][3] = {{8, 0, 9}, {8, 1, 10}, {8, 20, 11}, {13, 4, 14}, {13, 5, 15}, {13, 20, 16},
                                             {18, 20, 19}};   // ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ

static uint32_t ko_syllable(int initial, int vowel, int final) { return 0xAC00 + (initial * 21 + vowel) * 28 + final; }

static void hangul_type(char* out, int max, uint32_t j) {
    size_t at;
    uint32_t last = last_cp(out, &at);
    int syl = last >= 0xAC00 && last <= 0xD7A3, s = syl ? (int)(last - 0xAC00) : 0;
    int ini = s / 588, vow = s / 28 % 21, fin = s % 28;
    char one[8];
    if (j <= 0x314E) {   // a consonant: the final of the syllable before, or a double final
        uint32_t c[1];
        if (syl && !fin && ko_final[j - 0x3131]) {
            c[0] = ko_syllable(ini, vow, ko_final[j - 0x3131]);
            replace_tail(out, max, at, c, 1);
            return;
        }
        for (size_t k = 0; syl && fin && k < sizeof(ko_double_final) / sizeof(ko_double_final[0]); k++)
            if (ko_double_final[k][0] == fin && ko_double_final[k][1] == j) {
                c[0] = ko_syllable(ini, vow, ko_double_final[k][2]);
                replace_tail(out, max, at, c, 1);
                return;
            }
    } else {   // a vowel
        int v = (int)(j - 0x314F);
        if (last >= 0x3131 && last <= 0x314E && ko_initial[last - 0x3131] >= 0) {   // after a lone consonant
            uint32_t c[1] = {ko_syllable(ko_initial[last - 0x3131], v, 0)};
            replace_tail(out, max, at, c, 1);
            return;
        }
        if (syl && fin) {   // the final consonant starts the new syllable (or the second half of a double one)
            int keep = 0;
            uint32_t moved = ko_final_jamo[fin];
            for (size_t k = 0; k < sizeof(ko_double_final) / sizeof(ko_double_final[0]); k++)
                if (ko_double_final[k][2] == fin) keep = ko_double_final[k][0], moved = ko_double_final[k][1];
            uint32_t c[2] = {ko_syllable(ini, vow, keep), ko_syllable(ko_initial[moved - 0x3131], v, 0)};
            replace_tail(out, max, at, c, 2);
            return;
        }
        for (size_t k = 0; syl && k < sizeof(ko_double_vowel) / sizeof(ko_double_vowel[0]); k++)
            if (ko_double_vowel[k][0] == vow && ko_double_vowel[k][1] == v) {
                uint32_t c[1] = {ko_syllable(ini, ko_double_vowel[k][2], 0)};
                replace_tail(out, max, at, c, 1);
                return;
            }
    }
    one[utf8_put(j, one)] = 0;
    kb_append(out, max, one);
}

int kb_press(Kb* kb, char* out, int max) {
    const char* key = kb_key(kb, kb->row, kb->col);
    int kana = kb->page == 0 ? kana_key(kb->layout, key) : 0;
    if (!strcmp(key, "⏎")) return 1;
    if (!strcmp(key, "⌫")) kb_backspace(out);
    else if (!strcmp(key, "⇧")) kb->shift = !kb->shift;
    else if (!strcmp(key, "␣")) kb_append(out, max, " "), kb->dead = NULL;
    else if (!strcmp(key, "[emoji]")) kb->page = kb->last_emoji_page;
    else if (!strcmp(key, "[abc]")) kb->page = 0;
    else if (!strcmp(key, "[lang]")) kb->layout = kb_next_layout(kb), kb->shift = 0, kb->dead = NULL;
    else if (!strncmp(key, "[t", 2)) kb->page = kb->last_emoji_page = 1 + (key[2] - '0');
    else if (kana) {   // ゛ ゜ 小 change the kana before them (on anything else ゛ and ゜ are typed as they are)
        size_t at;
        uint32_t c = kana_change(last_cp(out, &at), kana);
        if (c) replace_tail(out, max, at, &c, 1);
        else if (kana < 3) kb_append(out, max, key);
    } else if (kb->page == 0 && is_dead(kb->layout, key)) {
        if (kb->dead && !strcmp(kb->dead, key)) kb_append(out, max, key), kb->dead = NULL;   // twice: the accent itself
        else kb->dead = key;
    } else {
        char s[32];
        kb_compose(key, kb->shift, kb->dead, s, sizeof(s));
        const char* q = s;
        uint32_t c = utf8_next(&q);
        if (kb->dead && !strcmp(s, key)) kb_append(out, max, kb->dead);   // accent that does not combine: keep it
        if (kb->layout == KB_KO && kb->page == 0 && !*q && c >= 0x3131 && c <= 0x3163) hangul_type(out, max, c);
        else kb_append(out, max, s);
        kb->dead = NULL;
        if (kb->layout != KB_JA) kb->shift = 0;   // shift applies to one letter, like a phone keyboard (katakana stay)
    }
    if (kb->page) kb->last_emoji_page = kb->page;
    return 0;
}

#define KEY_W 112
#define KEY_H 64
#define KEY_GAP 10
#define KEYS_W (KB_COLS * (KEY_W + KEY_GAP) - KEY_GAP)
#define FIELD_H 116
#define PANEL_PAD 30
#define PANEL_H (PANEL_PAD + FIELD_H + 24 + KB_ROWS * (KEY_H + KEY_GAP) - KEY_GAP + PANEL_PAD)
#define PANEL_Y (SCREEN_H - FOOTER_H - PANEL_H)

static void draw_field(int x, int y, int w, const char* text, double time) {
    rrect(x, y, w, FIELD_H, 20, 0x050E22, 255);
    rrect_line(x, y, w, FIELD_H, 20, 2, C_ACCENT_HI, 170);
    int pad = 28, inner_w = w - 2 * pad;
    int caret_x, caret_y, caret_h;
    int blink = fmod(time, 1.0) < 0.6;
    if (!text[0]) {   // (in a right-to-left language the empty field starts at its right edge)
        int r = lang_rtl(ui_lang);
        const char* ph = tr(S_PLACEHOLDER);
        draw_str(&font, r ? x + w - pad - str_width(&font, ph) : x + pad, y + (FIELD_H - font.line) / 2, ph, C_TEXT_MUTED);
        if (blink) rrect(r ? x + w - pad + 1 : x + pad - 4, y + (FIELD_H - 36) / 2, 3, 36, 1, C_ACCENT_HI, 255);
        return;
    }
    static Layout* L;
    Layout* l = layout_get(&L, text, strlen(text), inner_w, &PAL_FIELD, 1);
    int area = FIELD_H - 20;
    int oy = y + 10 + (l->height <= area ? (area - l->height) / 2 : area - l->height);   // keep the end in view
    set_clip(x + 4, y + 4, x + w - 4, y + FIELD_H - 4);
    draw_layout(l, x + pad, oy);
    int rtl;
    layout_end(l, &caret_x, &caret_y, &caret_h, &rtl);
    size_t n = strlen(text);
    for (size_t i = n; i > 0 && text[i - 1] == ' '; i--) caret_x += rtl ? -font.space : font.space;   // trailing spaces
    if (blink) rrect(x + pad + caret_x + (rtl ? -5 : 2), oy + caret_y + (caret_h - 36) / 2, 3, 36, 1, C_ACCENT_HI, 255);
    no_clip();
}

static void key_label(const Kb* kb, const char* key, char* out, int max, const Font** f) {
    *f = &font;
    if (!strcmp(key, "␣")) snprintf(out, max, "%s", tr(S_SPACE_KEY)), *f = &font_small;
    else if (!strcmp(key, "⏎")) snprintf(out, max, "%s", tr(S_SEND_KEY)), *f = &font_bold;
    else if (!strcmp(key, "[emoji]")) snprintf(out, max, "😀");
    else if (!strcmp(key, "[abc]")) snprintf(out, max, "%s", kb_abc[kb->layout]), *f = &font_bold;
    else if (!strcmp(key, "[lang]")) snprintf(out, max, "🌐 %s", kb_names[kb->layout]), *f = &font_small;
    else if (!strncmp(key, "[t", 2)) snprintf(out, max, "%s", emoji_tab[key[2] - '0']), *f = &font_big;
    else if (kb->page > 0) snprintf(out, max, "%s", key), *f = &font_big;   // emoji at ~25 px
    else if (is_dead(kb->layout, key)) snprintf(out, max, "%s", key), *f = &font_big;   // accents are tiny: bigger
    else if (kana_key(kb->layout, key)) snprintf(out, max, "%s", (const char*[]){"", "濁点", "半濁点", "小"}[kana_key(kb->layout, key)]);
    else if (!strcmp(key, "⇧") || !strcmp(key, "⌫")) snprintf(out, max, "%s", key);
    else kb_compose(key, kb->shift, kb->dead, out, max);
}

// One key (its first column k), plain or selected.
static void kb_draw_key(const Kb* kb, int r, int k, int x0, int y0, int sel) {
    const char* key = kb_key(kb, r, k);
    int span = kb_span(kb, r, k);
    int x = x0 + k * (KEY_W + KEY_GAP), ky = y0 + r * (KEY_H + KEY_GAP), w = span * (KEY_W + KEY_GAP) - KEY_GAP;
    int special = key[0] == '[' || !strcmp(key, "⇧") || !strcmp(key, "⌫") || !strcmp(key, "␣") ||
                  (kb->page == 0 && (is_dead(kb->layout, key) || kana_key(kb->layout, key)));
    int enter = !strcmp(key, "⏎");
    int on = (!strcmp(key, "⇧") && kb->shift) || (kb->dead && !strcmp(key, kb->dead)) ||
             (kb->page > 0 && !strncmp(key, "[t", 2) && key[2] - '0' == kb->page - 1);
    if (sel) {
        rrect(x - 7, ky - 7, w + 14, KEY_H + 14, 20, C_ACCENT_HI, 40);
        rrect(x - 3, ky - 3, w + 6, KEY_H + 6, 16, C_ACCENT_HI, 90);
        rrect(x, ky, w, KEY_H, 13, 0x1C86EA, 255);
    } else {
        rrect(x, ky, w, KEY_H, 13, enter ? C_ACCENT : special ? C_KEY_SPECIAL : C_KEY, 255);
        if (on) rrect_line(x, ky, w, KEY_H, 13, 2, C_ACCENT_HI, 255);
    }
    char label[48];
    const Font* f;
    key_label(kb, key, label, sizeof(label), &f);
    int tw = str_width(f, label);
    uint32_t col = sel || enter ? 0xFFFFFF : on ? C_ACCENT_HI : special ? C_TEXT_DIM : 0xEEF4FF;
    draw_str(f, x + (w - tw) / 2, ky + (KEY_H - f->line) / 2, label, col);
}

// The keyboard is drawn in two layers: everything that only changes with the layout, page, shift or accent
// (backdrop, panel, every key, footer) is kept in kb_cache; each frame copies it and draws only the selected
// key and the text field on top. So moving around costs almost nothing, whatever is behind. (kb_cache is declared
// with the backdrop; without memory for it, every frame draws the whole keyboard.)

void draw_keyboard(const Kb* kb, const char* text, double time) {
    int px = (SCREEN_W - KEYS_W) / 2 - PANEL_PAD;
    int x0 = px + PANEL_PAD, y = PANEL_Y + PANEL_PAD, y0 = y + FIELD_H + 24;
    int sig = kb->layout | kb->page << 4 | kb->shift << 7 | ui_lang << 8 |
              (kb->dead ? 1 + (int)((uintptr_t)kb->dead % 4093) : 0) << 12;
    if (!kb_cache) kb_cache = ui_screen_buf();
    if (sig != kb_cache_sig || !kb_cache) {
        ui_backdrop_draw();
        int rad[4] = {30, 30, 0, 0};
        rrect4(px, PANEL_Y, KEYS_W + 2 * PANEL_PAD, PANEL_H + 1, rad, 0, C_PANEL, 255);
        rrect4(px, PANEL_Y, KEYS_W + 2 * PANEL_PAD, PANEL_H + 1, rad, 1, C_LINE_HI, 255);
        for (int r = 0; r < KB_ROWS; r++)
            for (int k = 0; k < KB_COLS; k++)
                if (k == 0 || strcmp(kb_key(kb, r, k), kb_key(kb, r, k - 1))) kb_draw_key(kb, r, k, x0, y0, 0);
        // (in Chinese, Japanese and Korean, OPTIONS: the system keyboard, with its input methods)
        const Hint h_letters[] = {{BTN_CROSS, S_KB_TYPE}, {BTN_SQUARE, S_KB_DELETE}, {BTN_TRIANGLE, S_KB_SPACE},
                                  {BTN_L1, S_KB_SHIFT},   {BTN_R1, S_KB_EMOJI},      {BTN_L2, S_KB_DICTATE},
                                  {BTN_R2, S_KB_SEND},    {BTN_CIRCLE, S_KB_CLOSE},  {BTN_OPTIONS, S_KB_SYSTEM}};
        int cjk = ui_lang == LANG_JA || ui_lang == LANG_ZH_HANS || ui_lang == LANG_ZH_HANT || ui_lang == LANG_KO;
        const Hint h_emoji[] = {{BTN_CROSS, S_KB_TYPE}, {BTN_SQUARE, S_KB_DELETE}, {BTN_TRIANGLE, S_KB_SPACE},
                                {BTN_R1, S_KB_LETTERS}, {BTN_R2, S_KB_SEND},      {BTN_CIRCLE, S_KB_CLOSE}};
        if (kb->page) draw_footer(h_emoji, 6, emoji_titles[ui_lang][kb->page - 1]);
        else draw_footer(h_letters, cjk ? 9 : 8, NULL);
        if (kb_cache) {
            memcpy(kb_cache, back, SCREEN_W * SCREEN_H * 4);
            kb_cache_sig = sig;
        }
    } else {
        memcpy(back, kb_cache, SCREEN_W * SCREEN_H * 4);
    }
    int k = kb->col;
    while (k > 0 && !strcmp(kb_key(kb, kb->row, k - 1), kb_key(kb, kb->row, k))) k--;   // first column of a wide key
    kb_draw_key(kb, kb->row, k, x0, y0, 1);
    draw_field(x0, y, KEYS_W, text, time);
}

// ---- voice -------------------------------------------------------------------------------------------------------

static void mic_icon(float cx, float cy, float s, uint32_t col) {   // s = 1: about 60 px tall
    rrect((int)(cx - 13 * s), (int)(cy - 30 * s), (int)(26 * s), (int)(42 * s), (int)(13 * s), col, 255);
    int sx0 = clip_x0, sy0 = clip_y0, sx1 = clip_x1, sy1 = clip_y1;
    set_clip((int)(cx - 40 * s), (int)(cy - 6 * s), (int)(cx + 40 * s), (int)(cy + 40 * s));
    circle(cx, cy - 8 * s, 21 * s, 4 * s, col, 255);
    set_clip(sx0, sy0, sx1, sy1);
    line(cx, cy + 13 * s, cx, cy + 25 * s, 4 * s, col, 255);
    line(cx - 11 * s, cy + 26 * s, cx + 11 * s, cy + 26 * s, 4 * s, col, 255);
}

void draw_voice(int phase, const float* levels, int n_levels, float secs, float max_secs, double time) {
    int w = 760, h = 420, x = (SCREEN_W - w) / 2, y = HEADER_H + (SCREEN_H - HEADER_H - FOOTER_H - h) / 2;
    rrect(x, y, w, h, 34, C_PANEL, 255);
    rrect_line(x, y, w, h, 34, 1, C_LINE_HI, 255);
    int cx = SCREEN_W / 2, cy = y + 140;
    float level = n_levels ? levels[n_levels - 1] : 0;
    if (phase == VOICE_RECORDING) {
        float pulse = 0.5f + 0.5f * sinf((float)time * 4.0f);
        circle(cx, cy, 78 + 26 * level + 6 * pulse, 0, C_ACCENT_HI, 40);
        circle(cx, cy, 66 + 12 * level, 0, C_ACCENT_HI, 60);
        circle(cx, cy, 58, 0, C_ACCENT, 255);
        mic_icon(cx, cy + 2, 1.0f, 0xFFFFFF);
        draw_center(&font_big, cx, y + 236, tr(S_LISTENING), 0xFFFFFF);
        char t[64];
        snprintf(t, sizeof(t), "%d:%02d / %d:%02d", (int)secs / 60, (int)secs % 60, (int)max_secs / 60, (int)max_secs % 60);
        draw_center(&font_small, cx, y + 298, t, C_TEXT_DIM);
        const int bars = 40, bw = 9, gap = 5, total = bars * (bw + gap) - gap;
        for (int i = 0; i < bars; i++) {
            int src = n_levels - bars + i;
            float lv = src >= 0 ? levels[src] : 0;
            int bh = 6 + (int)(lv * 46);
            int bx = cx - total / 2 + i * (bw + gap), by = y + 372 - bh / 2;
            rrect(bx, by, bw, bh, 4, i > bars - 4 ? C_ACCENT_HI : 0x3C78C8, 255);
        }
        const Hint hh[] = {{BTN_SQUARE, S_RELEASE_TO_END}};
        draw_footer(hh, 1, NULL);
    } else {
        for (int k = 0; k < 12; k++) {   // spinner
            float a = (float)k / 12 * 6.2831853f + (float)time * 6.0f;
            float fade = (float)k / 12;
            circle(cx + 44 * cosf(a), cy + 44 * sinf(a), 7, 0, C_ACCENT_HI, (int)(40 + 215 * fade));
        }
        draw_center(&font_big, cx, y + 236, tr(S_TRANSCRIBING), 0xFFFFFF);
        draw_footer(NULL, 0, NULL);
    }
}

void draw_confirm(const char* text) {
    static Layout* L;
    int w = COL_W, pad = 36;
    Layout* l = layout_get(&L, text, strlen(text), w - 2 * pad - 90, &PAL_FIELD, 1);
    int text_h = l->height > 4 * TEXT_LINE_H ? 4 * TEXT_LINE_H : l->height;
    int h = 64 + text_h + 30, x = COL_X, y = SCREEN_H - FOOTER_H - 28 - h;
    rrect(x, y, w, h, 28, C_PANEL, 255);
    rrect_line(x, y, w, h, 28, 2, C_ACCENT_HI, 150);
    circle(x + pad + 30, y + h / 2, 30, 0, C_ACCENT, 255);
    mic_icon(x + pad + 30, y + h / 2 + 1, 0.5f, 0xFFFFFF);
    int tx = x + pad + 90;
    draw_str(&font_small, tx, y + 22, tr(S_YOUR_MESSAGE), C_ACCENT_HI);
    set_clip(tx, y + 58, x + w - pad, y + 58 + text_h + 4);
    draw_layout(l, tx, y + 58 + (l->height > text_h ? text_h - l->height : 0));
    no_clip();
    const Hint hh[] = {{BTN_CROSS, S_SEND}, {BTN_TRIANGLE, S_EDIT}, {BTN_SQUARE, S_RETRY}, {BTN_CIRCLE, S_DISCARD}};
    draw_footer(hh, 4, NULL);
}

#include "ui_run.inc"
#include "ui_voice.inc"
