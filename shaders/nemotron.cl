// Nemotron 3.5 ASR (speech to text) on the PS4 GPU: the kernels it needs beyond Parakeet's (parakeet.cl, whose
// mm_q4i, mm_f32t, layernorm_rows, glu_rows, lstm_cell, joint_in and argmax_tdt it shares). Activations are
// row-major [frames][channels]. Reference: pc/nemotron_ref.py.

#include "common.h"

#define NM_D 1024
#define NM_HEADS 8
#define NM_DH 128
#define NM_CHUNK 14               // attention chunks of 14 encoder frames (1.12 s)
#define NM_LEFT 4                 // chunks of left context (56 frames)
#define NM_DMIN (-13)             // relative distances i - j: NM_DMIN .. NM_DMIN + 82
#define NM_F1 65                  // frequency bins after each subsampling conv: 128 -> 65 -> 33 -> 17
#define NM_F2 33
#define NM_F3 17

// mm_q4i with 8-bit weights ("q8i", the joint head, which loses too much in 4 bits): the same tiles of 64 rows,
// per tile and block of 32 the 64 rows' 32 int8 weights back to back (2 x 16 bytes per lane), then the fp16 scales
// in the same order as q4i's. Weight = q * d.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mm_q8i(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict bias,
            global const float* restrict X, global float* restrict Y, uint cols, uint rows, uint n, uint ldx,
            uint ldy, uint act, float alpha, float beta) {
    local float part[4][MM_NB][64];
    uint l = LID & 63, nb = cols / 32, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID % tiles, r0 = (GID / tiles) * MM_NB;
    uint nv = n - r0 < MM_NB ? n - r0 : MM_NB;
    ulong tb = (ulong)tile * nb * 64;
    global const uint4v* wp = w + tb * 2 + l * 2;
    global const half_t* sp = sc + tb + l;
    float acc[MM_NB];
    for (uint v = 0; v < MM_NB; v++) acc[v] = 0.0f;
    for (uint b = wv; b < nb; b += 4) {
        uint4v q0 = wp[(ulong)b * 128], q1 = wp[(ulong)b * 128 + 1];
        float d = (float)sp[(ulong)b * 64];
        uint ws[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        float wf[32];
        for (uint k = 0; k < 8; k++) {
            int u = (int)ws[k];
            wf[4 * k] = (float)((u << 24) >> 24) * d;
            wf[4 * k + 1] = (float)((u << 16) >> 24) * d;
            wf[4 * k + 2] = (float)((u << 8) >> 24) * d;
            wf[4 * k + 3] = (float)(u >> 24) * d;
        }
        for (uint v = 0; v < MM_NB; v++) {
            if (v < nv) {
                global const float* xb = X + (ulong)(r0 + v) * ldx + b * 32;
                float s = 0.0f;
                for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < MM_NB; v++) part[wv][v][l] = acc[v];
    BARRIER();
    if (wv == 0) {
        uint o = tile * 64 + l;
        float bo = bias ? bias[o] : 0.0f;
        for (uint v = 0; v < nv; v++) {
            float r = act_fn(part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l] + bo, act);
            global float* y = Y + (ulong)(r0 + v) * ldy + o;
            *y = (beta != 0.0f ? beta * *y : 0.0f) + alpha * r;
        }
    }
}

// ---- subsampling (x8 in time and frequency), causal: each conv 3x3 stride 2 sees input rows 2t - 2 .. 2t and
// columns 2f - 2 .. 2f (zeros outside). Frames from the output length on are zero (they are masked).
// conv_in (1 -> 256, + ReLU) and the first depthwise conv in one pass, as sub_conv1_dw1: the mels [T][128] ->
// [T2][33][256]. T1, T2: valid output frames of the two convs.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void nm_sub_conv1_dw1(global const float* F, global const float* w0, global const float* b0, global const float* w1,
                      global const float* b1, global float* Y, uint T, uint T1, uint T2) {
    uint i = GID * 256 + LID, c = i % 256, f2 = (i / 256) % NM_F2, t2 = i / (256 * NM_F2);
    if (t2 >= T2) return;
    // branch-free (clamped indices, zero weight outside), as sub_conv1_dw1
    float s = b1[c];
    for (int ki = 0; ki < 3; ki++) {
        int t1 = 2 * (int)t2 - 2 + ki;
        float vt1 = (t1 >= 0 && t1 < (int)T1) ? 1.0f : 0.0f;
        for (int kj = 0; kj < 3; kj++) {
            int f1 = 2 * (int)f2 - 2 + kj;
            float v1 = (f1 >= 0 && f1 < NM_F1) ? vt1 : 0.0f;
            float a = b0[c];                               // conv_in output (t1, f1, c)
            for (int ui = 0; ui < 3; ui++) {
                int tt = 2 * t1 - 2 + ui;
                float vt = (tt >= 0 && tt < (int)T) ? 1.0f : 0.0f;
                int tc = tt < 0 ? 0 : tt >= (int)T ? (int)T - 1 : tt;
                for (int uj = 0; uj < 3; uj++) {
                    int ff = 2 * f1 - 2 + uj;
                    float v = (ff >= 0 && ff < 128) ? vt : 0.0f;
                    int fc = ff < 0 ? 0 : ff > 127 ? 127 : ff;
                    a += v * F[(ulong)tc * 128 + fc] * w0[c * 9 + ui * 3 + uj];
                }
            }
            s += v1 * __builtin_fmaxf(a, 0.0f) * w1[c * 9 + ki * 3 + kj];
        }
    }
    Y[i] = s;
}

// the second depthwise conv: [H][33][256] -> [H2][17][256] (ReLU comes after the pointwise conv)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void nm_sub_dwconv(global const float* X, global const float* w, global const float* b, global float* Y, uint H,
                   uint H2) {
    uint i = GID * 256 + LID, c = i % 256, x2 = (i / 256) % NM_F3, y2 = i / (256 * NM_F3);
    if (y2 >= H2) return;
    float s = b[c];
    for (int ki = 0; ki < 3; ki++) {
        int yy = 2 * (int)y2 - 2 + ki;
        float vy = (yy >= 0 && yy < (int)H) ? 1.0f : 0.0f;
        int yc = yy < 0 ? 0 : yy >= (int)H ? (int)H - 1 : yy;
        for (int kj = 0; kj < 3; kj++) {
            int xx = 2 * (int)x2 - 2 + kj;
            float v = (xx >= 0 && xx < NM_F2) ? vy : 0.0f;
            int xc = xx < 0 ? 0 : xx >= NM_F2 ? NM_F2 - 1 : xx;
            s += v * X[((ulong)yc * NM_F2 + xc) * 256 + c] * w[c * 9 + ki * 3 + kj];
        }
    }
    Y[i] = s;
}

// ---- attention over chunks: one group of 128 threads per (query frame i, head h). Frame i (chunk c = i / 14)
// sees frames (c - 4) * 14 .. (c + 1) * 14 - 1: at most 70, one per thread. QKV: [T][3 * 1024] (q | k | v);
// R: [83][1024] fp16, the positional rows of the distances i - j = -13 .. 69 (row i - j + 13).
// score(i, j) = ((q_i + bias_u) . k_j + (q_i + bias_v) . r_{i-j}) / sqrt(128).
kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void nm_attn(global const float* QKV, global const half_t* R, global const float* bu, global const float* bv,
             global float* O, uint T) {
    local float qu[NM_DH], qv[NM_DH], p[128], red[128];
    uint h = GID % NM_HEADS, i = GID / NM_HEADS, t = LID;
    float q = QKV[(ulong)i * 3 * NM_D + h * NM_DH + t];
    qu[t] = q + bu[h * NM_DH + t];
    qv[t] = q + bv[h * NM_DH + t];
    uint c = i / NM_CHUNK;
    uint j0 = c >= NM_LEFT ? (c - NM_LEFT) * NM_CHUNK : 0;
    uint j1 = (c + 1) * NM_CHUNK < T ? (c + 1) * NM_CHUNK : T;
    uint n = j1 - j0;
    BARRIER();
    float s = -3.0e38f;
    if (t < n) {
        uint j = j0 + t;
        global const float* k = QKV + (ulong)j * 3 * NM_D + NM_D + h * NM_DH;
        global const half_t* r = R + (ulong)((int)i - (int)j - NM_DMIN) * NM_D + h * NM_DH;
        float a = 0.0f;
        for (uint d = 0; d < NM_DH; d++) a += qu[d] * k[d] + qv[d] * (float)r[d];
        s = a * 0.08838834764831845f;
    }
    red[t] = s;
    BARRIER();
    for (uint st = 64; st > 0; st >>= 1) {
        if (t < st) red[t] = __builtin_fmaxf(red[t], red[t + st]);
        BARRIER();
    }
    float mx = red[0];
    BARRIER();
    float e = t < n ? exp_(s - mx) : 0.0f;
    p[t] = e;
    red[t] = e;
    BARRIER();
    for (uint st = 64; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        BARRIER();
    }
    float sum = red[0];
    float o = 0.0f;
    global const float* v = QKV + (ulong)j0 * 3 * NM_D + 2 * NM_D + h * NM_DH + t;
    for (uint j = 0; j < n; j++) o += p[j] * v[(ulong)j * 3 * NM_D];
    O[(ulong)i * NM_D + h * NM_DH + t] = o / sum;
}

// ---- conv module middle, after glu_rows: causal depthwise conv of 9 over time (frames t - 8 .. t, zeros before
// the first), LayerNorm (w, b) and SiLU. One group of 256 threads per frame, 4 channels each. G [T][1024] -> Y.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void nm_dwconv_ln_silu(global const float* G, global const float* dw, global const float* lw,
                       global const float* lb, global float* Y) {
    local float red[256];
    int t = (int)GID;
    float s[4], sum = 0.0f;
    for (uint k = 0; k < 4; k++) {
        uint ch = LID + 256 * k;
        float a = 0.0f;
        for (int j = 0; j < 9; j++) {
            int tt = t - 8 + j;
            float v = tt >= 0 ? 1.0f : 0.0f;
            a += v * G[(ulong)(tt < 0 ? 0 : tt) * NM_D + ch] * dw[ch * 9 + j];
        }
        s[k] = a;
        sum += a;
    }
    float mean = group_sum(red, sum) / (float)NM_D, var = 0.0f;
    for (uint k = 0; k < 4; k++) var += (s[k] - mean) * (s[k] - mean);
    float r = rsq(group_sum(red, var) / (float)NM_D + 1e-5f);
    for (uint k = 0; k < 4; k++) {
        uint ch = LID + 256 * k;
        float y = (s[k] - mean) * r * lw[ch] + lb[ch];
        Y[(ulong)t * NM_D + ch] = y / (1.0f + exp_(-y));
    }
}
