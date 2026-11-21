// Japanese kanji -> hiragana for the voice, as the official Chatterbox tokenizer does with pykakasi: at each
// kanji, the longest word of the dictionary (assets/ja_kana.bin, pc/export_ja_kana.py) that starts there, read
// with its first reading. The dictionary stays in the file: only its index is in memory, and each lookup reads
// the block of words that start with that kanji.
#pragma once
#include <stdint.h>
#include <stdio.h>

typedef struct {
    int ready, fd;
    char path[128];
    uint32_t n_var, n_first, n_words;
    uint32_t* var;      // n_var x (variant, usual form), sorted
    uint32_t* first;    // (n_first + 1) x (first character, block offset), sorted
    uint64_t data;      // where the blocks start in the file
} JaKana;

int jakana_open(JaKana* j, const char* path);   // 0 = ready
// Text in[0, n) (codepoints, lower case and NFKD, as the tokenizer has it) with its kanji words as hiragana, as
// Chatterbox's hiragana_normalize: readings composed (the caller decomposes them again), a space before a reading
// that starts with は or へ, kana and everything else as written. Returns the count written to out.
int jakana_convert(const JaKana* j, const uint32_t* in, int n, uint32_t* out, int max);
