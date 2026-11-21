// Japanese kanji -> hiragana for the voice, as the official Chatterbox tokenizer does with pykakasi: at each
// kanji, the longest word of the dictionary (assets/ja_kana.bin, pc/export_ja_kana.py) that starts there, read
// with its first reading (or a fixed one for a few very common words). Words are looked up in the text as written:
// the official tokenizer decomposes the text (NFKD) first, and then words with a voiced kana ending (食べる, 上がる)
// never match and are misread (くべる, うえがる). The dictionary stays in the file: only its index is in memory, and
// each lookup reads the block of words that start with that kanji.
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
// Text in[0, n) (codepoints, lower case, composed) with its kanji words as hiragana, as Chatterbox's
// hiragana_normalize: a space before a reading that starts with は or へ, kana and everything else as written (the
// caller decomposes the result, NFKD). Returns the count written to out.
int jakana_convert(const JaKana* j, const uint32_t* in, int n, uint32_t* out, int max);
