// Text tokenizer of Chatterbox Multilingual (assets/tts_tok.bin from pc/convert_chatterbox_t3.py --assets): text of
// one sentence -> T3 text tokens, as the official MTLTokenizer with punc_norm, lower case and NFKD.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "ja_kana.h"
#include "ru_stress.h"

typedef struct {
    const RuStress* ru;          // Russian stress marks before NFKD, as the official tokenizer (NULL: none)
    const JaKana* ja;            // Japanese kanji as hiragana, as the official tokenizer (NULL: kanji left as they are)
    uint32_t n_vocab, n_merges, n_added, n_lower, n_nfkd, n_ccc, n_wr;
    const uint8_t** vocab;       // per id: length byte + UTF-8
    const uint8_t* merges;       // per rank: <HHH left, right, merged
    const uint8_t** added;       // per added token (longest first): <H id, length byte, UTF-8
    const uint8_t* lower;        // <II pairs, sorted
    const uint8_t** nfkd;        // per codepoint: <I cp, n, n x <I
    const uint8_t* ccc;          // <II (first, last | combining class << 24), sorted
    const uint8_t* wr;           // <II (first, last): \w above U+FFFF, sorted
    const uint8_t* word_bits;    // \w for codepoints below 0x10000
} TtsTok;

int ttstok_load(TtsTok* t, const uint8_t* blob, size_t size);   // keeps pointers into blob
// Tokens of text for language lang ("es", "en", "pt", "ru", "ar", "ja"...), with start 255 and stop 0. Returns the
// count.
int ttstok_encode(const TtsTok* t, const char* text, const char* lang, uint16_t* out, int max);
