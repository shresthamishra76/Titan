# Tokenizer

**Subsystem:** `runtime/tokenizer` · **Milestone:** 1 · **Library:** `titan_tokenizer`

## Overview

The tokenizer converts between text and the integer token ids the model consumes. Titan implements a SentencePiece-style byte-pair encoder from scratch, matching the behavior of `llama.cpp`'s SPM tokenizer so that a TinyLlama GGUF tokenizes identically to the reference.

## Design rationale

- **Score-based greedy merge (not a merges list).** Llama's SentencePiece vocab ships per-token *scores*, not an ordered merge table. Encoding starts from single UTF-8 characters and repeatedly merges the adjacent pair whose combined string exists in the vocab with the **highest score**, breaking ties by lower left index. A priority queue over a doubly-linked list of symbols gives the standard `O(n log n)` implementation.
- **Byte fallback.** Any piece with no vocab entry is emitted as raw `<0xNN>` byte tokens, so the tokenizer is total — every input round-trips, including emoji and arbitrary bytes.
- **U+2581 space marker + dummy prefix.** Spaces are escaped to `▁` and a leading `▁` is prepended (SentencePiece's `add_dummy_prefix`), which decode reverses so round-trips are exact.
- **Config, not hard-coding.** All vocab data (tokens, scores, types, special-token ids, add-bos/eos flags) is supplied via `TokenizerConfig`, which the CLI builds from GGUF `tokenizer.ggml.*` metadata.

## Public interface

```cpp
Tokenizer tok(TokenizerConfig);
std::vector<int32_t> ids  = tok.encode(text, add_special=true);
std::string          text = tok.decode(ids, skip_special=true);
std::string          piece = tok.decode_piece(id);   // streaming
```

## Tradeoffs

- The score-based merge is faithful to Llama but assumes SPM semantics; a BPE-merges-list model (e.g. GPT-2 style) would need a rank-based variant (future work).
- Streaming `decode_piece` can emit partial UTF-8 when a code point spans several byte-fallback tokens; callers reassemble at the byte level.

## References

- Sennrich et al. 2016, *Neural Machine Translation of Rare Words with Subword Units* (BPE).
- Kudo & Richardson 2018, *SentencePiece*.
- GGUF spec, `tokenizer.ggml.*` metadata; `llama.cpp` `llm_tokenizer_spm`.

## Future improvements

- GPT-2/tiktoken-style rank-based BPE for non-SPM models.
- Precompiled-charsmap normalization; regex pre-tokenization.
