# Model Loader (GGUF)

**Subsystem:** `runtime/models` · **Milestone:** 1 · **Library:** `titan_models`

## Overview

The loader reads [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files — the single-file format used by `llama.cpp` that bundles model hyperparameters, the tokenizer, and weights. It parses the header, the typed metadata key/value table, and the tensor-info table, memory-maps the weight data, and materializes individual tensors as float32 `titan::Tensor`s.

## Design rationale

- **Memory-mapped weights (RAII).** `MappedFile` wraps `mmap`/`munmap` so the OS pages weights in on demand and the mapping is released deterministically on destruction. A multi-GB model costs no upfront copy.
- **Bounds-checked cursor.** All parsing goes through a little-endian `Cursor` that validates every read against the file size — untrusted files fail cleanly rather than reading out of bounds.
- **Config from metadata.** Hyperparameters (`<arch>.block_count`, `embedding_length`, `attention.head_count[_kv]`, `feed_forward_length`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`, …) are read from metadata, never hard-coded, so the same code loads any Llama-family GGUF.
- **ggml → row-major.** ggml stores dims with the contiguous axis first; the loader reverses them, so a ggml `[in, out]` weight becomes a natural row-major `[out, in]` `titan::Tensor`.
- **F16 widening; quant deferred.** F32 is copied directly and F16 is widened via a hand-written half→float. Quantized types are rejected with a message pointing at the deferred dequantization issue (#8 stretch), keeping V1 correctness-first.

## Public interface

```cpp
GgufFile gguf(path);
uint32_t v   = gguf.version();
uint32_t n   = gguf.get_u32("llama.block_count", 0);
auto     toks = gguf.get_str_array("tokenizer.ggml.tokens");
Tensor   w   = gguf.tensor("token_embd.weight");   // float32, row-major
```

## Tradeoffs

- Materializing tensors as F32 trades memory for simplicity; on-the-fly dequant (llama.cpp-style) is the memory-efficient path and is future work.
- The whole metadata table is parsed eagerly; fine for model files, but a lazy index would matter for huge metadata.

## References

- GGUF specification (ggml-org). `llama.cpp` GGUF reader and weight naming.
- IEEE-754 half precision (for F16 widening).

## Future improvements

- Quantized dequant (Q4_K, Q6_K, Q8_0) behind the same `tensor()` call.
- SafeTensors loader sharing the tensor-materialization path.
