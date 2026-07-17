# Transformer Runtime

**Subsystem:** `runtime/transformer` · **Milestone:** 1 · **Library:** `titan_transformer`

## Overview

The runtime executes a Llama-family decoder forward pass on CPU and drives autoregressive generation. It is the convergence point of the project: it consumes the tensor engine, the loaded weights, and (via the CLI) the tokenizer.

## Architecture

Per decoder block (pre-norm residual, matching Llama):

```
x ──▶ RMSNorm ──▶ Wq/Wk/Wv ──▶ RoPE ──▶ GQA causal attention ──▶ Wo ──▶ (+x)
                                                                          │
      ┌───────────────────────────────────────────────────────────────┘
      ▼
   RMSNorm ──▶ SwiGLU FFN (gate/up/down) ──▶ (+)
```

After all blocks: a final RMSNorm and the LM head (`output`, tied to `token_embd` when absent) produce vocab logits.

## Design rationale

- **Llama operator set.** RMSNorm (not LayerNorm), SwiGLU (not GELU), grouped-query attention, and rotary embeddings — the ops TinyLlama actually uses.
- **Interleaved-pair RoPE.** RoPE rotates adjacent element pairs `(2i, 2i+1)` — `llama.cpp`'s `NORM` convention, which is what standard GGUF conversions are permuted for. Getting this wrong produces plausible-looking but incorrect output, so it is called out explicitly.
- **GQA.** With `n_head_kv < n_head`, query heads are grouped so each group shares one KV head (`kv = head / (n_head / n_head_kv)`), matching how the weights were trained.
- **No KV cache in V1.** Each generation step re-runs the full forward over the growing sequence — `O(n^2)` but simple and correct. Adding a KV cache (and the attention read/append path it needs) is an explicit **V1 revision in V2** (issue #22).
- **Weights are `[out, in]`; linears use `x @ Wᵀ`** via a transpose view, keeping the layout identical to what GGUF loads.

## Public interface

```cpp
Model m = Model::from_gguf(gguf);
Tensor logits = m.forward(token_ids);                 // [n_vocab] for last pos
std::vector<int32_t> out = m.generate(prompt, max_new_tokens, eos_id);

void   apply_rope(Tensor& x, n_heads, head_dim, theta);
Tensor attention(q, k, v, n_head, n_head_kv, head_dim);
```

## Tradeoffs

- Recomputing the full forward each step (no cache) dominates cost for long sequences — intentional for V1, addressed by the KV cache in V2.
- Naive matmul underneath; Milestone 2 optimizes it (cache blocking, threads, NEON).

## References

- Vaswani et al. 2017, *Attention Is All You Need*.
- Su et al. 2021, *RoFormer* (RoPE). Ainslie et al. 2023, *GQA*.
- Zhang & Sennrich 2019, *RMSNorm*. Shazeer 2020, *GLU Variants* (SwiGLU).

## Future improvements

- KV cache + incremental decoding (V2, Milestone 3).
- Batched/ragged forward for continuous batching (V2).
- Fused attention/FFN kernels.
