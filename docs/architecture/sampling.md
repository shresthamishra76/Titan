# Sampling Engine

**Subsystem:** `runtime/sampling` · **Milestone:** 1 · **Library:** `titan_sampling`

## Overview

The sampler turns a logits vector into the next token id. It implements the standard decoding controls — greedy, temperature, top-k, top-p (nucleus), and repetition penalty — over a seeded RNG for reproducibility.

## Pipeline

```
logits ─▶ repetition penalty ─▶ temperature ─▶ top-k ─▶ top-p ─▶ categorical draw
```

Each stage is optional via a disable sentinel (`temperature<=0` ⇒ greedy, `top_k=0` ⇒ off, `top_p=1` ⇒ off, `repetition_penalty=1` ⇒ off). Order matches `llama.cpp`.

## Design rationale

- **Repetition penalty (llama.cpp convention).** Seen tokens have positive logits divided and negative logits multiplied by the penalty, pushing them toward zero probability.
- **Numerically stable softmax** over only the surviving candidates (max-subtraction), computed once after top-k sorts them descending, so top-p can walk the cumulative mass directly.
- **Seeded `std::mt19937` + `std::discrete_distribution`.** Identical params + seed ⇒ identical output, which the end-to-end correctness tests rely on.
- **`std::partial_sort` for top-k** avoids fully sorting a 32k-entry vocab.

## Public interface

```cpp
Sampler s(SamplingParams{temperature, top_k, top_p, repetition_penalty, seed});
int32_t id = s.sample(logits, recent_tokens);
int32_t g  = Sampler::argmax(logits);   // greedy
```

## Tradeoffs

- Repetition penalty applies uniformly to the supplied `recent` window; frequency/presence penalties (OpenAI-style) are future work.
- Full-vocab softmax each step; fine at V1 scale, fusable later.

## References

- Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration* (nucleus/top-p).
- `llama.cpp` sampling order and repetition-penalty semantics.

## Future improvements

- Speculative decoding (V2).
- Mirostat, min-p, frequency/presence penalties.
