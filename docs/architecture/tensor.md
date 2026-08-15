# Tensor Engine

**Subsystem:** `runtime/tensor` · **Milestone:** 1 · **Library:** `titan_tensor`

## Overview

The Tensor engine is the numerical foundation of Titan. Every other subsystem — the GGUF loader, the transformer runtime, sampling — is expressed in terms of tensors. It provides an N-dimensional array abstraction over a flat, contiguous, row-major `float` buffer, plus the operators the Llama forward pass needs: element-wise arithmetic with broadcasting, matrix multiplication, shape manipulation (reshape / transpose / views), and the neural ops (softmax, RMSNorm, SiLU, SwiGLU).

V1 is **correctness-first and CPU-only**. The naive implementations here are deliberately simple and readable; Milestone 2 replaces the hot loops (matmul especially) with cache-blocked, threaded, NEON-vectorized versions, justified by profiling data. Keeping V1 simple gives Milestone 2 a correct reference to benchmark and diff against.

## Design rationale

- **Single dtype (`float32`) in V1.** TinyLlama weights load as F16/F32 and are widened to F32 on load. A single dtype keeps the op implementations trivial and correct. Quantized dtypes (Q4_K, …) are deferred to a later issue, behind the same `Tensor` interface.
- **Owning buffer via `std::vector<float>`.** RAII gives us correct copy/move and leak-free lifetime with zero manual `new`/`delete`. This is the project's first lesson in C++ ownership.
- **Explicit shape + strides.** Storing strides (rather than assuming packed row-major everywhere) lets `transpose` and `reshape` produce cheap **views** that share the underlying buffer instead of copying. Ops that require contiguity call `contiguous()` first.
- **Value semantics with explicit views.** A `Tensor` owns its data. A `TensorView` is a non-owning window (pointer + shape + strides) into an existing tensor — this makes the owning-vs-borrowing distinction visible in the type system, which is exactly the C++ concept this subsystem teaches.

## Memory layout

A tensor of shape `(d0, d1, …, dn-1)` is stored in a flat buffer of `d0·d1·…·dn-1` floats in **row-major** (C) order. The stride of axis `i` is the product of all dimensions after it:

```
strides[n-1] = 1
strides[i]   = strides[i+1] * shape[i+1]
offset(idx)  = Σ idx[i] * strides[i]
```

A **contiguous** tensor is one whose strides match this formula. `transpose` swaps two entries in both `shape` and `strides`, producing a non-contiguous view without touching data.

## Public interface (V1)

```cpp
namespace titan {

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<std::size_t> shape);            // zero-filled
  Tensor(std::vector<std::size_t> shape, std::vector<float> data);

  // Factories
  static Tensor zeros(std::vector<std::size_t> shape);
  static Tensor ones(std::vector<std::size_t> shape);
  static Tensor full(std::vector<std::size_t> shape, float v);

  // Introspection
  const std::vector<std::size_t>& shape() const;
  const std::vector<std::size_t>& strides() const;
  std::size_t numel() const;
  std::size_t ndim() const;
  bool is_contiguous() const;

  // Data access (row-major, multi-index)
  float&       at(std::initializer_list<std::size_t> idx);
  const float& at(std::initializer_list<std::size_t> idx) const;
  float*       data();
  const float* data() const;

  // Shape ops (views share storage where possible)
  Tensor reshape(std::vector<std::size_t> shape) const;
  Tensor transpose(std::size_t a, std::size_t b) const;
  Tensor contiguous() const;
};

// Element-wise (with broadcasting)
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float s);

// Linear algebra
Tensor matmul(const Tensor& a, const Tensor& b);   // 2D: (m,k) x (k,n) -> (m,n)

// Neural ops (Llama operator set)
Tensor softmax(const Tensor& x, int axis = -1);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);
Tensor silu(const Tensor& x);                      // x * sigmoid(x)
Tensor swiglu(const Tensor& gate, const Tensor& up);   // silu(gate) * up

}  // namespace titan
```

## Broadcasting

Broadcasting follows NumPy rules: shapes are right-aligned, and a dimension is compatible if the sizes are equal or one of them is 1. This is what lets a `(seq, dim)` activation add a `(dim,)` bias, or a per-row scale multiply a matrix.

## Tradeoffs

- **Naive matmul is O(m·n·k) with poor cache behavior.** Accepted in V1 for readability; it is the headline target of Milestone 2. The benchmark harness will show why.
- **No SIMD / threading yet.** Deferred to M2 so improvements are measured, not assumed.
- **Views add stride bookkeeping.** Slightly more complex indexing, but avoids copies in the attention path (transpose is everywhere in attention).

## References

- *Attention Is All You Need* (Vaswani et al., 2017) — the ops this engine must support.
- *RoFormer* (Su et al., 2021) — RoPE, applied in the transformer runtime on top of these tensors.
- llama.cpp `ggml` — reference for the CPU tensor op set and GGUF dtypes.

## Future improvements

- Additional dtypes (F16 storage, quantized Q4_K/Q6_K/Q8_0) behind the same interface.
- Cache-blocked + NEON + threaded matmul (Milestone 2).
- Fused ops (e.g. fused RMSNorm, fused SwiGLU) to cut activation traffic.
- Arena/scratch allocator to eliminate per-op allocation (Milestone 2).
