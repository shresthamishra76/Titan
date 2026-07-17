#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "GGUF.h"
#include "Tensor.h"

namespace titan {

// Hyperparameters for a Llama-family model. All values come from GGUF metadata.
struct ModelConfig {
  int n_layers = 0;
  int n_embd = 0;         // hidden size
  int n_head = 0;         // query heads
  int n_head_kv = 0;      // key/value heads (GQA: n_head_kv <= n_head)
  int head_dim = 0;       // n_embd / n_head
  int n_ff = 0;           // feed-forward hidden size
  int n_vocab = 0;
  int context_length = 0;
  float rope_theta = 10000.0f;
  float rms_eps = 1e-5f;
};

// Per-decoder-block weights. Linear weights are row-major [out, in] (as loaded
// from GGUF), consumed via linear(x, W) = x @ W^T.
struct LayerWeights {
  Tensor attn_norm;   // [n_embd]
  Tensor wq;          // [n_head    * head_dim, n_embd]
  Tensor wk;          // [n_head_kv * head_dim, n_embd]
  Tensor wv;          // [n_head_kv * head_dim, n_embd]
  Tensor wo;          // [n_embd, n_head * head_dim]
  Tensor ffn_norm;    // [n_embd]
  Tensor w_gate;      // [n_ff, n_embd]
  Tensor w_up;        // [n_ff, n_embd]
  Tensor w_down;      // [n_embd, n_ff]
};

struct ModelWeights {
  Tensor token_embd;              // [n_vocab, n_embd]
  std::vector<LayerWeights> layers;
  Tensor output_norm;             // [n_embd]
  Tensor output;                  // [n_vocab, n_embd] (tied to token_embd if absent)
};

// Rotary position embedding (RoFormer), applied in place to a contiguous
// [seq, n_heads * head_dim] tensor. Uses llama.cpp's interleaved-pair
// convention, which is what standard GGUF conversions expect.
void apply_rope(Tensor& x, int n_heads, int head_dim, float theta);

// Grouped-query causal self-attention (no KV cache — a V1 revision in V2 adds
// one). q: [seq, n_head*head_dim]; k, v: [seq, n_head_kv*head_dim].
// Returns [seq, n_head*head_dim].
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int n_head,
                 int n_head_kv, int head_dim);

// A Llama-family transformer running a single sequence on CPU.
//
// V1 has no KV cache: each generation step re-runs the full forward pass over
// the growing sequence. Correct but O(n^2); the KV cache is V2 scope.
// See docs/architecture/transformer-runtime.md.
class Model {
 public:
  Model(ModelConfig config, ModelWeights weights);

  static Model from_gguf(const GgufFile& gguf);

  const ModelConfig& config() const { return config_; }

  // Logits for the final position: shape [n_vocab].
  Tensor forward(const std::vector<int32_t>& tokens) const;

  // Greedy autoregressive generation. Returns the new token ids (including the
  // terminating eos, if produced within max_new_tokens).
  std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_new_tokens,
                                int32_t eos_id) const;

 private:
  ModelConfig config_;
  ModelWeights weights_;
};

}  // namespace titan
