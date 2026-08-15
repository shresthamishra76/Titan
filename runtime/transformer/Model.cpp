#include "Model.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace titan {
namespace {

// y = x @ W^T, where W is row-major [out, in]. transpose is a view; matmul
// materializes it. Correctness-first; Milestone 2 fuses/optimizes this.
Tensor linear(const Tensor& x, const Tensor& w) {
  return matmul(x, w.transpose(0, 1));
}

}  // namespace

void apply_rope(Tensor& x, int n_heads, int head_dim, float theta) {
  const int seq = static_cast<int>(x.shape()[0]);
  const int width = n_heads * head_dim;
  const int half = head_dim / 2;
  float* d = x.data();
  for (int i = 0; i < seq; ++i) {
    for (int h = 0; h < n_heads; ++h) {
      float* base = d + static_cast<std::size_t>(i) * width + static_cast<std::size_t>(h) * head_dim;
      for (int p = 0; p < half; ++p) {
        const float freq = std::pow(theta, -2.0f * static_cast<float>(p) / head_dim);
        const float angle = static_cast<float>(i) * freq;
        const float c = std::cos(angle), s = std::sin(angle);
        const float x0 = base[2 * p], x1 = base[2 * p + 1];
        base[2 * p] = x0 * c - x1 * s;
        base[2 * p + 1] = x0 * s + x1 * c;
      }
    }
  }
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int n_head,
                 int n_head_kv, int head_dim) {
  const int seq = static_cast<int>(q.shape()[0]);
  const int qw = n_head * head_dim;
  const int kw = n_head_kv * head_dim;
  const int group = n_head / n_head_kv;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float* qd = q.data();
  const float* kd = k.data();
  const float* vd = v.data();

  Tensor out(Shape{static_cast<std::size_t>(seq), static_cast<std::size_t>(qw)});
  float* od = out.data();
  std::vector<float> scores(seq);

  for (int h = 0; h < n_head; ++h) {
    const int kvh = h / group;  // GQA: query heads share a kv head
    for (int i = 0; i < seq; ++i) {
      const float* qi = qd + static_cast<std::size_t>(i) * qw + static_cast<std::size_t>(h) * head_dim;
      float mx = -std::numeric_limits<float>::infinity();
      for (int j = 0; j <= i; ++j) {  // causal: only attend to <= i
        const float* kj = kd + static_cast<std::size_t>(j) * kw + static_cast<std::size_t>(kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
        dot *= scale;
        scores[j] = dot;
        mx = std::max(mx, dot);
      }
      float sum = 0.0f;
      for (int j = 0; j <= i; ++j) {
        scores[j] = std::exp(scores[j] - mx);
        sum += scores[j];
      }
      float* oi = od + static_cast<std::size_t>(i) * qw + static_cast<std::size_t>(h) * head_dim;
      for (int d = 0; d < head_dim; ++d) oi[d] = 0.0f;
      for (int j = 0; j <= i; ++j) {
        const float wj = scores[j] / sum;
        const float* vj = vd + static_cast<std::size_t>(j) * kw + static_cast<std::size_t>(kvh) * head_dim;
        for (int d = 0; d < head_dim; ++d) oi[d] += wj * vj[d];
      }
    }
  }
  return out;
}

Model::Model(ModelConfig config, ModelWeights weights)
    : config_(std::move(config)), weights_(std::move(weights)) {}

Model Model::from_gguf(const GgufFile& g) {
  const std::string arch = g.get_str("general.architecture", "llama");
  const std::string p = arch + ".";

  ModelConfig cfg;
  cfg.n_layers = static_cast<int>(g.get_u32(p + "block_count", 0));
  cfg.n_embd = static_cast<int>(g.get_u32(p + "embedding_length", 0));
  cfg.n_head = static_cast<int>(g.get_u32(p + "attention.head_count", 0));
  cfg.n_head_kv = static_cast<int>(g.get_u32(p + "attention.head_count_kv", cfg.n_head));
  cfg.n_ff = static_cast<int>(g.get_u32(p + "feed_forward_length", 0));
  cfg.context_length = static_cast<int>(g.get_u32(p + "context_length", 0));
  cfg.rms_eps = g.get_f32(p + "attention.layer_norm_rms_epsilon", 1e-5f);
  cfg.rope_theta = g.get_f32(p + "rope.freq_base", 10000.0f);
  const int derived_head_dim = cfg.n_head > 0 ? cfg.n_embd / cfg.n_head : 0;
  cfg.head_dim = static_cast<int>(g.get_u32(p + "rope.dimension_count", derived_head_dim));

  ModelWeights w;
  w.token_embd = g.tensor("token_embd.weight");
  cfg.n_vocab = static_cast<int>(w.token_embd.shape()[0]);
  w.output_norm = g.tensor("output_norm.weight");
  w.output = g.has_tensor("output.weight") ? g.tensor("output.weight") : w.token_embd;

  w.layers.reserve(cfg.n_layers);
  for (int i = 0; i < cfg.n_layers; ++i) {
    const std::string b = "blk." + std::to_string(i) + ".";
    LayerWeights L;
    L.attn_norm = g.tensor(b + "attn_norm.weight");
    L.wq = g.tensor(b + "attn_q.weight");
    L.wk = g.tensor(b + "attn_k.weight");
    L.wv = g.tensor(b + "attn_v.weight");
    L.wo = g.tensor(b + "attn_output.weight");
    L.ffn_norm = g.tensor(b + "ffn_norm.weight");
    L.w_gate = g.tensor(b + "ffn_gate.weight");
    L.w_up = g.tensor(b + "ffn_up.weight");
    L.w_down = g.tensor(b + "ffn_down.weight");
    w.layers.push_back(std::move(L));
  }
  return Model(std::move(cfg), std::move(w));
}

Tensor Model::forward(const std::vector<int32_t>& tokens) const {
  const int seq = static_cast<int>(tokens.size());
  const int D = config_.n_embd;

  // Token embedding lookup.
  Tensor x(Shape{static_cast<std::size_t>(seq), static_cast<std::size_t>(D)});
  {
    float* xd = x.data();
    const float* emb = weights_.token_embd.data();
    for (int i = 0; i < seq; ++i) {
      std::memcpy(xd + static_cast<std::size_t>(i) * D,
                  emb + static_cast<std::size_t>(tokens[i]) * D, D * sizeof(float));
    }
  }

  for (const LayerWeights& L : weights_.layers) {
    // Pre-norm attention with residual.
    Tensor h = rms_norm(x, L.attn_norm, config_.rms_eps);
    Tensor q = linear(h, L.wq);
    Tensor k = linear(h, L.wk);
    Tensor v = linear(h, L.wv);
    apply_rope(q, config_.n_head, config_.head_dim, config_.rope_theta);
    apply_rope(k, config_.n_head_kv, config_.head_dim, config_.rope_theta);
    Tensor a = attention(q, k, v, config_.n_head, config_.n_head_kv, config_.head_dim);
    x = add(x, linear(a, L.wo));

    // Pre-norm SwiGLU feed-forward with residual.
    Tensor h2 = rms_norm(x, L.ffn_norm, config_.rms_eps);
    Tensor act = swiglu(linear(h2, L.w_gate), linear(h2, L.w_up));
    x = add(x, linear(act, L.w_down));
  }

  x = rms_norm(x, weights_.output_norm, config_.rms_eps);

  // Only the last position is needed to predict the next token.
  Tensor last(Shape{1, static_cast<std::size_t>(D)});
  std::memcpy(last.data(), x.data() + static_cast<std::size_t>(seq - 1) * D, D * sizeof(float));
  Tensor logits = linear(last, weights_.output);  // [1, n_vocab]
  return logits.reshape({static_cast<std::size_t>(config_.n_vocab)});
}

std::vector<int32_t> Model::generate(const std::vector<int32_t>& prompt, int max_new_tokens,
                                     int32_t eos_id) const {
  std::vector<int32_t> tokens = prompt;
  std::vector<int32_t> out;
  for (int step = 0; step < max_new_tokens; ++step) {
    const Tensor logits = forward(tokens);
    const float* ld = logits.data();
    int best = 0;
    for (int j = 1; j < config_.n_vocab; ++j) {
      if (ld[j] > ld[best]) best = j;
    }
    tokens.push_back(best);
    out.push_back(best);
    if (best == eos_id) break;
  }
  return out;
}

}  // namespace titan
