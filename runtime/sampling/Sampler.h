#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "Tensor.h"

namespace titan {

struct SamplingParams {
  float temperature = 1.0f;          // <= 0 selects greedy (argmax)
  int top_k = 0;                     // 0 disables
  float top_p = 1.0f;                // 1 disables (nucleus sampling)
  float repetition_penalty = 1.0f;   // 1 disables
  uint64_t seed = 0;
};

// Turns a logits vector into a sampled token id. The pipeline mirrors
// llama.cpp: repetition penalty -> temperature -> top-k -> top-p -> sample.
// See docs/architecture/sampling.md.
class Sampler {
 public:
  explicit Sampler(SamplingParams params);

  // logits: shape [n_vocab]. `recent` holds previously produced tokens, used
  // for the repetition penalty.
  int32_t sample(const Tensor& logits, const std::vector<int32_t>& recent = {});

  static int32_t argmax(const Tensor& logits);

 private:
  SamplingParams params_;
  std::mt19937 rng_;
};

}  // namespace titan
