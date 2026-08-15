#include "Sampler.h"

#include <algorithm>
#include <cmath>

namespace titan {

Sampler::Sampler(SamplingParams params)
    : params_(params), rng_(params.seed) {}

int32_t Sampler::argmax(const Tensor& logits) {
  const float* d = logits.data();
  const int n = static_cast<int>(logits.numel());
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (d[i] > d[best]) best = i;
  }
  return best;
}

int32_t Sampler::sample(const Tensor& logits, const std::vector<int32_t>& recent) {
  const int n = static_cast<int>(logits.numel());
  std::vector<float> l(logits.data(), logits.data() + n);

  // Repetition penalty (llama.cpp convention: divide positive logits, multiply
  // negative ones, pushing seen tokens toward zero probability).
  if (params_.repetition_penalty != 1.0f) {
    for (const int32_t t : recent) {
      if (t >= 0 && t < n) {
        l[t] = l[t] > 0 ? l[t] / params_.repetition_penalty
                        : l[t] * params_.repetition_penalty;
      }
    }
  }

  // Greedy shortcut.
  if (params_.temperature <= 0.0f) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
      if (l[i] > l[best]) best = i;
    }
    return best;
  }

  for (float& x : l) x /= params_.temperature;

  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;

  // top-k: keep the k highest logits (descending).
  const int k = params_.top_k > 0 ? std::min(params_.top_k, n) : n;
  if (k < n) {
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return l[a] > l[b]; });
    idx.resize(k);
  } else {
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  }

  // Stable softmax over the surviving logits (already sorted descending).
  const float mx = l[idx[0]];
  std::vector<float> probs(idx.size());
  float sum = 0.0f;
  for (std::size_t i = 0; i < idx.size(); ++i) {
    probs[i] = std::exp(l[idx[i]] - mx);
    sum += probs[i];
  }
  for (float& p : probs) p /= sum;

  // top-p (nucleus): smallest descending prefix with cumulative prob >= top_p.
  if (params_.top_p < 1.0f) {
    float cum = 0.0f;
    std::size_t cut = probs.size();
    for (std::size_t i = 0; i < probs.size(); ++i) {
      cum += probs[i];
      if (cum >= params_.top_p) {
        cut = i + 1;
        break;
      }
    }
    idx.resize(cut);
    probs.resize(cut);
    float renorm = 0.0f;
    for (const float p : probs) renorm += p;
    for (float& p : probs) p /= renorm;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return idx[dist(rng_)];
}

}  // namespace titan
