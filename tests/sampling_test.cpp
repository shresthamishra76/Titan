#include <gtest/gtest.h>

#include <vector>

#include "Sampler.h"

using titan::Sampler;
using titan::SamplingParams;
using titan::Tensor;

TEST(SamplingTest, Argmax) {
  Tensor logits({4}, {1, 3, 2, 0});
  EXPECT_EQ(Sampler::argmax(logits), 1);
}

TEST(SamplingTest, TemperatureZeroIsGreedy) {
  SamplingParams p;
  p.temperature = 0.0f;
  Sampler s(p);
  Tensor logits({4}, {1, 3, 2, 5});
  EXPECT_EQ(s.sample(logits), 3);
}

TEST(SamplingTest, TopKOneAlwaysArgmax) {
  SamplingParams p;
  p.top_k = 1;
  p.temperature = 1.0f;
  p.seed = 123;
  Sampler s(p);
  Tensor logits({4}, {1, 2, 9, 0});
  for (int i = 0; i < 10; ++i) EXPECT_EQ(s.sample(logits), 2);
}

TEST(SamplingTest, DeterministicWithSeed) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 42;
  Tensor logits({5}, {0.5f, 1.0f, 0.2f, 2.0f, 0.1f});
  Sampler a(p), b(p);
  for (int i = 0; i < 20; ++i) EXPECT_EQ(a.sample(logits), b.sample(logits));
}

TEST(SamplingTest, RepetitionPenaltyDemotesSeenToken) {
  SamplingParams p;
  p.temperature = 0.0f;               // greedy over penalized logits
  p.repetition_penalty = 10.0f;
  Sampler s(p);
  Tensor logits({2}, {5.0f, 4.0f});   // argmax is 0...
  EXPECT_EQ(s.sample(logits, {0}), 1);  // ...until token 0 is penalized (0.5 < 4)
}

TEST(SamplingTest, PeakedDistributionFavorsPeak) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 7;
  Sampler s(p);
  Tensor logits({3}, {0.0f, 12.0f, 0.0f});  // token 1 dominates
  int count1 = 0;
  for (int i = 0; i < 200; ++i) {
    if (s.sample(logits) == 1) ++count1;
  }
  EXPECT_GT(count1, 190);
}
