#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "Model.h"

using titan::Model;
using titan::ModelConfig;
using titan::ModelWeights;
using titan::LayerWeights;
using titan::Tensor;

TEST(RopeTest, PositionZeroIsIdentity) {
  Tensor x({1, 4}, {1, 2, 3, 4});
  titan::apply_rope(x, /*n_heads=*/1, /*head_dim=*/4, /*theta=*/10000.0f);
  EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(x.at({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(x.at({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(x.at({0, 3}), 4.0f);
}

TEST(RopeTest, PreservesPairNorm) {
  Tensor x({2, 2}, {1, 0, 1, 0});
  titan::apply_rope(x, 1, 2, 10000.0f);
  EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);  // position 0 unchanged
  EXPECT_FLOAT_EQ(x.at({0, 1}), 0.0f);
  const float n = std::sqrt(x.at({1, 0}) * x.at({1, 0}) + x.at({1, 1}) * x.at({1, 1}));
  EXPECT_NEAR(n, 1.0f, 1e-6);  // rotation preserves norm
}

TEST(AttentionTest, FirstPositionAttendsOnlyToItself) {
  Tensor q({2, 2}, {1, 1, 1, 1});
  Tensor k({2, 2}, {1, 1, 1, 1});
  Tensor v({2, 2}, {5, 6, 7, 8});
  Tensor out = titan::attention(q, k, v, /*n_head=*/1, /*n_head_kv=*/1, /*head_dim=*/2);
  ASSERT_EQ(out.shape(), (std::vector<std::size_t>{2, 2}));
  EXPECT_FLOAT_EQ(out.at({0, 0}), 5.0f);  // only key 0 is visible
  EXPECT_FLOAT_EQ(out.at({0, 1}), 6.0f);
}

TEST(AttentionTest, GqaOutputShape) {
  Tensor q = Tensor::ones({3, 8});  // 4 heads * head_dim 2
  Tensor k = Tensor::ones({3, 4});  // 2 kv heads * head_dim 2
  Tensor v = Tensor::ones({3, 4});
  Tensor out = titan::attention(q, k, v, 4, 2, 2);
  EXPECT_EQ(out.shape(), (std::vector<std::size_t>{3, 8}));
}

namespace {

// A degenerate 1-layer model whose attention and FFN contribute zero (all
// projection weights are zero), so forward() reduces to
// output_identity @ rms_norm(embedding). Chosen so the argmax is predictable.
Model BuildTinyModel() {
  ModelConfig cfg;
  cfg.n_layers = 1;
  cfg.n_embd = 4;
  cfg.n_head = 1;
  cfg.n_head_kv = 1;
  cfg.head_dim = 4;
  cfg.n_ff = 4;
  cfg.n_vocab = 4;
  cfg.context_length = 8;
  cfg.rope_theta = 10000.0f;
  cfg.rms_eps = 1e-6f;

  ModelWeights w;
  w.token_embd = Tensor({4, 4}, {0, 0, 0, 10,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0});
  w.output_norm = Tensor::ones({4});
  w.output = Tensor({4, 4}, {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1});

  LayerWeights L;
  L.attn_norm = Tensor::ones({4});
  L.wq = Tensor::zeros({4, 4});
  L.wk = Tensor::zeros({4, 4});
  L.wv = Tensor::zeros({4, 4});
  L.wo = Tensor::zeros({4, 4});
  L.ffn_norm = Tensor::ones({4});
  L.w_gate = Tensor::zeros({4, 4});
  L.w_up = Tensor::zeros({4, 4});
  L.w_down = Tensor::zeros({4, 4});
  w.layers.push_back(std::move(L));

  return Model(std::move(cfg), std::move(w));
}

}  // namespace

TEST(ModelTest, ForwardProducesExpectedLogits) {
  Model m = BuildTinyModel();
  Tensor logits = m.forward({0});
  ASSERT_EQ(logits.shape(), (std::vector<std::size_t>{4}));
  EXPECT_NEAR(logits.at({0}), 0.0f, 1e-4);
  EXPECT_NEAR(logits.at({1}), 0.0f, 1e-4);
  EXPECT_NEAR(logits.at({2}), 0.0f, 1e-4);
  EXPECT_NEAR(logits.at({3}), 2.0f, 1e-4);  // 10 / sqrt(100/4)
}

TEST(ModelTest, GreedyGenerationIsDeterministic) {
  Model m = BuildTinyModel();
  EXPECT_EQ(m.generate({0}, /*max_new_tokens=*/2, /*eos_id=*/-1),
            (std::vector<int32_t>{3, 0}));
}

TEST(ModelTest, GenerationStopsAtEos) {
  Model m = BuildTinyModel();
  const std::vector<int32_t> out = m.generate({0}, 5, /*eos_id=*/3);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out.front(), 3);
}
