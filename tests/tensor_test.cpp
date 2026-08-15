#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Tensor.h"
#include "ThreadPool.h"

using titan::Tensor;

TEST(TensorTest, NumelComputesProductOfShape) {
  Tensor t({2, 3, 4});
  EXPECT_EQ(t.numel(), 24u);
}

TEST(TensorTest, DefaultConstructedTensorIsEmpty) {
  Tensor t;
  EXPECT_TRUE(t.shape().empty());
  EXPECT_EQ(t.numel(), 1u);
}

TEST(TensorTest, StridesAreRowMajor) {
  Tensor t({2, 3, 4});
  ASSERT_EQ(t.strides().size(), 3u);
  EXPECT_EQ(t.strides()[0], 12u);
  EXPECT_EQ(t.strides()[1], 4u);
  EXPECT_EQ(t.strides()[2], 1u);
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, Factories) {
  EXPECT_FLOAT_EQ(Tensor::zeros({2, 2}).at({1, 1}), 0.0f);
  EXPECT_FLOAT_EQ(Tensor::ones({2, 2}).at({0, 1}), 1.0f);
  EXPECT_FLOAT_EQ(Tensor::full({3}, 4.5f).at({2}), 4.5f);
}

TEST(TensorTest, DataConstructorAndAccess) {
  Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
  t.at({1, 0}) = 40.0f;
  EXPECT_FLOAT_EQ(t.at({1, 0}), 40.0f);
}

TEST(TensorTest, DataConstructorRejectsWrongSize) {
  EXPECT_THROW(Tensor({2, 2}, {1, 2, 3}), std::invalid_argument);
}

TEST(TensorTest, AtRejectsOutOfRange) {
  Tensor t({2, 2});
  EXPECT_THROW(t.at({2, 0}), std::out_of_range);
  EXPECT_THROW(t.at({0}), std::out_of_range);
}

TEST(TensorTest, Reshape) {
  Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor r = t.reshape({3, 2});
  EXPECT_EQ(r.shape(), (std::vector<std::size_t>{3, 2}));
  EXPECT_FLOAT_EQ(r.at({2, 1}), 6.0f);
  EXPECT_THROW(t.reshape({4, 2}), std::invalid_argument);
}

TEST(TensorTest, TransposeIsAViewMadeContiguous) {
  Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor tt = t.transpose(0, 1);
  EXPECT_EQ(tt.shape(), (std::vector<std::size_t>{3, 2}));
  EXPECT_FALSE(tt.is_contiguous());
  EXPECT_FLOAT_EQ(tt.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(tt.at({0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(tt.at({2, 1}), 6.0f);
  Tensor c = tt.contiguous();
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_FLOAT_EQ(c.at({0, 1}), 4.0f);
}

TEST(TensorTest, AddBroadcastsRowVector) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b({3}, {10, 20, 30});
  Tensor c = titan::add(a, b);
  EXPECT_FLOAT_EQ(c.at({0, 0}), 11.0f);
  EXPECT_FLOAT_EQ(c.at({1, 2}), 36.0f);
}

TEST(TensorTest, MulBroadcastsColumnVector) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b({2, 1}, {10, 100});
  Tensor c = titan::mul(a, b);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 20.0f);
  EXPECT_FLOAT_EQ(c.at({1, 2}), 600.0f);
}

TEST(TensorTest, IncompatibleBroadcastThrows) {
  Tensor a({2, 3});
  Tensor b({4});
  EXPECT_THROW(titan::add(a, b), std::invalid_argument);
}

TEST(TensorTest, Scale) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor c = titan::scale(a, 2.0f);
  EXPECT_FLOAT_EQ(c.at({1, 1}), 8.0f);
}

TEST(TensorTest, MatmulHandComputed) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
  Tensor c = titan::matmul(a, b);
  ASSERT_EQ(c.shape(), (std::vector<std::size_t>{2, 2}));
  EXPECT_FLOAT_EQ(c.at({0, 0}), 58.0f);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 64.0f);
  EXPECT_FLOAT_EQ(c.at({1, 0}), 139.0f);
  EXPECT_FLOAT_EQ(c.at({1, 1}), 154.0f);
}

TEST(TensorTest, MatmulIdentity) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor id({2, 2}, {1, 0, 0, 1});
  Tensor c = titan::matmul(a, id);
  EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(c.at({1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(c.at({1, 1}), 4.0f);
}

TEST(TensorTest, MatmulWithTransposedView) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b({2, 3}, {1, 0, 1, 0, 1, 0});
  Tensor c = titan::matmul(a, b.transpose(0, 1));
  ASSERT_EQ(c.shape(), (std::vector<std::size_t>{2, 2}));
  EXPECT_FLOAT_EQ(c.at({0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(c.at({0, 1}), 2.0f);
}

TEST(TensorTest, MatmulInnerMismatchThrows) {
  Tensor a({2, 3});
  Tensor b({2, 2});
  EXPECT_THROW(titan::matmul(a, b), std::invalid_argument);
}

TEST(TensorTest, SoftmaxSumsToOne) {
  Tensor x({2, 3}, {1, 2, 3, 1, 1, 1});
  Tensor s = titan::softmax(x);
  const float row0 = s.at({0, 0}) + s.at({0, 1}) + s.at({0, 2});
  const float row1 = s.at({1, 0}) + s.at({1, 1}) + s.at({1, 2});
  EXPECT_NEAR(row0, 1.0f, 1e-6);
  EXPECT_NEAR(row1, 1.0f, 1e-6);
  EXPECT_NEAR(s.at({1, 0}), 1.0f / 3.0f, 1e-6);
}

TEST(TensorTest, SoftmaxStableForLargeLogits) {
  Tensor x({1, 2}, {1000.0f, 1000.0f});
  Tensor s = titan::softmax(x);
  EXPECT_NEAR(s.at({0, 0}), 0.5f, 1e-6);
  EXPECT_NEAR(s.at({0, 1}), 0.5f, 1e-6);
}

TEST(TensorTest, RmsNorm) {
  Tensor x({1, 2}, {3.0f, 4.0f});
  Tensor w({2}, {1.0f, 1.0f});
  Tensor y = titan::rms_norm(x, w, 0.0f);
  const float rms = std::sqrt((9.0f + 16.0f) / 2.0f);
  EXPECT_NEAR(y.at({0, 0}), 3.0f / rms, 1e-5);
  EXPECT_NEAR(y.at({0, 1}), 4.0f / rms, 1e-5);
}

TEST(TensorTest, Silu) {
  Tensor x({3}, {0.0f, 1.0f, -1.0f});
  Tensor y = titan::silu(x);
  EXPECT_NEAR(y.at({0}), 0.0f, 1e-6);
  EXPECT_NEAR(y.at({1}), 1.0f / (1.0f + std::exp(-1.0f)), 1e-6);
  EXPECT_NEAR(y.at({2}), -1.0f / (1.0f + std::exp(1.0f)), 1e-6);
}

TEST(TensorTest, Swiglu) {
  Tensor gate({2}, {1.0f, 2.0f});
  Tensor up({2}, {3.0f, 4.0f});
  Tensor y = titan::swiglu(gate, up);
  EXPECT_NEAR(y.at({0}), (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f, 1e-6);
  EXPECT_NEAR(y.at({1}), (2.0f / (1.0f + std::exp(-2.0f))) * 4.0f, 1e-6);
}

TEST(TensorTest, CloneIsIndependent) {
  Tensor a({2}, {1.0f, 2.0f});
  Tensor b = a.clone();
  b.at({0}) = 99.0f;
  EXPECT_FLOAT_EQ(a.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(b.at({0}), 99.0f);
}

// Milestone 2: the optimized matmul (cache-friendly + NEON + threaded above a
// size threshold) must match a naive reference. 70x70x70 triggers the parallel
// path.
TEST(TensorTest, MatmulLargeMatchesNaiveReference) {
  constexpr std::size_t M = 70, K = 70, N = 70;
  std::vector<float> ad(M * K), bd(K * N);
  for (std::size_t i = 0; i < ad.size(); ++i) ad[i] = std::sin(0.1f * static_cast<float>(i));
  for (std::size_t i = 0; i < bd.size(); ++i) bd[i] = std::cos(0.07f * static_cast<float>(i));
  Tensor a({M, K}, ad), b({K, N}, bd);
  Tensor c = titan::matmul(a, b);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t p = 0; p < K; ++p) acc += ad[i * K + p] * bd[p * N + j];
      EXPECT_NEAR(c.at({i, j}), acc, 1e-3);
    }
  }
}

TEST(ThreadPoolTest, ParallelForCoversEveryIndexExactlyOnce) {
  titan::ThreadPool pool(3);
  std::vector<int> hits(1000, 0);
  pool.parallel_for(1000, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) hits[i] += 1;
  });
  for (int h : hits) EXPECT_EQ(h, 1);
}

TEST(ThreadPoolTest, ParallelForAccumulates) {
  titan::ThreadPool pool(4);
  std::atomic<long> sum{0};
  pool.parallel_for(10000, [&](std::size_t b, std::size_t e) {
    long s = 0;
    for (std::size_t i = b; i < e; ++i) s += static_cast<long>(i);
    sum += s;
  });
  EXPECT_EQ(sum.load(), 10000L * 9999 / 2);
}
