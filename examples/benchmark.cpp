// Titan matmul benchmark (Milestone 2, issue #16).
//
// Establishes the CPU matmul baseline and measures the optimized kernel
// (cache-friendly ikj ordering + NEON + thread pool) against a naive ijk
// reference. Matmul dominates transformer inference cost, so this is the
// headline Milestone 2 number.
//
//   titan_benchmark [N]      square N x N x N matmul (default 512)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Tensor.h"

using titan::Tensor;

namespace {

Tensor NaiveMatmul(const Tensor& a, const Tensor& b) {
  const std::size_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  Tensor out({m, n});
  const float* ad = a.data();
  const float* bd = b.data();
  float* od = out.data();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::size_t p = 0; p < k; ++p) acc += ad[i * k + p] * bd[p * n + j];
      od[i * n + j] = acc;
    }
  }
  return out;
}

template <typename Fn>
double Bench(const char* name, double flops, Fn fn) {
  fn();  // warmup
  constexpr int kIters = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < kIters; ++it) {
    volatile float sink = fn().data()[0];
    (void)sink;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count() / kIters;
  std::printf("  %-16s %9.2f ms   %7.2f GFLOP/s\n", name, sec * 1e3, flops / sec / 1e9);
  return sec;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t N = argc > 1 ? std::stoul(argv[1]) : 512;
  std::vector<float> ad(N * N), bd(N * N);
  for (std::size_t i = 0; i < ad.size(); ++i) {
    ad[i] = std::sin(0.001 * static_cast<double>(i));
    bd[i] = std::cos(0.001 * static_cast<double>(i));
  }
  const Tensor a({N, N}, ad), b({N, N}, bd);
  const double flops = 2.0 * N * N * N;

  std::printf("matmul %zux%zux%zu (float32), median of 5:\n", N, N, N);
  const double naive = Bench("naive ijk", flops, [&] { return NaiveMatmul(a, b); });
  const double opt = Bench("titan matmul", flops, [&] { return titan::matmul(a, b); });
  std::printf("  speedup: %.2fx\n", naive / opt);
  return 0;
}
