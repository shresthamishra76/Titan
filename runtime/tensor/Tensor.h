#pragma once

#include <cstddef>
#include <vector>

namespace titan {

// Placeholder scaffold for the Tensor abstraction described in
// docs/architecture (Milestone 1). Real implementation — memory layout,
// matmul, broadcasting, reshape/transpose, softmax/layernorm/gelu — lands
// via Milestone 1 issues, not here.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<std::size_t> shape);

  const std::vector<std::size_t>& shape() const { return shape_; }
  std::size_t numel() const;

 private:
  std::vector<std::size_t> shape_;
};

}  // namespace titan
