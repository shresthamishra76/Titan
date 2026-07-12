#include "Tensor.h"

#include <numeric>

namespace titan {

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)) {}

std::size_t Tensor::numel() const {
  return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                          std::multiplies<>());
}

}  // namespace titan
