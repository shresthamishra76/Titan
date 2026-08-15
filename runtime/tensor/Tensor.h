#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace titan {

// Row-major shape / stride vectors.
using Shape = std::vector<std::size_t>;

// N-dimensional array over a contiguous, row-major float32 buffer.
//
// Storage is reference-counted (shared_ptr): shape-manipulation ops
// (reshape / transpose) return cheap *views* that share the underlying
// buffer, NumPy-style. Use clone() for an independent deep copy, or
// contiguous() to materialize a packed buffer from a non-contiguous view.
//
// V1 is correctness-first and CPU-only; Milestone 2 optimizes the hot ops.
// See docs/architecture/tensor.md.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(Shape shape);                  // zero-filled, owns storage
  Tensor(Shape shape, std::vector<float> data);  // takes ownership of data

  static Tensor zeros(Shape shape);
  static Tensor ones(Shape shape);
  static Tensor full(Shape shape, float value);

  const Shape& shape() const { return shape_; }
  const Shape& strides() const { return strides_; }
  std::size_t numel() const;
  std::size_t ndim() const { return shape_.size(); }
  bool is_contiguous() const;

  // Multi-index element access (row-major). Bounds-checked.
  float& at(std::initializer_list<std::size_t> index);
  float at(std::initializer_list<std::size_t> index) const;

  // Raw pointer to the first logical element (includes any view offset).
  float* data();
  const float* data() const;

  Tensor reshape(Shape shape) const;                           // needs contiguous
  Tensor transpose(std::size_t dim0, std::size_t dim1) const;  // stride-swap view
  Tensor contiguous() const;                                   // packed copy or self
  Tensor clone() const;                                        // independent deep copy

 private:
  std::size_t flat_index(std::initializer_list<std::size_t> index) const;

  std::shared_ptr<std::vector<float>> storage_;
  Shape shape_;
  Shape strides_;
  std::size_t offset_ = 0;
};

// Element-wise ops with NumPy-style (right-aligned) broadcasting.
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float s);

// 2D matrix multiply: (m, k) x (k, n) -> (m, n).
Tensor matmul(const Tensor& a, const Tensor& b);

// Neural ops (Llama operator set).
Tensor softmax(const Tensor& x, int axis = -1);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);
Tensor silu(const Tensor& x);
Tensor swiglu(const Tensor& gate, const Tensor& up);

}  // namespace titan
