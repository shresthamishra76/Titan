#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "ThreadPool.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace titan {
namespace {

Shape packed_strides(const Shape& shape) {
  Shape s(shape.size());
  std::size_t acc = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

std::size_t product(const Shape& shape) {
  std::size_t n = 1;
  for (std::size_t d : shape) n *= d;
  return n;
}

Shape broadcast_shape(const Shape& a, const Shape& b) {
  const std::size_t n = std::max(a.size(), b.size());
  Shape out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t ad = i < n - a.size() ? 1 : a[i - (n - a.size())];
    const std::size_t bd = i < n - b.size() ? 1 : b[i - (n - b.size())];
    if (ad != bd && ad != 1 && bd != 1) {
      throw std::invalid_argument("broadcast: incompatible shapes");
    }
    out[i] = std::max(ad, bd);
  }
  return out;
}

// Read t at a broadcasted output index (right-aligned, size-1 dims repeat).
float read_broadcast(const Tensor& t, const Shape& out_idx) {
  const Shape& ts = t.shape();
  const Shape& st = t.strides();
  const std::size_t pad = out_idx.size() - ts.size();
  std::size_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const std::size_t idx = ts[i] == 1 ? 0 : out_idx[i + pad];
    off += idx * st[i];
  }
  return t.data()[off];
}

// Advance a row-major multi-index in place; wraps within `shape`.
void increment_index(Shape& idx, const Shape& shape) {
  for (std::size_t d = shape.size(); d-- > 0;) {
    if (++idx[d] < shape[d]) return;
    idx[d] = 0;
  }
}

}  // namespace

Tensor::Tensor(Shape shape)
    : storage_(std::make_shared<std::vector<float>>(product(shape), 0.0f)),
      shape_(std::move(shape)),
      strides_(packed_strides(shape_)) {}

Tensor::Tensor(Shape shape, std::vector<float> data)
    : storage_(std::make_shared<std::vector<float>>(std::move(data))),
      shape_(std::move(shape)),
      strides_(packed_strides(shape_)) {
  if (storage_->size() != product(shape_)) {
    throw std::invalid_argument("Tensor: data size does not match shape");
  }
}

Tensor Tensor::zeros(Shape shape) { return Tensor(std::move(shape)); }
Tensor Tensor::ones(Shape shape) { return full(std::move(shape), 1.0f); }

Tensor Tensor::full(Shape shape, float value) {
  Tensor t(std::move(shape));
  float* d = t.data();
  for (std::size_t i = 0; i < t.numel(); ++i) d[i] = value;
  return t;
}

std::size_t Tensor::numel() const { return product(shape_); }

bool Tensor::is_contiguous() const { return strides_ == packed_strides(shape_); }

std::size_t Tensor::flat_index(std::initializer_list<std::size_t> index) const {
  if (index.size() != shape_.size()) {
    throw std::out_of_range("Tensor::at: wrong number of indices");
  }
  std::size_t off = offset_;
  std::size_t d = 0;
  for (std::size_t i : index) {
    if (i >= shape_[d]) throw std::out_of_range("Tensor::at: index out of range");
    off += i * strides_[d];
    ++d;
  }
  return off;
}

float& Tensor::at(std::initializer_list<std::size_t> index) {
  return (*storage_)[flat_index(index)];
}
float Tensor::at(std::initializer_list<std::size_t> index) const {
  return (*storage_)[flat_index(index)];
}

float* Tensor::data() { return storage_ ? storage_->data() + offset_ : nullptr; }
const float* Tensor::data() const {
  return storage_ ? storage_->data() + offset_ : nullptr;
}

Tensor Tensor::reshape(Shape shape) const {
  if (product(shape) != numel()) {
    throw std::invalid_argument("Tensor::reshape: numel mismatch");
  }
  if (!is_contiguous()) {
    throw std::invalid_argument("Tensor::reshape: requires a contiguous tensor");
  }
  Tensor out;
  out.storage_ = storage_;
  out.shape_ = std::move(shape);
  out.strides_ = packed_strides(out.shape_);
  out.offset_ = offset_;
  return out;
}

Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
  if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
    throw std::out_of_range("Tensor::transpose: axis out of range");
  }
  Tensor out;
  out.storage_ = storage_;
  out.shape_ = shape_;
  out.strides_ = strides_;
  out.offset_ = offset_;
  std::swap(out.shape_[dim0], out.shape_[dim1]);
  std::swap(out.strides_[dim0], out.strides_[dim1]);
  return out;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;
  Tensor out(shape_);
  float* od = out.data();
  Shape idx(shape_.size(), 0);
  const std::size_t n = numel();
  for (std::size_t linear = 0; linear < n; ++linear) {
    std::size_t src = offset_;
    for (std::size_t d = 0; d < shape_.size(); ++d) src += idx[d] * strides_[d];
    od[linear] = (*storage_)[src];
    increment_index(idx, shape_);
  }
  return out;
}

Tensor Tensor::clone() const {
  Tensor c = contiguous();
  std::vector<float> buf(c.data(), c.data() + c.numel());
  return Tensor(c.shape_, std::move(buf));
}

namespace {
Tensor elementwise(const Tensor& a, const Tensor& b, float (*op)(float, float)) {
  const Shape os = broadcast_shape(a.shape(), b.shape());
  Tensor out(os);
  float* od = out.data();
  Shape idx(os.size(), 0);
  const std::size_t n = out.numel();
  for (std::size_t linear = 0; linear < n; ++linear) {
    od[linear] = op(read_broadcast(a, idx), read_broadcast(b, idx));
    increment_index(idx, os);
  }
  return out;
}
}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  return elementwise(a, b, [](float x, float y) { return x + y; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
  return elementwise(a, b, [](float x, float y) { return x * y; });
}

Tensor scale(const Tensor& a, float s) {
  const Tensor in = a.contiguous();
  Tensor out(in.shape());
  const float* id = in.data();
  float* od = out.data();
  for (std::size_t i = 0; i < out.numel(); ++i) od[i] = id[i] * s;
  return out;
}

Tensor matmul(const Tensor& A, const Tensor& B) {
  if (A.ndim() != 2 || B.ndim() != 2) {
    throw std::invalid_argument("matmul: expects 2D tensors");
  }
  if (A.shape()[1] != B.shape()[0]) {
    throw std::invalid_argument("matmul: inner dimension mismatch");
  }
  const Tensor a = A.contiguous();
  const Tensor b = B.contiguous();
  const std::size_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  Tensor out(Shape{m, n});  // zero-initialized; the kernel accumulates into it
  const float* ad = a.data();
  const float* bd = b.data();
  float* od = out.data();

  // Cache-friendly ikj ordering: stream B and the output row sequentially and
  // accumulate, rather than the pointer-chasing ijk dot product. NEON
  // vectorizes the inner j loop (Milestone 2).
  auto compute_rows = [&](std::size_t i0, std::size_t i1) {
    for (std::size_t i = i0; i < i1; ++i) {
      float* orow = od + i * n;
      for (std::size_t p = 0; p < k; ++p) {
        const float aip = ad[i * k + p];
        const float* brow = bd + p * n;
        std::size_t j = 0;
#if defined(__ARM_NEON)
        const float32x4_t va = vdupq_n_f32(aip);
        for (; j + 4 <= n; j += 4) {
          float32x4_t acc = vld1q_f32(orow + j);
          const float32x4_t vb = vld1q_f32(brow + j);
          acc = vfmaq_f32(acc, va, vb);
          vst1q_f32(orow + j, acc);
        }
#endif
        for (; j < n; ++j) orow[j] += aip * brow[j];
      }
    }
  };

  // Parallelize only when the work is large enough to amortize threading.
  if (m >= 64 && m * n * k >= (1u << 18)) {
    global_pool().parallel_for(m, compute_rows);
  } else {
    compute_rows(0, m);
  }
  return out;
}

Tensor softmax(const Tensor& x, int axis) {
  const Tensor xc = x.contiguous();
  const Shape& sh = xc.shape();
  const int nd = static_cast<int>(sh.size());
  if (nd == 0) throw std::invalid_argument("softmax: scalar tensor");
  const int ax = axis < 0 ? nd + axis : axis;
  if (ax < 0 || ax >= nd) throw std::out_of_range("softmax: axis out of range");
  Tensor out(sh);
  const float* in = xc.data();
  float* od = out.data();
  const std::size_t axis_len = sh[ax];
  std::size_t inner = 1;
  for (int d = ax + 1; d < nd; ++d) inner *= sh[d];
  std::size_t outer = 1;
  for (int d = 0; d < ax; ++d) outer *= sh[d];
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t s = 0; s < inner; ++s) {
      const std::size_t base = o * axis_len * inner + s;
      float mx = -std::numeric_limits<float>::infinity();
      for (std::size_t a = 0; a < axis_len; ++a) {
        mx = std::max(mx, in[base + a * inner]);
      }
      float sum = 0.0f;
      for (std::size_t a = 0; a < axis_len; ++a) {
        const float e = std::exp(in[base + a * inner] - mx);
        od[base + a * inner] = e;
        sum += e;
      }
      for (std::size_t a = 0; a < axis_len; ++a) od[base + a * inner] /= sum;
    }
  }
  return out;
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  const Tensor xc = x.contiguous();
  const Shape& sh = xc.shape();
  if (sh.empty()) throw std::invalid_argument("rms_norm: scalar tensor");
  const std::size_t dim = sh.back();
  if (weight.numel() != dim) {
    throw std::invalid_argument("rms_norm: weight size does not match last dim");
  }
  const Tensor wc = weight.contiguous();
  Tensor out(sh);
  const float* in = xc.data();
  const float* w = wc.data();
  float* od = out.data();
  const std::size_t rows = xc.numel() / dim;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = in + r * dim;
    float* orow = od + r * dim;
    float ss = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) ss += xr[i] * xr[i];
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
    for (std::size_t i = 0; i < dim; ++i) orow[i] = xr[i] * inv * w[i];
  }
  return out;
}

Tensor silu(const Tensor& x) {
  const Tensor xc = x.contiguous();
  Tensor out(xc.shape());
  const float* in = xc.data();
  float* od = out.data();
  for (std::size_t i = 0; i < out.numel(); ++i) {
    const float v = in[i];
    od[i] = v / (1.0f + std::exp(-v));
  }
  return out;
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
  if (gate.shape() != up.shape()) {
    throw std::invalid_argument("swiglu: gate and up shapes differ");
  }
  const Tensor g = gate.contiguous();
  const Tensor u = up.contiguous();
  Tensor out(g.shape());
  const float* gd = g.data();
  const float* ud = u.data();
  float* od = out.data();
  for (std::size_t i = 0; i < out.numel(); ++i) {
    const float v = gd[i];
    od[i] = (v / (1.0f + std::exp(-v))) * ud[i];
  }
  return out;
}

}  // namespace titan
