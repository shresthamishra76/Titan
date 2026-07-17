#include "GGUF.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace titan {
namespace {

// GGUF metadata value type tags.
enum : uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5,
  kF32 = 6, kBool = 7, kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
};

// ggml tensor data types we can materialize in V1.
enum : uint32_t { kGgmlF32 = 0, kGgmlF16 = 1 };

// Little-endian cursor over the mapped bytes with bounds checks.
class Cursor {
 public:
  Cursor(const uint8_t* base, std::size_t size) : base_(base), size_(size) {}

  std::size_t pos() const { return pos_; }

  template <typename T>
  T read() {
    T v;
    need(sizeof(T));
    std::memcpy(&v, base_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }

  std::string read_string() {
    const uint64_t len = read<uint64_t>();
    need(len);
    std::string s(reinterpret_cast<const char*>(base_ + pos_), len);
    pos_ += len;
    return s;
  }

 private:
  void need(std::size_t n) const {
    if (pos_ + n > size_) throw std::runtime_error("GGUF: unexpected end of file");
  }
  const uint8_t* base_;
  std::size_t size_;
  std::size_t pos_ = 0;
};

GgufValue read_typed(Cursor& c, uint32_t type) {
  GgufValue v;
  v.type = type;
  switch (type) {
    case kU8: v.u = c.read<uint8_t>(); break;
    case kI8: v.i = c.read<int8_t>(); break;
    case kU16: v.u = c.read<uint16_t>(); break;
    case kI16: v.i = c.read<int16_t>(); break;
    case kU32: v.u = c.read<uint32_t>(); break;
    case kI32: v.i = c.read<int32_t>(); break;
    case kF32: v.f = c.read<float>(); break;
    case kBool: v.u = c.read<uint8_t>(); break;
    case kString: v.s = c.read_string(); break;
    case kU64: v.u = c.read<uint64_t>(); break;
    case kI64: v.i = c.read<int64_t>(); break;
    case kF64: v.f = c.read<double>(); break;
    case kArray: {
      v.arr_type = c.read<uint32_t>();
      const uint64_t n = c.read<uint64_t>();
      v.arr.reserve(n);
      for (uint64_t k = 0; k < n; ++k) v.arr.push_back(read_typed(c, v.arr_type));
      break;
    }
    default: throw std::runtime_error("GGUF: unknown metadata value type");
  }
  return v;
}

float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FF;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

}  // namespace

uint64_t GgufValue::to_u64() const {
  switch (type) {
    case kU8: case kU16: case kU32: case kBool: case kU64: return u;
    case kI8: case kI16: case kI32: case kI64: return static_cast<uint64_t>(i);
    case kF32: case kF64: return static_cast<uint64_t>(f);
    default: return 0;
  }
}

double GgufValue::to_f64() const {
  switch (type) {
    case kF32: case kF64: return f;
    case kI8: case kI16: case kI32: case kI64: return static_cast<double>(i);
    default: return static_cast<double>(u);
  }
}

MappedFile::MappedFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("MappedFile: cannot open " + path);
  struct stat st{};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("MappedFile: cannot stat " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  addr_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (addr_ == MAP_FAILED) {
    ::close(fd_);
    throw std::runtime_error("MappedFile: mmap failed for " + path);
  }
}

MappedFile::~MappedFile() {
  if (addr_ && addr_ != MAP_FAILED) ::munmap(addr_, size_);
  if (fd_ >= 0) ::close(fd_);
}

GgufFile::GgufFile(const std::string& path)
    : file_(std::make_unique<MappedFile>(path)) {
  Cursor c(file_->data(), file_->size());
  const uint32_t magic = c.read<uint32_t>();
  if (magic != 0x46554747u /* "GGUF" */) throw std::runtime_error("GGUF: bad magic");
  version_ = c.read<uint32_t>();
  if (version_ < 2 || version_ > 3) throw std::runtime_error("GGUF: unsupported version");
  const uint64_t n_tensors = c.read<uint64_t>();
  const uint64_t n_kv = c.read<uint64_t>();

  for (uint64_t k = 0; k < n_kv; ++k) {
    std::string key = c.read_string();
    const uint32_t type = c.read<uint32_t>();
    metadata_.emplace(std::move(key), read_typed(c, type));
  }

  for (uint64_t t = 0; t < n_tensors; ++t) {
    GgufTensorInfo info;
    info.name = c.read_string();
    const uint32_t ndim = c.read<uint32_t>();
    info.dims.resize(ndim);
    for (uint32_t d = 0; d < ndim; ++d) info.dims[d] = c.read<uint64_t>();
    info.type = c.read<uint32_t>();
    info.offset = c.read<uint64_t>();
    tensor_infos_.emplace(info.name, std::move(info));
  }

  const uint32_t alignment = get_u32("general.alignment", 32);
  const std::size_t here = c.pos();
  data_start_ = (here + alignment - 1) / alignment * alignment;
}

bool GgufFile::has(const std::string& key) const { return metadata_.count(key) > 0; }

const GgufValue& GgufFile::value(const std::string& key) const {
  const auto it = metadata_.find(key);
  if (it == metadata_.end()) throw std::runtime_error("GGUF: missing metadata key " + key);
  return it->second;
}

uint32_t GgufFile::get_u32(const std::string& key, uint32_t fallback) const {
  return has(key) ? static_cast<uint32_t>(value(key).to_u64()) : fallback;
}
float GgufFile::get_f32(const std::string& key, float fallback) const {
  return has(key) ? static_cast<float>(value(key).to_f64()) : fallback;
}
std::string GgufFile::get_str(const std::string& key, const std::string& fallback) const {
  return has(key) ? value(key).s : fallback;
}

std::vector<std::string> GgufFile::get_str_array(const std::string& key) const {
  std::vector<std::string> out;
  if (!has(key)) return out;
  for (const GgufValue& e : value(key).arr) out.push_back(e.s);
  return out;
}
std::vector<float> GgufFile::get_f32_array(const std::string& key) const {
  std::vector<float> out;
  if (!has(key)) return out;
  for (const GgufValue& e : value(key).arr) out.push_back(static_cast<float>(e.to_f64()));
  return out;
}
std::vector<int32_t> GgufFile::get_i32_array(const std::string& key) const {
  std::vector<int32_t> out;
  if (!has(key)) return out;
  for (const GgufValue& e : value(key).arr) out.push_back(static_cast<int32_t>(e.to_u64()));
  return out;
}

bool GgufFile::has_tensor(const std::string& name) const {
  return tensor_infos_.count(name) > 0;
}

const GgufTensorInfo& GgufFile::tensor_info(const std::string& name) const {
  const auto it = tensor_infos_.find(name);
  if (it == tensor_infos_.end()) throw std::runtime_error("GGUF: missing tensor " + name);
  return it->second;
}

Tensor GgufFile::tensor(const std::string& name) const {
  const GgufTensorInfo& info = tensor_info(name);

  std::size_t numel = 1;
  for (const uint64_t d : info.dims) numel *= static_cast<std::size_t>(d);

  // ggml dims are reversed relative to row-major titan shape.
  Shape shape(info.dims.rbegin(), info.dims.rend());

  std::size_t elem_size = 0;
  if (info.type == kGgmlF32) {
    elem_size = 4;
  } else if (info.type == kGgmlF16) {
    elem_size = 2;
  } else {
    throw std::runtime_error(
        "GGUF: tensor '" + name +
        "' uses an unsupported (quantized) type; V1 supports F16/F32 only "
        "(see issue #8 stretch goal: quantized dequantization)");
  }

  const std::size_t byte_off = data_start_ + info.offset;
  if (byte_off + numel * elem_size > file_->size()) {
    throw std::runtime_error("GGUF: tensor '" + name + "' data out of bounds");
  }
  const uint8_t* src = file_->data() + byte_off;

  std::vector<float> data(numel);
  if (info.type == kGgmlF32) {
    std::memcpy(data.data(), src, numel * sizeof(float));
  } else {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(src);
    for (std::size_t k = 0; k < numel; ++k) data[k] = half_to_float(h[k]);
  }
  return Tensor(std::move(shape), std::move(data));
}

}  // namespace titan
