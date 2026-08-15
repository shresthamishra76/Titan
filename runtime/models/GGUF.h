#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Tensor.h"

namespace titan {

// A single GGUF metadata value. GGUF stores typed key/value metadata; numeric
// values are normalized into `u`/`i`/`f`, strings into `s`, and arrays into
// `arr` (with `arr_type` giving the element type).
struct GgufValue {
  uint32_t type = 0;       // gguf_metadata_value_type
  uint64_t u = 0;          // unsigned / bool
  int64_t i = 0;           // signed
  double f = 0.0;          // float32 / float64
  std::string s;           // string
  uint32_t arr_type = 0;   // element type when type == ARRAY
  std::vector<GgufValue> arr;

  uint64_t to_u64() const;
  double to_f64() const;
};

// Location + type of one tensor's data within the file's data section.
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> dims;  // ggml order (ne[0] is the contiguous dim)
  uint32_t type = 0;           // ggml_type
  uint64_t offset = 0;         // relative to the data section start
};

// RAII read-only memory-mapping of a file (mmap/munmap).
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return static_cast<const uint8_t*>(addr_); }
  std::size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void* addr_ = nullptr;
  std::size_t size_ = 0;
};

// Parses a GGUF file: header, typed metadata, and the tensor-info table, and
// materializes tensors as float32 titan::Tensors (F16 is widened; quantized
// types are rejected in V1 — see issue #8 stretch goal).
//
// See docs/architecture/model-loader.md and the GGUF specification.
class GgufFile {
 public:
  explicit GgufFile(const std::string& path);

  uint32_t version() const { return version_; }
  std::size_t tensor_count() const { return tensor_infos_.size(); }

  bool has(const std::string& key) const;
  const GgufValue& value(const std::string& key) const;  // throws if absent
  uint32_t get_u32(const std::string& key, uint32_t fallback) const;
  float get_f32(const std::string& key, float fallback) const;
  std::string get_str(const std::string& key, const std::string& fallback) const;
  std::vector<std::string> get_str_array(const std::string& key) const;
  std::vector<float> get_f32_array(const std::string& key) const;
  std::vector<int32_t> get_i32_array(const std::string& key) const;

  bool has_tensor(const std::string& name) const;
  const GgufTensorInfo& tensor_info(const std::string& name) const;

  // Materialize a tensor as float32. Shape is the ggml dims reversed, so a
  // ggml [in, out] weight becomes a row-major [out, in] titan::Tensor.
  Tensor tensor(const std::string& name) const;

 private:
  std::unique_ptr<MappedFile> file_;
  uint32_t version_ = 0;
  std::size_t data_start_ = 0;
  std::unordered_map<std::string, GgufValue> metadata_;
  std::unordered_map<std::string, GgufTensorInfo> tensor_infos_;
};

}  // namespace titan
