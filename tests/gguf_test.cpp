#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "GGUF.h"

namespace {

// Minimal little-endian GGUF writer for building a fixture in tests.
struct GgufWriter {
  std::vector<uint8_t> buf;

  template <typename T>
  void put(T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
  }
  void put_str(const std::string& s) {
    put<uint64_t>(s.size());
    buf.insert(buf.end(), s.begin(), s.end());
  }
  void pad_to(std::size_t alignment) {
    while (buf.size() % alignment != 0) buf.push_back(0);
  }
};

std::string WriteFixture() {
  GgufWriter w;
  w.buf.insert(w.buf.end(), {'G', 'G', 'U', 'F'});
  w.put<uint32_t>(3);   // version
  w.put<uint64_t>(2);   // tensor count
  w.put<uint64_t>(2);   // metadata kv count

  // metadata: general.alignment (u32) = 32
  w.put_str("general.alignment");
  w.put<uint32_t>(4);   // U32
  w.put<uint32_t>(32);
  // metadata: answer (i32) = 42
  w.put_str("answer");
  w.put<uint32_t>(5);   // I32
  w.put<int32_t>(42);

  // tensor info: "w", F32, ggml dims [3, 2] -> titan shape [2, 3], offset 0
  w.put_str("w");
  w.put<uint32_t>(2);
  w.put<uint64_t>(3);
  w.put<uint64_t>(2);
  w.put<uint32_t>(0);   // ggml F32
  w.put<uint64_t>(0);
  // tensor info: "h", F16, dims [2], offset 32
  w.put_str("h");
  w.put<uint32_t>(1);
  w.put<uint64_t>(2);
  w.put<uint32_t>(1);   // ggml F16
  w.put<uint64_t>(32);

  // data section: aligned to 32
  w.pad_to(32);
  for (float v : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) w.put<float>(v);  // "w" (24 bytes)
  w.pad_to(32);             // advance to data-section offset 32 for "h"
  w.put<uint16_t>(0x3C00);  // half 1.0
  w.put<uint16_t>(0x4000);  // half 2.0

  const std::string path =
      (std::filesystem::temp_directory_path() / "titan_fixture.gguf").string();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(w.buf.data()), w.buf.size());
  out.close();
  return path;
}

}  // namespace

TEST(GgufTest, ParsesHeaderAndMetadata) {
  titan::GgufFile gguf(WriteFixture());
  EXPECT_EQ(gguf.version(), 3u);
  EXPECT_EQ(gguf.tensor_count(), 2u);
  EXPECT_EQ(gguf.get_u32("general.alignment", 0), 32u);
  EXPECT_EQ(gguf.value("answer").to_u64(), 42u);
  EXPECT_FALSE(gguf.has("missing"));
}

TEST(GgufTest, LoadsF32TensorWithReversedDims) {
  titan::GgufFile gguf(WriteFixture());
  ASSERT_TRUE(gguf.has_tensor("w"));
  titan::Tensor w = gguf.tensor("w");
  EXPECT_EQ(w.shape(), (std::vector<std::size_t>{2, 3}));
  EXPECT_FLOAT_EQ(w.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(w.at({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(w.at({1, 2}), 6.0f);
}

TEST(GgufTest, WidensF16Tensor) {
  titan::GgufFile gguf(WriteFixture());
  titan::Tensor h = gguf.tensor("h");
  EXPECT_EQ(h.shape(), (std::vector<std::size_t>{2}));
  EXPECT_FLOAT_EQ(h.at({0}), 1.0f);
  EXPECT_FLOAT_EQ(h.at({1}), 2.0f);
}

TEST(GgufTest, MissingTensorThrows) {
  titan::GgufFile gguf(WriteFixture());
  EXPECT_THROW(gguf.tensor("nope"), std::runtime_error);
}
