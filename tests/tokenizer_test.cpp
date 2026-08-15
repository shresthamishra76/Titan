#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Tokenizer.h"

using titan::Tokenizer;
using titan::TokenizerConfig;
using titan::TokenType;

namespace {

// U+2581 space marker, as used inside vocab pieces.
const std::string kSM = "\xe2\x96\x81";

// A tiny hand-built vocab exercising merges, byte fallback, and the marker.
TokenizerConfig MakeConfig(bool dummy_prefix, bool add_bos) {
  TokenizerConfig c;
  c.tokens = {"<unk>", "<s>", "</s>", "<0x7A>", "a", "b", "c", "ab", "abc", kSM, kSM + "a"};
  c.scores = {0, 0, 0, 0, -3, -3, -3, -1, -0.5f, -3, -2};
  c.token_types = {
      static_cast<int32_t>(TokenType::kUnknown), static_cast<int32_t>(TokenType::kControl),
      static_cast<int32_t>(TokenType::kControl), static_cast<int32_t>(TokenType::kByte),
      static_cast<int32_t>(TokenType::kNormal),  static_cast<int32_t>(TokenType::kNormal),
      static_cast<int32_t>(TokenType::kNormal),  static_cast<int32_t>(TokenType::kNormal),
      static_cast<int32_t>(TokenType::kNormal),  static_cast<int32_t>(TokenType::kNormal),
      static_cast<int32_t>(TokenType::kNormal),
  };
  c.bos_id = 1;
  c.eos_id = 2;
  c.unk_id = 0;
  c.add_bos = add_bos;
  c.add_eos = false;
  c.add_dummy_prefix = dummy_prefix;
  return c;
}

}  // namespace

TEST(TokenizerTest, VocabSize) {
  Tokenizer tok(MakeConfig(false, false));
  EXPECT_EQ(tok.vocab_size(), 11u);
}

TEST(TokenizerTest, EncodeMergesByHighestScore) {
  Tokenizer tok(MakeConfig(false, false));
  // "a","b","c" -> merge ab (-1) then abc (-0.5): a single token id 8.
  EXPECT_EQ(tok.encode("abc", false), (std::vector<int32_t>{8}));
}

TEST(TokenizerTest, RoundTrip) {
  Tokenizer tok(MakeConfig(false, false));
  EXPECT_EQ(tok.decode(tok.encode("abc", false)), "abc");
}

TEST(TokenizerTest, ByteFallback) {
  Tokenizer tok(MakeConfig(false, false));
  // 'z' (0x7A) is not a normal piece -> falls back to the <0x7A> byte token.
  EXPECT_EQ(tok.encode("z", false), (std::vector<int32_t>{3}));
  EXPECT_EQ(tok.decode(std::vector<int32_t>{3}), "z");
}

TEST(TokenizerTest, DummyPrefixAndSpaceMarker) {
  Tokenizer tok(MakeConfig(true, false));
  // add_dummy_prefix escapes to "\u2581a" which merges to token 10.
  EXPECT_EQ(tok.encode("a", false), (std::vector<int32_t>{10}));
  EXPECT_EQ(tok.decode(std::vector<int32_t>{10}), "a");
}

TEST(TokenizerTest, SpecialTokensAdded) {
  Tokenizer tok(MakeConfig(false, true));
  const std::vector<int32_t> ids = tok.encode("abc");  // add_special default true
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.front(), 1);  // BOS
  EXPECT_EQ(ids.back(), 8);
}

TEST(TokenizerTest, DecodeSkipsSpecialTokens) {
  Tokenizer tok(MakeConfig(false, true));
  EXPECT_EQ(tok.decode(std::vector<int32_t>{1, 8, 2}), "abc");
}

TEST(TokenizerTest, DecodePieceStreaming) {
  Tokenizer tok(MakeConfig(false, false));
  EXPECT_EQ(tok.decode_piece(8), "abc");
  EXPECT_EQ(tok.decode_piece(1), "");  // control token contributes nothing
}
