#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace titan {

// SentencePiece token types (as stored in GGUF tokenizer.ggml.token_type).
enum class TokenType : int32_t {
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

// Everything needed to build a tokenizer, as read from a GGUF file's
// tokenizer.ggml.* metadata (or hand-built in tests).
struct TokenizerConfig {
  std::vector<std::string> tokens;   // id -> piece (UTF-8; spaces encoded as U+2581)
  std::vector<float> scores;         // id -> merge score (higher merges first)
  std::vector<int32_t> token_types;  // id -> TokenType (optional)
  int32_t bos_id = -1;
  int32_t eos_id = -1;
  int32_t unk_id = -1;
  bool add_bos = true;
  bool add_eos = false;
  bool add_dummy_prefix = true;  // SPM prepends a space before encoding
};

// From-scratch SentencePiece-style BPE tokenizer.
//
// Encoding follows the llama.cpp SPM algorithm: the text is escaped (spaces ->
// U+2581), split into UTF-8 characters, then adjacent symbols are greedily
// merged, always taking the highest-scoring merge that exists in the vocab.
// Pieces with no vocab entry fall back to raw <0xNN> byte tokens.
//
// See docs/architecture/tokenizer.md.
class Tokenizer {
 public:
  explicit Tokenizer(TokenizerConfig config);

  // Text -> token ids. When add_special is set, prepends BOS / appends EOS
  // according to the config.
  std::vector<int32_t> encode(const std::string& text, bool add_special = true) const;

  // Token ids -> text. Skips BOS/EOS/control tokens when skip_special is set.
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;

  // Single token -> its text contribution (for streaming). May emit partial
  // UTF-8 when a code point is split across byte-fallback tokens.
  std::string decode_piece(int32_t id) const;

  std::size_t vocab_size() const { return config_.tokens.size(); }
  int32_t bos_id() const { return config_.bos_id; }
  int32_t eos_id() const { return config_.eos_id; }

 private:
  int32_t piece_to_id(const std::string& piece) const;
  float score_of(int32_t id) const;
  bool is_control(int32_t id) const;
  int byte_value(int32_t id) const;  // raw byte for a <0xNN> token, else -1
  std::string piece_text(int32_t id) const;

  TokenizerConfig config_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<int32_t> byte_to_id_;  // 256 entries, -1 if absent
};

}  // namespace titan
