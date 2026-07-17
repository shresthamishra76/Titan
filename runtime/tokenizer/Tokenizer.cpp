#include "Tokenizer.h"

#include <array>
#include <cstdio>
#include <queue>
#include <vector>

namespace titan {
namespace {

// U+2581 "LOWER ONE EIGHTH BLOCK" — SentencePiece's visible space marker.
constexpr const char* kSpaceMarker = "\xe2\x96\x81";

bool is_utf8_continuation(unsigned char b) { return (b & 0xC0) == 0x80; }

// Parse a "<0xNN>" byte token into its byte value, or return -1.
int parse_byte_token(const std::string& piece) {
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' ||
      (piece[2] != 'x' && piece[2] != 'X') || piece[5] != '>') {
    return -1;
  }
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  const int hi = hex(piece[3]), lo = hex(piece[4]);
  if (hi < 0 || lo < 0) return -1;
  return (hi << 4) | lo;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

}  // namespace

Tokenizer::Tokenizer(TokenizerConfig config)
    : config_(std::move(config)), byte_to_id_(256, -1) {
  for (int32_t id = 0; id < static_cast<int32_t>(config_.tokens.size()); ++id) {
    token_to_id_.emplace(config_.tokens[id], id);
    const int b = byte_value(id);
    if (b >= 0) byte_to_id_[b] = id;
  }
}

int32_t Tokenizer::piece_to_id(const std::string& piece) const {
  const auto it = token_to_id_.find(piece);
  return it == token_to_id_.end() ? -1 : it->second;
}

float Tokenizer::score_of(int32_t id) const {
  return id >= 0 && id < static_cast<int32_t>(config_.scores.size()) ? config_.scores[id]
                                                                     : 0.0f;
}

bool Tokenizer::is_control(int32_t id) const {
  if (id == config_.bos_id || id == config_.eos_id) return true;
  if (id >= 0 && id < static_cast<int32_t>(config_.token_types.size())) {
    return config_.token_types[id] == static_cast<int32_t>(TokenType::kControl);
  }
  return false;
}

int Tokenizer::byte_value(int32_t id) const {
  if (id < 0 || id >= static_cast<int32_t>(config_.tokens.size())) return -1;
  if (id < static_cast<int32_t>(config_.token_types.size()) &&
      config_.token_types[id] != static_cast<int32_t>(TokenType::kByte)) {
    // If types are known and this isn't a byte token, skip the string parse.
    return -1;
  }
  return parse_byte_token(config_.tokens[id]);
}

std::string Tokenizer::piece_text(int32_t id) const {
  const int b = byte_value(id);
  if (b >= 0) return std::string(1, static_cast<char>(b));
  if (id < 0 || id >= static_cast<int32_t>(config_.tokens.size())) return "";
  return replace_all(config_.tokens[id], kSpaceMarker, " ");
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_special) const {
  // 1. Escape: spaces -> U+2581, optional leading space (SPM dummy prefix).
  std::string norm = replace_all(text, " ", kSpaceMarker);
  if (config_.add_dummy_prefix) norm = std::string(kSpaceMarker) + norm;

  // 2. Split into UTF-8 characters as a doubly-linked list of symbols.
  struct Symbol {
    std::size_t start;
    std::size_t len;
    int prev;
    int next;
  };
  std::vector<Symbol> syms;
  for (std::size_t i = 0; i < norm.size();) {
    std::size_t len = 1;
    while (i + len < norm.size() &&
           is_utf8_continuation(static_cast<unsigned char>(norm[i + len]))) {
      ++len;
    }
    const int idx = static_cast<int>(syms.size());
    syms.push_back({i, len, idx - 1, -1});
    i += len;
  }
  for (std::size_t i = 0; i + 1 < syms.size(); ++i) syms[i].next = static_cast<int>(i + 1);
  if (!syms.empty()) syms.back().next = -1;

  // 3. Greedy highest-score merge via a priority queue of candidate bigrams.
  struct Bigram {
    int left;
    int right;
    float score;
    std::size_t size;
  };
  struct Cmp {
    bool operator()(const Bigram& a, const Bigram& b) const {
      return a.score < b.score || (a.score == b.score && a.left > b.left);
    }
  };
  std::priority_queue<Bigram, std::vector<Bigram>, Cmp> work;

  auto try_bigram = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const std::size_t size = syms[left].len + syms[right].len;
    const std::string piece = norm.substr(syms[left].start, size);
    const int32_t id = piece_to_id(piece);
    if (id >= 0) work.push({left, right, score_of(id), size});
  };

  for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
    try_bigram(static_cast<int>(i), static_cast<int>(i + 1));
  }

  while (!work.empty()) {
    const Bigram top = work.top();
    work.pop();
    Symbol& left = syms[top.left];
    Symbol& right = syms[top.right];
    // Skip stale entries (already merged / no longer adjacent).
    if (left.len == 0 || right.len == 0 || left.len + right.len != top.size ||
        left.next != top.right) {
      continue;
    }
    left.len += right.len;
    right.len = 0;
    left.next = right.next;
    if (right.next >= 0) syms[right.next].prev = top.left;
    try_bigram(left.prev, top.left);
    try_bigram(top.left, left.next);
  }

  // 4. Emit ids; fall back to raw bytes for pieces without a vocab entry.
  std::vector<int32_t> out;
  if (add_special && config_.add_bos && config_.bos_id >= 0) out.push_back(config_.bos_id);
  for (int i = 0; i >= 0; i = syms[i].next) {
    if (syms[i].len == 0) continue;
    const std::string piece = norm.substr(syms[i].start, syms[i].len);
    const int32_t id = piece_to_id(piece);
    if (id >= 0) {
      out.push_back(id);
      continue;
    }
    for (std::size_t b = 0; b < piece.size(); ++b) {
      const unsigned char byte = static_cast<unsigned char>(piece[b]);
      if (byte_to_id_[byte] >= 0) {
        out.push_back(byte_to_id_[byte]);
      } else if (config_.unk_id >= 0) {
        out.push_back(config_.unk_id);
      }
    }
    if (syms[i].next < 0) break;
  }
  if (add_special && config_.add_eos && config_.eos_id >= 0) out.push_back(config_.eos_id);
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  std::string out;
  for (const int32_t id : ids) {
    if (skip_special && is_control(id)) continue;
    out += piece_text(id);
  }
  // Undo the SPM dummy prefix so round-trips are exact.
  if (config_.add_dummy_prefix && !out.empty() && out.front() == ' ') out.erase(out.begin());
  return out;
}

std::string Tokenizer::decode_piece(int32_t id) const {
  if (is_control(id)) return "";
  return piece_text(id);
}

}  // namespace titan
