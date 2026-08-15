// Titan generation CLI (Milestone 1, issue #15).
//
// Loads a Llama-family GGUF model, tokenizes a prompt, and streams generated
// text to stdout. This is the end-to-end wiring of every V1 subsystem:
// GGUF loader -> tokenizer -> transformer forward -> sampler.
//
// Usage:
//   titan_generate -m model.gguf -p "Once upon a time" -n 64 [-g] [-t 0.8] ...
//
// Note: V1 has no KV cache, so each step re-runs the full forward pass over the
// growing sequence. Correct, but expect it to be slow on a full model — that is
// exactly what Milestone 2 (and the KV cache in V2) address.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "GGUF.h"
#include "Model.h"
#include "Sampler.h"
#include "Tokenizer.h"

using namespace titan;

namespace {

TokenizerConfig BuildTokenizerConfig(const GgufFile& g) {
  TokenizerConfig c;
  c.tokens = g.get_str_array("tokenizer.ggml.tokens");
  c.scores = g.get_f32_array("tokenizer.ggml.scores");
  c.token_types = g.get_i32_array("tokenizer.ggml.token_type");
  c.bos_id = static_cast<int32_t>(g.get_u32("tokenizer.ggml.bos_token_id", 1));
  c.eos_id = static_cast<int32_t>(g.get_u32("tokenizer.ggml.eos_token_id", 2));
  c.unk_id = static_cast<int32_t>(g.get_u32("tokenizer.ggml.unknown_token_id", 0));
  c.add_bos = g.get_u32("tokenizer.ggml.add_bos_token", 1) != 0;
  c.add_eos = g.get_u32("tokenizer.ggml.add_eos_token", 0) != 0;
  c.add_dummy_prefix = true;
  return c;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " -m <model.gguf> [options]\n"
      << "  -m, --model PATH        GGUF model file (required)\n"
      << "  -p, --prompt TEXT       prompt text (default: \"Hello\")\n"
      << "  -n, --max-tokens N      max new tokens (default: 64)\n"
      << "  -g, --greedy            greedy decoding (argmax)\n"
      << "  -t, --temp F            temperature (default: 0.8)\n"
      << "      --top-k N           top-k (default: 40, 0=off)\n"
      << "      --top-p F           top-p / nucleus (default: 0.95)\n"
      << "      --repeat-penalty F  repetition penalty (default: 1.1)\n"
      << "  -s, --seed N            RNG seed (default: 0)\n"
      << "  -h, --help              show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt = "Hello";
  int max_new = 64;
  bool greedy = false;
  SamplingParams sp;
  sp.temperature = 0.8f;
  sp.top_k = 40;
  sp.top_p = 0.95f;
  sp.repetition_penalty = 1.1f;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const std::string& def) -> std::string {
      return i + 1 < argc ? std::string(argv[++i]) : def;
    };
    if (a == "-m" || a == "--model") {
      model_path = next("");
    } else if (a == "-p" || a == "--prompt") {
      prompt = next("Hello");
    } else if (a == "-n" || a == "--max-tokens") {
      max_new = std::stoi(next("64"));
    } else if (a == "-g" || a == "--greedy") {
      greedy = true;
    } else if (a == "-t" || a == "--temp") {
      sp.temperature = std::stof(next("0.8"));
    } else if (a == "--top-k") {
      sp.top_k = std::stoi(next("40"));
    } else if (a == "--top-p") {
      sp.top_p = std::stof(next("0.95"));
    } else if (a == "--repeat-penalty") {
      sp.repetition_penalty = std::stof(next("1.1"));
    } else if (a == "-s" || a == "--seed") {
      sp.seed = static_cast<uint64_t>(std::stoull(next("0")));
    } else if (a == "-h" || a == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (model_path.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  try {
    GgufFile gguf(model_path);
    Tokenizer tokenizer(BuildTokenizerConfig(gguf));
    Model model = Model::from_gguf(gguf);

    const ModelConfig& cfg = model.config();
    std::cerr << "[titan] loaded " << gguf.tensor_count() << " tensors | "
              << cfg.n_layers << " layers, d=" << cfg.n_embd << ", heads=" << cfg.n_head
              << "/" << cfg.n_head_kv << ", vocab=" << cfg.n_vocab << "\n";

    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    std::vector<int32_t> generated;
    Sampler sampler(sp);

    std::cout << prompt << std::flush;

    const int ctx = cfg.context_length > 0 ? cfg.context_length : 2048;
    for (int step = 0; step < max_new && static_cast<int>(tokens.size()) < ctx; ++step) {
      const Tensor logits = model.forward(tokens);
      const int32_t next =
          greedy ? Sampler::argmax(logits) : sampler.sample(logits, generated);
      if (next == tokenizer.eos_id()) break;
      std::cout << tokenizer.decode_piece(next) << std::flush;
      tokens.push_back(next);
      generated.push_back(next);
    }
    std::cout << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[titan] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
