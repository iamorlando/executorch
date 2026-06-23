/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Engine/Session adapter for Gemma 4 31B exported methods. CUDA builds use
// separate prefill/decode methods; MLX builds use one dynamic forward method.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

#ifdef EXECUTORCH_BUILD_CUDA
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#elif defined(EXECUTORCH_BUILD_MLX)
#include <executorch/backends/mlx/runtime/mlx_mutable_state.h>
#endif

#if defined(EXECUTORCH_BUILD_CUDA) || defined(EXECUTORCH_BUILD_MLX)
#define GEMMA_HAS_MUTABLE_STATE 1
#endif

namespace executorch::extension::llm {

#if defined(EXECUTORCH_BUILD_CUDA)
using GemmaMutableStateContextOwner =
    ::executorch::backends::cuda::MutableStateContextOwner;
constexpr int kGemmaNoMutableSession =
    ::executorch::backends::cuda::kNoMutableSession;
#elif defined(EXECUTORCH_BUILD_MLX)
using GemmaMutableStateContextOwner =
    ::executorch::backends::mlx::MutableStateContextOwner;
constexpr int kGemmaNoMutableSession =
    ::executorch::backends::mlx::kNoMutableSession;
#endif

struct Gemma4_31BConfig {
  std::string model_path;
  std::string data_path;
  std::string tokenizer_path;
  int32_t max_sessions = 1;
  int64_t eos_id = 1;
  bool enable_cuda_graph = false;
};

class ET_EXPERIMENTAL Gemma4_31BEngine : public LLMEngine {
 public:
  static ::executorch::runtime::Result<std::unique_ptr<Gemma4_31BEngine>>
  create(const Gemma4_31BConfig& config);

  ~Gemma4_31BEngine() override;

  ::executorch::runtime::Result<std::unique_ptr<LLMSession>> create_session()
      override;

  LLMServingCapacity serving_capacity() const override;

  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return metadata_;
  }

  ::tokenizers::Tokenizer* tokenizer() const {
    return tokenizer_.get();
  }

  Gemma4_31BEngine(const Gemma4_31BEngine&) = delete;
  Gemma4_31BEngine& operator=(const Gemma4_31BEngine&) = delete;

 private:
  Gemma4_31BEngine(
      Gemma4_31BConfig config,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids,
      std::unique_ptr<Module> shared_module,
      int64_t max_prefill_chunk,
      int64_t min_prefill_chunk,
      bool rebind_available
#ifdef GEMMA_HAS_MUTABLE_STATE
      ,
      std::unique_ptr<GemmaMutableStateContextOwner> mutable_state
#endif
      )
      : config_(std::move(config)),
        tokenizer_(std::move(tokenizer)),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        shared_module_(std::move(shared_module)),
        max_prefill_chunk_(max_prefill_chunk),
        min_prefill_chunk_(min_prefill_chunk),
        rebind_available_(rebind_available)
#ifdef GEMMA_HAS_MUTABLE_STATE
        ,
        mutable_state_(std::move(mutable_state))
#endif
  {
  }

  Gemma4_31BConfig config_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;
  std::unique_ptr<Module> shared_module_;
  std::mutex exec_mutex_;
  int64_t max_prefill_chunk_ = 0;
  int64_t min_prefill_chunk_ = 1;
  bool rebind_available_ = false;
#ifdef GEMMA_HAS_MUTABLE_STATE
  std::unique_ptr<GemmaMutableStateContextOwner> mutable_state_;
#endif
  std::atomic<int> live_sessions_{0};
};

} // namespace executorch::extension::llm
