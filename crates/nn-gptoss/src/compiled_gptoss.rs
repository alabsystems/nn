// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compiled gpt-oss-20b model for production inference.
//!
//! Wraps [`GptOssModel`] with Metal GPU optimization, BF16 autocasting,
//! and end-to-end generation support. This is the primary interface for
//! running Context-1 inference.
//!
//! # Usage
//!
//! ```no_run
//! use nn_gptoss::compiled_gptoss::CompiledGptOss;
//! use nn_core::Device;
//!
//! let model = CompiledGptOss::load_default().unwrap();
//! let output = model.generate(&[1, 2, 3], &Default::default()).unwrap();
//! println!("Generated {} tokens in {:.1}ms", output.generated_tokens, output.total_time_ms);
//! ```

use std::path::Path;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::{DType, Device, Result};

use crate::config::GptOssConfig;
use crate::generate::GenerateConfig;
use crate::sampling::SamplingConfig;
use crate::GptOssModel;

// ---------------------------------------------------------------------------
// GenerationOutput
// ---------------------------------------------------------------------------

/// Output from an autoregressive generation run.
///
/// Contains the generated token sequence plus timing metrics for
/// performance analysis.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenerationOutput {
    /// Generated token IDs (excluding the prompt).
    pub tokens: Vec<usize>,
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Number of tokens generated during the decode phase.
    pub generated_tokens: usize,
    /// Total wall-clock time for the full generation (ms).
    pub total_time_ms: f64,
    /// Time spent in the prefill (prompt processing) phase (ms).
    pub prefill_time_ms: f64,
    /// Time spent in the decode (token generation) phase (ms).
    pub decode_time_ms: f64,
    /// Tokens generated per second (decode phase only).
    pub tokens_per_second: f64,
}

impl GenerationOutput {
    /// Constructor for `#[non_exhaustive]` cross-crate use.
    #[must_use]
    pub fn new(
        tokens: Vec<usize>,
        prompt_tokens: usize,
        generated_tokens: usize,
        total_time_ms: f64,
        prefill_time_ms: f64,
        decode_time_ms: f64,
        tokens_per_second: f64,
    ) -> Self {
        Self {
            tokens,
            prompt_tokens,
            generated_tokens,
            total_time_ms,
            prefill_time_ms,
            decode_time_ms,
            tokens_per_second,
        }
    }
}

// ---------------------------------------------------------------------------
// CompiledGptOss
// ---------------------------------------------------------------------------

/// Compiled gpt-oss-20b model for production Metal GPU inference.
///
/// Manages the model, device placement, dtype selection, and provides
/// high-level APIs for forward pass, generation, and multi-turn sessions.
///
/// When loaded with `Device::Metal { .. }`, all DynTensor operations
/// dispatch to Metal GPU automatically via the nn-metal backend.
pub struct CompiledGptOss {
    model: GptOssModel,
    device: Device,
    dtype: DType,
    config: GptOssConfig,
}

impl CompiledGptOss {
    /// Load from a safetensors file with explicit device and dtype.
    ///
    /// Use `Device::metal()` for GPU inference on Apple Silicon.
    /// Use `DType::BF16` for native BF16 on M4 Max (halves memory).
    ///
    /// # Errors
    ///
    /// Returns an error if weights cannot be read, shapes mismatch, or
    /// the config is invalid.
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self> {
        let config = GptOssConfig::gptoss_20b();
        let model = GptOssModel::load_safetensors_to_device(path, config.clone(), dtype, &device)?;
        Ok(Self {
            model,
            device,
            dtype,
            config,
        })
    }

    /// Load from a safetensors file with a custom config.
    ///
    /// # Errors
    ///
    /// Returns an error if weights cannot be read or the config is invalid.
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: GptOssConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let model = GptOssModel::load_safetensors_to_device(path, config.clone(), dtype, &device)?;
        Ok(Self {
            model,
            device,
            dtype,
            config,
        })
    }

    /// Load from the `CONTEXT1_WEIGHTS` env var with auto-detected settings.
    ///
    /// - Device: `Metal(0)` if target is aarch64-apple (Apple Silicon), else `Cpu`.
    /// - DType: `BF16` on Metal (2x memory savings), `F32` on CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the `CONTEXT1_WEIGHTS` env var is not set,
    /// the file cannot be read, or model loading fails.
    pub fn load_default() -> Result<Self> {
        let path =
            std::env::var("CONTEXT1_WEIGHTS").map_err(|_| crate::GptOssError::WeightLoad {
                reason: "CONTEXT1_WEIGHTS env var not set".into(),
            })?;
        let (device, dtype) = default_device_and_dtype();
        Self::load(path, device, dtype)
    }

    /// Build from a pre-loaded [`GptOssModel`].
    ///
    /// Useful when the caller manages weight loading directly.
    #[must_use]
    pub fn from_model(model: GptOssModel, config: GptOssConfig) -> Self {
        let device = model.device();
        let dtype = model.dtype();
        Self {
            model,
            device,
            dtype,
            config,
        }
    }

    /// Forward pass: token IDs to logits.
    ///
    /// Returns logits of shape `[1, seq_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass produces NaN/Inf or
    /// tensor operations fail.
    pub fn forward(&self, input_ids: &[usize]) -> Result<DynTensor> {
        let positions: Vec<usize> = (0..input_ids.len()).collect();
        self.model.forward(input_ids, &positions)
    }

    /// Greedy autoregressive generation (temperature=0).
    ///
    /// Runs prompt prefill, then generates tokens one at a time until
    /// EOS or `config.max_tokens` is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if generation config validation fails or the
    /// model forward pass fails.
    pub fn generate(
        &self,
        prompt_ids: &[usize],
        config: &GenerateConfig,
    ) -> Result<GenerationOutput> {
        self.generate_inner(prompt_ids, config, None)
    }

    /// Generation with sampling (temperature, top-k, top-p).
    ///
    /// Applies the full sampling pipeline at each decode step.
    ///
    /// # Errors
    ///
    /// Returns an error if config validation fails or the model
    /// forward pass fails.
    pub fn generate_sampled(
        &self,
        prompt_ids: &[usize],
        gen_cfg: &GenerateConfig,
        sampling_cfg: &SamplingConfig,
    ) -> Result<GenerationOutput> {
        self.generate_inner(prompt_ids, gen_cfg, Some(sampling_cfg))
    }

    /// Shared generation implementation.
    fn generate_inner(
        &self,
        prompt_ids: &[usize],
        gen_cfg: &GenerateConfig,
        sampling_cfg: Option<&SamplingConfig>,
    ) -> Result<GenerationOutput> {
        gen_cfg.validate()?;

        if gen_cfg.max_tokens == 0 || prompt_ids.is_empty() {
            return Ok(GenerationOutput::new(
                Vec::new(),
                prompt_ids.len(),
                0,
                0.0,
                0.0,
                0.0,
                0.0,
            ));
        }

        let eos = self.config.eos_token_id;
        let mut cache = self.model.new_cache();
        let mut generated = Vec::with_capacity(gen_cfg.max_tokens);

        let total_start = std::time::Instant::now();

        // Prefill: run full prompt through model.
        let prefill_start = std::time::Instant::now();
        let positions: Vec<usize> = (0..prompt_ids.len()).collect();
        let logits = self
            .model
            .forward_cached(prompt_ids, &positions, Some(&mut cache))?;
        let prefill_time_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Decode loop.
        let decode_start = std::time::Instant::now();
        let first_token = self.sample_last_token(&logits, gen_cfg, sampling_cfg, &generated)?;
        if first_token == eos {
            let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return Ok(GenerationOutput::new(
                generated,
                prompt_ids.len(),
                0,
                total_time_ms,
                prefill_time_ms,
                0.0,
                0.0,
            ));
        }
        generated.push(first_token);

        for _ in 1..gen_cfg.max_tokens {
            let pos = prompt_ids.len() + generated.len() - 1;
            let last_token = *generated.last().expect("invariant: generated is non-empty");
            let logits = self
                .model
                .forward_cached(&[last_token], &[pos], Some(&mut cache))?;
            let token = self.sample_last_token(&logits, gen_cfg, sampling_cfg, &generated)?;
            if token == eos {
                break;
            }
            generated.push(token);
        }

        let decode_time_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let gen_count = generated.len();
        let tps = if decode_time_ms > 0.0 {
            gen_count as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        };

        Ok(GenerationOutput::new(
            generated,
            prompt_ids.len(),
            gen_count,
            total_time_ms,
            prefill_time_ms,
            decode_time_ms,
            tps,
        ))
    }

    /// Sample a token from the last position of the logits tensor.
    fn sample_last_token(
        &self,
        logits: &DynTensor,
        gen_cfg: &GenerateConfig,
        sampling_cfg: Option<&SamplingConfig>,
        past_tokens: &[usize],
    ) -> Result<usize> {
        let dims = logits.dims();
        let seq_len = dims[1];
        let vocab_size = dims[2];

        let last_logits = logits.narrow(1, seq_len - 1, 1)?;
        let flat = last_logits.reshape([vocab_size])?;
        let logit_vec = flat.to_flat_vec::<f32>()?;

        if let Some(scfg) = sampling_cfg {
            let seed = past_tokens.len() as u64;
            Ok(crate::sampling::sample_token(
                &logit_vec,
                scfg,
                past_tokens,
                seed,
            ))
        } else {
            // Build a SamplingConfig from the GenerateConfig for the
            // non-sampling path. Temperature 0.0 maps to greedy (top_k=1).
            let scfg = if gen_cfg.temperature == 0.0 {
                SamplingConfig::greedy()
            } else {
                let mut sc = SamplingConfig::default().with_temperature(gen_cfg.temperature);
                if let Some(k) = gen_cfg.top_k {
                    sc = sc.with_top_k(k);
                }
                if let Some(p) = gen_cfg.top_p {
                    sc = sc.with_top_p(p);
                }
                if let Some(rp) = gen_cfg.repetition_penalty {
                    sc = sc.with_repetition_penalty(rp);
                }
                sc
            };
            let seed = past_tokens.len() as u64;
            Ok(crate::sampling::sample_token(
                &logit_vec,
                &scfg,
                past_tokens,
                seed,
            ))
        }
    }

    /// Create a new stateful inference session for multi-turn conversation.
    #[must_use]
    pub fn new_session(&self) -> InferenceSession<'_> {
        InferenceSession {
            model: self,
            cache: self.model.new_cache(),
            position: 0,
        }
    }

    /// Model configuration reference.
    #[must_use]
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    /// Device the model weights are on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Model weight dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Reference to the underlying model.
    #[must_use]
    pub fn model(&self) -> &GptOssModel {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// InferenceSession
// ---------------------------------------------------------------------------

/// Stateful inference session with KV cache for multi-turn conversation.
///
/// Maintains the KV cache across calls to [`step`](InferenceSession::step),
/// allowing efficient multi-turn inference without reprocessing the full
/// context on each turn.
pub struct InferenceSession<'a> {
    model: &'a CompiledGptOss,
    cache: KvCache,
    position: usize,
}

impl InferenceSession<'_> {
    /// Run one forward step: process input tokens and return logits.
    ///
    /// The KV cache is updated in-place, so subsequent calls only need
    /// to process new tokens. Returns logits of shape
    /// `[1, input_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn step(&mut self, input_ids: &[usize]) -> Result<DynTensor> {
        let positions: Vec<usize> = (0..input_ids.len()).map(|i| self.position + i).collect();
        let logits =
            self.model
                .model
                .forward_cached(input_ids, &positions, Some(&mut self.cache))?;
        self.position += input_ids.len();
        Ok(logits)
    }

    /// Reset the session: clear KV cache and reset position to 0.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.position = 0;
    }

    /// Current sequence length (total tokens processed so far).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.position
    }

    /// Reference to the underlying KV cache.
    #[must_use]
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Auto-detect the best device and dtype for the current platform.
///
/// On Apple Silicon (`aarch64-apple-*`), returns Metal(0) + BF16.
/// Otherwise returns Cpu + F32.
#[must_use]
pub(crate) fn default_device_and_dtype() -> (Device, DType) {
    if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
        (Device::metal(), DType::BF16)
    } else {
        (Device::Cpu, DType::F32)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "compiled_gptoss_tests.rs"]
mod tests;
