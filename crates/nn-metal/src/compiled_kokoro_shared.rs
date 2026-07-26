// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared state for multi-instance [`CompiledKokoro`] dispatch.
//!
//! [`SharedKokoroState`] holds model weights, verifier, and iSTFT basis
//! in an `Arc` so multiple [`CompiledKokoro`] instances can share them
//! via [`clone_dispatch()`](super::CompiledKokoro::clone_dispatch).
//!
//! Memory: 7 voices share ~400MB weights (1.02x overhead vs 7x without sharing).
//!
//! ## Weight release (#3079)
//!
//! After all segments are compiled, [`CompiledKokoro::release_model_weights()`]
//! drops the model's CPU weight data (~320 MB for Kokoro-82M). The compiled
//! segments' GPU MetalBuffers remain in `SegmentCache.shared_weights`.
//! Source module and config are extracted before release for bridge operations.
//!
//! Part of #2740, #2218, #3079.

use std::sync::{Arc, OnceLock};

use nn_core::TensorError;
use nn_models::kokoro_source::SourceModule;
use nn_models::kokoro_tts::{KokoroConfig, KokoroModel};
use nn_tts_verify::{HardBoundsConfig, TtsVerifier};

use super::helpers::gpu;
use crate::istft_gpu::IstftGpuBasis;
use crate::stft_gpu::StftGpuBasis;

/// Shared state across multiple [`CompiledKokoro`] instances.
///
/// All fields are immutable or interior-mutable after initialization.
/// Thread-safe via `Arc<SharedKokoroState>`.
///
/// - `model`: `Some` during tracing/compilation, `None` after weight release.
/// - `config`: always available (extracted at init).
/// - `source_module`: always available for bridge operations.
/// - `verifier`: stateless config (~100 bytes).
/// - `istft_basis` / `stft_basis`: lazy-initialized GPU buffers via `OnceLock`.
pub(crate) struct SharedKokoroState {
    /// The original model (sub-modules used for tracing).
    /// `None` after `release_model_weights()` frees CPU weight data (#3079).
    pub(super) model: Option<KokoroModel>,
    /// Model config (always available, even after weight release).
    pub(super) config: KokoroConfig,
    /// Source module (GPU weights for harmonic source bridge).
    /// Extracted at init, persists after weight release.
    pub(super) source_module: Option<SourceModule>,
    /// TTS audio quality verifier (stateless config).
    pub(super) verifier: TtsVerifier,
    /// Cached GPU iSTFT basis. Lazy-initialized on first iSTFT call.
    /// `OnceLock` ensures one thread initializes, all others share.
    /// Uses `Result` wrapper per design doc `OnceLock<Result<T, String>>` pattern
    /// since `OnceLock::get_or_try_init()` is unstable.
    pub(super) istft_basis: OnceLock<Result<IstftGpuBasis, String>>,
    /// Cached GPU forward STFT basis. Used by `build_harmonic_source`.
    /// Same `Result` wrapper pattern as `istft_basis`.
    pub(super) stft_basis: OnceLock<Result<StftGpuBasis, String>>,
}

impl SharedKokoroState {
    /// Create shared state from a loaded model with default verification config.
    ///
    /// Transfers SourceModule `l_linear` weights to GPU at init time so the
    /// harmonic source bridge doesn't hit a CPU/GPU device mismatch at runtime.
    /// Extracts config and source_module for persistence after weight release.
    pub(super) fn new(model: KokoroModel) -> Result<Arc<Self>, TensorError> {
        Self::with_hard_bounds(model, HardBoundsConfig::default())
    }

    /// Create shared state from a loaded model with custom hard bounds config.
    ///
    /// Same as [`new`](Self::new) but allows callers to configure per-check
    /// threshold overrides and rejection policy on the embedded
    /// [`TtsVerifier`].
    ///
    /// Part of #3780, #3758, #3760.
    pub(super) fn with_hard_bounds(
        mut model: KokoroModel,
        hard_bounds: HardBoundsConfig,
    ) -> Result<Arc<Self>, TensorError> {
        // SourceModule l_linear weights must be on GPU for build_harmonic_source.
        // Compiled segments upload their own weights during compilation, but
        // bridge stages use eager DynTensor ops that require matching devices.
        model.ensure_source_device(&gpu())?;
        let config = model.config().clone();
        // Extract SourceModule to GPU for bridge operations. SourceModule
        // doesn't impl Clone — use to_device() which creates a new instance
        // sharing Arc-backed DynTensor weight data (zero-copy on same device).
        let source_module = model
            .source_module()
            .map(|sm| sm.to_device(&gpu()))
            .transpose()?;
        Ok(Arc::new(Self {
            model: Some(model),
            config,
            source_module,
            verifier: TtsVerifier::builder()
                .hard_bounds(hard_bounds)
                .build()
                .map_err(|e| TensorError::Unsupported(e.to_string()))?,
            istft_basis: OnceLock::new(),
            stft_basis: OnceLock::new(),
        }))
    }

    /// Access the model, returning error if weights have been released.
    pub(super) fn model(&self) -> Result<&KokoroModel, super::CompiledKokoroError> {
        self.model
            .as_ref()
            .ok_or(super::CompiledKokoroError::WeightsReleased)
    }
}
