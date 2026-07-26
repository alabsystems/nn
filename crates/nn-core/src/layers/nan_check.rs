// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NaN check policy — thread-local opt-out for per-layer finiteness checks.
//!
//! Allows callers to skip per-layer NaN/Inf checks during stable inference.
//! Default is `Always` (safe). `Skip` eliminates GPU→CPU readbacks inside
//! `with_gpu_scope` where buffer contents are stale until scope exit.
//!
//! See [`NanCheckPolicy`] and [`with_nan_check_policy`].

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

/// Controls whether [`check_output_finite`] performs actual checks.
///
/// Used with [`with_nan_check_policy`] for scoped opt-out during stable
/// inference paths. Default is `Always`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanCheckPolicy {
    /// Check every layer output (default). Use during development and validation.
    Always,
    /// Skip NaN checks. Use for stable inference after model validation.
    /// Caller guarantees inputs are finite.
    Skip,
}

thread_local! {
    static NAN_CHECK_POLICY: std::cell::Cell<NanCheckPolicy> =
        const { std::cell::Cell::new(NanCheckPolicy::Always) };
}

/// RAII guard that restores the prior [`NanCheckPolicy`] on drop, even on panic.
struct NanCheckPolicyGuard {
    prev: NanCheckPolicy,
}

impl Drop for NanCheckPolicyGuard {
    fn drop(&mut self) {
        NAN_CHECK_POLICY.set(self.prev);
    }
}

/// Run `f` with the given [`NanCheckPolicy`]. Restores the prior policy on return,
/// including on panic (via RAII drop guard).
///
/// Composable with `with_gpu_scope`: wrap the outer scope in `Skip` to eliminate
/// GPU→CPU readbacks for per-layer NaN checks while retaining model-level output
/// validation after the scope exits.
///
/// # Example
///
/// ```ignore
/// use nn::layers::{NanCheckPolicy, with_nan_check_policy};
///
/// let output = with_nan_check_policy(NanCheckPolicy::Skip, || {
///     model.forward(&input)
/// });
/// ```
pub fn with_nan_check_policy<T>(policy: NanCheckPolicy, f: impl FnOnce() -> T) -> T {
    let _guard = NanCheckPolicyGuard {
        prev: NAN_CHECK_POLICY.get(),
    };
    NAN_CHECK_POLICY.set(policy);
    f()
}

/// Returns the current thread-local [`NanCheckPolicy`].
pub fn nan_check_policy() -> NanCheckPolicy {
    NAN_CHECK_POLICY.get()
}

// -- Layer output finiteness check (defense-in-depth, #1202) ------------------
//
// Tiering policy (#1209):
//
// - **Tier 1 (per-layer check):** Layers that can silently amplify NaN/Inf
//   through mathematical operations are checked after every forward call.
//   These "silent amplifiers" include: SwiGLU (exp in sigmoid), MoeRouter
//   (softmax), MoeLayer (softmax + expert dispatch), DiTBlock / DiTBlockDual
//   (complex composition with modulation), attention, batch_norm, instance_norm,
//   joint_attention, GatedDeltaNet (sigmoid+tanh gates), Lstm (sigmoid+tanh),
//   QLinear (dequantize path), SqueezeExcitation (sigmoid gating),
//   DeformableAttention (grid_sample + softmax).
//
// - **Tier 2 (model-level check):** Simple linear/normalization layers (Linear,
//   LayerNorm, GroupNorm, RmsNorm, Conv1d, Conv2d, ConvTranspose1d,
//   ConvTranspose2d, Embedding, etc.) rely on model-level finiteness checks at
//   stage boundaries (#941, #958). These operations cannot independently produce
//   NaN/Inf from finite inputs. Also includes: BiLstm (composes Lstm which is
//   Tier 1 — inner checks provide coverage), Res2NetBlock (composes Conv1d +
//   BatchNorm), MBConv (composes Conv2d + BatchNorm + SqueezeExcitation),
//   WeightNormConv1d (g/||v|| division at construction, forward is linear),
//   PatchEmbedding (delegates to Conv2d), Rvq/VqCodebook (CPU-only, sqrt/div
//   in quantize path — acceptable at Tier 2 since codebook lookup is the
//   common forward path).
//
// When adding new nn layers, classify them:
// - Contains division, sqrt, exp, softmax, or multi-step composition → Tier 1
// - Pure linear/elementwise on finite inputs → Tier 2

/// Check a layer output for NaN/Inf.
///
/// For CPU tensors, checks the data directly. For GPU tensors, delegates to
/// [`GpuBackend::count_non_finite`] which reads the GPU buffer without
/// constructing a full CPU tensor. If no GPU backend is registered or the
/// backend does not support `count_non_finite`, the check is skipped for GPU
/// tensors (model-level forward paths (#941, #958) serve as the backstop).
///
/// Called by Tier 1 layers (those that can silently amplify NaN/Inf):
/// Attention, LSTM, SwiGLU, BatchNorm, InstanceNorm, etc. (#1202, #1209).
pub fn check_output_finite(output: &DynTensor, layer_name: &str) -> Result<()> {
    if NAN_CHECK_POLICY.get() == NanCheckPolicy::Skip {
        return Ok(());
    }
    if output.device().is_cpu() {
        // Try zero-copy f32 view first (O(n) scan, no allocation for F32 tensors).
        // Falls back to to_f32_array() for BF16/F16 (O(n) scan + O(n) allocation).
        let non_finite = match output.as_cpu_f32() {
            Ok(view) => view.iter().filter(|v| !v.is_finite()).count(),
            Err(_) => {
                let data = output.to_f32_array()?;
                data.iter().filter(|v| !v.is_finite()).count()
            }
        };
        if non_finite > 0 {
            return Err(TensorError::NonFiniteData {
                name: format!("{layer_name} output"),
                count: non_finite,
            });
        }
        return Ok(());
    }

    // GPU path: use backend's count_non_finite if available (#1320).
    if let Some(result) = crate::dyn_tensor::gpu::gpu_backend_dispatch_count_non_finite(output) {
        let non_finite = result?;
        if non_finite > 0 {
            return Err(TensorError::NonFiniteData {
                name: format!("{layer_name} output"),
                count: non_finite,
            });
        }
    }
    Ok(())
}

#[cfg(kani)]
#[path = "kani_nan_check_proofs.rs"]
mod kani_proofs;
