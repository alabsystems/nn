// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compound and parametric mathematical operations for [`DynTensor`].
//!
//! Contains multi-step operations (elu, leaky_relu, snake, clamp) and
//! binary math ops (maximum, minimum, atan2) that don't reduce to a single
//! `UnaryOp` dispatch. Also includes `repair_non_finite` and `any_non_finite`
//! utility methods.
//!
//! Extracted from `math.rs` (#2018) for file-size compliance.

use crate::dyn_tensor::gpu::gpu_backend_dispatch;
use crate::dyn_tensor::trace::{self, KokoroFusedOp, TraceOp};
use crate::dyn_tensor::{BinaryOp, DynTensor};
use crate::layers::check_output_finite;
use crate::{DType, Device, Result};

impl DynTensor {
    /// ELU activation: x if x > 0, else alpha * (exp(x) - 1).
    ///
    /// Works on both CPU and GPU tensors. GPU tensors decompose into
    /// existing GPU-native ops: `relu(x) + alpha * (exp(-relu(-x)) - 1)`.
    /// Uses `traced_forward` so decomposed GPU ops don't create redundant
    /// trace nodes — only the composite Elu op is recorded.
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Elu { alpha }),
            || {
                if self.device().is_gpu() {
                    // GPU path: decompose into GPU-native ops to avoid round-trip.
                    // elu(x) = relu(x) + alpha * (exp(min(x, 0)) - 1)
                    // where min(x, 0) = -relu(-x)
                    let pos = self.relu()?;
                    let neg_input = self.neg()?.relu()?.neg()?;
                    let exp_neg = neg_input.exp()?;
                    let exp_m1 = exp_neg.add_scalar(-1.0)?;
                    let scaled = exp_m1.mul_scalar(alpha)?;
                    pos.add(&scaled)
                } else {
                    // Promote bf16/f16 to f32, compute, convert back (#1646 D3).
                    let arr = self.to_f32_array()?;
                    let a = crate::dyn_tensor::checked_f64_to_f32(alpha, "elu() alpha")?;
                    let computed = arr.mapv(|x| if x > 0.0 { x } else { a * x.exp_m1() });
                    Self::from_f32_result(computed, self.dtype)
                }
            },
        )
    }

    /// Leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
    ///
    /// Works on both CPU and GPU tensors via composed ops (relu, neg, mul_scalar, add).
    /// Uses `traced_forward` so decomposed ops don't create redundant trace nodes.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::LeakyRelu {
                    slope: negative_slope,
                })
            },
            || {
                let positive = self.relu()?;
                let negative = self.neg()?.relu()?.neg()?.mul_scalar(negative_slope)?;
                positive.add(&negative)
            },
        )
    }

    /// Softplus: `log(1 + exp(x))`. Smooth approximation of ReLU.
    ///
    /// CPU path decomposes to `log(exp(x) + 1)` using existing ops.
    /// The compiled model pipeline compiles to a single `TensorOpKind::Softplus`
    /// MSL kernel. Uses `traced_forward` so decomposed ops create a single trace node.
    pub fn softplus(&self) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Softplus),
            || self.exp()?.add_scalar(1.0)?.log(),
        )
    }

    /// SELU (Scaled ELU): `lambda * (x if x >= 0, else alpha * (exp(x) - 1))`.
    ///
    /// Uses fixed constants: alpha ~= 1.6733, lambda ~= 1.0507.
    pub fn selu(&self) -> Result<Self> {
        const SELU_ALPHA: f64 = 1.6732632423543772;
        const SELU_LAMBDA: f64 = 1.0507009873554805;
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Selu),
            || {
                let arr = self.to_f32_array()?;
                let alpha = SELU_ALPHA as f32;
                let lambda = SELU_LAMBDA as f32;
                let computed = arr.mapv(|x| {
                    if x >= 0.0 {
                        lambda * x
                    } else {
                        lambda * alpha * x.exp_m1()
                    }
                });
                Self::from_f32_result(computed, self.dtype)
            },
        )
    }

    /// CELU (Continuous ELU): `max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
    pub fn celu(&self, alpha: f64) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Celu { alpha }),
            || {
                let arr = self.to_f32_array()?;
                let a = crate::dyn_tensor::checked_f64_to_f32(alpha, "celu() alpha")?;
                let computed = arr.mapv(|x| if x >= 0.0 { x } else { a * (x / a).exp_m1() });
                Self::from_f32_result(computed, self.dtype)
            },
        )
    }

    /// HardSigmoid: `max(0, min(1, x/6 + 0.5))`.
    pub fn hard_sigmoid(&self) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::HardSigmoid),
            || {
                let arr = self.to_f32_array()?;
                let computed = arr.mapv(|x| (x / 6.0 + 0.5).clamp(0.0, 1.0));
                Self::from_f32_result(computed, self.dtype)
            },
        )
    }

    /// HardSwish: `x * HardSigmoid(x)`.
    pub fn hard_swish(&self) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::HardSwish),
            || {
                let arr = self.to_f32_array()?;
                let computed = arr.mapv(|x| x * (x / 6.0 + 0.5).clamp(0.0, 1.0));
                Self::from_f32_result(computed, self.dtype)
            },
        )
    }

    /// Mish activation: `x * tanh(softplus(x))`.
    pub fn mish(&self) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Mish),
            || {
                let arr = self.to_f32_array()?;
                let computed = arr.mapv(|x| {
                    let sp = x.exp().ln_1p();
                    x * sp.tanh()
                });
                Self::from_f32_result(computed, self.dtype)
            },
        )
    }

    /// Softsign activation: `x / (1 + |x|)`.
    ///
    /// Output range is (-1, 1). Similar to tanh but with lighter tails
    /// (polynomial vs. exponential decay). Matches PyTorch `nn.Softsign`.
    pub fn softsign(&self) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Softsign),
            || {
                let arr = self.to_f32_array()?;
                let computed = arr.mapv(|x| x / (1.0 + x.abs()));
                Self::from_f32_result(computed, self.dtype)
            },
        )
    }

    /// Snake activation: `x + (1/alpha) * sin²(alpha * x)`.
    ///
    /// Used by Kokoro TTS ISTFTNet decoder. Verified kernel exists in nn-dsl.
    /// Alpha is clamped to `[1e-8, 1e6]` to match GPU/DSL `SNAKE_MIN_ALPHA`.
    pub fn snake(&self, alpha: f64) -> Result<Self> {
        let a = alpha.clamp(1e-8, 1e6);
        let scaled = self.mul_scalar(a)?;
        let sin_sq = scaled.sin()?.sqr()?;
        let inv_alpha = 1.0 / a;
        self.add(&sin_sq.mul_scalar(inv_alpha)?)
    }

    /// Per-channel snake activation: `x + (1/alpha) * sin²(alpha * x)`.
    ///
    /// `alpha` is a tensor (typically shape `[1, C, 1]`) that broadcasts over `self`.
    /// Each channel gets its own alpha value, matching Kokoro's ISTFTNet decoder.
    /// Alpha values are clamped to a minimum of `1e-8` to match GPU/DSL `SNAKE_MIN_ALPHA`.
    ///
    /// Records `TraceOp::SnakeTensor` as a composite node when tracing is active,
    /// so the trace graph sees a single Snake op instead of 5 primitives
    /// (mul, sin, sqr, recip, mul, add). This reduces Kokoro trace nodes from
    /// ~216 to ~36 for the Generator's snake_tensor calls.
    pub fn snake_tensor(&self, alpha: &Self) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor {
                    alpha: alpha.to_weight_ref()?,
                }))
            },
            || {
                // GPU fused path: single dispatch for snake (#2226)
                if self.device().is_gpu() {
                    if let Some(result) = gpu_backend_dispatch(|b| b.snake_tensor(self, alpha)) {
                        let r = result?;
                        check_output_finite(&r, "SnakeTensor")?;
                        return Ok(r);
                    }
                }

                // CPU/fallback: decomposed ops
                let alpha_safe = alpha.clamp(1e-8, 1e6)?;
                let scaled = self.broadcast_mul(&alpha_safe)?;
                let sin_sq = scaled.sin()?.sqr()?;
                let inv_alpha = alpha_safe.recip()?;
                self.add(&sin_sq.broadcast_mul(&inv_alpha)?)
            },
        )
    }

    /// Clamp every element to [min, max].
    ///
    /// Works on both CPU and GPU tensors. GPU tensors use a fused single-dispatch
    /// kernel when available (#1815 D2a), falling back to relu decomposition.
    /// Uses `traced_forward` so decomposed GPU ops don't create redundant
    /// trace nodes — only the composite Clamp op is recorded.
    pub fn clamp(&self, min: f64, max: f64) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::Clamp {
                    min: Some(min),
                    max: Some(max),
                })
            },
            || {
                if self.device().is_gpu() {
                    // Fused GPU path: single dispatch (#1815 D2a)
                    if let Some(result) = gpu_backend_dispatch(|b| b.clamp(self, min, max)) {
                        return result;
                    }
                    // Fallback: relu decomposition (8 encodings)
                    self.clamp_min(min)?.clamp_max(max)
                } else {
                    // Promote bf16/f16 to f32, compute, convert back (#1646 D3).
                    let arr = self.to_f32_array()?;
                    let lo = crate::dyn_tensor::checked_f64_to_f32(min, "clamp() min")?;
                    let hi = crate::dyn_tensor::checked_f64_to_f32(max, "clamp() max")?;
                    let clamped = arr.mapv(|x| x.clamp(lo, hi));
                    Self::from_f32_result(clamped, self.dtype)
                }
            },
        )
    }

    /// Element-wise maximum of two tensors.
    ///
    /// For each element, returns `max(self[i], rhs[i])`.
    /// Shapes must be broadcast-compatible. Dispatches to GPU when both
    /// tensors are on a GPU device.
    ///
    /// # CPU/GPU behavioral divergence
    ///
    /// - **CPU:** Uses `f32::max()`, which returns the non-NaN operand when
    ///   one input is NaN (IEEE 754 `maxNum` semantics).
    /// - **GPU:** Uses Compare+Select IR, which may return NaN when NaN is
    ///   in the else branch (select propagates NaN unchanged).
    ///
    /// Inputs should be finite for consistent cross-device results.
    /// Use [`repair_non_finite`](Self::repair_non_finite) to sanitize
    /// tensors containing NaN before calling this method.
    pub fn maximum(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Maximum, rhs)
    }

    /// Element-wise minimum of two tensors.
    ///
    /// For each element, returns `min(self[i], rhs[i])`.
    /// Shapes must be broadcast-compatible. Dispatches to GPU when both
    /// tensors are on a GPU device.
    ///
    /// # CPU/GPU behavioral divergence
    ///
    /// Same as [`maximum`](Self::maximum) — CPU uses `f32::min()` (returns
    /// non-NaN), GPU Compare+Select may propagate NaN. Inputs should be
    /// finite for consistent results.
    pub fn minimum(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Minimum, rhs)
    }

    /// Element-wise two-argument arctangent.
    ///
    /// For each element, returns `atan2(self[i], rhs[i])` — the angle in
    /// radians between the positive x-axis and the point `(rhs[i], self[i])`.
    /// Result range is `(-π, π]`. Shapes must be broadcast-compatible.
    ///
    /// Follows Rust `f32::atan2(self, other)` convention: `self` is `y`,
    /// `rhs` is `x`. MSL `atan2(y, x)` has the same parameter order.
    ///
    /// Dispatches to native Metal `atan2` intrinsic when both tensors are
    /// on a GPU device.
    pub fn atan2(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Atan2, rhs)
    }

    /// Replace every NaN/Inf element with `fallback`.
    ///
    /// Finite values are preserved. This matches NY's
    /// `repair_non_finite_lower`/`repair_non_finite_upper` pattern for
    /// sanitizing bound tensors after propagation.
    ///
    /// Requires F32 dtype — returns `DTypeMismatch` for integer tensors.
    ///
    /// GPU tensors are round-tripped through CPU. This is intentional:
    /// `repair_non_finite` is applied to small output bound tensors, not
    /// to large intermediate matrices where GPU dispatch matters.
    pub fn repair_non_finite(&self, fallback: f64) -> Result<Self> {
        let fallback_f32 =
            crate::dyn_tensor::checked_f64_to_f32(fallback, "repair_non_finite() fallback")?;
        let original_device = self.device();
        // Round-trip GPU tensors to CPU for element-wise repair.
        let cpu_self = if original_device.is_gpu() {
            self.to_device(&Device::Cpu)?
        } else {
            self.clone()
        };
        let input_dtype = self.dtype;
        let arr = cpu_self.to_f32_array()?;
        let result = arr.mapv(|x| if x.is_finite() { x } else { fallback_f32 });
        let repaired = Self::from_f32_result(result, input_dtype)?;
        // Return on the original device.
        if original_device.is_gpu() {
            repaired.to_device(&original_device)
        } else {
            Ok(repaired)
        }
    }

    /// Check whether this tensor contains any non-finite (NaN or Inf) elements.
    ///
    /// Returns `true` if at least one element is NaN or ±Inf, `false` if all
    /// elements are finite. Non-float dtypes (U32, U8, I64) always return `false`.
    ///
    /// # GPU dispatch
    ///
    /// - **CPU tensors:** scans the ndarray slice directly (zero allocation).
    /// - **GPU tensors:** delegates to [`GpuBackend::count_non_finite`] which
    ///   reads unified memory without constructing a CPU `DynTensor`. Falls back
    ///   to full readback via `to_flat_vec::<f32>()` if no GPU backend is registered.
    pub fn any_non_finite(&self) -> Result<bool> {
        // Non-float dtypes are always finite.
        if !matches!(
            self.dtype(),
            DType::F32 | DType::BF16 | DType::F16 | DType::F64
        ) {
            return Ok(false);
        }

        if self.device().is_cpu() {
            // Zero-copy view for F32 (O(n) scan, no allocation), converting for BF16/F16 (O(n) + alloc).
            // bf16/f16 NaN maps to f32 NaN, so promotion is safe.
            return match self.as_cpu_f32() {
                Ok(view) => Ok(view.iter().any(|v| !v.is_finite())),
                Err(_) => {
                    let arr = self.to_f32_array()?;
                    Ok(arr.iter().any(|v| !v.is_finite()))
                }
            };
        }

        // GPU path: use backend's count_non_finite if available.
        if let Some(result) = crate::dyn_tensor::gpu::gpu_backend_dispatch_count_non_finite(self) {
            return Ok(result? > 0);
        }

        // Fallback: full readback (shouldn't happen with Metal backend registered).
        let arr = self.to_device(&Device::Cpu)?.to_f32_array()?;
        Ok(arr.iter().any(|v| !v.is_finite()))
    }
}
