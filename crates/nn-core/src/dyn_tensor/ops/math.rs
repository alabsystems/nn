// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unary and elementwise mathematical operations for [`DynTensor`].
//!
//! Includes standard math functions (exp, log, sqrt, sin, cos, etc.) and
//! simple activation functions (relu, gelu, silu, sigmoid, tanh).
//! Compound/parametric ops (elu, snake, clamp, maximum, minimum, atan2,
//! repair_non_finite, any_non_finite) live in [`math_compound`].
//! Extracted from `dyn_tensor_ops.rs` (#2018) for file-size compliance.

use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend, trace, DynTensor, TensorStorage, UnaryOp};
use crate::{DType, Result};

/// Convert a UnaryOp to its corresponding TraceOp.
fn unary_op_to_trace_op(op: UnaryOp) -> TraceOp {
    match op {
        UnaryOp::Relu => TraceOp::Relu,
        UnaryOp::Gelu => TraceOp::Gelu,
        UnaryOp::GeluErf => TraceOp::GeluErf,
        UnaryOp::Silu => TraceOp::Silu,
        UnaryOp::Tanh => TraceOp::Tanh,
        UnaryOp::Sigmoid => TraceOp::Sigmoid,
        UnaryOp::Exp => TraceOp::Exp,
        UnaryOp::Log => TraceOp::Log,
        UnaryOp::Sqrt => TraceOp::Sqrt,
        UnaryOp::Sqr => TraceOp::Sqr,
        UnaryOp::Abs => TraceOp::Abs,
        UnaryOp::Neg => TraceOp::Neg,
        UnaryOp::Recip => TraceOp::Recip,
        UnaryOp::Sin => TraceOp::Sin,
        UnaryOp::Cos => TraceOp::Cos,
        UnaryOp::Floor => TraceOp::Floor,
        UnaryOp::Round => TraceOp::Round,
        UnaryOp::Fract => TraceOp::Fract,
        UnaryOp::Tan => TraceOp::Tan,
        UnaryOp::Ceil => TraceOp::Ceil,
        UnaryOp::Sign => TraceOp::Sign,
    }
}

/// Abramowitz & Stegun approximation of erf(x), max error ~1.5e-7.
///
/// Used by `gelu_erf` for the exact GELU formula. This avoids a dependency
/// on `libm::erff` while maintaining sufficient precision for ML inference.
fn erf_f32(x: f32) -> f32 {
    // Abramowitz & Stegun formula 7.1.26 (f32-truncated coefficients)
    let a1: f32 = 0.254_829_6;
    let a2: f32 = -0.284_496_74;
    let a3: f32 = 1.421_413_8;
    let a4: f32 = -1.453_152;
    let a5: f32 = 1.061_405_4;
    let p: f32 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Operations that require F32 precision due to BF16/F16 overflow or precision
/// loss. These get auto-upcast at the public API level (#2013).
///
/// - `Exp`: BF16 overflows at x > ~10 (F32 safe to x > 88)
/// - `Log`: precision loss near 1.0 in BF16 (7-bit mantissa)
/// - `Sqrt`: precision loss near 0.0 in BF16
/// - `Recip`: same precision concern as division
/// - `Sin`/`Cos`: precision loss for large arguments in BF16
/// - `Sigmoid`/`Silu`/`Gelu`/`GeluErf`/`Tanh`: contain exp() internally
const PRECISION_SENSITIVE_OPS: &[UnaryOp] = &[
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sqrt,
    UnaryOp::Recip,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tan,
    UnaryOp::Sigmoid,
    UnaryOp::Silu,
    UnaryOp::Gelu,
    UnaryOp::GeluErf,
    UnaryOp::Tanh,
];

/// Returns `true` if the given op requires F32 precision for correctness.
fn is_precision_sensitive(op: UnaryOp) -> bool {
    PRECISION_SENSITIVE_OPS.contains(&op)
}

impl DynTensor {
    /// Apply a unary operation, auto-upcasting BF16/F16 for precision-sensitive ops.
    fn unary_op_impl(&self, op: UnaryOp) -> Result<Self> {
        // Auto-upcast BF16/F16 to F32 for precision-sensitive ops (#2013).
        // GPU kernels use float accumulators internally (D1 of #2981), so skip
        // the upcast+downcast round-trip — saves 2 GPU dispatches per call.
        if is_precision_sensitive(op)
            && matches!(self.dtype(), DType::BF16 | DType::F16)
            && self.device().is_cpu()
        {
            let original_dtype = self.dtype();
            let f32_self = self.to_dtype(DType::F32)?;
            let result = f32_self.unary_op_impl(op)?;
            return result.to_dtype(original_dtype);
        }
        // Auto-dequantize quantized tensors before CPU/GPU dispatch.
        if self.is_quantized() {
            return self.dequantize()?.unary_op_impl(op);
        }
        let mut result = match &self.storage {
            TensorStorage::Cpu(_) => self.cpu_unary(op),
            TensorStorage::Gpu { .. } => {
                let backend = gpu_backend()?;
                backend.unary_op(op, self)
            }
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            let trace_op = unary_op_to_trace_op(op);
            if let Some(id) = trace::record_op(trace_op, &input_ids, result.dims(), result.dtype())
            {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    fn cpu_unary(&self, op: UnaryOp) -> Result<Self> {
        // Promote bf16/f16 to f32, compute, convert back (#1646 D3).
        let f32_arr = self.to_f32_array()?;
        let result = match op {
            UnaryOp::Relu => f32_arr.mapv(|x| x.max(0.0)),
            UnaryOp::Gelu => f32_arr.mapv(|x| {
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                x * 0.5 * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
            }),
            UnaryOp::Silu => f32_arr.mapv(|x| x / (1.0 + (-x).exp())),
            UnaryOp::Tanh => f32_arr.mapv(f32::tanh),
            UnaryOp::Sigmoid => f32_arr.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            UnaryOp::Exp => f32_arr.mapv(f32::exp),
            UnaryOp::Log => f32_arr.mapv(f32::ln),
            UnaryOp::Sqrt => f32_arr.mapv(f32::sqrt),
            UnaryOp::Sqr => f32_arr.mapv(|x| x * x),
            UnaryOp::Abs => f32_arr.mapv(f32::abs),
            UnaryOp::Neg => f32_arr.mapv(|x| -x),
            UnaryOp::Recip => f32_arr.mapv(|x| 1.0 / x),
            UnaryOp::Sin => f32_arr.mapv(f32::sin),
            UnaryOp::Cos => f32_arr.mapv(f32::cos),
            UnaryOp::GeluErf => f32_arr.mapv(|x| {
                // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
                x * 0.5 * (1.0 + erf_f32(x * std::f32::consts::FRAC_1_SQRT_2))
            }),
            UnaryOp::Floor => f32_arr.mapv(f32::floor),
            UnaryOp::Round => f32_arr.mapv(f32::round_ties_even),
            UnaryOp::Fract => f32_arr.mapv(|x| x - x.floor()),
            UnaryOp::Tan => f32_arr.mapv(f32::tan),
            UnaryOp::Ceil => f32_arr.mapv(f32::ceil),
            UnaryOp::Sign => f32_arr.mapv(|x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
        };
        Self::from_f32_result(result, self.dtype)
    }

    pub fn sqrt(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Sqrt)
    }
    pub fn sqr(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Sqr)
    }
    pub fn abs(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Abs)
    }
    pub fn exp(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Exp)
    }
    pub fn log(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Log)
    }
    /// Element-wise reciprocal: `1.0 / x`.
    ///
    /// # CPU/GPU behavioral divergence
    ///
    /// - **CPU:** Returns `Err(Unsupported)` if any output is non-finite
    ///   (Inf/NaN from zero or near-zero input). Matches the finiteness guard
    ///   on [`div()`](Self::div) and [`div_scalar()`](Self::div_scalar).
    /// - **GPU:** Returns `Ok(tensor)` containing `Inf` for zero inputs.
    ///   Model-level finiteness guards (#941/#958) catch non-finite values
    ///   at stage boundaries.
    ///
    /// This divergence is by design: GPU finiteness checks require a costly
    /// readback, and the established pattern is model-level validation
    /// rather than per-op validation on GPU. Code that relies on `recip()`
    /// failing for zero inputs should either validate inputs beforehand or
    /// use model-level guards on GPU.
    pub fn recip(&self) -> Result<Self> {
        let result = self.unary_op_impl(UnaryOp::Recip)?;
        // CPU finiteness check — matches check_div_result_finite in ops/mod.rs.
        // GPU tensors skip this (model-level NaN guards handle it).
        if result.device().is_cpu() {
            // Promote to f32 for finiteness check (bf16/f16 NaN maps to f32 NaN).
            let arr = result.to_f32_array()?;
            let non_finite = arr.iter().filter(|v| !v.is_finite()).count();
            if non_finite > 0 {
                return Err(crate::TensorError::Unsupported(format!(
                    "recip produced {non_finite} non-finite value(s) (Inf/NaN from zero or near-zero input)"
                )));
            }
        }
        Ok(result)
    }
    pub fn neg(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Neg)
    }
    pub fn sin(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Sin)
    }
    pub fn cos(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Cos)
    }
    pub fn relu(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Relu)
    }
    pub fn gelu(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Gelu)
    }
    pub fn silu(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Silu)
    }
    pub fn tanh(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Tanh)
    }
    pub fn sigmoid(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Sigmoid)
    }

    /// Exact erf-based GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
    ///
    /// Matches candle's `gelu_erf()` and PyTorch's `nn.GELU(approximate='none')`.
    /// More accurate than the tanh approximation in [`gelu()`](Self::gelu).
    ///
    /// CPU: uses A&S erf approximation (max error ~1.5e-7).
    /// GPU: fused Metal kernel using same A&S erf polynomial (single dispatch).
    pub fn gelu_erf(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::GeluErf)
    }

    /// Element-wise floor: largest integer ≤ x.
    ///
    /// GPU tensors dispatch to Metal `floor()` via the builder pipeline.
    pub fn floor(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Floor)
    }

    /// Element-wise round to nearest integer (banker's rounding on ties).
    pub fn round(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Round)
    }

    /// Element-wise fractional part: `x - floor(x)`.
    ///
    /// Matches MSL/GLSL `fract()` semantics (floor-based). Result is always
    /// in `[0, 1)` for finite inputs. Differs from Rust `f32::fract()` which
    /// uses trunc (`x - trunc(x)`) and can be negative for negative inputs.
    pub fn fract(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Fract)
    }

    /// Element-wise inverse square root: `1.0 / sqrt(x)`.
    ///
    /// Composed from [`sqrt()`](Self::sqrt) + [`recip()`](Self::recip).
    /// Matches PyTorch `torch.rsqrt()`.
    pub fn rsqrt(&self) -> Result<Self> {
        self.sqrt()?.recip()
    }

    /// Element-wise tangent: `tan(x)`.
    ///
    /// Auto-upcasts BF16/F16 to F32 on CPU for precision.
    /// GPU tensors fall back to CPU (no native GPU `tan` kernel yet).
    pub fn tan(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Tan)
    }

    /// Element-wise ceiling: smallest integer >= x.
    ///
    /// Matches PyTorch `torch.ceil()`.
    pub fn ceil(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Ceil)
    }

    /// Element-wise sign: -1 if x < 0, 0 if x == 0, 1 if x > 0.
    ///
    /// Matches PyTorch `torch.sign()`. NaN inputs produce 0.0 (same as
    /// the comparison-based definition, since NaN fails both `> 0` and `< 0`).
    pub fn sign(&self) -> Result<Self> {
        self.unary_op_impl(UnaryOp::Sign)
    }
}

#[path = "math_compound.rs"]
mod math_compound;

#[cfg(test)]
#[path = "tests_rsqrt.rs"]
mod tests_rsqrt;
