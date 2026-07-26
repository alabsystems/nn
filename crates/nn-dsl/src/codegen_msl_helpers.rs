// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation helper functions.
//!
//! Pure mapping functions for MSL types, operators, precision tiers,
//! and literal formatting. Extracted from `codegen_msl.rs` to keep
//! the main emission module focused on structure.

use crate::ir::{BinOpKind, CompareOpKind, IRError, ScalarType, UnaryFnKind};
use crate::precision::PrecisionTier;

/// Map `BinOpKind` to its MSL operator string.
///
/// Decoupled from `BinOpKind::Display` so that Display can evolve for
/// human-readable purposes (logging, errors) without affecting MSL output.
pub(crate) fn msl_binop(op: BinOpKind) -> &'static str {
    match op {
        BinOpKind::Add => "+",
        BinOpKind::Sub => "-",
        BinOpKind::Mul => "*",
        BinOpKind::Div => "/",
    }
}

pub(crate) const MSL_PRELUDE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\n";

/// Metal hardware limit: buffer argument indices 0..=30 (31 slots total).
pub(crate) const MAX_METAL_BUFFER_INDEX: usize = 30;

/// Maximum number of direct-binding inputs for Stack/Concat kernels.
///
/// When a Stack or Concat operation has more than this many inputs, MSL
/// codegen switches to a "packed" kernel variant that packs all input
/// buffers into one contiguous buffer with an offsets array. This uses
/// only 4 buffer slots (packed_inputs, offsets, output, total) regardless
/// of input count, staying within the Metal hardware limit.
///
/// 28 inputs + 1 output + 1 total = 30, which fits indices 0..=29 with
/// one slot to spare for future metadata.
///
/// Re-exported as `pub` so the dispatch layer (nn-metal) can use the
/// same threshold to decide between direct-binding and packed-buffer
/// encoding. Part of #1649.
pub const MAX_DIRECT_BINDING_INPUTS: usize = 28;

pub(crate) fn compare_op(op: CompareOpKind) -> &'static str {
    match op {
        CompareOpKind::Eq => "==",
        CompareOpKind::Ne => "!=",
        CompareOpKind::Lt => "<",
        CompareOpKind::Le => "<=",
        CompareOpKind::Gt => ">",
        CompareOpKind::Ge => ">=",
    }
}

/// MSL scalar type name. Delegates to [`ScalarType::msl_str`].
pub(crate) fn msl_type(ty: ScalarType) -> &'static str {
    ty.msl_str()
}

/// MSL accumulator type. Delegates to [`ScalarType::msl_accumulator_str`].
pub(crate) fn msl_accumulator_type(ty: ScalarType) -> &'static str {
    ty.msl_accumulator_str()
}

/// Result of mapping a [`UnaryFnKind`] to its MSL emission pattern.
///
/// Most unary functions map to a named Metal intrinsic (`sin`, `exp`, etc.),
/// but `Recip` has no single-function equivalent — it must be emitted as
/// `T(1) / arg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MslUnaryOp {
    /// A named Metal intrinsic: emitted as `fn_name(arg)`.
    Named(&'static str),
    /// Reciprocal: emitted as `T(1) / arg` (no named MSL function).
    Reciprocal,
    /// Negation: emitted as `(-arg)`.
    Negation,
}

pub(crate) fn msl_fn(op: UnaryFnKind, tier: PrecisionTier) -> MslUnaryOp {
    match op {
        UnaryFnKind::Sin => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::sin",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::sin",
        }),
        UnaryFnKind::Cos => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::cos",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::cos",
        }),
        UnaryFnKind::Sqrt => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::sqrt",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::sqrt",
        }),
        UnaryFnKind::Rsqrt => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::rsqrt",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::rsqrt",
        }),
        UnaryFnKind::Exp => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::exp",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::exp",
        }),
        UnaryFnKind::Abs => MslUnaryOp::Named("metal::abs"),
        UnaryFnKind::Neg => MslUnaryOp::Negation,
        UnaryFnKind::Recip => MslUnaryOp::Reciprocal,
        UnaryFnKind::Tanh => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::tanh",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::tanh",
        }),
        UnaryFnKind::Log => MslUnaryOp::Named(match tier {
            PrecisionTier::Relaxed => "metal::log",
            PrecisionTier::Strict | PrecisionTier::Normal => "metal::precise::log",
        }),
        UnaryFnKind::Floor => MslUnaryOp::Named("metal::floor"),
        UnaryFnKind::Round => MslUnaryOp::Named("metal::rint"),
        UnaryFnKind::Fract => MslUnaryOp::Named("metal::fract"),
    }
}

pub(crate) const fn wrapper_out_buffer_index(param_count: usize) -> usize {
    param_count
}

pub(crate) const fn wrapper_total_buffer_index(param_count: usize) -> usize {
    param_count + 1
}

/// Validate that a scalar kernel's buffer count fits within Metal's limit.
///
/// Scalar kernels use `param_count + 2` buffers: one per param, plus `out`
/// and `total`. The highest buffer index is `param_count + 1`, which must
/// be at most `MAX_METAL_BUFFER_INDEX` (30).
pub(crate) fn validate_buffer_count(param_count: usize) -> Result<(), IRError> {
    let highest_index = wrapper_total_buffer_index(param_count);
    if highest_index > MAX_METAL_BUFFER_INDEX {
        return Err(IRError::BufferLimitExceeded {
            required: param_count + 2,
            max: MAX_METAL_BUFFER_INDEX + 1,
            max_index: MAX_METAL_BUFFER_INDEX,
        });
    }
    Ok(())
}

/// Ensure a literal value is representable in the target scalar type.
///
/// For F32, values beyond `f32::MAX` / below `f32::MIN` are clamped to prevent
/// overflow to infinity in the emitted MSL literal.
///
/// For F16/BF16, non-zero values smaller than the minimum positive f16 normal
/// would silently flush to zero, which can cause division-by-zero in safety
/// clamps (e.g., `max(alpha, 1e-8)` where 1e-8 underflows to 0.0h). This
/// function clamps such values to f16 MIN_POSITIVE to preserve intent.
///
/// BF16 uses F16 clamping because Apple GPUs emit bf16 as MSL `half` (f16).
pub(crate) fn clamp_literal_for_type(v: f64, ty: ScalarType) -> f64 {
    // NaN and infinity pass through — format_float handles them with MSL
    // macros (NAN, INFINITY). Clamping NaN is undefined (all comparisons
    // return false), so we must guard it explicitly.
    if !v.is_finite() {
        return v;
    }
    match ty {
        ScalarType::F32 => {
            let f32_max = f64::from(f32::MAX);
            if v > f32_max {
                f32_max
            } else if v < -f32_max {
                -f32_max
            } else {
                v
            }
        }
        // BF16 maps to MSL "half" (f16) on Apple GPUs, so literal clamping
        // uses f16 range to prevent overflow/underflow in the emitted MSL.
        ScalarType::F16 | ScalarType::BF16 => {
            let f16_min = f64::from(half::f16::MIN_POSITIVE);
            let f16_max = f64::from(half::f16::MAX);
            // Guard underflow: non-zero values below f16 MIN_POSITIVE flush to zero.
            if v > 0.0 && v < f16_min {
                f16_min
            } else if v < 0.0 && v > -f16_min {
                -f16_min
            // Guard overflow: values beyond f16 MAX become infinity in half.
            } else if v > f16_max {
                f16_max
            } else if v < -f16_max {
                -f16_max
            } else {
                v
            }
        }
    }
}

pub(crate) fn format_float(v: f64) -> String {
    // Guard: NaN/Infinity are not valid MSL float literals. The IR validator
    // rejects non-finite Literal values, but defend at the emission layer too
    // so that a future decoupling cannot produce silently invalid MSL.
    if v.is_nan() {
        return "NAN".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() {
            "(-INFINITY)".to_string()
        } else {
            "INFINITY".to_string()
        };
    }
    if v == 0.0 {
        // IEEE 754: distinguish -0.0 from +0.0 since 1.0/-0.0 = -inf in MSL.
        if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        }
    } else if v == v.floor() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

// `powi_stmts` moved to `crate::codegen_shared` (shared with HIP backend).
// Part of #3338.
