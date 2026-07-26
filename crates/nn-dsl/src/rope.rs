// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RoPE (K6) kernel — Rotary Position Embedding.
//!
//! Applies rotation to paired elements using precomputed frequencies:
//!
//! ```text
//! y_even = x_even * cos(freq) - x_odd * sin(freq)
//! y_odd  = x_even * sin(freq) + x_odd * cos(freq)
//! ```
//!
//! The tensor builder decomposes a single tensor rotation (`rope_rotate`) into:
//!
//! 1. Reshape `[BH, S, D]` → `[BH, S, D/2, 2]`
//! 2. AxisSelect even (index=0) and odd (index=1) pairs
//! 3. Broadcast freqs `[S, D/2]` → `[BH, S, D/2]`
//! 4. Elementwise `rope_cos` and `rope_sin` on `(x_even, x_odd, freq)`
//! 5. Stack results → `[BH, S, D/2, 2]`
//! 6. Reshape back to `[BH, S, D]`
//!
//! The original dvoice MSL kernel applies rotation to both Q and K in a
//! single dispatch. This design uses two separate `rope_rotate` invocations
//! (pure functional, no in-place mutation). Joint dispatch is a fusion
//! optimization (Phase 5).
//!
//! Part of #19 (K2-K8 kernel ports).
//!
//! # Naming convention (#336)
//!
//! - `rope_cos_scalar` / `rope_sin_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `rope_rotate_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_rope_rotate_kernel` — `KernelDef` IR builder

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, checked_slice_output, validate_finite_inputs,
    validate_finite_slice, validate_nonzero_dims,
};
use crate::lower::LowerError;
use crate::tensor_builders::{broadcast_node, elementwise_node, input_node};
use crate::tensor_ir::{
    BroadcastAlignment, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build the `rope_cos` scalar KernelDef.
///
/// Parameters: `x0`, `x1`, `freq` (3 params).
/// Computes: `x0 * cos(freq) - x1 * sin(freq)`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rope_cos_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn rope_cos(x0: f32, x1: f32, freq: f32) -> f32 {
            x0 * freq.cos() - x1 * freq.sin()
        }",
    )
}

/// Build the `rope_sin` scalar KernelDef.
///
/// Parameters: `x0`, `x1`, `freq` (3 params).
/// Computes: `x0 * sin(freq) + x1 * cos(freq)`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rope_sin_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn rope_sin(x0: f32, x1: f32, freq: f32) -> f32 {
            x0 * freq.sin() + x1 * freq.cos()
        }",
    )
}

/// Validate RoPE dimensions: all nonzero and `head_dim` must be even.
fn validate_dims(bh: usize, seq_len: usize, head_dim: usize) -> Result<(), KernelError> {
    validate_nonzero_dims(&[("bh", bh), ("seq_len", seq_len), ("head_dim", head_dim)])?;
    if !head_dim.is_multiple_of(2) {
        return Err(KernelError::InvalidDimension {
            name: "head_dim (must be even)",
            value: head_dim,
        });
    }
    Ok(())
}

/// Build the RoPE rotation (K6) `TensorKernelDef` for shape `[BH, S, D]`.
///
/// Decomposes into 9 nodes following the design in
/// `designs/2026-02-26-kernelir-paired-element-access.md`:
///
/// ```text
/// Node 0: x [BH, S, D]                 (Input)
/// Node 1: freqs [S, D/2]               (Input)
/// Node 2: x_pairs [BH, S, D/2, 2]      (Reshape)
/// Node 3: x_even [BH, S, D/2]          (AxisSelect axis=3, index=0)
/// Node 4: x_odd [BH, S, D/2]           (AxisSelect axis=3, index=1)
/// Node 5: freqs_bc [BH, S, D/2]        (Broadcast Right)
/// Node 6: y_even [BH, S, D/2]          (Elementwise: rope_cos(x0, x1, freq))
/// Node 7: y_odd [BH, S, D/2]           (Elementwise: rope_sin(x0, x1, freq))
/// Node 8: y_pairs [BH, S, D/2, 2]      (Stack axis=3)
/// Node 9: y [BH, S, D]                 (Reshape)
/// ```
///
/// # Arguments
///
/// * `bh` — batch × heads (first dimension)
/// * `seq_len` — sequence length
/// * `head_dim` — head dimension (must be even)
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if any dimension is 0 or `head_dim` is odd.
/// Returns [`TensorIRError::ScalarKernelBuild`] if the scalar kernel builders fail.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rope_rotate_kernel(
    bh: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_dims(bh, seq_len, head_dim)?;

    let half_dim = head_dim / 2;
    let full_shape = vec![bh, seq_len, head_dim];
    let pairs_shape = vec![bh, seq_len, half_dim, 2];
    let half_shape = vec![bh, seq_len, half_dim];

    let rope_cos =
        build_rope_cos_kernel().map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;
    let rope_sin =
        build_rope_sin_kernel().map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;

    Ok(TensorKernelDef {
        name: "rope_rotate".into(),
        nodes: vec![
            // Node 0: x input [BH, S, D]
            input_node(0, "x", &full_shape),
            // Node 1: freqs input [S, D/2]
            input_node(1, "freqs", &[seq_len, half_dim]),
            // Node 2: reshape to expose pairs [BH, S, D/2, 2]
            TensorNode {
                id: TensorNodeId::new(2),
                kind: TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: pairs_shape.clone(),
                },
                shape: pairs_shape.clone(),
            },
            // Node 3: x_even = axis_select(pairs, axis=3, index=0) → [BH, S, D/2]
            TensorNode {
                id: TensorNodeId::new(3),
                kind: TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(2),
                    axis: 3,
                    index: 0,
                },
                shape: half_shape.clone(),
            },
            // Node 4: x_odd = axis_select(pairs, axis=3, index=1) → [BH, S, D/2]
            TensorNode {
                id: TensorNodeId::new(4),
                kind: TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(2),
                    axis: 3,
                    index: 1,
                },
                shape: half_shape.clone(),
            },
            // Node 5: freqs broadcast [S, D/2] → [BH, S, D/2] (Right-aligned)
            broadcast_node(5, 1, &half_shape, BroadcastAlignment::Right),
            // Node 6: y_even = rope_cos(x_even, x_odd, freqs) → [BH, S, D/2]
            elementwise_node(6, rope_cos, &[3, 4, 5], &half_shape),
            // Node 7: y_odd = rope_sin(x_even, x_odd, freqs) → [BH, S, D/2]
            elementwise_node(7, rope_sin, &[3, 4, 5], &half_shape),
            // Node 8: stack [y_even, y_odd] at axis=3 → [BH, S, D/2, 2]
            TensorNode {
                id: TensorNodeId::new(8),
                kind: TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(6), TensorNodeId::new(7)],
                    axis: 3,
                },
                shape: pairs_shape,
            },
            // Node 9: reshape back to [BH, S, D]
            TensorNode {
                id: TensorNodeId::new(9),
                kind: TensorOpKind::Reshape {
                    input: TensorNodeId::new(8),
                    target_shape: full_shape.clone(),
                },
                shape: full_shape,
            },
        ],
        output: TensorNodeId::new(9),
    })
}

// Builder Kani proofs (Part of #659 AC3: rope_rotate builder harnesses).
// No stubbing needed — builder constructs graph, no trig calls.
#[cfg(kani)]
#[path = "rope_kani_builder.rs"]
mod kani_builder_proofs;

// Scalar bounds extracted to rope_bounds.rs (500-line file limit, #175 pattern).
#[path = "rope_bounds.rs"]
mod bounds;
pub use bounds::{rope_cos_scalar_bounds, rope_sin_scalar_bounds};

// --- Scalar reference implementations ---

/// Scalar reference for `rope_cos`: `x0 * cos(freq) - x1 * sin(freq)`.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed
/// result is non-finite despite all inputs being finite.
#[must_use = "returns a Result that may contain an error"]
pub fn rope_cos_scalar(x0: f32, x1: f32, freq: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x0", x0), ("x1", x1), ("freq", freq)])?;
    let result = x0 * freq.cos() - x1 * freq.sin();
    checked_scalar_output(result)
}

/// Scalar reference for `rope_sin`: `x0 * sin(freq) + x1 * cos(freq)`.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed
/// result is non-finite despite all inputs being finite.
#[must_use = "returns a Result that may contain an error"]
pub fn rope_sin_scalar(x0: f32, x1: f32, freq: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x0", x0), ("x1", x1), ("freq", freq)])?;
    let result = x0 * freq.sin() + x1 * freq.cos();
    checked_scalar_output(result)
}

// --- Tensor reference implementation ---

/// Rust reference implementation of RoPE rotation for differential testing.
///
/// Applies rotary position embedding to tensor `x` of shape `[BH, S, D]`
/// using precomputed frequencies `freqs` of shape `[S, D/2]`.
///
/// Returns the rotated tensor in the same layout.
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] if any dimension is 0 or `head_dim` is odd.
/// Returns [`KernelError::ShapeMismatch`] if input lengths don't match expected shapes.
#[must_use = "returns a Result that may contain an error"]
pub fn rope_rotate_ref(
    x: &[f32],
    freqs: &[f32],
    bh: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>, KernelError> {
    validate_dims(bh, seq_len, head_dim)?;

    let half_dim = head_dim / 2;
    let x_len = bh
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| KernelError::DimensionOverflow {
            dims: format!("{bh} * {seq_len} * {head_dim}"),
        })?;

    if x.len() != x_len {
        return Err(KernelError::ShapeMismatch {
            expected: x_len,
            got: x.len(),
        });
    }
    let freqs_len =
        seq_len
            .checked_mul(half_dim)
            .ok_or_else(|| KernelError::DimensionOverflow {
                dims: format!("{seq_len} * {half_dim}"),
            })?;
    if freqs.len() != freqs_len {
        return Err(KernelError::ShapeMismatch {
            expected: freqs_len,
            got: freqs.len(),
        });
    }
    validate_finite_slice("x", x)?;
    validate_finite_slice("freqs", freqs)?;

    let mut output = vec![0.0f32; x_len];

    for b in 0..bh {
        for s in 0..seq_len {
            for p in 0..half_dim {
                let freq = freqs[s * half_dim + p];
                let cos_f = freq.cos();
                let sin_f = freq.sin();
                let base = b * seq_len * head_dim + s * head_dim + p * 2;

                let x0 = x[base];
                let x1 = x[base + 1];
                output[base] = x0 * cos_f - x1 * sin_f;
                output[base + 1] = x0 * sin_f + x1 * cos_f;
            }
        }
    }

    checked_slice_output(&output)?;
    Ok(output)
}

// Kani proofs for scalar RoPE kernels (finiteness + norm preservation).
// Extracted to rope_kani_proofs.rs (500-line file limit, #175 pattern).
#[cfg(all(kani, feature = "kani-stubbing"))]
#[path = "rope_kani_proofs.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "rope_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rope_tests_bounds.rs"]
mod bounds_tests;

#[cfg(test)]
#[path = "rope_tests_error_paths.rs"]
mod error_path_tests;
