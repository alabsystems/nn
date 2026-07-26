// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-kernel output bounds dispatch for verification.
//!
//! Registry maps kernel names to analytical bounds functions. Kernels without
//! a registry entry fall through to the ±1e6 heuristic.
//!
//! Extracted from `ay/prove_dispatch.rs` (#859) to be always-available
//! without the `ay-smt` feature flag. Pure Rust math — no ay-bindings dependency.
//!
//! Refactored from string-match dispatch to registry pattern (#465).
//! Per-kernel bounds functions extracted to `dispatch_kernel_bounds.rs` (#2218).

use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::smt_error::SmtError;

// Per-kernel analytical bounds functions, extracted to stay under 450 lines.
#[path = "dispatch_kernel_bounds.rs"]
mod kernel_bounds;
// Re-export so `bounds/mod.rs` re-exports (`dispatch::bounds_*`) still resolve.
pub(crate) use kernel_bounds::*;

/// Analytical bounds function signature.
///
/// Arguments: `(constant_params, input_lower, input_upper)`.
/// Returns `Ok(Some((lo, hi)))` for analytical bounds, `Ok(None)` to fall
/// through to the heuristic (e.g. snake with non-positive alpha), or
/// `Err(...)` on validation failure.
type AnalyticalBoundsFn = fn(&[f32], f32, f32) -> Result<Option<(f64, f64)>, VerifyError>;

/// Registry entry mapping a kernel name to its analytical bounds function.
struct BoundsEntry {
    name: &'static str,
    min_constant_params: usize,
    bounds_fn: AnalyticalBoundsFn,
}

/// Registry of kernels with known analytical bounds.
///
/// When adding a new kernel to `nn-dsl`, add a corresponding entry here.
/// Missing entries fall through to the ±1e6 heuristic, which is logged to
/// stderr — grep for "using ±1e6 fallback" to detect missing registrations.
const BOUNDS_REGISTRY: &[BoundsEntry] = &[
    BoundsEntry {
        name: "snake",
        min_constant_params: 1,
        bounds_fn: bounds_snake,
    },
    BoundsEntry {
        name: "silu_mul",
        min_constant_params: 1,
        bounds_fn: bounds_silu_mul,
    },
    BoundsEntry {
        name: "rope_cos",
        min_constant_params: 2,
        bounds_fn: bounds_rope_cos,
    },
    BoundsEntry {
        name: "rope_sin",
        min_constant_params: 2,
        bounds_fn: bounds_rope_sin,
    },
    BoundsEntry {
        name: "rms_norm_scalar",
        min_constant_params: 2,
        bounds_fn: bounds_rms_norm_scalar,
    },
    BoundsEntry {
        name: "layer_norm_scalar",
        min_constant_params: 5,
        bounds_fn: bounds_norm_affine,
    },
    BoundsEntry {
        name: "instance_norm_scalar",
        min_constant_params: 3,
        bounds_fn: bounds_instance_norm,
    },
    BoundsEntry {
        name: "instance_norm_affine_scalar",
        min_constant_params: 5,
        bounds_fn: bounds_norm_affine,
    },
    BoundsEntry {
        name: "adain",
        min_constant_params: 5,
        bounds_fn: bounds_adain,
    },
    BoundsEntry {
        name: "adain_snake",
        min_constant_params: 6,
        bounds_fn: bounds_adain_snake,
    },
    BoundsEntry {
        name: "gelu",
        min_constant_params: 0,
        bounds_fn: bounds_gelu,
    },
    BoundsEntry {
        name: "sigmoid",
        min_constant_params: 0,
        bounds_fn: bounds_sigmoid,
    },
    BoundsEntry {
        name: "relu",
        min_constant_params: 0,
        bounds_fn: bounds_relu,
    },
    BoundsEntry {
        name: "tanh_act",
        min_constant_params: 0,
        bounds_fn: bounds_tanh_act,
    },
    BoundsEntry {
        name: "leaky_relu",
        min_constant_params: 1,
        bounds_fn: bounds_leaky_relu,
    },
    BoundsEntry {
        name: "exp",
        min_constant_params: 0,
        bounds_fn: bounds_exp,
    },
    BoundsEntry {
        name: "softplus",
        min_constant_params: 0,
        bounds_fn: bounds_softplus,
    },
    BoundsEntry {
        name: "add",
        min_constant_params: 1,
        bounds_fn: bounds_binary_add,
    },
    BoundsEntry {
        name: "mul",
        min_constant_params: 1,
        bounds_fn: bounds_binary_mul,
    },
    BoundsEntry {
        name: "conv1d_k1_scalar",
        min_constant_params: 2,
        bounds_fn: bounds_conv1d_k1_scalar,
    },
    BoundsEntry {
        name: "adain_leaky_relu",
        min_constant_params: 6,
        bounds_fn: bounds_adain_leaky_relu,
    },
    BoundsEntry {
        name: "ada_layer_norm",
        min_constant_params: 7,
        bounds_fn: bounds_ada_layer_norm,
    },
];

/// Compute output bounds heuristic for kernels.
///
/// For kernels where we have analytical bounds, use those.
/// For other kernels, use a conservative heuristic based on input range.
///
/// Returns `(lower, upper, is_heuristic)` where `is_heuristic` is `true`
/// when the conservative ±1e6 fallback was used instead of analytical bounds.
/// Callers must not treat results from heuristic bounds as meaningful proofs (#385).
///
/// These heuristics are a Phase A stopgap. Phase B replaces them with
/// NY IBP output bounds, making the SMT check a cross-verification
/// of the IBP result rather than a standalone proof.
///
/// **Parameter convention (NY, #448):**
/// `constant_params` holds values for kernel params 1..N (params after the
/// variable). Param 0 is always the symbolic variable bounded by
/// `[input_lower, input_upper]`. So `constant_params[0]` = kernel param 1,
/// `constant_params[1]` = kernel param 2, etc.
///
/// **#459 fix:** All bounds functions now use the #448 variable-first
/// convention. `constant_params[i]` maps to kernel param `i+1`.
pub(crate) fn compute_output_bounds_heuristic(
    kernel: &KernelDef,
    constant_params: &[f32],
    input_lower: f32,
    input_upper: f32,
) -> Result<(f64, f64, bool), VerifyError> {
    // Defense-in-depth: validate input bounds finiteness (#471, #394 convention).
    if !input_lower.is_finite() || !input_upper.is_finite() {
        return Err(SmtError::NonFiniteInputBound {
            lower: f64::from(input_lower),
            upper: f64::from(input_upper),
        }
        .into());
    }

    // Validate ALL constant_params finiteness before use.
    for (i, &val) in constant_params.iter().enumerate() {
        if !val.is_finite() {
            return Err(SmtError::NonFiniteConstantParam {
                index: i + 1,
                value: f64::from(val),
            }
            .into());
        }
    }

    // Look up analytical bounds in the registry (#465).
    for entry in BOUNDS_REGISTRY {
        if kernel.name == entry.name && constant_params.len() >= entry.min_constant_params {
            if let Some((lo, hi)) = (entry.bounds_fn)(constant_params, input_lower, input_upper)? {
                return Ok((lo, hi, false));
            }
        }
    }

    // Conservative heuristic: input range ± a large margin.
    let lo_f64 = f64::from(input_lower);
    let hi_f64 = f64::from(input_upper);
    let _ = std::io::Write::write_fmt(
        &mut std::io::stderr(),
        format_args!(
            "nn-verify: compute_output_bounds_heuristic: kernel '{}' using ±1e6 fallback \
             — no analytical bounds available\n",
            kernel.name
        ),
    );
    Ok((lo_f64 - 1e6, hi_f64 + 1e6, true))
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod bounds_dispatch_tests;

#[cfg(test)]
#[path = "dispatch_error_tests.rs"]
mod bounds_error_tests;
