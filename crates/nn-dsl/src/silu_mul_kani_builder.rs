// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `build_silu_mul_tensor`.
//!
//! Tensor builder: no-panic and shape-preserved proofs.
//! Scalar builder tests in `#[cfg(test)] mod tests` (deterministic, no symbolic inputs).
//!
//! Part of #752 AC2.

use super::{build_silu_mul_kernel, build_silu_mul_tensor};

// --- Tensor TensorKernelDef builder ---

/// Proves `build_silu_mul_tensor` does not panic for any bounded params,
/// including invalid ones (0 dimensions) that return `Err`.
#[kani::unwind(1)]
#[kani::proof]
fn silu_mul_tensor_build_no_panic() {
    let n: usize = kani::any();
    let dim: usize = kani::any();

    kani::assume(n <= 4);
    kani::assume(dim <= 4);

    let _ = build_silu_mul_tensor(n, dim);
}

/// Proves the output shape matches the input shape `[N, dim]`.
///
/// SiLU-Mul is element-wise: output has the same shape as input.
#[kani::unwind(1)]
#[kani::proof]
fn silu_mul_tensor_output_shape_preserved() {
    let n: usize = kani::any();
    let dim: usize = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(dim >= 1 && dim <= 4);

    let def = build_silu_mul_tensor(n, dim).expect("valid params must succeed");

    let input_shape = &def.nodes[0].shape;
    let output_shape = &def.nodes[def.output.index()].shape;

    assert_eq!(
        input_shape, output_shape,
        "SiLU-Mul output shape must equal input shape [N, dim]"
    );
    assert_eq!(output_shape.len(), 2, "output rank must be 2");
    assert_eq!(output_shape[0], n, "output dim 0 must equal N");
    assert_eq!(output_shape[1], dim, "output dim 1 must equal dim");
}

// ---------------------------------------------------------------------------
// Deterministic structural tests (converted from tautological Kani harnesses)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ScalarType;

    #[test]
    fn silu_mul_build_no_panic() {
        let def = build_silu_mul_kernel().expect("silu_mul kernel build must succeed");
        assert!(!def.name.is_empty(), "kernel name must not be empty");
    }

    #[test]
    fn silu_mul_param_count() {
        let def = build_silu_mul_kernel().expect("build must succeed");
        assert_eq!(def.params.len(), 2);
    }

    #[test]
    fn silu_mul_return_type() {
        let def = build_silu_mul_kernel().expect("build must succeed");
        assert_eq!(def.return_type, ScalarType::F32);
    }
}
