// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for AdaIN tensor builder functions.
//!
//! Tensor builders (symbolic proofs via kani::any()):
//! - `build_adain1d(channels, time)` — tensor TensorKernelDef
//! - `build_snake_tensor(channels, time)` — tensor TensorKernelDef
//! - `build_adain_snake_tensor(channels, time)` — tensor TensorKernelDef
//!
//! Scalar builders (deterministic, in `#[cfg(test)] mod tests`):
//! - `build_adain_scalar_kernel` — scalar KernelDef (6 params)
//! - `build_snake_scalar_kernel` — scalar KernelDef (2 params)
//! - `build_adain_snake_fused_kernel` — scalar KernelDef (7 params)
//!
//! Part of #752 AC3.

use super::{
    build_adain1d, build_adain_scalar_kernel, build_adain_snake_fused_kernel,
    build_adain_snake_tensor, build_snake_scalar_kernel, build_snake_tensor,
};

// ---------------------------------------------------------------------------
// Tensor TensorKernelDef builders — symbolic proofs
// ---------------------------------------------------------------------------

/// Proves `build_adain1d` does not panic for any bounded params.
#[kani::unwind(1)]
#[kani::proof]
fn adain1d_tensor_build_no_panic() {
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(channels <= 4);
    kani::assume(time <= 4);

    let _ = build_adain1d(channels, time);
}

/// Proves the AdaIN1d tensor output shape is `[channels, time]`.
#[kani::unwind(1)]
#[kani::proof]
fn adain1d_tensor_output_shape_preserved() {
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 4);
    kani::assume(time >= 1 && time <= 4);

    let def = build_adain1d(channels, time).expect("valid params must succeed");

    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape.len(), 2, "output rank must be 2");
    assert_eq!(
        output_shape[0], channels,
        "output dim 0 must equal channels"
    );
    assert_eq!(output_shape[1], time, "output dim 1 must equal time");
}

/// Proves `build_snake_tensor` does not panic for any bounded params.
#[kani::unwind(1)]
#[kani::proof]
fn snake_tensor_build_no_panic() {
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(channels <= 4);
    kani::assume(time <= 4);

    let _ = build_snake_tensor(channels, time);
}

/// Proves the Snake K1 tensor output shape is `[channels, time]`.
#[kani::unwind(1)]
#[kani::proof]
fn snake_tensor_output_shape_preserved() {
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 4);
    kani::assume(time >= 1 && time <= 4);

    let def = build_snake_tensor(channels, time).expect("valid params must succeed");

    let input_shape = &def.nodes[0].shape;
    let output_shape = &def.nodes[def.output.index()].shape;

    assert_eq!(
        input_shape, output_shape,
        "Snake K1 output shape must equal input shape [C, T]"
    );
    assert_eq!(
        output_shape[0], channels,
        "output dim 0 must equal channels"
    );
    assert_eq!(output_shape[1], time, "output dim 1 must equal time");
}

/// Proves `build_adain_snake_tensor` does not panic for any bounded params.
#[kani::unwind(1)]
#[kani::proof]
fn adain_snake_tensor_build_no_panic() {
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(channels <= 4);
    kani::assume(time <= 4);

    let _ = build_adain_snake_tensor(channels, time);
}

/// Proves the fused AdaIN+Snake K4 tensor output shape is `[channels, time]`.
#[kani::unwind(1)]
#[kani::proof]
fn adain_snake_tensor_output_shape_preserved() {
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 4);
    kani::assume(time >= 1 && time <= 4);

    let def = build_adain_snake_tensor(channels, time).expect("valid params must succeed");

    let input_shape = &def.nodes[0].shape;
    let output_shape = &def.nodes[def.output.index()].shape;

    assert_eq!(
        input_shape, output_shape,
        "AdaIN+Snake K4 output shape must equal input shape [C, T]"
    );
    assert_eq!(
        output_shape[0], channels,
        "output dim 0 must equal channels"
    );
    assert_eq!(output_shape[1], time, "output dim 1 must equal time");
}

// ---------------------------------------------------------------------------
// Deterministic structural tests (converted from tautological Kani harnesses)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ScalarType;

    #[test]
    fn adain_scalar_build_no_panic() {
        let def = build_adain_scalar_kernel().expect("adain scalar kernel build must succeed");
        assert!(!def.name.is_empty(), "kernel name must not be empty");
    }

    #[test]
    fn snake_scalar_build_no_panic() {
        let def = build_snake_scalar_kernel().expect("snake scalar kernel build must succeed");
        assert!(!def.name.is_empty(), "kernel name must not be empty");
    }

    #[test]
    fn adain_snake_fused_build_no_panic() {
        let def =
            build_adain_snake_fused_kernel().expect("adain_snake fused kernel build must succeed");
        assert!(!def.name.is_empty(), "kernel name must not be empty");
    }

    #[test]
    fn adain_scalar_param_count() {
        let def = build_adain_scalar_kernel().expect("build must succeed");
        assert_eq!(def.params.len(), 6);
    }

    #[test]
    fn adain_scalar_return_type() {
        let def = build_adain_scalar_kernel().expect("build must succeed");
        assert_eq!(def.return_type, ScalarType::F32);
    }

    #[test]
    fn snake_scalar_param_count() {
        let def = build_snake_scalar_kernel().expect("build must succeed");
        assert_eq!(def.params.len(), 2);
    }

    #[test]
    fn snake_scalar_return_type() {
        let def = build_snake_scalar_kernel().expect("build must succeed");
        assert_eq!(def.return_type, ScalarType::F32);
    }

    #[test]
    fn adain_snake_fused_param_count() {
        let def = build_adain_snake_fused_kernel().expect("build must succeed");
        assert_eq!(def.params.len(), 7);
    }

    #[test]
    fn adain_snake_fused_return_type() {
        let def = build_adain_snake_fused_kernel().expect("build must succeed");
        assert_eq!(def.return_type, ScalarType::F32);
    }
}
