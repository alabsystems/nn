// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: verify `#[nn::kernel]` syntax works via the top-level crate.
//!
//! Requires the `dsl` feature (proc-macro + nn-dsl types).

#[nn::kernel]
fn add_one(x: f32) -> f32 {
    x + 1.0
}

#[test]
fn test_nn_kernel_attribute_works() {
    // The proc-macro expands via `#[nn::kernel]` — if this compiles,
    // the re-export is working.
    assert!((add_one(2.0) - 3.0).abs() < 1e-6);
}

#[test]
fn test_nn_kernel_generates_msl_constant() {
    // `#[nn::kernel]` should generate ADD_ONE_MSL alongside the function.
    assert!(
        ADD_ONE_MSL.contains("_nn_add_one"),
        "MSL should contain _nn_ prefixed helper: {ADD_ONE_MSL}"
    );
}

#[test]
fn test_nn_kernel_generates_descriptor() {
    let desc: nn::KernelDescriptor = ADD_ONE_DESCRIPTOR;
    assert_eq!(desc.param_count, 1);
    assert_eq!(desc.entry_point, "add_one_kernel");
}

#[test]
fn test_core_type_re_exports() {
    // Verify core types are accessible through `nn::`.
    fn _assert_tensor_type<const D: usize, T: nn::TensorElement, B: nn::Backend>() {}
    let _: nn::PrecisionTier = nn::PrecisionTier::Normal;
}
