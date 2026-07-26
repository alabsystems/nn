#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for nn-metal test files.
//!
//! Consolidates `init()`, `assert_close()`, `assert_gpu_vals()`,
//! `assert_gpu_matches_cpu()`, and `make_cache()` which were previously
//! duplicated across 15+ test files.
//!
//! Usage from any `#[cfg(test)]` submodule:
//! ```ignore
//! // NOTE: ignore — uses crate-internal paths only valid within nn-metal
//! use crate::test_common::{init, assert_close};
//! ```
//!
//! Part of #1204: Metal test helper consolidation.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::Device;

use crate::metal_backend::MetalBackend;
use crate::PipelineCache;

/// Initialize Metal backend + register DynTensor GPU backend. Idempotent.
pub(crate) fn init() {
    let _ = MetalBackend::init();
    crate::register_metal_dyn_backend();
}

/// Assert two f32 slices are equal within tolerance, with per-element diff reporting.
///
/// Delegates to [`nn_core::test_utils::assert_close_with_label`].
pub(crate) fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    nn_core::test_utils::assert_close_with_label(actual, expected, tol, label);
}

/// Assert GPU tensor values match expected within tolerance.
///
/// Validates the tensor is on GPU, transfers to CPU, then compares.
pub(crate) fn assert_gpu_vals(t: &DynTensor, expected: &[f32], tol: f32, label: &str) {
    assert_eq!(t.device(), Device::metal(), "{label}: must stay on GPU");
    let vals = t
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in vals.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "{label} [{i}]: gpu={g}, expected={e}, diff={}",
            (g - e).abs()
        );
    }
}

/// Run nn layer forward on both CPU and GPU, compare results within tolerance.
pub(crate) fn assert_gpu_matches_cpu<F>(build_layer: F, tol: f32, label: &str)
where
    F: Fn(&Device) -> (Box<dyn Module>, DynTensor),
{
    init();
    // CPU forward
    let (cpu_layer, cpu_input) = build_layer(&Device::Cpu);
    let cpu_out = cpu_layer.forward(&cpu_input).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU forward
    let (gpu_layer, gpu_input) = build_layer(&Device::metal());
    assert_eq!(gpu_input.device(), Device::metal());
    let gpu_out = gpu_layer.forward(&gpu_input).unwrap();
    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "{label}: output should stay on GPU"
    );
    assert_eq!(gpu_out.dims(), cpu_out.dims(), "{label}: shape mismatch");

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, tol, label);
}

/// Create a `PipelineCache` backed by a Metal context, or `None` on
/// non-Metal platforms.
pub(crate) fn make_cache() -> Option<PipelineCache> {
    let backend = MetalBackend::init().ok()?;
    Some(PipelineCache::new(backend.context().clone()))
}
