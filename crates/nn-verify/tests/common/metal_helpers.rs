// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Metal parity testing helpers for nn-verify integration tests.
//!
//! Deduplicates PRNG, Metal setup, and assertion utilities that were
//! previously copy-pasted between `metal_ir_parity.rs` and
//! `metal_ir_parity_reduction.rs`.
//!
//! Usage from any integration test:
//! ```ignore
//! // NOTE: ignore — file-system relative #[path] declaration for integration tests
//! #[path = "common/metal_helpers.rs"]
//! mod metal_helpers;
//! use metal_helpers::{rand_f32_vec, metal_setup, ...};
//! ```

#![allow(dead_code, unreachable_pub)]

use nn_dsl::{within_differential_budget, PrecisionContract, PrecisionTier, ScalarType};
use nn_metal::{MetalBackend, MetalContext, PipelineCache};
use nn_reftest::{compare_tensors, ComparisonConfig, NamedTensor};

// ---------------------------------------------------------------------------
// Deterministic PRNG — re-exported from nn_core::test_prng (#1411)
// ---------------------------------------------------------------------------

pub use nn_core::test_prng::rand_f32_vec;

// ---------------------------------------------------------------------------
// Metal setup
// ---------------------------------------------------------------------------

pub(crate) fn metal_setup() -> PipelineCache {
    // Initialize the global MetalBackend singleton before any GPU dispatch.
    // Element-wise/elementwise dispatch (`dispatch_elementwise`) routes through
    // the global backend, which otherwise fails with "Metal backend not
    // initialized -- call MetalBackend::init() first". `init()` is idempotent;
    // the `let _ =` mirrors the existing convention in
    // metal_ir_parity_reduction.rs and nn-metal's own test utils. We do not
    // unwrap so a machine without a usable Metal device fails later at the
    // dispatch site with a precise error rather than here.
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal device required");
    PipelineCache::new(ctx)
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// Compare GPU output vs CPU reference via nn-reftest tensor comparison.
pub(crate) fn assert_metal_cpu_parity(name: &str, gpu: &[f32], cpu: &[f32]) {
    let config = ComparisonConfig::default();
    let ref_tensor = NamedTensor::new(format!("{name}_cpu"), vec![cpu.len()], cpu.to_vec())
        .expect("valid test tensor");
    let cand_tensor = NamedTensor::new(format!("{name}_gpu"), vec![gpu.len()], gpu.to_vec())
        .expect("valid test tensor");
    let result =
        compare_tensors(&ref_tensor, &cand_tensor, &config).expect("comparison should succeed");
    assert!(
        result.passed,
        "{name}: Metal/IR parity failed — max_abs={:.6e}, mean_abs={:.6e}, cosine={:.8}",
        result.max_abs_diff, result.mean_abs_diff, result.cosine_similarity,
    );
}

/// Per-element precision check with named-input diagnostics on failure.
pub(crate) fn assert_within_budget(
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    inputs: &[(&str, &[f32])],
) {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    for (i, (&r, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        if within_differential_budget(r, g, contract) {
            continue;
        }
        let in_vals: Vec<String> = inputs
            .iter()
            .filter(|(_, s)| i < s.len())
            .map(|(n, s)| format!("{n}={}", s[i]))
            .collect();
        let extra = if in_vals.is_empty() {
            String::new()
        } else {
            format!(", {}", in_vals.join(", "))
        };
        panic!(
            "{name}[{i}]: out of budget — cpu={r}, gpu={g}, delta={:.6e}{extra}",
            (r - g).abs()
        );
    }
}
