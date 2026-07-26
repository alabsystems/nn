// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Step-by-step GroupNorm trace tests for HTDemucs DConv debugging.
//!
//! Extracted from `demucs_dconv_debug.rs` for the 500-line limit.
//!
//! Part of #887, Part of #779.

use std::collections::HashMap;

use super::demucs_test_utils::*;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::ReduceOp;
use nn_dsl::ScalarType;
use nn_metal::{MetalBackend, PipelineCache};

/// Step-by-step GroupNorm: early steps (mean, broadcast, square, variance).
#[test]
fn step_trace_early_steps() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let flat = 6usize;
    let cpu_mean = 3.5f32;
    let cpu_var = input.iter().map(|v| (v - cpu_mean).powi(2)).sum::<f32>() / flat as f32;

    // Step 2: Reduce(Mean)
    let mean_result = {
        let mut b = TensorBlockBuilder::new("s2_mean");
        let inp = b.add_input("data", &[1, flat]);
        let mean = b.add_reduce(inp, ReduceOp::Mean, 1, false, &[1]);
        let def = b.build(mean).expect("build");
        let mut m: HashMap<&str, &[f32]> = HashMap::new();
        m.insert("data", &input);
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("step2")
    };
    eprintln!("Step 2 (mean): {mean_result:?} (expected {cpu_mean})");
    assert!((mean_result[0] - cpu_mean).abs() < 1e-5, "mean wrong");

    // Step 5: squared = centered^2
    let cpu_centered: Vec<f32> = input.iter().map(|v| v - cpu_mean).collect();
    let sq_result = {
        let mut b = TensorBlockBuilder::new("s5_sq");
        let a = b.add_input("a", &[1, flat]);
        let bv = b.add_input("b", &[1, flat]);
        let sq = b.add_binary_mul(a, bv, &[1, flat]);
        let def = b.build(sq).expect("build");
        let mut m: HashMap<&str, &[f32]> = HashMap::new();
        m.insert("a", &cpu_centered);
        m.insert("b", &cpu_centered);
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("step5")
    };
    let cpu_sq: Vec<f32> = cpu_centered.iter().map(|v| v * v).collect();
    eprintln!("Step 5 (squared): {sq_result:?}");
    eprintln!("  CPU expected:   {cpu_sq:?}");

    // Step 6: Reduce(Mean) of squared → variance
    let var_result = {
        let mut b = TensorBlockBuilder::new("s6_var");
        let inp = b.add_input("data", &[1, flat]);
        let var = b.add_reduce(inp, ReduceOp::Mean, 1, false, &[1]);
        let def = b.build(var).expect("build");
        let mut m: HashMap<&str, &[f32]> = HashMap::new();
        m.insert("data", &sq_result);
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("step6")
    };
    eprintln!(
        "Step 6 (variance): {var_result:?} (expected {cpu_var:.8})"
    );
}

/// Step-by-step GroupNorm: full comparison (step-by-step vs batched).
#[test]
fn step_trace_full_compare() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let flat = 6usize;
    let cpu_expected = cpu_expected_gn(&input);

    // Step-by-step: centered * rsqrt(var + eps)
    let cpu_mean = 3.5f32;
    let cpu_centered: Vec<f32> = input.iter().map(|v| v - cpu_mean).collect();
    let cpu_var = input.iter().map(|v| (v - cpu_mean).powi(2)).sum::<f32>() / flat as f32;
    let rsqrt_val = 1.0 / (cpu_var + 1e-5f32).sqrt();
    let rsqrt_bc = vec![rsqrt_val; flat];

    let mul_result = {
        let mut b = TensorBlockBuilder::new("s10_mul");
        let a = b.add_input("a", &[1, flat]);
        let bv = b.add_input("b", &[1, flat]);
        let mul = b.add_binary_mul(a, bv, &[1, flat]);
        let def = b.build(mul).expect("build");
        let mut m: HashMap<&str, &[f32]> = HashMap::new();
        m.insert("a", &cpu_centered);
        m.insert("b", &rsqrt_bc);
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("step10")
    };
    let err_steps = max_abs_err(&mul_result, &cpu_expected);
    eprintln!("Step-by-step max_err: {err_steps:.8}");

    // Full GroupNorm as single dispatch
    let full_gn = dispatch_gn_noaffine(&cache, &input, 2, 3);
    let err_full = max_abs_err(&full_gn, &cpu_expected);
    eprintln!("Full GN max_err:      {err_full:.8}");

    eprintln!("STEP-BY-STEP works: {}", err_steps < 1e-4);
    eprintln!("FULL PIPELINE works: {}", err_full < 1e-4);
    if err_steps < 1e-4 && err_full > 1e-4 {
        eprintln!("BUG: step-by-step correct but batched GroupNorm broken");
    }
}
