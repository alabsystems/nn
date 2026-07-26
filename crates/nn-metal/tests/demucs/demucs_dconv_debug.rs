// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Isolated DConv sublayer debugging tests for HTDemucs parity.
//!
//! These tests dispatch individual DConv sub-steps (compress Conv1d,
//! GroupNorm, GELU, expand Conv1d, GLU, LayerScale) on GPU and compare
//! against known Python reference values.
//!
//! Part of #779 — Milestone 1, AC4 debugging.

use std::collections::HashMap;

use super::demucs_test_utils::*;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::ReduceOp;
use nn_dsl::ScalarType;
use nn_metal::{MetalBackend, PipelineCache};

// ---------------------------------------------------------------------------
// Tests: DConv parity with real weights
// ---------------------------------------------------------------------------

/// Test compress Conv1d (with causal padding) matches Python.
#[test]
fn dconv_compress_conv_parity() {
    let (weights, cache, audio) = match load_test_weights() {
        Some(t) => t,
        None => return,
    };
    let normalized = normalize_audio(&audio, 1024);
    let gelu_out = dispatch_conv_gelu(&cache, &normalized, &weights, 1024);
    eprintln!("Conv+GELU: len={}", gelu_out.len());

    let compress_out = dispatch_compress_conv(&cache, &gelu_out, &weights, 48, 256);
    eprintln!(
        "Compress Conv: len={}, first8={:?}",
        compress_out.len(),
        &compress_out[..8]
    );

    let py_compress = [
        0.8272429_f32,
        1.8584472,
        3.3166769,
        2.529_765,
        2.449_966,
        1.985_945,
        2.1772158,
        3.2681124,
    ];
    let v = compare_first_n("compress_conv", &compress_out, &py_compress, 8, 1e-4);
    assert_eq!(v, 0, "compress Conv1d should match Python within 1e-4");
}

/// Test GroupNorm(G=1) after compress Conv matches Python.
#[test]
fn dconv_group_norm_parity() {
    let (weights, cache, audio) = match load_test_weights() {
        Some(t) => t,
        None => return,
    };
    let normalized = normalize_audio(&audio, 1024);
    let gelu_out = dispatch_conv_gelu(&cache, &normalized, &weights, 1024);
    let compress_out = dispatch_compress_conv(&cache, &gelu_out, &weights, 48, 256);

    let dc = &weights.encoder.blocks[0].dconv[0];
    let gn_out = dispatch_group_norm_g1(
        &cache,
        &compress_out,
        &dc.norm_compress_gamma,
        &dc.norm_compress_beta,
        6,
        256,
    );
    eprintln!(
        "GroupNorm1: len={}, first8={:?}",
        gn_out.len(),
        &gn_out[..8]
    );

    let unique_count = {
        let mut s: Vec<u32> = gn_out.iter().map(|v| v.to_bits()).collect();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    eprintln!("  unique values: {unique_count}/{}", gn_out.len());

    let py_gn1 = [
        0.458_398_4_f32,
        0.701_507_4,
        1.045_288_7,
        0.85977226,
        0.840_959_4,
        0.73156536,
        0.77665794,
        1.033_839_5,
    ];
    let v = compare_first_n("group_norm1", &gn_out, &py_gn1, 8, 1e-3);
    if v > 0 {
        eprintln!("GroupNorm1 diverges — this is the likely root cause");
    }
}

/// Test GroupNorm(G=1) without affine (normalize only) to isolate the issue.
#[test]
fn dconv_group_norm_no_affine() {
    let (weights, cache, audio) = match load_test_weights() {
        Some(t) => t,
        None => return,
    };
    let normalized = normalize_audio(&audio, 1024);
    let gelu_out = dispatch_conv_gelu(&cache, &normalized, &weights, 1024);
    let compress_out = dispatch_compress_conv(&cache, &gelu_out, &weights, 48, 256);

    let result = dispatch_gn_noaffine(&cache, &compress_out, 6, 256);
    eprintln!(
        "GN1 no-affine: len={}, first8={:?}",
        result.len(),
        &result[..8]
    );

    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    let var: f32 = result.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / result.len() as f32;
    eprintln!("  output mean={:.8}, std={:.8}", mean, var.sqrt());
}

/// Test Reshape → Reduce(Mean) to isolate where values go wrong.
#[test]
fn dconv_group_norm_mean_only() {
    let (weights, cache, audio) = match load_test_weights() {
        Some(t) => t,
        None => return,
    };
    let normalized = normalize_audio(&audio, 1024);
    let gelu_out = dispatch_conv_gelu(&cache, &normalized, &weights, 1024);
    let compress_out = dispatch_compress_conv(&cache, &gelu_out, &weights, 48, 256);

    let channels: usize = 6;
    let t_len: usize = 256;
    let flat = channels * t_len;

    let mut b = TensorBlockBuilder::new("mean_only");
    let inp = b.add_input("data", &[channels, t_len]);
    let reshaped = b.add_reshape(inp, &[1, flat]);
    let mean = b.add_reduce(reshaped, ReduceOp::Mean, 1, false, &[1]);
    let def = b.build(mean).expect("build");

    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
    inputs.insert("data", &compress_out);
    let result =
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("mean");
    eprintln!("Mean: {:?} (len={})", &result, result.len());

    let cpu_mean: f32 = compress_out.iter().sum::<f32>() / compress_out.len() as f32;
    eprintln!("CPU mean: {cpu_mean:.8}");

    let norm_result = dispatch_gn_noaffine(&cache, &compress_out, channels, t_len);
    let nonzero = norm_result.iter().filter(|v| v.abs() > 1e-6).count();
    eprintln!("  GN no-affine nonzero: {nonzero}/{}", norm_result.len());
}

// ---------------------------------------------------------------------------
// Tests: GroupNorm synthetic (small known values)
// ---------------------------------------------------------------------------

/// GroupNorm G=1: mean-only and broadcast steps (Steps A, B).
#[test]
fn gn_synthetic_mean_broadcast() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let flat = 6usize;

    // Step A: Reshape → Reduce(Mean)
    let mut b = TensorBlockBuilder::new("step_a");
    let inp = b.add_input("data", &[2, 3]);
    let reshaped = b.add_reshape(inp, &[1, flat]);
    let mean = b.add_reduce(reshaped, ReduceOp::Mean, 1, false, &[1]);
    let def = b.build(mean).expect("build");
    let mut m: HashMap<&str, &[f32]> = HashMap::new();
    m.insert("data", &input);
    let r = nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("step_a");
    eprintln!("A (mean): {r:?} (expected 3.5)");
    assert!((r[0] - 3.5).abs() < 1e-5);

    // Step B: Reduce(Mean) → Broadcast
    let mut b2 = TensorBlockBuilder::new("step_b");
    let inp2 = b2.add_input("data", &[1, flat]);
    let mean2 = b2.add_reduce(inp2, ReduceOp::Mean, 1, false, &[1]);
    let bc = b2.add_broadcast_left(mean2, &[1, flat]);
    let def2 = b2.build(bc).expect("build");
    let mut m2: HashMap<&str, &[f32]> = HashMap::new();
    m2.insert("data", &input);
    let r2 =
        nn_metal::execute_tensor_dispatch(&cache, &def2, ScalarType::F32, &m2).expect("step_b");
    eprintln!("B (broadcast mean): {r2:?} (expected all 3.5)");
    assert!(r2.iter().all(|&v| (v - 3.5).abs() < 1e-5));
}

/// GroupNorm G=1: full pipeline with power-of-2 and non-power-of-2 sizes.
#[test]
fn gn_synthetic_sizes() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());

    // 8 elements (power-of-2)
    let input8 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result8 = dispatch_gn_noaffine(&cache, &input8, 2, 4);
    let expected8 = cpu_expected_gn(&input8);
    let err8 = max_abs_err(&result8, &expected8);
    eprintln!("C8 (GroupNorm [2,4]): {:?}", &result8);
    eprintln!("   expected:         {:?}", &expected8);
    eprintln!("max error (8 elems): {err8:.8}");

    // 6 elements (non-power-of-2)
    let input6 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result6 = dispatch_gn_noaffine(&cache, &input6, 2, 3);
    let expected6 = cpu_expected_gn(&input6);
    let err6 = max_abs_err(&result6, &expected6);
    eprintln!("D (GroupNorm [2,3]): {:?}", &result6);
    eprintln!("   expected:        {:?}", &expected6);
    eprintln!("max error (6 elems): {err6:.8}");
    assert!(err6 < 1e-4, "GroupNorm G=1 failed, max_err={err6}");
}

/// Test Reduce(Mean) with 6 elements — non-power-of-2.
#[test]
fn reduce_mean_6_elements() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b = TensorBlockBuilder::new("reduce6");
    let inp = b.add_input("data", &[1, 6]);
    let mean = b.add_reduce(inp, ReduceOp::Mean, 1, false, &[1]);
    let def = b.build(mean).expect("build");

    let mut m: HashMap<&str, &[f32]> = HashMap::new();
    m.insert("data", &input);
    let result =
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("reduce");
    eprintln!("Reduce(Mean, 6 elements): {result:?} (expected 3.5)");
    assert!(
        (result[0] - 3.5).abs() < 1e-6,
        "mean mismatch: got {}",
        result[0]
    );
}

/// Dump the dispatch plan for GroupNorm to understand step sequence.
#[test]
fn group_norm_g1_dump_plan() {
    let mut b = TensorBlockBuilder::new("gn_dump");
    let inp = b.add_input("data", &[2, 3]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(inp, eps, None, None, 2, 3);
    let def = b.build(out).expect("build");

    let (plan, effective_output) =
        nn_dsl::build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    eprintln!("GroupNorm G=1 dispatch plan ({} steps):", plan.len());
    eprintln!("effective_output = {effective_output:?}");
    for (i, step) in plan.iter().enumerate() {
        eprintln!("  step {i}: {step:?}");
    }

    let contract =
        nn_dsl::PrecisionContract::bootstrap(nn_dsl::PrecisionTier::Normal, ScalarType::F32);
    let msl = nn_dsl::emit_tensor_msl_with_contract(&def, ScalarType::F32, contract).expect("msl");
    eprintln!("\n--- Generated MSL ---\n{msl}\n--- End MSL ---");
}

// ---------------------------------------------------------------------------
// Tests: Full DConv sublayer 0 parity
// ---------------------------------------------------------------------------

/// Full DConv sublayer 0 parity: build whole sublayer and compare.
#[test]
fn dconv_sublayer0_full_parity() {
    let (weights, cache, audio) = match load_test_weights() {
        Some(t) => t,
        None => return,
    };
    let normalized = normalize_audio(&audio, 1024);
    let gelu_out = dispatch_conv_gelu(&cache, &normalized, &weights, 1024);

    let t_len: usize = 256;
    let eps_val = [1e-5f32];
    let dc = &weights.encoder.blocks[0].dconv[0];

    let mut b = TensorBlockBuilder::new("dconv0_full");
    let out = build_dconv0_graph(&mut b, 48, t_len);
    let def = b.build(out).expect("build dconv0");

    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
    inputs.insert("data", &gelu_out);
    inputs.insert("cw", &dc.conv_compress_weight);
    inputs.insert("cb", &dc.conv_compress_bias);
    inputs.insert("eps1", &eps_val);
    inputs.insert("ng", &dc.norm_compress_gamma);
    inputs.insert("nb", &dc.norm_compress_beta);
    inputs.insert("ew", &dc.conv_expand_weight);
    inputs.insert("eb", &dc.conv_expand_bias);
    inputs.insert("eps2", &eps_val);
    inputs.insert("eng", &dc.norm_expand_gamma);
    inputs.insert("enb", &dc.norm_expand_beta);
    inputs.insert("ls", &dc.layer_scale);

    let result = nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dconv0 dispatch");
    eprintln!(
        "DConv0 full: len={}, first8={:?}",
        result.len(),
        &result[..8]
    );

    let py_dconv0 = [
        -0.11032045_f32,
        -0.15330403,
        0.39912808,
        0.58027005,
        0.07495014,
        -0.06050082,
        0.029_700_1,
        0.08373211,
    ];
    let v = compare_first_n("dconv0_full", &result, &py_dconv0, 8, 1e-3);
    if v > 0 {
        eprintln!("DConv sublayer 0 diverges from Python by more than 1e-3");
    }
}

// ---------------------------------------------------------------------------
// Tests: Batch pipeline progressive
// ---------------------------------------------------------------------------

/// Reshape → Reduce(Mean) (2 steps).
#[test]
fn batch_reshape_reduce() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut b = TensorBlockBuilder::new("test_a");
    let inp = b.add_input("data", &[2, 3]);
    let reshaped = b.add_reshape(inp, &[1, 6]);
    let mean = b.add_reduce(reshaped, ReduceOp::Mean, 1, false, &[1]);
    let def = b.build(mean).expect("build");
    let mut m: HashMap<&str, &[f32]> = HashMap::new();
    m.insert("data", &input);
    let result =
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("test_a");
    eprintln!("A (Reshape→Reduce): {result:?} (expected [3.5])");
    assert!(
        (result[0] - 3.5).abs() < 1e-5,
        "test A: mean wrong: {}",
        result[0]
    );
}

/// Reshape → Reduce(Mean) → Broadcast (3 steps).
#[test]
fn batch_reshape_reduce_broadcast() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut b = TensorBlockBuilder::new("test_b");
    let inp = b.add_input("data", &[2, 3]);
    let reshaped = b.add_reshape(inp, &[1, 6]);
    let mean = b.add_reduce(reshaped, ReduceOp::Mean, 1, false, &[1]);
    let bc = b.add_broadcast_left(mean, &[1, 6]);
    let def = b.build(bc).expect("build");
    let mut m: HashMap<&str, &[f32]> = HashMap::new();
    m.insert("data", &input);
    let result =
        nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("test_b");
    eprintln!("B (→Broadcast): {result:?} (expected all 3.5)");
    assert!(
        result.iter().all(|&v| (v - 3.5).abs() < 1e-5),
        "test B: broadcast wrong"
    );
}

/// Full GroupNorm G=1 (12 steps) via batch dispatch.
#[test]
fn batch_full_group_norm() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = dispatch_gn_noaffine(&cache, &input, 2, 3);
    let expected = cpu_expected_gn(&input);
    let err = max_abs_err(&result, &expected);
    eprintln!("Full GN: {result:?}");
    eprintln!("Expected: {expected:?}");
    eprintln!("max_err: {err}");
    let all_zero = result.iter().all(|&v| v == 0.0);
    eprintln!("ALL ZEROS: {all_zero}");
    assert!(!all_zero, "GroupNorm output is all zeros");
}

/// Reduce(Mean) → Broadcast as a single batch (basic inter-step test).
#[test]
fn batch_reduce_broadcast_pipeline() {
    let backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new(backend.context().clone());
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut b = TensorBlockBuilder::new("rb");
    let inp = b.add_input("data", &[1, 6]);
    let mean = b.add_reduce(inp, ReduceOp::Mean, 1, false, &[1]);
    let bc = b.add_broadcast_left(mean, &[1, 6]);
    let def = b.build(bc).expect("build");

    let mut m: HashMap<&str, &[f32]> = HashMap::new();
    m.insert("data", &input);
    let result = nn_metal::execute_tensor_dispatch(&cache, &def, ScalarType::F32, &m).expect("rb");
    eprintln!("Reduce→Broadcast: {result:?}");
    assert!(
        result.iter().all(|&v| (v - 3.5).abs() < 1e-5),
        "expected all 3.5, got {result:?}"
    );
}
