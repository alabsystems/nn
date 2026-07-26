// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the KernelPipeline bridge (nn-dsl → nn-metal).
//!
//! Exercises the compile path: Rust source → Lowerer → KernelDef → MSL →
//! Metal pipeline → GPU dispatch → differential comparison with Rust reference.
//!
//! The `from_msl` path is tested in `kernel_pipeline_from_msl.rs`.

use nn_dsl::lower::Lowerer;
use nn_metal::{KernelPipeline, MetalBackend, PipelineCache};

/// Initialize global Metal backend and return a PipelineCache (#2424).
fn metal_cache() -> PipelineCache {
    let _ = MetalBackend::init().expect("Metal device required");
    PipelineCache::new_global().expect("Metal global cache")
}

fn lower(src: &str) -> nn_dsl::ir::KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    Lowerer::lower_fn(&func).expect("lower")
}

// --- Reference implementations for differential testing ---

fn snake_ref(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

fn relu_ref(x: f32) -> f32 {
    x.max(0.0)
}

fn add_ref(a: f32, b: f32) -> f32 {
    a + b
}

fn add_ref_f16(a: half::f16, b: half::f16) -> half::f16 {
    half::f16::from_f32(a.to_f32() + b.to_f32())
}

// --- Tests ---

#[test]
fn test_kernel_pipeline_compile_snake() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    assert_eq!(pipeline.name(), "snake");
    assert_eq!(pipeline.param_count(), 2);
    assert!(!pipeline.fast_math());
}

#[test]
fn test_kernel_pipeline_dispatch_add() {
    let kernel = lower("fn add(a: f32, b: f32) -> f32 { a + b }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&a, &b])
        .expect("dispatch");

    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| add_ref(*x, *y)).collect();
    assert_eq!(result, expected);
}

#[test]
fn test_kernel_pipeline_dispatch_add_f16() {
    let kernel = lower("fn add_half(a: f16, b: f16) -> f16 { a + b }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let a = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(-3.0),
        half::f16::from_f32(4.5),
    ];
    let b = vec![
        half::f16::from_f32(10.0),
        half::f16::from_f32(-2.5),
        half::f16::from_f32(0.5),
        half::f16::from_f32(1.25),
    ];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&a, &b])
        .expect("dispatch");

    let expected: Vec<half::f16> = a.iter().zip(&b).map(|(x, y)| add_ref_f16(*x, *y)).collect();
    for (index, (lhs, rhs)) in result.iter().zip(&expected).enumerate() {
        let diff = (lhs.to_f32() - rhs.to_f32()).abs();
        assert!(
            diff <= 1e-3,
            "f16 add mismatch at {index}: lhs={}, rhs={}, diff={diff}",
            lhs.to_f32(),
            rhs.to_f32()
        );
    }
}

#[test]
fn test_kernel_pipeline_dispatch_relu() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let x = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 100.0];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&x])
        .expect("dispatch");

    let expected: Vec<f32> = x.iter().map(|v| relu_ref(*v)).collect();
    assert_eq!(result, expected);
}

#[test]
fn test_kernel_pipeline_dispatch_snake_differential() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let x: Vec<f32> = vec![0.0, 0.5, 1.0, -1.0, 2.0, -0.5, 2.71, -2.71];
    let alpha: Vec<f32> = vec![1.0; x.len()];
    let gpu_result = pipeline
        .dispatch_elementwise(cache.context(), &[&x, &alpha])
        .expect("dispatch");

    for i in 0..x.len() {
        let rust_val = snake_ref(x[i], alpha[i]);
        let gpu_val = gpu_result[i];
        let diff = (rust_val - gpu_val).abs();
        assert!(
            diff < 1e-5,
            "snake divergence at i={i}: rust={rust_val} gpu={gpu_val} diff={diff} (x={}, alpha={})",
            x[i],
            alpha[i]
        );
    }
}

#[test]
fn test_kernel_pipeline_param_count_mismatch() {
    let kernel = lower("fn add(a: f32, b: f32) -> f32 { a + b }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let a = vec![1.0f32];
    let err = pipeline
        .dispatch_elementwise(cache.context(), &[&a])
        .expect_err("should fail with wrong param count");
    let msg = err.to_string();
    assert!(
        msg.contains("expects 2 parameters but got 1"),
        "error message: {msg}"
    );
}

#[test]
fn test_kernel_pipeline_input_len_mismatch() {
    let kernel = lower("fn add(a: f32, b: f32) -> f32 { a + b }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let a = vec![1.0f32, 2.0];
    let b = vec![3.0f32];
    let err = pipeline
        .dispatch_elementwise(cache.context(), &[&a, &b])
        .expect_err("shape mismatch should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("input slice 1 has length 1, expected 2"),
        "error message: {msg}"
    );
}

#[test]
fn test_kernel_pipeline_empty_input() {
    let kernel = lower("fn id(x: f32) -> f32 { x }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let empty: Vec<f32> = vec![];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&empty])
        .expect("dispatch");
    assert!(result.is_empty());
}

#[test]
fn test_kernel_pipeline_cache_reuse() {
    let kernel = lower("fn id(x: f32) -> f32 { x }");
    let cache = metal_cache();
    assert!(cache.is_empty());

    let _ = KernelPipeline::compile(&cache, &kernel).expect("first compile");
    assert_eq!(cache.len(), 1);

    let _ = KernelPipeline::compile(&cache, &kernel).expect("second compile (cache hit)");
    assert_eq!(cache.len(), 1, "cache should reuse the compiled pipeline");
}

#[test]
fn test_kernel_pipeline_dispatch_batch_f32() {
    let kernel = lower("fn add(a: f32, b: f32) -> f32 { a + b }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let a0 = vec![1.0f32, 2.0, 3.0, 4.0];
    let b0 = vec![10.0f32, 20.0, 30.0, 40.0];
    let launch0: [&[f32]; 2] = [&a0, &b0];

    let a1 = vec![0.5f32, -1.0, 3.25];
    let b1 = vec![0.25f32, 2.0, -0.25];
    let launch1: [&[f32]; 2] = [&a1, &b1];

    let empty: Vec<f32> = Vec::new();
    let launch2: [&[f32]; 2] = [&empty, &empty];

    let outputs = pipeline
        .dispatch_elementwise_batch(cache.context(), &[&launch0, &launch1, &launch2])
        .expect("batch dispatch");
    assert_eq!(outputs.len(), 3);

    let expected0: Vec<f32> = a0.iter().zip(&b0).map(|(x, y)| add_ref(*x, *y)).collect();
    let expected1: Vec<f32> = a1.iter().zip(&b1).map(|(x, y)| add_ref(*x, *y)).collect();

    assert_eq!(outputs[0], expected0);
    assert_eq!(outputs[1], expected1);
    assert!(outputs[2].is_empty());
}

#[test]
fn test_kernel_pipeline_dispatch_batch_f16() {
    let kernel = lower("fn add_half(a: f16, b: f16) -> f16 { a + b }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let a0 = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(-2.0),
        half::f16::from_f32(3.5),
        half::f16::from_f32(0.125),
    ];
    let b0 = vec![
        half::f16::from_f32(0.5),
        half::f16::from_f32(4.0),
        half::f16::from_f32(-1.0),
        half::f16::from_f32(2.0),
    ];
    let launch0: [&[half::f16]; 2] = [&a0, &b0];

    let a1 = vec![half::f16::from_f32(-0.75), half::f16::from_f32(8.0)];
    let b1 = vec![half::f16::from_f32(0.25), half::f16::from_f32(-2.0)];
    let launch1: [&[half::f16]; 2] = [&a1, &b1];

    let outputs = pipeline
        .dispatch_elementwise_batch(cache.context(), &[&launch0, &launch1])
        .expect("batch dispatch");
    assert_eq!(outputs.len(), 2);

    let expected0: Vec<half::f16> = a0
        .iter()
        .zip(&b0)
        .map(|(x, y)| add_ref_f16(*x, *y))
        .collect();
    let expected1: Vec<half::f16> = a1
        .iter()
        .zip(&b1)
        .map(|(x, y)| add_ref_f16(*x, *y))
        .collect();

    for (index, (lhs, rhs)) in outputs[0].iter().zip(&expected0).enumerate() {
        let diff = (lhs.to_f32() - rhs.to_f32()).abs();
        assert!(
            diff <= 1e-3,
            "batch f16 mismatch (launch0) at {index}: lhs={}, rhs={}",
            lhs.to_f32(),
            rhs.to_f32()
        );
    }
    for (index, (lhs, rhs)) in outputs[1].iter().zip(&expected1).enumerate() {
        let diff = (lhs.to_f32() - rhs.to_f32()).abs();
        assert!(
            diff <= 1e-3,
            "batch f16 mismatch (launch1) at {index}: lhs={}, rhs={}",
            lhs.to_f32(),
            rhs.to_f32()
        );
    }
}

/// Regression for #89: f16 dispatch must use `half::f16`, not nightly `std::f16`.
///
/// Exercises edge-case f16 values (zero, negative zero, smallest subnormal,
/// largest finite) through the GPU dispatch path to confirm the `half` crate
/// handles the full f16 range correctly after the `std::f16` removal.
#[test]
fn test_f16_dispatch_edge_values_regression_89() {
    let kernel = lower("fn id_half(x: f16) -> f16 { x }");
    let cache = metal_cache();
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile");

    let edge_values = vec![
        half::f16::ZERO,
        half::f16::NEG_ZERO,
        half::f16::ONE,
        half::f16::NEG_ONE,
        half::f16::MIN_POSITIVE,      // smallest positive normal
        half::f16::from_bits(0x0001), // smallest subnormal
        half::f16::MAX,               // 65504.0
        half::f16::MIN,               // -65504.0
        half::f16::INFINITY,          // +inf
        half::f16::NEG_INFINITY,      // -inf
        half::f16::NAN,               // NaN (IEEE 754 #66)
    ];

    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&edge_values])
        .expect("dispatch");

    assert_eq!(result.len(), edge_values.len());
    for (index, (got, expected)) in result.iter().zip(&edge_values).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f16 identity mismatch at {index}: got bits=0x{:04X} ({}), expected bits=0x{:04X} ({})",
            got.to_bits(),
            got.to_f32(),
            expected.to_bits(),
            expected.to_f32(),
        );
    }
}
