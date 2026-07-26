// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `KernelPipeline::from_msl` path.
//!
//! Exercises the proc-macro-generated MSL path: pre-generated MSL → Metal
//! pipeline → GPU dispatch → differential comparison with Rust reference.

use nn_dsl::lower::Lowerer;
use nn_dsl::{PrecisionContract, PrecisionTier};
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

fn snake_ref(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

fn add_ref(a: f32, b: f32) -> f32 {
    a + b
}

fn add_ref_f16(a: half::f16, b: half::f16) -> half::f16 {
    half::f16::from_f32(a.to_f32() + b.to_f32())
}

fn relu_ref(x: f32) -> f32 {
    x.max(0.0)
}

#[test]
fn test_from_msl_snake_differential() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    let msl = nn_dsl::emit_msl_with_contract(&kernel, contract).expect("emit");

    let cache = metal_cache();
    let pipeline =
        KernelPipeline::from_msl(&cache, &msl, "snake_kernel", 2, false).expect("from_msl");

    assert_eq!(pipeline.name(), "snake");
    assert_eq!(pipeline.param_count(), 2);
    assert!(!pipeline.fast_math());

    let n = 256;
    // Include negative x values (sin(alpha*x) differs for negative x)
    let x_data: Vec<f32> = (0..n)
        .map(|j| -5.0 + (((j + 1) as f32) * 0.173).sin() * 10.0)
        .collect();
    let alpha_data: Vec<f32> = (0..n)
        .map(|j| 0.5 + (((j + 1) as f32) * 0.346).sin().abs() * 9.0)
        .collect();

    let rust_out: Vec<f32> = (0..n)
        .map(|i| snake_ref(x_data[i], alpha_data[i]))
        .collect();
    let metal_out = pipeline
        .dispatch_elementwise(cache.context(), &[&x_data, &alpha_data])
        .expect("dispatch");

    for i in 0..n {
        assert!(
            nn_dsl::within_differential_budget(rust_out[i], metal_out[i], contract),
            "from_msl snake mismatch at {i}: rust={}, metal={}, delta={}",
            rust_out[i],
            metal_out[i],
            (rust_out[i] - metal_out[i]).abs()
        );
    }
}

#[test]
fn test_from_msl_add_exact() {
    let kernel = lower("fn add(a: f32, b: f32) -> f32 { a + b }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    let msl = nn_dsl::emit_msl_with_contract(&kernel, contract).expect("emit");

    let cache = metal_cache();
    let pipeline =
        KernelPipeline::from_msl(&cache, &msl, "add_kernel", 2, false).expect("from_msl");

    let a = vec![1.0f32, 2.0, 3.0, 4.0, -1.5, 0.0, 100.0, -100.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0, 1.5, 0.0, -50.0, 100.0];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&a, &b])
        .expect("dispatch");

    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| add_ref(*x, *y)).collect();
    assert_eq!(
        result, expected,
        "f32 add should be exact via from_msl path"
    );
}

#[test]
fn test_from_msl_relu_single_param() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    let msl = nn_dsl::emit_msl_with_contract(&kernel, contract).expect("emit");

    let cache = metal_cache();
    let pipeline =
        KernelPipeline::from_msl(&cache, &msl, "relu_kernel", 1, false).expect("from_msl");

    assert_eq!(pipeline.name(), "relu");
    assert_eq!(pipeline.param_count(), 1);

    let x = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 100.0];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&x])
        .expect("dispatch");
    let expected: Vec<f32> = x.iter().map(|v| relu_ref(*v)).collect();
    assert_eq!(result, expected);
}

#[test]
fn test_from_msl_cache_reuse() {
    let kernel = lower("fn id(x: f32) -> f32 { x }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    let msl = nn_dsl::emit_msl_with_contract(&kernel, contract).expect("emit");

    let cache = metal_cache();
    assert!(cache.is_empty());

    let _ = KernelPipeline::from_msl(&cache, &msl, "id_kernel", 1, false).expect("first");
    assert_eq!(cache.len(), 1);

    let _ = KernelPipeline::from_msl(&cache, &msl, "id_kernel", 1, false).expect("second");
    assert_eq!(cache.len(), 1, "from_msl should reuse cached pipeline");
}

#[test]
fn test_from_msl_relaxed_fast_math() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let contract = PrecisionContract::bootstrap(PrecisionTier::Relaxed, kernel.return_type);
    let msl = nn_dsl::emit_msl_with_contract(&kernel, contract).expect("emit");

    let cache = metal_cache();
    let pipeline =
        KernelPipeline::from_msl(&cache, &msl, "snake_kernel", 2, true).expect("from_msl");

    assert!(pipeline.fast_math(), "relaxed tier should enable fast_math");

    let x_data: Vec<f32> = (0..64)
        .map(|j| -5.0 + (((j + 1) as f32) * 0.173).sin() * 10.0)
        .collect();
    let alpha_data: Vec<f32> = (0..64)
        .map(|j| 0.5 + (((j + 1) as f32) * 0.346).sin().abs() * 9.0)
        .collect();
    let rust_out: Vec<f32> = (0..64)
        .map(|i| snake_ref(x_data[i], alpha_data[i]))
        .collect();
    let metal_out = pipeline
        .dispatch_elementwise(cache.context(), &[&x_data, &alpha_data])
        .expect("dispatch");

    for i in 0..64 {
        assert!(
            nn_dsl::within_differential_budget(rust_out[i], metal_out[i], contract),
            "relaxed snake mismatch at {i}: rust={}, metal={}, delta={}",
            rust_out[i],
            metal_out[i],
            (rust_out[i] - metal_out[i]).abs()
        );
    }
}

#[test]
fn test_from_msl_add_f16() {
    let kernel = lower("fn add_half(a: f16, b: f16) -> f16 { a + b }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    let msl = nn_dsl::emit_msl_with_contract(&kernel, contract).expect("emit");

    let cache = metal_cache();
    let pipeline =
        KernelPipeline::from_msl(&cache, &msl, "add_half_kernel", 2, false).expect("from_msl");

    assert_eq!(pipeline.name(), "add_half");
    assert_eq!(pipeline.param_count(), 2);

    let a = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(-2.5),
        half::f16::from_f32(0.0),
        half::f16::from_f32(3.25),
        half::f16::from_f32(-0.125),
        half::f16::from_f32(100.0),
        half::f16::from_f32(-100.0),
        half::f16::from_f32(0.5),
    ];
    let b = vec![
        half::f16::from_f32(10.0),
        half::f16::from_f32(2.5),
        half::f16::from_f32(0.0),
        half::f16::from_f32(-1.25),
        half::f16::from_f32(0.125),
        half::f16::from_f32(-50.0),
        half::f16::from_f32(100.0),
        half::f16::from_f32(-0.25),
    ];
    let result = pipeline
        .dispatch_elementwise(cache.context(), &[&a, &b])
        .expect("dispatch");

    let expected: Vec<half::f16> = a.iter().zip(&b).map(|(x, y)| add_ref_f16(*x, *y)).collect();
    for (index, (lhs, rhs)) in result.iter().zip(&expected).enumerate() {
        let diff = (lhs.to_f32() - rhs.to_f32()).abs();
        assert!(
            diff <= 1e-3,
            "from_msl f16 add mismatch at {index}: lhs={}, rhs={}, diff={diff}",
            lhs.to_f32(),
            rhs.to_f32()
        );
    }
}
