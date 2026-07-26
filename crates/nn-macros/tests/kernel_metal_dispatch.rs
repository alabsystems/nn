// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(unexpected_cfgs)]

//! End-to-end proc-macro dispatch test for issue #18.
//!
//! Verifies the generated `SNAKE_DESCRIPTOR` from `#[nn_macros::kernel]` can
//! be compiled and dispatched through `KernelPipeline::from_descriptor`, and
//! that GPU outputs match the Rust reference implementation within tolerance.

use nn_dsl::{
    differential_tolerance, snake_scalar_bounds, within_differential_budget, PrecisionContract,
    PrecisionTier, ScalarType,
};
use nn_metal::{KernelPipeline, MetalContext, MetalError, PipelineCache};

#[nn_macros::kernel(bounds(alpha = "0.1..1e6"))]
fn snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

fn issue18_domain_inputs(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let denom = (samples.saturating_sub(1)) as f32;

    let x = (0..samples)
        .map(|i| {
            let t = if denom > 0.0 { (i as f32) / denom } else { 0.0 };
            -10.0 + (20.0 * t)
        })
        .collect();

    let alpha = (0..samples)
        .map(|i| {
            let j = (i * 37) % samples.max(1);
            let t = if denom > 0.0 { (j as f32) / denom } else { 0.0 };
            0.01 + (99.99 * t)
        })
        .collect();

    (x, alpha)
}

fn issue18_kani_domain_inputs() -> (Vec<f32>, Vec<f32>) {
    let xs = [
        -1.0e4f32, -1.0e3, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1.0e3, 1.0e4,
    ];
    let alphas = [
        1.0e-8f32, 1.0e-7, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3,
    ];

    let mut x = Vec::with_capacity(xs.len() * alphas.len());
    let mut alpha = Vec::with_capacity(xs.len() * alphas.len());

    for xv in xs {
        for av in alphas {
            x.push(xv);
            alpha.push(av);
        }
    }

    (x, alpha)
}

fn compile_generated_snake_pipeline(cache: &PipelineCache) -> Result<KernelPipeline, MetalError> {
    KernelPipeline::from_descriptor(cache, &SNAKE_DESCRIPTOR)
}

#[test]
fn test_generated_snake_msl_dispatch_matches_reference_issue18_domain() {
    let (x, alpha) = issue18_domain_inputs(512);
    assert_eq!(
        x.len(),
        alpha.len(),
        "issue18 domain helper must generate aligned input vectors"
    );

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let gpu = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect("dispatch generated snake kernel");
    assert_eq!(
        gpu.len(),
        x.len(),
        "dispatch output must preserve element count"
    );

    let (bound_lo, bound_hi) =
        snake_scalar_bounds(-10.0, 10.0, 0.01, 100.0).expect("finite bounds");

    for (index, ((xv, av), y_gpu)) in x.iter().zip(&alpha).zip(&gpu).enumerate() {
        let y_ref = snake(*xv, *av);
        let y_gpu = *y_gpu;
        let diff = (y_ref - y_gpu).abs();

        assert!(
            diff <= 1e-5,
            "snake mismatch at {index}: ref={y_ref}, gpu={y_gpu}, diff={diff}, x={xv}, alpha={av}"
        );
        assert!(
            y_gpu >= bound_lo - 1e-5 && y_gpu <= bound_hi + 1e-5,
            "output out of conservative bounds at {index}: y={y_gpu}, bounds=[{bound_lo}, {bound_hi}]"
        );
    }
}

#[test]
fn test_generated_snake_msl_dispatch_matches_reference_kani_domain() {
    let (x, alpha) = issue18_kani_domain_inputs();
    assert_eq!(
        x.len(),
        alpha.len(),
        "Kani-domain helper must generate aligned input vectors"
    );

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let gpu = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect("dispatch generated snake kernel");
    assert_eq!(
        gpu.len(),
        x.len(),
        "dispatch output must preserve element count"
    );

    let contract = PrecisionContract::bootstrap(
        PrecisionTier::parse(__snake_kernel_meta::PRECISION_TIER).expect("precision tier"),
        ScalarType::F32,
    );
    assert_eq!(
        contract.differential_abs_budget,
        __snake_kernel_meta::DIFFERENTIAL_ABS_BUDGET
    );
    assert_eq!(
        contract.differential_rel_budget,
        __snake_kernel_meta::DIFFERENTIAL_REL_BUDGET
    );

    let (bound_lo, bound_hi) =
        snake_scalar_bounds(-1.0e4, 1.0e4, 1.0e-8, 1.0e3).expect("finite bounds");
    for (index, ((xv, av), y_gpu)) in x.iter().zip(&alpha).zip(&gpu).enumerate() {
        let y_ref = snake(*xv, *av);
        let y_gpu = *y_gpu;
        let tolerance = differential_tolerance(y_ref, contract);

        assert!(
            y_gpu.is_finite(),
            "GPU output must remain finite at {index}: y={y_gpu}, x={xv}, alpha={av}"
        );
        assert!(
            within_differential_budget(y_ref, y_gpu, contract),
            "snake mismatch at {index}: ref={y_ref}, gpu={y_gpu}, tol={tolerance}, x={xv}, alpha={av}"
        );
        assert!(
            y_gpu >= bound_lo - tolerance && y_gpu <= bound_hi + tolerance,
            "output out of conservative bounds at {index}: y={y_gpu}, bounds=[{bound_lo}, {bound_hi}], tol={tolerance}"
        );
    }
}

#[test]
fn test_generated_snake_msl_dispatch_empty_inputs_returns_empty_output() {
    let x = Vec::<f32>::new();
    let alpha = Vec::<f32>::new();

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let gpu = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect("dispatch generated snake kernel with empty inputs");

    assert!(
        gpu.is_empty(),
        "empty input dispatch should return empty output"
    );
}

#[test]
fn test_generated_snake_msl_dispatch_rejects_param_count_mismatch() {
    let x = vec![0.0f32; 8];

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let err = pipeline
        .dispatch_elementwise(&ctx, &[&x])
        .expect_err("dispatch should reject missing alpha input");
    assert!(
        matches!(
            err,
            MetalError::ParamCountMismatch {
                expected: 2,
                got: 1
            }
        ),
        "unexpected error for param mismatch: {err:?}"
    );
}

#[test]
fn test_generated_snake_msl_dispatch_rejects_extra_param_count() {
    let x = vec![0.0f32; 8];
    let alpha = vec![1.0f32; 8];
    let extra = vec![2.0f32; 8];

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let err = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha, &extra])
        .expect_err("dispatch should reject unexpected third parameter");
    assert!(
        matches!(
            err,
            MetalError::ParamCountMismatch {
                expected: 2,
                got: 3
            }
        ),
        "unexpected error for param mismatch: {err:?}"
    );
}

#[test]
fn test_generated_snake_msl_dispatch_rejects_zero_param_count() {
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let err = pipeline
        .dispatch_elementwise::<f32>(&ctx, &[])
        .expect_err("dispatch should reject missing all parameters");
    assert!(
        matches!(
            err,
            MetalError::ParamCountMismatch {
                expected: 2,
                got: 0
            }
        ),
        "unexpected error for zero-param mismatch: {err:?}"
    );
}

#[test]
fn test_generated_snake_msl_dispatch_rejects_mismatched_input_lengths() {
    let x = vec![0.0f32; 8];
    let alpha = vec![1.0f32; 7];

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let err = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect_err("dispatch should reject mismatched input lengths");
    assert!(
        matches!(
            err,
            MetalError::InputLenMismatch {
                expected: 8,
                got: 7,
                index: 1
            }
        ),
        "unexpected error for input length mismatch: {err:?}"
    );
}

#[test]
fn test_generated_snake_msl_dispatch_rejects_mismatched_input_lengths_lhs_empty() {
    let x = Vec::<f32>::new();
    let alpha = vec![1.0f32; 9];

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let err = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect_err("dispatch should reject mismatched input lengths even when lhs is empty");
    assert!(
        matches!(
            err,
            MetalError::InputLenMismatch {
                expected: 0,
                got: 9,
                index: 1
            }
        ),
        "unexpected error for input length mismatch with empty lhs: {err:?}"
    );
}

#[test]
fn test_generated_snake_msl_dispatch_rejects_mismatched_input_lengths_rhs_longer() {
    let x = vec![0.0f32; 8];
    let alpha = vec![1.0f32; 9];

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        compile_generated_snake_pipeline(&cache).expect("compile generated snake MSL for dispatch");

    let err = pipeline
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect_err("dispatch should reject mismatched input lengths");
    assert!(
        matches!(
            err,
            MetalError::InputLenMismatch {
                expected: 8,
                got: 9,
                index: 1
            }
        ),
        "unexpected error for input length mismatch: {err:?}"
    );
}

/// Verify the proc-macro-generated descriptor carries correct metadata.
///
/// This is the core contract test for issue #99: the `#[kernel]` proc-macro
/// must generate a `KernelDescriptor` whose `param_count` matches the actual
/// Rust function signature arity. If codegen drifts, this test catches it.
#[test]
fn test_snake_descriptor_fields_match_kernel_signature() {
    // snake(x: f32, alpha: f32) -> f32  =>  2 params
    assert_eq!(
        SNAKE_DESCRIPTOR.param_count, 2,
        "SNAKE_DESCRIPTOR.param_count must match fn snake(x, alpha) arity"
    );
    assert_eq!(
        SNAKE_DESCRIPTOR.entry_point, "snake_kernel",
        "entry_point must be <name>_kernel by convention"
    );
    assert!(
        !SNAKE_DESCRIPTOR.msl_source.is_empty(),
        "MSL source must be non-empty"
    );
    assert!(
        SNAKE_DESCRIPTOR.msl_source.contains("snake_kernel"),
        "MSL source must contain the entry point function"
    );
}

/// Prove `from_descriptor` and `from_msl` with identical args produce the same
/// GPU output, confirming the descriptor is a faithful bundle of its fields.
///
/// This demonstrates that the typed contract (issue #99) doesn't alter behavior
/// — it only prevents mis-pairing the components.
#[test]
fn test_from_descriptor_matches_from_msl_output() {
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());

    let via_descriptor =
        KernelPipeline::from_descriptor(&cache, &SNAKE_DESCRIPTOR).expect("from_descriptor");
    let via_from_msl = KernelPipeline::from_msl(
        &cache,
        SNAKE_DESCRIPTOR.msl_source,
        SNAKE_DESCRIPTOR.entry_point,
        SNAKE_DESCRIPTOR.param_count,
        SNAKE_DESCRIPTOR.fast_math,
    )
    .expect("from_msl with descriptor fields");

    let x = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0, 10.0];
    let alpha = vec![0.5f32, 1.0, 2.0, 0.1, 5.0, 0.01];

    let out_desc = via_descriptor
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect("dispatch via descriptor");
    let out_msl = via_from_msl
        .dispatch_elementwise(&ctx, &[&x, &alpha])
        .expect("dispatch via from_msl");

    assert_eq!(out_desc.len(), out_msl.len(), "output lengths must match");
    for (i, (d, m)) in out_desc.iter().zip(&out_msl).enumerate() {
        assert_eq!(
            d.to_bits(),
            m.to_bits(),
            "bit-exact match required at index {i}: descriptor={d}, from_msl={m}"
        );
    }
}
