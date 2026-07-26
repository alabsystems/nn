// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end pipeline test: `#[nn_macros::kernel]` annotation through Metal
//! GPU dispatch through NY bounds verification.
//!
//! This test validates the full nn pipeline composition for the Snake
//! activation kernel. Each stage is documented with what it proves.
//!
//! Split into sub-modules to stay under 500 lines (#533, #542):
//! - `pipeline/contract.rs` — relu/clamped/bounds tightening contract tests
//! - `pipeline/contract_extended.rs` — silu_mul/gelu/inv_norm/rope contract tests
//! - `pipeline/contract_norm.rs` — rms_norm/layer_norm/instance_norm/instance_norm_affine
//! - `pipeline/contract_adain.rs` — adain (K3)/adain_snake (K4) contract tests (#570)
//!
//! # Pipeline stages demonstrated
//!
//! 1. **Proc-macro expansion** (`#[kernel]`): Rust source -> MSL constant +
//!    Kani harness + differential test + metadata
//! 2. **Rust reference**: The original function runs on CPU
//! 3. **Metal GPU dispatch**: Generated MSL compiles and runs on GPU
//! 4. **Differential comparison**: Rust and GPU outputs match within precision
//!    budget
//! 5. **NY bounds verification**: Formal proof that output is bounded
//!    for all inputs in range
//! 6. **Kani harness generation**: Formal verification source for exhaustive
//!    model checking (shown, not run — `cargo kani` runs these)
//!
//! # Using this as a template
//!
//! To port a dvoice kernel to nn:
//! 1. Write the kernel in Rust, annotate with `#[nn_macros::kernel]`
//! 2. The proc-macro generates MSL and verification artifacts automatically
//! 3. Use `KernelPipeline::from_descriptor` with the generated descriptor
//! 4. Use `VerifyRequest` builder to prove bounds for your input domain
//! 5. Run `cargo kani` to model-check the generated harness

#![allow(unexpected_cfgs)]

// ============================================================================
// Stage 1: Define kernel with the #[kernel] proc-macro
// ============================================================================
//
// The `#[kernel]` attribute does the following at compile time:
// - Preserves the original Rust function as a reference implementation
// - Lowers the function body to KernelIR (DAG of typed scalar ops)
// - Translates the IR to MSL and emits it as `const SNAKE_MSL: &str`
// - Generates `const SNAKE_DESCRIPTOR: KernelDescriptor` bundling MSL + metadata
// - Generates a `#[cfg(kani)]` verification harness
// - Generates a `#[cfg(test)]` differential test (Rust vs Metal)
// - Emits metadata (node count, param count, precision tier)
//
// What this proves: Rust source compiles to GPU shader code with no manual
// translation step.

// Note: 1e-8 must match nn_dsl::SNAKE_MIN_ALPHA — see #325.
#[nn_macros::kernel(bounds(alpha = "0.1..1e6"))]
fn snake(x: f32, alpha: f32) -> f32 {
    let a = alpha.max(1e-8);
    x + (1.0 / a) * (a * x).sin().powi(2)
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Build a `KernelDef` from the same source for programmatic verification.
///
/// The proc-macro consumes the IR at compile time to generate MSL. For
/// runtime verification, we re-lower the identical source.
///
/// Includes the `SNAKE_MIN_ALPHA` clamp matching production
/// `build_snake_scalar_kernel()` — see #325.
fn snake_kernel_def() -> nn_dsl::ir::KernelDef {
    let src = format!(
        "fn snake(x: f32, alpha: f32) -> f32 {{
        let a = alpha.max({alpha_min:e});
        x + (1.0 / a) * (a * x).sin().powi(2)
    }}",
        alpha_min = nn_dsl::SNAKE_MIN_ALPHA,
    );
    let func: syn::ItemFn = syn::parse_str(&src).expect("parse snake source");
    nn_dsl::Lowerer::lower_fn(&func).expect("lower snake to KernelDef")
}

/// Compile the proc-macro-generated kernel descriptor into a Metal pipeline.
fn compile_snake_pipeline() -> (nn_metal::PipelineCache, nn_metal::KernelPipeline) {
    // Initialize the global Metal backend so gpu_scope::get_or_create_batch()
    // can find the global context for lazy command buffer batching (#2424).
    let _ = nn_metal::MetalBackend::init().expect("Metal device required");
    let cache = nn_metal::PipelineCache::new_global().expect("Metal global cache");
    let pipeline = nn_metal::KernelPipeline::from_descriptor(&cache, &SNAKE_DESCRIPTOR)
        .expect("proc-macro descriptor should compile on Metal");
    (cache, pipeline)
}

// ============================================================================
// Tests — each function covers one or two pipeline stages
// ============================================================================

/// Stage 2: Verify proc-macro generated artifacts exist and are valid.
///
/// What this proves: The Rust kernel was successfully lowered and
/// translated to a GPU shader without manual intervention.
#[test]
fn test_proc_macro_artifacts() {
    assert!(!SNAKE_MSL.is_empty(), "MSL source must be non-empty");
    assert!(
        SNAKE_MSL.contains("snake_kernel"),
        "MSL must contain the entry point function"
    );
    let param_count = __snake_kernel_meta::PARAM_COUNT;
    assert_eq!(param_count, 2, "snake has 2 parameters (x, alpha)");
    let node_count = __snake_kernel_meta::NODE_COUNT;
    assert!(node_count > 0, "IR graph must have nodes");
}

/// Stages 3-5: Rust reference -> Metal GPU dispatch -> differential comparison.
///
/// What this proves: The generated Metal shader compiles on real GPU hardware,
/// executes correctly, and matches the Rust reference within precision budget.
#[test]
fn test_rust_vs_metal_differential() {
    // Stage 3: Run Rust reference
    let test_x: Vec<f32> = vec![0.0, 1.0, -1.0, 5.0, -5.0, 0.5, -0.5, 2.0];
    let test_alpha: Vec<f32> = vec![1.0; test_x.len()];
    let rust_out: Vec<f32> = test_x
        .iter()
        .zip(&test_alpha)
        .map(|(&x, &a)| snake(x, a))
        .collect();

    // Sanity: snake(0, 1) = 0 + sin(0)^2 = 0
    assert!(
        rust_out[0].abs() < 1e-6,
        "snake(0, 1) should be ~0, got {}",
        rust_out[0]
    );

    // Stage 4: Compile MSL and dispatch on Metal GPU
    let (cache, pipeline) = compile_snake_pipeline();
    let gpu_out = pipeline
        .dispatch_elementwise(cache.context(), &[&test_x, &test_alpha])
        .expect("Metal dispatch should succeed");

    // Stage 5: Differential comparison
    assert_eq!(rust_out.len(), gpu_out.len(), "output length mismatch");
    let contract = nn_dsl::PrecisionContract::bootstrap(
        nn_dsl::PrecisionTier::Normal,
        nn_dsl::ScalarType::F32,
    );
    for (i, (r, g)) in rust_out.iter().zip(&gpu_out).enumerate() {
        assert!(
            nn_dsl::within_differential_budget(*r, *g, contract),
            "mismatch at {i}: rust={r}, metal={g}, delta={}",
            (r - g).abs(),
        );
    }
}

/// Stage 6: NY bounds verification — exhaustive mathematical proof.
///
/// What this proves: For ALL inputs in the specified range, the kernel output
/// is provably bounded and finite. This is exhaustive, not sampled.
#[test]
fn test_gamma_crown_bounds_verification() {
    let kernel_def = snake_kernel_def();

    // Multi-variable verification: both x and alpha are symbolic
    let bindings = vec![
        nn_verify::ParamBinding::Variable, // x
        nn_verify::ParamBinding::Variable, // alpha
    ];
    let variable_bounds: &[(f32, f32)] = &[
        (-10.0, 10.0), // x in [-10, 10]
        (0.01, 100.0), // alpha in [0.01, 100]
    ];

    let verification = nn_verify::VerifyRequest::new(&kernel_def)
        .bindings(&bindings)
        .variable_bounds(variable_bounds)
        .verify_bounds()
        .expect("NY verification should succeed");

    assert!(
        verification.is_finite,
        "output bounds must be finite: [{}, {}]",
        verification.output_lower, verification.output_upper,
    );
    assert!(
        verification.output_lower > -1000.0 && verification.output_upper < 1000.0,
        "output bounds should be reasonable: [{}, {}]",
        verification.output_lower,
        verification.output_upper,
    );
}

/// Stage 7: Kani harness source inspection.
///
/// Kani harnesses run under `cargo kani`, not at test time. Here we verify
/// the harness source was generated correctly by the codegen pipeline.
///
/// What the Kani harness proves (when run): For ALL possible f32 inputs,
/// the kernel does not overflow, produce NaN, or violate finiteness assertions.
#[test]
fn test_kani_harness_generation() {
    let kernel_def = snake_kernel_def();
    let kani_source = nn_dsl::emit_kani_harness(&kernel_def).expect("Kani harness generation");

    assert!(kani_source.contains("kani::proof"), "must have proof attr");
    assert!(
        kani_source.contains("kani::any"),
        "must use symbolic inputs"
    );
    assert!(kani_source.contains("snake"), "must reference kernel name");
}

/// Validate that proc-macro metadata is consistent with the kernel source.
#[test]
fn test_proc_macro_metadata_consistency() {
    let param_count = __snake_kernel_meta::PARAM_COUNT;
    assert_eq!(param_count, 2, "snake kernel has exactly 2 parameters");
    let node_count = __snake_kernel_meta::NODE_COUNT;
    assert!(node_count >= 5, "snake IR should have at least 5 nodes");
    assert_eq!(
        __snake_kernel_meta::PRECISION_TIER,
        "normal",
        "default precision tier is normal"
    );

    let ir = __snake_kernel_meta::IR_DEBUG;
    assert!(!ir.is_empty(), "IR debug string should be non-empty");
}

/// Validate that the generated MSL matches what programmatic codegen produces.
#[test]
fn test_proc_macro_msl_matches_programmatic() {
    let kernel_def = snake_kernel_def();
    let contract = nn_dsl::PrecisionContract::bootstrap(
        nn_dsl::PrecisionTier::Normal,
        kernel_def.return_type,
    );
    let programmatic_msl =
        nn_dsl::emit_msl_with_contract(&kernel_def, contract).expect("emit MSL");

    assert_eq!(
        SNAKE_MSL, programmatic_msl,
        "proc-macro MSL must match programmatic codegen"
    );
}

/// Verify Metal dispatch across varying alpha values, including domain boundaries.
///
/// The NY verification in this file proves bounds for alpha in [0.01, 100].
/// This test validates that the GPU matches the Rust reference at those extremes,
/// where floating-point divergence is most likely (large sin arguments from high alpha).
#[test]
fn test_pipeline_varying_alpha() {
    let (cache, pipeline) = compile_snake_pipeline();

    // Include values at the verified domain boundaries (alpha=0.01, alpha=100)
    // to exercise sin of large arguments where CPU/GPU can diverge.
    let x_vals: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5, 5.0, -5.0];
    let alpha_vals: Vec<f32> = vec![0.1, 1.0, 10.0, 50.0, 0.01, 100.0];

    let gpu_out = pipeline
        .dispatch_elementwise(cache.context(), &[&x_vals, &alpha_vals])
        .expect("dispatch");

    let contract = nn_dsl::PrecisionContract::bootstrap(
        nn_dsl::PrecisionTier::Normal,
        nn_dsl::ScalarType::F32,
    );
    for (i, ((&x, &a), &g)) in x_vals.iter().zip(&alpha_vals).zip(&gpu_out).enumerate() {
        let r = snake(x, a);
        assert!(
            nn_dsl::within_differential_budget(r, g, contract),
            "mismatch at {i}: x={x}, alpha={a}, rust={r}, metal={g}, delta={}",
            (r - g).abs(),
        );
    }
}

/// Cross-backend contract test: GPU output falls within NY proved bounds.
///
/// This is the missing link between differential testing (GPU ≈ CPU) and formal
/// verification (output ∈ [lower, upper] for all inputs). Neither alone guarantees
/// the GPU computes what was formally verified. This test chains them:
///
/// 1. NY proves output bounds for all x ∈ [-10, 10] with alpha=1.0
/// 2. Metal GPU dispatches snake on sampled x values in that range
/// 3. Every GPU output is asserted to fall within the proven bounds
///
/// If this test fails, either the MSL codegen diverges from the IR that
/// NY verified, or the bounds are unsound. Both are critical bugs.
///
/// Part of #506.
#[test]
fn test_gpu_output_within_verified_bounds() {
    let kernel_def = snake_kernel_def();

    // Step 1: NY verification — prove bounds for x ∈ [-10, 10], alpha=1.0.
    let bindings = vec![
        nn_verify::ParamBinding::Variable,      // x
        nn_verify::ParamBinding::Constant(1.0), // alpha
    ];
    let verification = nn_verify::VerifyRequest::new(&kernel_def)
        .bindings(&bindings)
        .variable_bounds(&[(-10.0, 10.0)])
        .verify_bounds()
        .expect("NY verification should succeed");

    assert!(
        verification.is_finite,
        "verified bounds must be finite: [{}, {}]",
        verification.output_lower, verification.output_upper,
    );

    let proved_lower = verification.output_lower;
    let proved_upper = verification.output_upper;

    // Step 2: Metal GPU dispatch on sampled inputs within the verified range.
    let (cache, pipeline) = compile_snake_pipeline();
    let test_x: Vec<f32> = vec![
        -10.0, -7.5, -5.0, -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0,
    ];
    let test_alpha: Vec<f32> = vec![1.0; test_x.len()];
    let gpu_out = pipeline
        .dispatch_elementwise(cache.context(), &[&test_x, &test_alpha])
        .expect("Metal dispatch should succeed");

    // Step 3: Assert every GPU output falls within the proved bounds.
    // ULP margin accounts for GPU floating-point rounding vs the real-valued
    // bounds that NY operates on. One ULP in each direction.
    let ulp_margin = (proved_upper - proved_lower) * f32::EPSILON;
    let safe_lower = proved_lower - ulp_margin;
    let safe_upper = proved_upper + ulp_margin;

    for (i, (&x, &g)) in test_x.iter().zip(&gpu_out).enumerate() {
        assert!(
            g >= safe_lower && g <= safe_upper,
            "GPU output at x={x} violates proved bounds: \
             gpu={g}, proved=[{proved_lower}, {proved_upper}], \
             safe=[{safe_lower}, {safe_upper}] (index {i})",
        );
    }
}

// Shared contract test harness: verify-dispatch-assert pipeline (#700)
#[path = "contract_harness.rs"]
mod contract_harness;

// Cross-backend contract tests: relu, clamped, bounds tightening
#[path = "contract.rs"]
mod contract;

// Extended cross-backend contract tests: silu_mul, gelu, inv_norm, rope (#542)
#[path = "contract_extended.rs"]
mod contract_extended;

// Normalization kernel contract tests (rms_norm, layer_norm, instance_norm)
#[path = "contract_norm.rs"]
mod contract_norm;

// AdaIN (K3) and fused AdaIN+Snake (K4) contract tests (#570)
#[path = "contract_adain.rs"]
mod contract_adain;

// Tanh standalone GPU contract test (#776)
#[path = "contract_tanh.rs"]
mod contract_tanh;
