// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differential test generation from KernelIR.
//!
//! Generates a `#[test]` function that:
//! 1. Creates random input vectors within declared bounds for each kernel parameter
//! 2. Runs the Rust reference function element-wise
//! 3. Dispatches the generated MSL on Metal via `nn_metal::KernelPipeline`
//! 4. Asserts all outputs match within the kernel's precision budget
//! 5. Tests edge cases: zero-length, single element, threadgroup boundary,
//!    denormals, and bound edges
//!
//! The generated test is gated behind `#[cfg(test)]` and
//! `#[cfg(target_os = "macos")]` since it requires a Metal GPU.

#[path = "codegen_difftest_edges.rs"]
mod edges;

use crate::ir::{IRError, KernelDef, ScalarType};
use crate::precision::{InputBounds, PrecisionTier};

/// Number of random samples for the main differential sweep.
const RANDOM_SAMPLE_COUNT: usize = 1024;

/// Emit a differential test as Rust source text (default bounds).
///
/// Convenience wrapper that uses default bounds for all parameters.
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "differential test string is computed but not used"]
pub(crate) fn emit_differential_test(
    kernel: &KernelDef,
    tier: PrecisionTier,
) -> Result<String, IRError> {
    emit_differential_test_with_bounds(kernel, tier, &InputBounds::new())
}

/// Emit a differential test with explicit per-parameter input bounds.
///
/// The generated test has three phases:
/// 1. **Random sweep**: 1024 random samples within declared bounds
/// 2. **Edge cases**: zero-length, single element, threadgroup boundary (64/65),
///    denormals, and values at bound edges
/// 3. **Precision comparison**: per-element assertion with full diagnostic output
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[must_use = "differential test string is computed but not used"]
pub fn emit_differential_test_with_bounds(
    kernel: &KernelDef,
    tier: PrecisionTier,
    bounds: &InputBounds,
) -> Result<String, IRError> {
    kernel.validate()?;
    let name = &kernel.name;
    let mut lines = Vec::new();

    lines.push("#[cfg(test)]".to_string());
    lines.push("#[cfg(target_os = \"macos\")]".to_string());
    lines.push("#[allow(clippy::excessive_precision)]".to_string());
    lines.push("#[test]".to_string());
    lines.push(format!("fn test_{name}_differential() {{"));

    // Metal context setup (shared across all phases)
    emit_metal_setup(&mut lines, kernel);

    // Phase 1: Random sweep
    lines.push(format!(
        "    // Phase 1: Random sweep ({RANDOM_SAMPLE_COUNT} samples)"
    ));
    lines.push("    {".to_string());
    lines.push(format!("    let n: usize = {RANDOM_SAMPLE_COUNT};"));
    emit_input_vectors(&mut lines, kernel, bounds);
    emit_rust_reference(&mut lines, kernel);
    emit_metal_dispatch(&mut lines, kernel);
    emit_comparison(&mut lines, kernel, tier);
    lines.push("    }".to_string());

    // Phase 2: Edge cases
    edges::emit_edge_cases(&mut lines, kernel, bounds, tier);

    lines.push("}".to_string());
    Ok(lines.join("\n"))
}

/// Emit Metal context and pipeline creation (once, reused across phases).
fn emit_metal_setup(lines: &mut Vec<String>, kernel: &KernelDef) {
    let upper = kernel.name.to_uppercase();
    // Initialize global Metal backend (#2424) — required for lazy batch dispatch.
    lines.push(
        "    let _ = nn_metal::MetalBackend::init()\
         .expect(\"Metal device required for differential test\");"
            .to_string(),
    );
    lines.push(
        "    let cache = nn_metal::PipelineCache::new_global()\
         .expect(\"Metal global cache\");"
            .to_string(),
    );
    lines.push(format!(
        "    let kp = nn_metal::KernelPipeline::from_descriptor(\
         &cache, &{upper}_DESCRIPTOR)\
         .expect(\"MSL should compile\");"
    ));
}

/// Emit random input vectors sampled within declared bounds.
///
/// Uses a seeded deterministic PRNG so tests are reproducible.
/// Each parameter gets a different seed derived from its index.
fn emit_input_vectors(lines: &mut Vec<String>, kernel: &KernelDef, bounds: &InputBounds) {
    for (i, param) in kernel.params.iter().enumerate() {
        let pname = &param.name;
        let bound = bounds.get(pname, param.ty);
        let lo = bound.lo();
        let hi = bound.hi();
        // Per-parameter seed: SplitMix64 step from the parameter index.
        let seed: u64 = splitmix64_next((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        // Generated code uses SplitMix64 mixing per element for good distribution.
        // t ∈ [0, 1): take top 53 bits, scale by 2^-53 (standard double conversion).
        let splitmix_expr =
            "let mut z = {SEED}_u64.wrapping_add((j as u64).wrapping_mul(0x9E3779B97F4A7C15));\
                     z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);\
                     z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);\
                     z ^= z >> 31;\
                     let t = (z >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64);"
                .replace("{SEED}", &seed.to_string());
        match param.ty {
            ScalarType::F32 => {
                lines.push(format!(
                    "    let {pname}_data: Vec<f32> = (0..n).map(|j| {{\
                     {splitmix_expr}\
                     ({lo}_f64 + t * ({hi}_f64 - {lo}_f64)) as f32\
                     }}).collect();"
                ));
            }
            ScalarType::F16 => {
                lines.push(format!(
                    "    let {pname}_data: Vec<half::f16> = (0..n).map(|j| {{\
                     {splitmix_expr}\
                     half::f16::from_f32(({lo}_f64 + t * ({hi}_f64 - {lo}_f64)) as f32)\
                     }}).collect();"
                ));
            }
            ScalarType::BF16 => {
                lines.push(format!(
                    "    let {pname}_data: Vec<half::bf16> = (0..n).map(|j| {{\
                     {splitmix_expr}\
                     half::bf16::from_f32(({lo}_f64 + t * ({hi}_f64 - {lo}_f64)) as f32)\
                     }}).collect();"
                ));
            }
        }
    }
}

/// Single SplitMix64 mixing step (Vigna, 2015).
///
/// Used at codegen time to derive per-parameter seeds. The generated test
/// code also uses the same mixing function inline for per-element randomness.
fn splitmix64_next(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Emit the Rust reference computation: call the original function per element.
pub(crate) fn emit_rust_reference(lines: &mut Vec<String>, kernel: &KernelDef) {
    let name = &kernel.name;
    let call_args: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{}_data[i]", p.name))
        .collect();
    let ret_convert = match kernel.return_type {
        ScalarType::F32 => "",
        ScalarType::F16 | ScalarType::BF16 => ".to_f32()",
    };
    lines.push(format!(
        "    let rust_out: Vec<f32> = (0..n).map(|i| {name}({}){ret_convert}).collect();",
        call_args.join(", "),
    ));
}

/// Emit GPU dispatch using the already-created pipeline.
pub(crate) fn emit_metal_dispatch(lines: &mut Vec<String>, kernel: &KernelDef) {
    let input_refs: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{}_data.as_slice()", p.name))
        .collect();
    match kernel.return_type {
        ScalarType::F32 => {
            lines.push(format!(
                "    let metal_out = kp.dispatch_elementwise(cache.context(), &[{}])\
                 .expect(\"Metal dispatch should succeed\");",
                input_refs.join(", "),
            ));
        }
        ScalarType::F16 => {
            lines.push(format!(
                "    let metal_out_f16 = kp.dispatch_elementwise(cache.context(), &[{}])\
                 .expect(\"Metal dispatch should succeed\");",
                input_refs.join(", "),
            ));
            lines.push(
                "    let metal_out: Vec<f32> = metal_out_f16.iter().map(|v| v.to_f32()).collect();"
                    .to_string(),
            );
        }
        ScalarType::BF16 => {
            lines.push(format!(
                "    let metal_out_bf16 = kp.dispatch_elementwise(cache.context(), &[{}])\
                 .expect(\"Metal dispatch should succeed\");",
                input_refs.join(", "),
            ));
            lines.push(
                "    let metal_out: Vec<f32> = metal_out_bf16.iter().map(|v| v.to_f32()).collect();"
                    .to_string(),
            );
        }
    }
}

/// Emit precision budget comparison with diagnostic output.
fn emit_comparison(lines: &mut Vec<String>, kernel: &KernelDef, tier: PrecisionTier) {
    let tier_str = tier.as_str();
    let dtype_str = match kernel.return_type {
        ScalarType::F32 => "ScalarType::F32",
        ScalarType::F16 => "ScalarType::F16",
        ScalarType::BF16 => "ScalarType::BF16",
    };
    lines.push(format!(
        "    let contract = nn_dsl::PrecisionContract::bootstrap(\
         nn_dsl::PrecisionTier::parse(\"{tier_str}\").expect(\"invariant: valid tier\"), \
         nn_dsl::{dtype_str});"
    ));

    // Build input-value diagnostic string per element
    let input_fmts: Vec<String> = kernel
        .params
        .iter()
        .map(|p| {
            let pname = &p.name;
            format!("{pname}={{{pname}_data_i}}")
        })
        .collect();
    let input_fmt_str = input_fmts.join(", ");

    let input_let_binds: Vec<String> = kernel
        .params
        .iter()
        .map(|p| {
            let pname = &p.name;
            match p.ty {
                ScalarType::F32 => format!("let {pname}_data_i = {pname}_data[i];"),
                ScalarType::F16 | ScalarType::BF16 => {
                    format!("let {pname}_data_i = {pname}_data[i].to_f32();")
                }
            }
        })
        .collect();

    lines.push(
        "    for (i, (r, m)) in rust_out.iter().zip(metal_out.iter()).enumerate() {".to_string(),
    );
    for bind in &input_let_binds {
        lines.push(format!("        {bind}"));
    }
    lines.push(format!(
        "        assert!(nn_dsl::within_differential_budget(*r, *m, contract), \
         \"differential mismatch at index {{i}}: {input_fmt_str}, \
         rust={{r}}, metal={{m}}, delta={{}}\", (r - m).abs());"
    ));
    lines.push("    }".to_string());
}

#[cfg(test)]
#[path = "codegen_difftest_tests.rs"]
mod tests;
