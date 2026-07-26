// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edge-case test generation for differential tests.
//!
//! Split from `codegen_difftest.rs` to stay within file-size limits.
//! Generates sub-tests for: single element, threadgroup boundary,
//! denormals, bound edges, and zero-length dispatch.

use crate::ir::{KernelDef, ScalarType};
use crate::precision::{InputBounds, PrecisionTier};

/// Emit edge-case test phases.
pub(super) fn emit_edge_cases(
    lines: &mut Vec<String>,
    kernel: &KernelDef,
    bounds: &InputBounds,
    tier: PrecisionTier,
) {
    let tier_str = tier.as_str();
    let dtype_str = match kernel.return_type {
        ScalarType::F32 => "ScalarType::F32",
        ScalarType::F16 => "ScalarType::F16",
        ScalarType::BF16 => "ScalarType::BF16",
    };

    let params: Vec<(&str, ScalarType)> = kernel
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.ty))
        .collect();

    emit_single_element(lines, kernel, &params, bounds, tier_str, dtype_str);
    emit_threadgroup_boundary(lines, kernel, &params, bounds, tier_str, dtype_str);
    emit_denormals(lines, kernel, &params, bounds, tier_str, dtype_str);
    emit_bound_edges(lines, kernel, &params, bounds, tier_str, dtype_str);
    emit_zero_length(lines, kernel);
}

/// Emit a sub-test block that dispatches specific input values through
/// both Rust and Metal, then compares with the precision contract.
fn emit_subtest(
    lines: &mut Vec<String>,
    kernel: &KernelDef,
    label: &str,
    inputs: &[(&str, ScalarType, &str)],
    tier_str: &str,
    dtype_str: &str,
) {
    lines.push(format!("    // Edge case: {label}"));
    lines.push("    {".to_string());
    for (pname, ty, expr) in inputs {
        match ty {
            ScalarType::F32 => {
                lines.push(format!("    let {pname}_data: Vec<f32> = {expr};"));
            }
            ScalarType::F16 => {
                lines.push(format!(
                    "    let {pname}_data: Vec<half::f16> = ({expr}).into_iter()\
                     .map(half::f16::from_f32).collect();"
                ));
            }
            ScalarType::BF16 => {
                lines.push(format!(
                    "    let {pname}_data: Vec<half::bf16> = ({expr}).into_iter()\
                     .map(half::bf16::from_f32).collect();"
                ));
            }
        }
    }
    let first_pname = &inputs[0].0;
    lines.push(format!("    let n = {first_pname}_data.len();"));
    super::emit_rust_reference(lines, kernel);
    super::emit_metal_dispatch(lines, kernel);
    lines.push(format!(
        "    let contract = nn_dsl::PrecisionContract::bootstrap(\
         nn_dsl::PrecisionTier::parse(\"{tier_str}\").expect(\"invariant: valid tier\"), \
         nn_dsl::{dtype_str});"
    ));
    lines.push(
        "    for (i, (r, m)) in rust_out.iter().zip(metal_out.iter()).enumerate() {".to_string(),
    );
    lines.push(format!(
        "        assert!(nn_dsl::within_differential_budget(*r, *m, contract), \
         \"{label} mismatch at {{i}}: rust={{r}}, metal={{m}}, delta={{}}\", (r - m).abs());"
    ));
    lines.push("    }".to_string());
    lines.push("    }".to_string());
}

fn emit_single_element(
    lines: &mut Vec<String>,
    kernel: &KernelDef,
    params: &[(&str, ScalarType)],
    bounds: &InputBounds,
    tier_str: &str,
    dtype_str: &str,
) {
    let inputs: Vec<(&str, ScalarType, String)> = params
        .iter()
        .map(|(pname, ty)| {
            let b = bounds.get(pname, *ty);
            let mid = f64::midpoint(b.lo(), b.hi());
            let expr = format!("vec![{mid}_f32]");
            (*pname, *ty, expr)
        })
        .collect();
    let refs: Vec<(&str, ScalarType, &str)> = inputs
        .iter()
        .map(|(n, t, e)| (*n, *t, e.as_str()))
        .collect();
    emit_subtest(lines, kernel, "single element", &refs, tier_str, dtype_str);
}

fn emit_threadgroup_boundary(
    lines: &mut Vec<String>,
    kernel: &KernelDef,
    params: &[(&str, ScalarType)],
    bounds: &InputBounds,
    tier_str: &str,
    dtype_str: &str,
) {
    for tg_size in [64_usize, 65] {
        let inputs: Vec<(&str, ScalarType, String)> = params
            .iter()
            .map(|(pname, ty)| {
                let b = bounds.get(pname, *ty);
                let mid = f64::midpoint(b.lo(), b.hi());
                let expr = format!("vec![{mid}_f32; {tg_size}]");
                (*pname, *ty, expr)
            })
            .collect();
        let refs: Vec<(&str, ScalarType, &str)> = inputs
            .iter()
            .map(|(n, t, e)| (*n, *t, e.as_str()))
            .collect();
        emit_subtest(
            lines,
            kernel,
            &format!("threadgroup boundary n={tg_size}"),
            &refs,
            tier_str,
            dtype_str,
        );
    }
}

fn emit_denormals(
    lines: &mut Vec<String>,
    kernel: &KernelDef,
    params: &[(&str, ScalarType)],
    bounds: &InputBounds,
    tier_str: &str,
    dtype_str: &str,
) {
    // Skip denormal tests for kernels with no float params at all.
    // Both F32 and F16 have denormal ranges that should be tested.
    if kernel.params.is_empty() {
        return;
    }

    // Skip denormal tests for kernels containing rsqrt, recip, or div.
    // Metal GPUs use FTZ (flush-to-zero) which flushes denormals to zero,
    // causing rsqrt(0)→NaN or x/0→NaN where Rust CPU computes correctly.
    // This is expected hardware behavior, not a codegen bug.
    if kernel.has_ftz_sensitive_op() {
        return;
    }

    // Candidate denormal values per dtype.  Only include values that fall
    // within the parameter's declared bounds — denormals outside the valid
    // input range would test undefined behaviour, not GPU/CPU divergence.
    //
    // All parameter vectors must have the same length.  Compute valid
    // candidates per-param first, find the uniform length, then build
    // expressions.
    let per_param: Vec<(&str, ScalarType, Vec<f64>)> = params
        .iter()
        .map(|(pname, ty)| {
            let b = bounds.get(pname, *ty);
            let candidates: Vec<f64> = match ty {
                ScalarType::F32 => vec![1.0e-38, -1.0e-38, 5.0e-39],
                // f16 MIN_POSITIVE ≈ 6.1e-5; actual denormals are below that.
                // 5.96e-8 = smallest positive f16 subnormal (2^-24).
                // BF16 uses f16 denormals since Metal computes bf16 as f16.
                ScalarType::F16 | ScalarType::BF16 => vec![5.96e-8, -5.96e-8, 3.0e-5],
            };
            let valid: Vec<f64> = candidates
                .into_iter()
                .filter(|v| *v >= b.lo() && *v <= b.hi())
                .collect();
            (*pname, *ty, valid)
        })
        .collect();

    // Uniform length: minimum of all valid counts.  If any parameter has
    // zero valid denormals, fall back to one element at a small in-range
    // value for every parameter.
    let min_len = per_param.iter().map(|(_, _, v)| v.len()).min().unwrap_or(0);
    let inputs: Vec<(&str, ScalarType, String)> = if min_len == 0 {
        per_param
            .iter()
            .map(|(pname, ty, _)| {
                let b = bounds.get(pname, *ty);
                let near_lo = b.lo() + (b.hi() - b.lo()) * 1e-12;
                let expr = format!("vec![{near_lo}_f32]");
                (*pname, *ty, expr)
            })
            .collect()
    } else {
        per_param
            .iter()
            .map(|(pname, ty, valid)| {
                let elems: Vec<String> = valid
                    .iter()
                    .take(min_len)
                    .map(|v| format!("{v}_f32"))
                    .collect();
                let expr = format!("vec![{}]", elems.join(", "));
                (*pname, *ty, expr)
            })
            .collect()
    };

    let refs: Vec<(&str, ScalarType, &str)> = inputs
        .iter()
        .map(|(n, t, e)| (*n, *t, e.as_str()))
        .collect();
    emit_subtest(lines, kernel, "denormal inputs", &refs, tier_str, dtype_str);
}

fn emit_bound_edges(
    lines: &mut Vec<String>,
    kernel: &KernelDef,
    params: &[(&str, ScalarType)],
    bounds: &InputBounds,
    tier_str: &str,
    dtype_str: &str,
) {
    let inputs: Vec<(&str, ScalarType, String)> = params
        .iter()
        .map(|(pname, ty)| {
            let b = bounds.get(pname, *ty);
            let lo = b.lo();
            let hi = b.hi();
            let mid = f64::midpoint(lo, hi);
            let expr = format!("vec![{lo}_f32, {hi}_f32, {mid}_f32]");
            (*pname, *ty, expr)
        })
        .collect();
    let refs: Vec<(&str, ScalarType, &str)> = inputs
        .iter()
        .map(|(n, t, e)| (*n, *t, e.as_str()))
        .collect();
    emit_subtest(
        lines,
        kernel,
        "bound edge values",
        &refs,
        tier_str,
        dtype_str,
    );
}

fn emit_zero_length(lines: &mut Vec<String>, kernel: &KernelDef) {
    let name = &kernel.name;
    lines.push("    // Edge case: zero-length input".to_string());
    lines.push("    {".to_string());
    for param in &kernel.params {
        let pname = &param.name;
        match param.ty {
            ScalarType::F32 => {
                lines.push(format!("    let {pname}_empty: Vec<f32> = Vec::new();"));
            }
            ScalarType::F16 => {
                lines.push(format!(
                    "    let {pname}_empty: Vec<half::f16> = Vec::new();"
                ));
            }
            ScalarType::BF16 => {
                lines.push(format!(
                    "    let {pname}_empty: Vec<half::bf16> = Vec::new();"
                ));
            }
        }
    }
    let empty_refs: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{}_empty.as_slice()", p.name))
        .collect();
    match kernel.return_type {
        ScalarType::F32 => {
            lines.push(format!(
                "    let empty_out: Result<Vec<f32>, _> = kp.dispatch_elementwise(\
                 cache.context(), &[{}]);",
                empty_refs.join(", "),
            ));
        }
        ScalarType::F16 => {
            lines.push(format!(
                "    let empty_out: Result<Vec<half::f16>, _> = kp.dispatch_elementwise(\
                 cache.context(), &[{}]);",
                empty_refs.join(", "),
            ));
        }
        ScalarType::BF16 => {
            lines.push(format!(
                "    let empty_out: Result<Vec<half::bf16>, _> = kp.dispatch_elementwise(\
                 cache.context(), &[{}]);",
                empty_refs.join(", "),
            ));
        }
    }
    lines.push(format!(
        "    assert!(empty_out.is_ok(), \
         \"zero-length dispatch should succeed for {name}\");"
    ));
    lines.push(format!(
        "    assert!(empty_out.expect(\"invariant: zero-length ok\").is_empty(), \
         \"zero-length dispatch should produce empty output for {name}\");"
    ));
    lines.push("    }".to_string());
}
