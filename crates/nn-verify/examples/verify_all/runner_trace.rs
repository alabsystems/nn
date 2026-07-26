// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based model verification runner extracted from `runner.rs` (#2074).
//!
//! Addresses R1-1143 findings:
//! - Finding 1: Derives soundness from graph+method via gamma-propagate
//! - Finding 2: Keeps runner.rs under 500 lines via extraction
//! - Finding 3: Uses NaN-propagating min/max for output bound folds

use ny_propagate::soundness::soundness_provenance_for_graph;
use ny_propagate::PropagationMethod;

use nn_verify::{
    certificate_from_pipeline_enriched, CertificateBundle, CertificateEnrichment, ParamInputRecord,
    PropMethod, VerifyStatus,
};

use super::super::trace_model_configs::TraceModelConfig;
use super::runner_helpers;

/// Map nn's `PropMethod` to gamma-propagate's `PropagationMethod`.
///
/// Mirrors the private `to_propagation_method` in `nn-verify/src/soundness.rs`.
fn to_propagation_method(method: &PropMethod) -> PropagationMethod {
    match method {
        PropMethod::Ibp => PropagationMethod::Ibp,
        PropMethod::Crown => PropagationMethod::Crown,
        // PropMethod is #[non_exhaustive]; fail-safe to IBP for unknown variants.
        _ => PropagationMethod::Ibp,
    }
}

/// Fold minimum that propagates NaN (IEEE 754: f32::min returns the non-NaN
/// operand, which silently hides NaN bounds).
fn fold_min_propagate_nan(vals: &[f32]) -> f32 {
    vals.iter().copied().fold(f32::INFINITY, |acc, v| {
        if v.is_nan() || acc.is_nan() {
            f32::NAN
        } else {
            acc.min(v)
        }
    })
}

/// Fold maximum that propagates NaN.
fn fold_max_propagate_nan(vals: &[f32]) -> f32 {
    vals.iter().copied().fold(f32::NEG_INFINITY, |acc, v| {
        if v.is_nan() || acc.is_nan() {
            f32::NAN
        } else {
            acc.max(v)
        }
    })
}

/// Verify trace-based model configurations (#2074 AC0).
///
/// Each `TraceModelConfig` holds a pre-built NY `GraphNetwork` from
/// `trace_to_graph_model()`. Verification propagates bounds (IBP → CROWN
/// escalation), records the result, and builds a model-level certificate.
///
/// This validates the trace-to-graph pipeline in production, exercising the
/// automated DynTensor tracing path that real model consumers use.
///
/// Returns `(passed, failed)`.
pub(crate) fn run_trace_model_verification(
    status: &mut VerifyStatus,
    configs: Vec<TraceModelConfig>,
    bundle: &mut CertificateBundle,
) -> (usize, usize) {
    let mut passed = 0;
    let mut failed = 0;

    println!(
        "\n{:<30} {:>10} {:>10} {:>12} {:>10}",
        "TraceModel", "Status", "Method", "OutWidth", "Soundness"
    );
    println!("{}", "-".repeat(76));

    for config in configs {
        let name = config.name;

        // Propagate bounds: try CROWN first, fall back to IBP.
        let (method, output, crown_err) =
            match nn_verify::propagate_with_crown_fallback(&config.graph, &config.input_bounds) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("{name:<30} FAIL      {e}");
                    failed += 1;
                    continue;
                }
            };

        // Extract scalar summary from output bounds.
        // Uses NaN-propagating folds (R1-1143 Finding 3) so NaN bounds
        // don't silently become finite via f32::min/f32::max.
        let (lo_arr, up_arr) = output.lower_upper();
        let lo_vals: Vec<f32> = lo_arr.iter().copied().collect();
        let up_vals: Vec<f32> = up_arr.iter().copied().collect();

        let output_lower = fold_min_propagate_nan(&lo_vals);
        let output_upper = fold_max_propagate_nan(&up_vals);
        let output_width = output_upper - output_lower;
        let is_finite = output_lower.is_finite() && output_upper.is_finite();

        // Derive soundness from graph layers and propagation method
        // (R1-1143 Finding 1). Matches the kernel path pattern at
        // nn-verify/src/verify.rs:86-92 via soundness_for_graph.
        let prop_method = to_propagation_method(&method);
        let provenance = soundness_provenance_for_graph(&config.graph, &prop_method);
        let soundness = provenance.mode();

        let status_str = if is_finite { "Verified" } else { "Infinite" };
        let method_str = format!("{method:?}");
        let width_str = if is_finite {
            format!("{output_width:.4}")
        } else {
            "inf".to_string()
        };
        let soundness_str = format!("{soundness:?}");
        println!(
            "{name:<30} {status_str:>10} {method_str:>10} {width_str:>12} {soundness_str:>10}"
        );

        if let Some(reason) = &crown_err {
            eprintln!("  {name}: CROWN fallback to IBP — {reason}");
        }

        // Record to status file.
        let verification = nn_verify::KernelVerification::new(
            name.to_string(),
            method,
            output_lower,
            output_upper,
            output_width,
            is_finite,
        )
        .with_crown_fallback_reason(crown_err)
        .with_soundness_mode(soundness);
        let variable_inputs = vec![ParamInputRecord::new(
            0,
            config.input_lower,
            config.input_upper,
        )];
        if let Err(e) = status.record_with_variable_inputs(
            &verification,
            &variable_inputs,
            &[],
            Some(name),
            None, // trace runner — scalar bounds
        ) {
            eprintln!("  {name}: warning: failed to record status: {e}");
        }

        // Build proof certificate.
        let enrichment = CertificateEnrichment {
            source_path: source_path_for_trace_model(name),
            verifier_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            ..CertificateEnrichment::default()
        };
        let mut cert = certificate_from_pipeline_enriched(
            &verification,
            &variable_inputs,
            &[],  // No scalar constant params.
            None, // No SMT for trace models.
            Some(&enrichment),
        );
        cert.kernel_name = name.to_string();
        bundle.push(cert);
        passed += 1;
    }

    (passed, failed)
}

/// Source path for trace-based model verification configs (#2074 AC0).
fn source_path_for_trace_model(name: &str) -> Option<std::path::PathBuf> {
    let _ = name; // All trace models live in the same file.
    let root = runner_helpers::workspace_root();
    let path = root.join("crates/nn-verify/examples/verify_all/trace_model_configs.rs");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}
