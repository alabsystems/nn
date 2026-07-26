// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification runner functions extracted from `main.rs` (#909).
//!
//! Contains the kernel, fusion, and model verification loops plus
//! certificate generation helpers.

use std::path::Path;

use nn_verify::{
    certificate_from_pipeline_enriched, verify_and_record_full, CertificateBundle,
    CertificateEnrichment, FusionVerification, ParamInputRecord, PipelineResult, ProofCertificate,
    PropMethod, ScalarInputBounds, SmtStatusRecord, VerifyStatus,
};

use super::configs::{BuilderFailure, KernelConfig};
use super::fusion_configs::FusionConfig;
use super::model_configs::ModelConfig;
#[path = "runner_helpers.rs"]
mod runner_helpers;
use runner_helpers::{
    extract_layer_bounds_for_kernel, source_path_for_config, source_path_for_model,
};

#[path = "runner_trace.rs"]
mod runner_trace;
pub(super) use runner_trace::run_trace_model_verification;

/// Write proof certificate bundle to disk (#802).
pub(super) fn save_certificate_bundle(bundle: &CertificateBundle, cert_path: &Path) {
    if !bundle.is_empty() {
        match bundle.save(cert_path) {
            Ok(()) => println!(
                "Saved {} proof certificate(s) to {}",
                bundle.len(),
                cert_path.display()
            ),
            Err(e) => eprintln!("warning: failed to save certificate bundle: {e}"),
        }
    }
}

/// Run NY + ay verification for each config.
///
/// Returns `(passed, failed, certificate_bundle)`. Each successfully verified
/// kernel produces a `ProofCertificate` in the bundle (#802).
pub(super) fn run_verification(
    status: &mut VerifyStatus,
    configs: Vec<KernelConfig>,
    builder_failures: usize,
    kani_status_path: Option<&Path>,
) -> (usize, usize, CertificateBundle) {
    let mut passed = 0;
    let mut failed = builder_failures;
    let mut bundle = CertificateBundle::new("nn_verify_all");

    println!(
        "{:<30} {:>10} {:>10} {:>12} {:>14} {:>10}",
        "Kernel", "Status", "Method", "Width", "SMT", "Soundness"
    );
    println!("{}", "-".repeat(90));

    for config in configs {
        // Pass config_name as status_key so each configuration gets a
        // distinct entry in nn_verify_status.json (#513) without mutating
        // kernel.name, which must stay as the base name for BOUNDS_REGISTRY
        // dispatch (#521).
        let name = config.config_name;
        let bounds = match ScalarInputBounds::new(config.input_lower, config.input_upper) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{name:<30} SKIP      bounds error: {e}");
                failed += 1;
                continue;
            }
        };

        match verify_and_record_full(
            status,
            &config.kernel,
            &config.constant_params,
            bounds,
            Some(name),
        ) {
            Ok(result) => {
                print_result(name, &result);
                let cert = build_certificate(
                    name,
                    &result,
                    bounds,
                    &config.kernel,
                    &config.constant_params,
                    kani_status_path,
                );
                bundle.push(cert);
                passed += 1;
            }
            Err(e) => {
                eprintln!("{name:<30} FAIL      {e}");
                record_pipeline_failure(status, name, bounds, &config.constant_params, &e);
                failed += 1;
            }
        }
    }

    (passed, failed, bundle)
}

/// Persist builder failures to the status file so they are visible in
/// nn_verify_status.json, not just in transient stderr output (#558 AC2).
pub(super) fn record_builder_failures(status: &mut VerifyStatus, failures: &[BuilderFailure]) {
    for f in failures {
        let bounds = match ScalarInputBounds::new(f.input_lower, f.input_upper) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "  warning: cannot record {}: bad bounds: {e}",
                    f.config_name
                );
                continue;
            }
        };
        if let Err(e) =
            status.record_failure(f.config_name, PropMethod::Ibp, bounds, &f.constant_params)
        {
            eprintln!(
                "  warning: failed to record failure for {}: {e}",
                f.config_name
            );
            continue;
        }
        let smt_record = SmtStatusRecord::execution_failed(&format!("builder error: {}", f.error));
        if let Err(e) = status.record_smt(f.config_name, smt_record) {
            eprintln!(
                "  warning: failed to record SMT status for {}: {e}",
                f.config_name
            );
        }
    }
}

/// Persist a pipeline failure to the status file so it is visible in
/// nn_verify_status.json, not just in transient stderr output.
fn record_pipeline_failure(
    status: &mut VerifyStatus,
    name: &str,
    bounds: ScalarInputBounds,
    constant_params: &[f32],
    error: &nn_verify::VerifyError,
) {
    // If NY itself failed (no entry exists), record_failure creates
    // a Failed entry. Then record_smt attaches the error detail.
    if !status.has_kernel(name) {
        if let Err(e) = status.record_failure(name, PropMethod::Ibp, bounds, constant_params) {
            eprintln!("  warning: failed to record failure for {name}: {e}");
        }
    }
    let failure_record = SmtStatusRecord::execution_failed(&format!("pipeline error: {error}"));
    if let Err(e) = status.record_smt(name, failure_record) {
        eprintln!("  warning: failed to record SMT status for {name}: {e}");
    }
}

fn print_result(name: &str, result: &PipelineResult) {
    let gc = &result.gamma_crown;
    let status = if gc.is_finite { "Verified" } else { "Infinite" };
    let method = format!("{:?}", gc.method);
    let width = if gc.is_finite {
        format!("{:.4}", gc.output_upper - gc.output_lower)
    } else {
        "inf".to_string()
    };
    let smt = format!("{:?}", result.smt.outcome);
    let soundness = format!("{:?}", gc.soundness_mode);
    println!("{name:<30} {status:>10} {method:>10} {width:>12} {smt:>14} {soundness:>10}");

    // Surface CROWN fallback diagnostics so consumers (e.g. dvoice) can
    // distinguish "CROWN not attempted" from "CROWN failed, fell back to IBP".
    if let Some(reason) = &gc.crown_fallback_reason {
        eprintln!("  {name}: CROWN fallback to IBP — {reason}");
    }
}

/// Build a proof certificate from a pipeline result (#802).
///
/// For scalar kernels verified via `verify_and_record_full`, the certificate
/// includes: NY output bounds, SMT outcome, Kani proof records
/// (when kani_status.json is available), per-layer bound trace, and the
/// verification method + soundness provenance.
///
/// Layer bounds are extracted by rebuilding the NY graph from the
/// kernel definition and calling `extract_layer_bounds`. Graph construction
/// is cheap compared to the verification itself.
fn build_certificate(
    config_name: &str,
    result: &PipelineResult,
    bounds: ScalarInputBounds,
    kernel: &nn_dsl::ir::KernelDef,
    constant_params: &[f32],
    kani_status_path: Option<&Path>,
) -> ProofCertificate {
    let variable_inputs = vec![ParamInputRecord::new(0, bounds.lower(), bounds.upper())];
    let smt_outcome = format!("{:?}", result.smt.outcome);

    // Extract per-layer bound trace by rebuilding the graph (#802 AC3).
    let layer_bounds = extract_layer_bounds_for_kernel(kernel, constant_params, bounds);

    let enrichment = CertificateEnrichment {
        source_path: source_path_for_config(config_name),
        kani_status_path: kani_status_path.map(Path::to_path_buf),
        verifier_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        layer_bounds,
        ..CertificateEnrichment::default()
    };
    let mut cert = certificate_from_pipeline_enriched(
        &result.gamma_crown,
        &variable_inputs,
        constant_params,
        Some(&smt_outcome),
        Some(&enrichment),
    );
    // Override kernel_name to config_name for distinct certificate identity.
    cert.kernel_name = config_name.to_string();
    cert
}

/// Run fusion equivalence verification for each config (#803 AC3).
///
/// Each fusion config calls its convenience wrapper (which builds both kernels,
/// constructs the `FusionSpec`, and runs NY diamond DAG diff), then
/// records the result to `nn_verify_status.json` via `record_fusion`.
///
/// Returns `(passed, failed)`.
pub(super) fn run_fusion_verification(
    status: &mut VerifyStatus,
    configs: Vec<FusionConfig>,
) -> (usize, usize) {
    let mut passed = 0;
    let mut failed = 0;

    println!(
        "\n{:<30} {:>10} {:>10} {:>12} {:>10}",
        "Fusion", "Status", "Method", "MaxDiff", "Conclusive"
    );
    println!("{}", "-".repeat(76));

    for config in configs {
        let name = config.config_name;
        match (config.verify_fn)(
            &config.variable_bounds,
            config.epsilon,
            &config.verify_config,
        ) {
            Ok(result) => {
                print_fusion_result(name, &result);
                if let Err(e) = status.record_fusion(&result, &config.variable_bounds, Some(name)) {
                    eprintln!("  {name}: warning: failed to record fusion: {e}");
                }
                passed += 1;
            }
            Err(e) => {
                eprintln!("{name:<30} FAIL      {e}");
                failed += 1;
            }
        }
    }

    (passed, failed)
}

fn print_fusion_result(name: &str, result: &FusionVerification) {
    let status = if result.within_epsilon {
        "Proved"
    } else {
        "Exceeded"
    };
    let method = format!("{:?}", result.method);
    let max_diff = format!("{:.6}", result.max_abs_diff);
    let conclusive = if result.is_conclusive() { "Yes" } else { "No" };
    println!("{name:<30} {status:>10} {method:>10} {max_diff:>12} {conclusive:>10}");
}

// ---------------------------------------------------------------------------
// Model-level verification (#839 AC4)
// ---------------------------------------------------------------------------

/// Run NY verification for composed model-level configurations.
///
/// Each model config produces a `TensorKernelDef` representing the full model
/// graph. Verification translates to NY, propagates bounds (IBP →
/// CROWN escalation), records the result, and builds a model-level certificate.
///
/// Returns `(passed, failed)`.
pub(super) fn run_model_verification(
    status: &mut VerifyStatus,
    configs: Vec<ModelConfig>,
    bundle: &mut CertificateBundle,
) -> (usize, usize) {
    let mut passed = 0;
    let mut failed = 0;

    println!(
        "\n{:<30} {:>10} {:>10} {:>12} {:>10}",
        "Model", "Status", "Method", "OutShape", "Soundness"
    );
    println!("{}", "-".repeat(76));

    let verify_config = nn_verify::VerifyConfig::default().with_collect_layer_bounds(true);

    for config in configs {
        let name = config.name;
        match nn_verify::verify_tensor_and_record_with_config(
            status,
            &config.def,
            &config.bindings,
            &config.input_bounds,
            Some(name),
            &verify_config,
        ) {
            Ok(result) => {
                let method = format!("{:?}", result.verification.method);
                let soundness = format!("{:?}", result.verification.soundness_mode);
                let shape = result.output_bounds.lower_upper().0.shape().to_vec();
                let shape_str = format!("{shape:?}");
                let status_str = if result.verification.is_finite {
                    "Verified"
                } else {
                    "Infinite"
                };
                println!(
                    "{name:<30} {status_str:>10} {method:>10} {shape_str:>12} {soundness:>10}"
                );

                // Build model-level certificate.
                let variable_inputs = vec![ParamInputRecord::new(
                    0,
                    config.input_lower,
                    config.input_upper,
                )];
                let enrichment = CertificateEnrichment {
                    source_path: source_path_for_model(name),
                    verifier_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                    layer_bounds: result.layer_bounds,
                    ..CertificateEnrichment::default()
                };
                let mut cert = certificate_from_pipeline_enriched(
                    &result.verification,
                    &variable_inputs,
                    &[],  // No scalar constant params — all are tensor bindings.
                    None, // No SMT for model-level (tensor graphs).
                    Some(&enrichment),
                );
                cert.kernel_name = name.to_string();
                bundle.push(cert);
                passed += 1;
            }
            Err(e) => {
                eprintln!("{name:<30} FAIL      {e}");
                failed += 1;
            }
        }
    }

    (passed, failed)
}
