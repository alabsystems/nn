// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verify all kernels and models, persist results to per-model status files.
//!
//! Run with: `cargo run -p nn-verify --example verify_all`
//!
//! This is the production entry point for "verify all kernels with their
//! standard configurations and persist results." Each kernel goes through:
//!   1. NY bounds verification (IBP)
//!   2. ay SMT cross-verification
//!   3. Result persistence to `nn_verify_status_{model}.json` (#2577)
//!   4. Proof certificate generation to `nn_verify.proof.json` (#802)
//!
//! Status files are split per-model (kokoro, demucs, silero, whisper, qwen3,
//! shared) to prevent concurrent modification races between Workers (#2577).
//!
//! Addresses #424: no automated verification runner.
//! Config builders extracted to `configs.rs` (#571 AC1).
//! Certificate generation added for #802 (proof certificate format).
//! Runner functions extracted to `runner.rs` (#909).

use std::path::Path;
use std::process;

use nn_verify::{sign_bundle, SigningKey, VerifyStatus};

mod configs;
mod fusion_configs;
mod model_configs;
mod runner;
mod trace_model_configs;

use configs::build_kernel_configs;
use fusion_configs::build_fusion_configs;
use model_configs::build_model_configs;
use runner::{
    record_builder_failures, run_fusion_verification, run_model_verification,
    run_trace_model_verification, run_verification, save_certificate_bundle,
};
use trace_model_configs::build_trace_model_configs;

/// Resolve the workspace root directory.
fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Print a verification tier summary line.
fn print_tier_summary(label: &str, passed: usize, total: usize, failed: usize) {
    if total > 0 {
        println!(
            "{}\n{passed}/{total} {label} verified, {failed} failed\n",
            "-".repeat(90)
        );
    }
}

fn main() {
    let root = workspace_root();
    let cert_path = root.join("nn_verify.proof.json");
    let kani_status_path = root.join("kani_status.json");
    let signing_key = SigningKey::from_env();

    // Migrate from monolithic file to per-model files if needed (#2577).
    VerifyStatus::migrate_to_per_model(&root).expect("migrate to per-model status files");

    // Load merged status from all per-model files.
    let mut status = VerifyStatus::load_merged(&root).expect("load per-model status files");
    let pre_count = status.kernel_count();
    println!("Loaded {pre_count} entries from per-model status files\n");

    let (configs, builder_failures) = build_kernel_configs();
    let num_build_failures = builder_failures.len();
    if num_build_failures > 0 {
        eprintln!("{num_build_failures} kernel builder(s) failed (see BUILD_ERR above)\n");
    }
    record_builder_failures(&mut status, &builder_failures);
    let total = configs.len() + num_build_failures;

    let kani_path = kani_status_path.exists().then_some(kani_status_path);
    let (passed, failed, mut bundle) = run_verification(
        &mut status,
        configs,
        num_build_failures,
        kani_path.as_deref(),
    );

    print_tier_summary("kernels", passed, total, failed);

    // Fusion equivalence verification (#803 AC3).
    let fusion_configs = build_fusion_configs();
    let fusion_total = fusion_configs.len();
    let (fusion_passed, fusion_failed) = run_fusion_verification(&mut status, fusion_configs);
    print_tier_summary("fusions", fusion_passed, fusion_total, fusion_failed);

    // Model-level verification (#839 AC4).
    let model_configs = build_model_configs();
    let model_total = model_configs.len();
    let (model_passed, model_failed) =
        run_model_verification(&mut status, model_configs, &mut bundle);
    print_tier_summary("models", model_passed, model_total, model_failed);

    // Trace-based model verification (#2074 AC0).
    let trace_configs = build_trace_model_configs();
    let trace_total = trace_configs.len();
    let (trace_passed, trace_failed) =
        run_trace_model_verification(&mut status, trace_configs, &mut bundle);
    print_tier_summary("trace models", trace_passed, trace_total, trace_failed);

    // Save split across per-model files (#2577).
    status
        .save_per_model(&root)
        .expect("save per-model status files");
    let final_count = status.kernel_count();
    println!("Saved {final_count} kernel entries across per-model files");
    if final_count > pre_count {
        println!("  (+{} new entries)", final_count - pre_count);
    }

    // Sign certificates if signing key is configured (#3253).
    if let Some(key) = signing_key.as_bytes() {
        sign_bundle(&mut bundle, key).expect("certificate signing failed");
        println!("Signed {} certificates", bundle.certificates.len());
    }

    save_certificate_bundle(&bundle, &cert_path);

    if failed > 0 || fusion_failed > 0 || model_failed > 0 || trace_failed > 0 {
        process::exit(1);
    }
}
