// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generate a model-specific proof certificate bundle from the full
//! `nn_verify.proof.json` bundle.
//!
//! Run with: `cargo run -p nn-verify --example generate_proof_bundle -- --model silero_vad --output silero_vad.proof.json`
//!
//! Reads the workspace-root `nn_verify.proof.json`, filters to the
//! kernel certificates relevant to the specified model, validates the
//! resulting bundle, and writes it to the output path.
//!
//! Addresses #1680: V1 G2 proof certificate bundle for dvoice.

use std::path::{Path, PathBuf};
use std::process;

use nn_verify::{check_bundle, CertificateBundle};

/// Kernel certificates that compose the Silero VAD model.
///
/// The Silero VAD post-STFT pipeline uses:
/// - Conv1d + ReLU encoder (4 blocks) → verified as `relu`, `relu_wide`
/// - LSTM cell (sigmoid + tanh gates) → verified as `sigmoid`, `tanh_act`
/// - Linear + Sigmoid output → verified as `sigmoid`
/// - Full model composition → verified as `silero_vad_full`
///
/// We include all kernel configs that could appear in the model's
/// compute graph, plus the model-level composition certificate.
const SILERO_VAD_KERNEL_NAMES: &[&str] = &[
    // Activation functions used in the model
    "relu",
    "relu_wide",
    "sigmoid",
    "sigmoid_wide",
    "tanh_act",
    "tanh_act_wide",
    // Full model composition certificate
    "silero_vad_full",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model = find_arg(&args, "--model").unwrap_or_else(|| {
        eprintln!(
            "Usage: generate_proof_bundle --model <name> [--output <path>] [--input <proof.json>]"
        );
        eprintln!("  Supported models: silero_vad");
        process::exit(2);
    });

    let input_path = find_arg(&args, "--input")
        .map(PathBuf::from)
        .unwrap_or_else(default_proof_path);

    let output_path = find_arg(&args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let name = format!("{model}.proof.json");
            input_path.parent().unwrap_or(Path::new(".")).join(name)
        });

    let kernel_names: &[&str] = match model.as_str() {
        "silero_vad" => SILERO_VAD_KERNEL_NAMES,
        other => {
            eprintln!("Unknown model: {other}");
            eprintln!("Supported models: silero_vad");
            process::exit(2);
        }
    };

    let bundle = load_and_filter(&input_path, &model, kernel_names);
    validate_bundle(&bundle);
    print_summary(&bundle);
    save_bundle(&bundle, &output_path);
}

/// Load the full verification bundle and filter to model-specific certificates.
fn load_and_filter(input_path: &Path, model: &str, kernel_names: &[&str]) -> CertificateBundle {
    eprintln!("Loading {}", input_path.display());
    let full_bundle = match CertificateBundle::load(input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error loading bundle: {e}");
            process::exit(1);
        }
    };
    eprintln!("  {} certificates in full bundle", full_bundle.len());

    let model_name = format!("{model}_verified");
    let bundle = full_bundle.filter_by_names(&model_name, kernel_names);
    eprintln!(
        "  {} certificates after filtering for {model}",
        bundle.len()
    );
    bundle
}

/// Validate structural consistency and run the certificate checker.
fn validate_bundle(bundle: &CertificateBundle) {
    if let Err((idx, err)) = bundle.validate_all() {
        eprintln!(
            "Validation error in certificate {idx} ({}): {err}",
            bundle.certificates[idx].kernel_name
        );
        process::exit(1);
    }

    let results = check_bundle(bundle, None, None);
    let failed: Vec<_> = results.iter().filter(|r| !r.is_valid()).collect();
    if !failed.is_empty() {
        eprintln!("{} certificate(s) have issues:", failed.len());
        for r in &failed {
            for issue in &r.issues {
                eprintln!("  {}: {issue}", r.kernel_name);
            }
        }
        process::exit(1);
    }
}

/// Print bundle summary to stderr.
fn print_summary(bundle: &CertificateBundle) {
    eprintln!("\nBundle summary:");
    eprintln!("  Model: {}", bundle.model_name);
    eprintln!("  Certificates: {}", bundle.len());
    eprintln!("  Verified (finite): {}", bundle.verified_count());
    eprintln!("  Sound: {}", bundle.sound_count());
    eprintln!("  All have source_hash: {}", bundle.all_have_source_hash());
    eprintln!("  All sound: {}", bundle.all_sound());
    for cert in &bundle.certificates {
        eprintln!(
            "    {} — {:?} {:?}",
            cert.kernel_name, cert.soundness_mode, cert.method,
        );
    }
}

/// Save the bundle to disk.
fn save_bundle(bundle: &CertificateBundle, output_path: &Path) {
    match bundle.save(output_path) {
        Ok(()) => eprintln!("\nSaved to {}", output_path.display()),
        Err(e) => {
            eprintln!("Error saving bundle: {e}");
            process::exit(1);
        }
    }
}

/// Resolve the workspace-root `nn_verify.proof.json` path.
fn default_proof_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("nn_verify.proof.json")
}

/// Find the value of a flag like `--model <name>` in the arg list.
fn find_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
