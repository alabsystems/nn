// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generate a moonshot TTS verification certificate from the workspace
//! artifact registry as a single [`MoonshotCertificate`]. CROWN/SMT/dispatch
//! evidence is attached through the library API (`FullCertificateBuilder`).
//!
//! ```text
//! targo --unverified run -p nn-tts-verify --example generate_moonshot_certificate -- \
//!   --model kokoro \
//!   --input-spec "English text up to 100 tokens" \
//!   --output kokoro_moonshot.proof.json
//! ```
//!
//! The certificate captures all 8 moonshot properties (P1-P8) with evidence
//! from whichever verification sources are available in the workspace.

use std::path::{Path, PathBuf};
use std::process;

use nn_tts_verify::moonshot::{FullCertificateBuilder, MoonshotCertificate, VerificationLevel};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model = find_arg(&args, "--model").unwrap_or_else(|| {
        print_usage();
        process::exit(2);
    });

    let input_spec =
        find_arg(&args, "--input-spec").unwrap_or_else(|| "unspecified input".to_string());

    let source_hash = find_arg(&args, "--source-hash").unwrap_or_else(compute_source_hash);

    let output_path = find_arg(&args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{model}_moonshot.proof.json")));

    eprintln!("Generating moonshot certificate for model: {model}");
    eprintln!("  source_hash:  {source_hash}");

    let cert = FullCertificateBuilder::new(&model, &input_spec, &source_hash).build();

    print_summary(&cert);
    save_certificate(&cert, &output_path);
}

fn print_usage() {
    eprintln!(
        "Usage: generate_moonshot_certificate --model <name> [OPTIONS]\n\
         \n\
         Required:\n\
         \x20 --model <name>         Model name (e.g., kokoro, silero_vad, htdemucs)\n\
         \n\
         Optional:\n\
         \x20 --input-spec <text>    Input specification for the certificate\n\
         \x20 --source-hash <hex>    Source hash (auto-computed from git if omitted)\n\
         \x20 --output <path>        Output JSON path (default: <model>_moonshot.proof.json)"
    );
}

fn find_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Compute a source hash from git HEAD.
fn compute_source_hash() -> String {
    process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn print_summary(cert: &MoonshotCertificate) {
    eprintln!("\n--- Moonshot Certificate Summary ---");
    eprintln!("Model:        {}", cert.model_name);
    eprintln!("Source hash:  {}", cert.source_hash);
    eprintln!("Input spec:   {}", cert.input_specification);
    eprintln!("All proven:   {}", cert.all_proven);
    eprintln!("All partial+: {}", cert.all_at_least_partial);

    let proven_count = cert
        .properties
        .iter()
        .filter(|p| {
            matches!(
                p.level,
                VerificationLevel::CrownProven | VerificationLevel::SmtProven
            )
        })
        .count();
    let partial_count = cert
        .properties
        .iter()
        .filter(|p| p.level == VerificationLevel::CrownPartial)
        .count();
    let total = cert.properties.len();

    eprintln!("Properties:   {proven_count}/{total} proven, {partial_count}/{total} partial");

    for prop in &cert.properties {
        let level_str = match prop.level {
            VerificationLevel::CrownProven => "CROWN  ",
            VerificationLevel::CrownProbabilistic => "PROB   ",
            VerificationLevel::SmtProven => "SMT    ",
            VerificationLevel::CrownPartial => "PARTIAL",
            VerificationLevel::Empirical => "EMPIR  ",
            VerificationLevel::None => "NONE   ",
        };
        eprintln!(
            "  P{}: [{}] {}",
            prop.property_index + 1,
            level_str,
            prop.property_name,
        );
    }
}

fn save_certificate(cert: &MoonshotCertificate, path: &Path) {
    let json = serde_json::to_string_pretty(cert).unwrap_or_else(|e| {
        eprintln!("Failed to serialize certificate: {e}");
        process::exit(1);
    });

    std::fs::write(path, &json).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", path.display());
        process::exit(1);
    });

    eprintln!("\nCertificate written to: {}", path.display());
    eprintln!("({} bytes)", json.len());
}
