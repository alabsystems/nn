// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generate a moonshot TTS verification certificate from workspace artifacts.
//!
//! Combines Kani harness results (from `kani_status.json` or workspace scan)
//! with optional CROWN/SMT/dispatch evidence into a single
//! [`MoonshotCertificate`].
//!
//! ```text
//! cargo run -p nn-tts-verify --example generate_moonshot_certificate -- \
//!   --model kokoro \
//!   --input-spec "English text up to 100 tokens" \
//!   --output kokoro_moonshot.proof.json
//! ```
//!
//! The certificate captures all 8 moonshot properties (P1-P8) with evidence
//! from whichever verification sources are available in the workspace.

use std::path::{Path, PathBuf};
use std::process;

use nn_tts_verify::moonshot::{
    build_certificate_from_workspace, MoonshotCertificate, VerificationLevel,
};

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

    let workspace_root = find_arg(&args, "--workspace")
        .map(PathBuf::from)
        .unwrap_or_else(discover_workspace_root);

    let kani_status_path = workspace_root.join("kani_status.json");
    let crates_dir = workspace_root.join("crates");

    let assume_pass = args.iter().any(|a| a == "--assume-pass");

    eprintln!("Generating moonshot certificate for model: {model}");
    eprintln!("  workspace:    {}", workspace_root.display());
    eprintln!("  kani_status:  {}", kani_status_path.display());
    eprintln!("  crates_dir:   {}", crates_dir.display());
    eprintln!("  source_hash:  {source_hash}");
    eprintln!("  assume_pass:  {assume_pass}");

    let cert = build_certificate_from_workspace(
        &model,
        &input_spec,
        &source_hash,
        &kani_status_path,
        &crates_dir,
        assume_pass,
        None, // CROWN bundle — pass via library API when available
        None, // SMT evidence — pass via library API when available
        None, // Dispatch evidence — pass via library API when available
    );

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
         \x20 --output <path>        Output JSON path (default: <model>_moonshot.proof.json)\n\
         \x20 --workspace <path>     Workspace root (auto-discovered if omitted)\n\
         \x20 --assume-pass          Assume all Kani harnesses pass in fallback scan"
    );
}

fn find_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Discover workspace root by walking up from the current directory.
fn discover_workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("crates").is_dir() {
            return dir;
        }
        if !dir.pop() {
            eprintln!("Could not discover workspace root. Use --workspace.");
            process::exit(1);
        }
    }
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
                VerificationLevel::CrownProven
                    | VerificationLevel::KaniProven
                    | VerificationLevel::SmtProven
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
            VerificationLevel::KaniProven => "KANI   ",
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
