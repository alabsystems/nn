// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CLI tool for validating Kokoro deployment certificates and constructive
//! proof certificates.
//!
//! # Usage
//!
//! ```bash
//! # Validate a Kokoro certificate:
//! cargo run -p nn-verify --bin nn_verify_cert -- verify kokoro.proof.json
//!
//! # Validate with model hash check:
//! cargo run -p nn-verify --bin nn_verify_cert -- verify kokoro.proof.json \
//!   --model-hash abc123...
//!
//! # Show certificate summary:
//! cargo run -p nn-verify --bin nn_verify_cert -- summary kokoro.proof.json
//!
//! # Generate a certificate from the status file:
//! cargo run -p nn-verify --bin nn_verify_cert -- generate \
//!   --status nn_verify_status_kokoro.json \
//!   --model-hash abc123... \
//!   --output kokoro.proof.json
//!
//! # Compare tightening reports:
//! cargo run -p nn-verify --bin tighten -- kokoro report-diff \
//!   --baseline previous_report.json \
//!   --candidate current_report.json
//!
//! # Validate a constructive proof certificate:
//! cargo run -p nn-verify --bin nn_verify_cert -- check-proof model.constructive.json
//!
//! # Show constructive proof summary:
//! cargo run -p nn-verify --bin nn_verify_cert -- proof-summary model.constructive.json
//! ```
//!
//! Part of #4254, #4315.

use std::path::PathBuf;
use std::process;

use nn_verify::kokoro_certificate::{
    generate_kokoro_certificate, verify_certificate, CertificateConfig, KokoroCertificate,
};
use nn_verify::ConstructiveProofData;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    match args[1].as_str() {
        "verify" => cmd_verify(&args[2..]),
        "summary" => cmd_summary(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "check-proof" => cmd_check_proof(&args[2..]),
        "proof-summary" => cmd_proof_summary(&args[2..]),
        "--help" | "-h" | "help" => {
            print_usage(&args[0]);
        }
        other => {
            eprintln!("Unknown command: {other}");
            print_usage(&args[0]);
            process::exit(1);
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("nn verify certificate - certificate validator");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  {program} verify <cert.json> [--model-hash <hash>]");
    eprintln!("      Validate a Kokoro certificate file. Checks structural integrity,");
    eprintln!("      content hash, junction bounds, and proof strength.");
    eprintln!("      If --model-hash is provided, checks that the certificate");
    eprintln!("      was generated for that specific model weights file.");
    eprintln!();
    eprintln!("  {program} summary <cert.json>");
    eprintln!("      Print a one-line summary of a Kokoro certificate.");
    eprintln!();
    eprintln!("  {program} generate --status <status.json> --model-hash <hash>");
    eprintln!("                     [--output <cert.json>] [--include-stale]");
    eprintln!("      Generate a new Kokoro certificate from a verification status file.");
    eprintln!();
    eprintln!("  Tightening report diffs live in the standalone `tighten` binary.");
    eprintln!("      Use `tighten --help` for the report-diff surface.");
    eprintln!();
    eprintln!("  {program} check-proof <proof.json>");
    eprintln!("      Validate a constructive proof certificate. Checks structural");
    eprintln!("      consistency, bound finiteness, bound chain containment,");
    eprintln!("      and replay verification.");
    eprintln!();
    eprintln!("  {program} proof-summary <proof.json>");
    eprintln!("      Print a summary of a constructive proof certificate.");
    eprintln!();
    eprintln!("  {program} help");
    eprintln!("      Show this help message.");
    eprintln!();
    eprintln!("Part of nn verified ML framework. See #4254, #4315.");
}

fn cmd_verify(args: &[String]) {
    if args.is_empty() {
        eprintln!("verify: missing certificate path");
        process::exit(1);
    }

    let cert_path = PathBuf::from(&args[0]);
    let model_hash = parse_flag(args, "--model-hash");

    let cert = match KokoroCertificate::load(&cert_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load certificate: {e}");
            process::exit(1);
        }
    };

    // Use provided hash or the certificate's own hash for structural checks.
    let check_hash = model_hash.as_deref().unwrap_or(&cert.model_hash);
    let verdict = verify_certificate(&cert, check_hash);

    println!("{verdict}");

    // Print detailed entry info.
    println!(
        "Entries: {} active / {} total",
        cert.summary.active_entries, cert.summary.total_entries
    );
    println!("Sound: {}", cert.summary.sound_count);
    println!("Heuristic: {}", cert.summary.heuristic_count);

    if !cert.summary.proof_strength_breakdown.is_empty() {
        println!("\nProof strength breakdown:");
        for (strength, count) in &cert.summary.proof_strength_breakdown {
            println!("  {strength}: {count}");
        }
    }

    if !cert.summary.method_breakdown.is_empty() {
        println!("\nMethod breakdown:");
        for (method, count) in &cert.summary.method_breakdown {
            println!("  {method}: {count}");
        }
    }

    println!("\nJunction bounds: {}", cert.junction_bounds.len());
    for jb in &cert.junction_bounds {
        println!(
            "  {}: {} [{:.1}, {:.1}]",
            jb.name, jb.zone, jb.lower, jb.upper
        );
    }

    if !verdict.is_valid() {
        process::exit(1);
    }
}

fn cmd_summary(args: &[String]) {
    if args.is_empty() {
        eprintln!("summary: missing certificate path");
        process::exit(1);
    }

    let cert_path = PathBuf::from(&args[0]);
    let cert = match KokoroCertificate::load(&cert_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load certificate: {e}");
            process::exit(1);
        }
    };

    let valid = if cert.content_hash.is_some() {
        let verdict = verify_certificate(&cert, &cert.model_hash);
        if verdict.is_valid() {
            "VALID"
        } else {
            "INVALID"
        }
    } else {
        "UNSIGNED"
    };

    let vacuous = cert
        .summary
        .proof_strength_breakdown
        .get("vacuous")
        .copied()
        .unwrap_or(0);

    println!(
        "{valid} v{} sound={}/{} heuristic={} vacuous={} junctions={} rev={} generated={}",
        cert.schema_version,
        cert.summary.sound_count,
        cert.summary.active_entries,
        cert.summary.heuristic_count,
        vacuous,
        cert.junction_bounds.len(),
        truncate_rev(&cert.gamma_crown_rev),
        cert.generated_at,
    );
}

fn cmd_generate(args: &[String]) {
    let status_path = parse_flag(args, "--status");
    let model_hash = parse_flag(args, "--model-hash");
    let output_path = parse_flag(args, "--output");
    let include_stale = args.iter().any(|a| a == "--include-stale");

    let status_path = match status_path {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("generate: --status <path> is required");
            process::exit(1);
        }
    };

    let model_hash = match model_hash {
        Some(h) => h,
        None => {
            eprintln!("generate: --model-hash <hash> is required");
            process::exit(1);
        }
    };

    let config = CertificateConfig::new(&model_hash, &status_path).with_stale(include_stale);

    let cert = match generate_kokoro_certificate(&config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to generate certificate: {e}");
            process::exit(1);
        }
    };

    if let Some(out) = output_path {
        let out_path = PathBuf::from(out);
        if let Err(e) = cert.save(&out_path) {
            eprintln!("Failed to save certificate: {e}");
            process::exit(1);
        }
        println!("Certificate saved to {}", out_path.display());

        // Print summary.
        println!(
            "  sound={}/{} heuristic={} junctions={}",
            cert.summary.sound_count,
            cert.summary.active_entries,
            cert.summary.heuristic_count,
            cert.junction_bounds.len(),
        );
    } else {
        // Print to stdout.
        match cert.to_json() {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("Failed to serialize certificate: {e}");
                process::exit(1);
            }
        }
    }
}

/// Validate a constructive proof certificate file.
///
/// Loads the proof, runs structural validation, and performs replay
/// verification (bound chain containment check). Reports pass/fail
/// with detailed diagnostics.
fn cmd_check_proof(args: &[String]) {
    if args.is_empty() {
        eprintln!("check-proof: missing proof certificate path");
        process::exit(1);
    }

    let proof_path = PathBuf::from(&args[0]);
    let proof = match ConstructiveProofData::load(&proof_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load constructive proof: {e}");
            process::exit(1);
        }
    };

    // Structural validation.
    print!("Structural validation: ");
    match proof.validate() {
        Ok(()) => println!("PASS"),
        Err(e) => {
            println!("FAIL - {e}");
            process::exit(1);
        }
    }

    // Replay verification (bound chain containment).
    print!("Replay verification:   ");
    if proof.replay_verify() {
        println!("PASS");
    } else {
        println!("FAIL");
        process::exit(1);
    }

    // Machine-checkable assessment.
    print!("Machine-checkable:     ");
    if proof.is_machine_checkable() {
        println!("YES");
    } else {
        println!("NO (missing verified bounds or Lean4 export)");
    }

    // Method tightness.
    print!("Method tightness:      ");
    if proof.method.is_tight() {
        println!("TIGHT ({:?})", proof.method);
    } else {
        println!("LOOSE ({:?}) - may be vacuously wide", proof.method);
    }

    println!("\nVERIFIED - constructive proof certificate is valid");
}

/// Print a summary of a constructive proof certificate.
fn cmd_proof_summary(args: &[String]) {
    if args.is_empty() {
        eprintln!("proof-summary: missing proof certificate path");
        process::exit(1);
    }

    let proof_path = PathBuf::from(&args[0]);
    let proof = match ConstructiveProofData::load(&proof_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load constructive proof: {e}");
            process::exit(1);
        }
    };

    let valid = proof.validate().is_ok();
    let status = if valid { "VALID" } else { "INVALID" };
    let tight = if proof.method.is_tight() {
        "tight"
    } else {
        "loose"
    };
    let composition = if proof.method.is_composition() {
        "composition"
    } else {
        "single-layer"
    };
    let checkable = if proof.is_machine_checkable() {
        "machine-checkable"
    } else {
        "not-machine-checkable"
    };

    println!(
        "{status} method={:?} ({tight}, {composition}) layers={} \
         inputs={} outputs={} verified={} {checkable} \
         lean4={} composition_lean4={} generated={}",
        proof.method,
        proof.num_layers,
        proof.input_lower.len(),
        proof.output_lower.len(),
        proof.verified,
        proof.lean4_export.is_some(),
        proof.has_composition_proof(),
        proof.generated_at,
    );

    if let Some(ref layers) = proof.layer_proofs {
        println!("Layer proofs: {}", layers.len());
        for layer in layers {
            println!(
                "  [{:>2}] {:<20} in={} out={}",
                layer.layer_index,
                layer.layer_type,
                layer.input_lower.len(),
                layer.output_lower.len(),
            );
        }
    }
}

/// Parse a `--flag value` pair from args.
fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Truncate a git rev to first 12 chars for display.
fn truncate_rev(rev: &str) -> &str {
    if rev.len() > 12 {
        &rev[..12]
    } else {
        rev
    }
}
