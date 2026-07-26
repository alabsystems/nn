// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `tighten` CLI for progressive tightening report diffs.
//!
//! Current surface:
//! - `kokoro report-diff` compares two bound-analysis reports and prints the
//!   tightening delta.
//! - `kokoro run` is intentionally unavailable in `nn-verify` for now and
//!   exits with a clear error instead of pretending a live model runner exists.
//!
//! Part of #2456.

use std::process;

use nn_verify::tighten::{
    compare_report_path_sequence, compare_report_paths, format_tightening_diff,
    format_tightening_sequence, kokoro_runner_unavailable,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    match args[1].as_str() {
        "kokoro" => cmd_kokoro(&args[2..], &args[0]),
        "--help" | "-h" | "help" => print_usage(&args[0]),
        other => {
            eprintln!("Unknown command: {other}");
            print_usage(&args[0]);
            process::exit(1);
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("tighten - progressive tightening report diff");
    eprintln!();
    eprintln!("Usage:");
    eprintln!(
        "  {program} kokoro report-diff --baseline <report.json> --candidate <report.json> [--candidate <report.json> ...] [--json]"
    );
    eprintln!("      Compare a baseline against one or more Kokoro bound-analysis reports.");
    eprintln!("      Repeated candidates are treated as successive tightening passes.");
    eprintln!();
    eprintln!("  {program} kokoro run");
    eprintln!("      Live Kokoro tightening is not wired in nn-verify yet.");
    eprintln!("      Use report-diff on precomputed report JSON instead.");
    eprintln!();
    eprintln!("  {program} help");
    eprintln!("      Show this help message.");
}

fn cmd_kokoro(args: &[String], program: &str) {
    if args.is_empty() {
        print_usage(program);
        process::exit(1);
    }

    match args[0].as_str() {
        "report-diff" => cmd_kokoro_report_diff(&args[1..], program),
        "run" => cmd_kokoro_run(program),
        "--help" | "-h" | "help" => print_usage(program),
        other => {
            eprintln!("Unknown Kokoro command: {other}");
            print_usage(program);
            process::exit(1);
        }
    }
}

fn cmd_kokoro_report_diff(args: &[String], program: &str) {
    let mut baseline: Option<String> = None;
    let mut candidates = Vec::new();
    let mut json = false;
    let mut positional = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => {
                baseline = take_value(args, &mut i, "--baseline");
            }
            "--candidate" => {
                candidates.push(
                    take_value(args, &mut i, "--candidate").expect("missing candidate value"),
                );
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage(program);
                return;
            }
            other if other.starts_with("--") => {
                eprintln!("Unknown flag for report-diff: {other}");
                print_usage(program);
                process::exit(1);
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    if baseline.is_none() && !positional.is_empty() {
        baseline = Some(positional.remove(0));
    }
    candidates.extend(positional);

    if baseline.is_none() || candidates.is_empty() {
        eprintln!("report-diff: expected a baseline and at least one candidate report path");
        print_usage(program);
        process::exit(1);
    }

    if candidates.len() == 1 {
        let diff = match compare_report_paths(
            baseline.as_deref().expect("baseline checked above"),
            candidates[0].as_str(),
        ) {
            Ok(diff) => diff,
            Err(e) => {
                eprintln!("Failed to compare reports: {e}");
                process::exit(1);
            }
        };

        if json {
            match serde_json::to_string_pretty(&diff) {
                Ok(json) => println!("{json}"),
                Err(e) => {
                    eprintln!("Failed to serialize diff: {e}");
                    process::exit(1);
                }
            }
            return;
        }

        print!("{}", format_tightening_diff(&diff));
        return;
    }

    let mut report_paths = Vec::with_capacity(candidates.len() + 1);
    report_paths.push(baseline.expect("baseline checked above"));
    report_paths.extend(candidates);

    let sequence = match compare_report_path_sequence(&report_paths) {
        Ok(sequence) => sequence,
        Err(e) => {
            eprintln!("Failed to compare reports: {e}");
            process::exit(1);
        }
    };

    if json {
        match serde_json::to_string_pretty(&sequence) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("Failed to serialize diff: {e}");
                process::exit(1);
            }
        }
        return;
    }

    print!("{}", format_tightening_sequence(&sequence));
}

fn cmd_kokoro_run(program: &str) {
    let err = kokoro_runner_unavailable();
    eprintln!("{err}");
    eprintln!();
    eprintln!("Use:");
    eprintln!("  {program} kokoro report-diff --baseline <report.json> --candidate <report.json>");
    process::exit(1);
}

fn take_value(args: &[String], idx: &mut usize, flag: &str) -> Option<String> {
    *idx += 1;
    if *idx >= args.len() {
        eprintln!("{flag}: missing value");
        process::exit(1);
    }
    let value = args[*idx].clone();
    if value.starts_with("--") {
        eprintln!("{flag}: missing value");
        process::exit(1);
    }
    *idx += 1;
    Some(value)
}
