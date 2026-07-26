// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CLI tool for optimization report management.
//!
//! Loads, compares, and analyzes optimization reports produced by the
//! progressive tightening loop.
//!
//! # Usage
//!
//! ```bash
//! # View a report:
//! cargo run -p nn-metal --bin nn_report -- view reports/kokoro_iter_0.json
//!
//! # Compare two reports:
//! cargo run -p nn-metal --bin nn_report -- diff \
//!   reports/kokoro_iter_0.json reports/kokoro_iter_1.json
//!
//! # Validate against a behavioral contract:
//! cargo run -p nn-metal --bin nn_report -- validate \
//!   reports/kokoro_iter_1.json --contract reports/kokoro_contract.json
//! ```
//!
//! Part of #2456, #2218.

use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    match args[1].as_str() {
        "view" => cmd_view(&args[2..]),
        "diff" => cmd_diff(&args[2..]),
        "summary" => cmd_summary(&args[2..]),
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
    eprintln!("Usage:");
    eprintln!("  {program} view <report.json>          Print report summary");
    eprintln!("  {program} diff <prev.json> <curr.json> Compare two reports");
    eprintln!("  {program} summary <report.json>        Print one-line summary");
    eprintln!("  {program} help                         Show this help");
}

fn cmd_view(args: &[String]) {
    if args.is_empty() {
        eprintln!("view: missing report path");
        process::exit(1);
    }
    let path = PathBuf::from(&args[0]);
    let report = match nn_metal::OptimizationReport::load(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load report: {e}");
            process::exit(1);
        }
    };

    println!("Model:      {}", report.model_name);
    println!("Iteration:  {}", report.iteration);
    println!("Version:    {}", report.version);
    println!("Generated:  {}", report.generated_at);

    print_performance_summary(&report.performance);
    print_recommendations_and_contract(&report);
}

fn print_performance_summary(perf: &serde_json::Value) {
    if let Some(dispatches) = perf
        .get("total_dispatches")
        .and_then(serde_json::Value::as_u64)
    {
        println!("Dispatches: {dispatches}");
    }
    if let Some(metal) = perf
        .get("total_metal_dispatches")
        .and_then(serde_json::Value::as_u64)
    {
        println!("Metal:      {metal}");
    }
    if let Some(mem) = perf
        .get("memory")
        .and_then(|m| m.get("total_buffer_bytes"))
        .and_then(serde_json::Value::as_u64)
    {
        println!("Memory:     {mem} bytes");
    }
    if let Some(segs) = perf.get("segments").and_then(serde_json::Value::as_array) {
        println!("\nSegments:");
        for seg in segs {
            let name = seg
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("?");
            let d = seg
                .get("dispatches")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let lat = seg
                .get("latency_us")
                .and_then(serde_json::Value::as_f64)
                .map(|l| format!("{l:.1} us"))
                .unwrap_or_else(|| "-".into());
            println!("  {name:<25} dispatches={d:<5} latency={lat}");
        }
    }
}

fn print_recommendations_and_contract(report: &nn_metal::OptimizationReport) {
    if !report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for rec in &report.recommendations {
            println!("  - {rec}");
        }
    }
    if let Some(status) = &report.contract_status {
        println!(
            "\nContract: {}",
            if status.all_bounds_satisfied {
                "PASS"
            } else {
                "FAIL"
            }
        );
        for v in &status.violations {
            println!("  VIOLATION: {v}");
        }
        for t in &status.tightened_bounds {
            println!("  TIGHTENED: {t}");
        }
    }
}

fn cmd_diff(args: &[String]) {
    if args.len() < 2 {
        eprintln!("diff: requires two report paths");
        process::exit(1);
    }
    let prev_path = PathBuf::from(&args[0]);
    let curr_path = PathBuf::from(&args[1]);

    let prev = match nn_metal::OptimizationReport::load(&prev_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load prev report: {e}");
            process::exit(1);
        }
    };
    let curr = match nn_metal::OptimizationReport::load(&curr_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load curr report: {e}");
            process::exit(1);
        }
    };

    let delta = nn_metal::diff_reports(&prev, &curr);

    println!(
        "Iteration {} -> {}:",
        delta.prev_iteration, delta.curr_iteration
    );
    println!("  Dispatch delta:       {:+}", delta.dispatch_delta);
    println!("  Metal dispatch delta: {:+}", delta.metal_dispatch_delta);
    println!("  Memory delta:         {:+} bytes", delta.memory_delta);

    if !delta.new_violations.is_empty() {
        println!("  New violations:");
        for v in &delta.new_violations {
            println!("    - {v}");
        }
    }
    if !delta.resolved_violations.is_empty() {
        println!("  Resolved violations:");
        for v in &delta.resolved_violations {
            println!("    + {v}");
        }
    }

    match &delta.verdict {
        nn_metal::IterationVerdict::Improved { summary } => {
            println!("\nVerdict: IMPROVED — {summary}");
        }
        nn_metal::IterationVerdict::Regressed { summary } => {
            println!("\nVerdict: REGRESSED — {summary}");
        }
        nn_metal::IterationVerdict::Stalled { consecutive_stalls } => {
            println!("\nVerdict: STALLED ({consecutive_stalls} consecutive)");
        }
        nn_metal::IterationVerdict::Mixed {
            improved,
            regressed,
        } => {
            println!("\nVerdict: MIXED");
            for i in improved {
                println!("  + {i}");
            }
            for r in regressed {
                println!("  - {r}");
            }
        }
        _ => {
            println!("\nVerdict: UNKNOWN");
        }
    }

    // Also emit machine-readable JSON.
    if let Ok(json) = serde_json::to_string_pretty(&delta) {
        println!("\n--- DELTA JSON ---\n{json}");
    }
}

fn cmd_summary(args: &[String]) {
    if args.is_empty() {
        eprintln!("summary: missing report path");
        process::exit(1);
    }
    let path = PathBuf::from(&args[0]);
    let report = match nn_metal::OptimizationReport::load(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load report: {e}");
            process::exit(1);
        }
    };

    let dispatches = report
        .performance
        .get("total_dispatches")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let metal = report
        .performance
        .get("total_metal_dispatches")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let contract = report
        .contract_status
        .as_ref()
        .map(|s| {
            if s.all_bounds_satisfied {
                "PASS"
            } else {
                "FAIL"
            }
        })
        .unwrap_or("N/A");

    println!(
        "{} iter={} dispatches={} metal={} contract={}",
        report.model_name, report.iteration, dispatches, metal, contract
    );
}
