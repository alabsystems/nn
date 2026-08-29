// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Op-coverage opt-in plumbing and table rendering for `verify_all`.
//!
//! Holds the `NN_VERIFY_OP_COVERAGE` switch and the small amount of formatting
//! the `ny_conformance` wiring adds to the kernel and model tier tables. Kept
//! separate so `runner.rs` stays about the verification loops.
//!
//! The rendering rule that matters: when the switch is off, every row and header
//! is byte-identical to the pre-adapter output. The `Coverage` column only
//! exists in a run that asked for it.

use nn_verify::ny_conformance::ModelCoverage;
use nn_verify::pipeline::OpCoverageMode;

/// Resolve the op-coverage opt-in from the environment.
///
/// `NN_VERIFY_OP_COVERAGE`:
/// * unset / `off` — the conformance gate does not run (default; historical
///   behavior).
/// * `report` — classify every op, attach the tally, and record the summary
///   into the status entry.
/// * `gate` — as `report`, plus refuse (fail closed) any kernel or model
///   containing an `Unsupported`-class op, before anything is recorded.
pub(crate) fn op_coverage_mode_from_env() -> OpCoverageMode {
    match std::env::var("NN_VERIFY_OP_COVERAGE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "report" | "1" | "true" => OpCoverageMode::Report,
        "gate" => OpCoverageMode::Gate,
        _ => OpCoverageMode::Off,
    }
}

/// Render an op-coverage tally as a table cell, or `None` when the run did not
/// ask for coverage.
///
/// `None` keeps the row byte-identical to the pre-adapter output, so the
/// default (opt-in off) run of `verify_all` prints exactly what it always did.
pub(crate) fn coverage_cell(coverage: Option<&ModelCoverage>) -> Option<String> {
    coverage.map(|c| match c.unsupported {
        0 => format!("{}ok", c.total()),
        n => format!("{n}UNSUP"),
    })
}

/// Print the header for a tier table, widening it only when the coverage column
/// is actually going to be filled.
pub(crate) fn print_tier_header(columns: &[(&str, usize)], coverage: OpCoverageMode, rule: usize) {
    let mut header = String::new();
    for (idx, (label, width)) in columns.iter().enumerate() {
        if idx == 0 {
            header.push_str(&format!("{label:<0$}", *width));
        } else {
            header.push_str(&format!(" {label:>0$}", *width));
        }
    }
    if coverage.is_enabled() {
        header.push_str(&format!(" {:>9}", "Coverage"));
    }
    println!("{header}");
    println!(
        "{}",
        "-".repeat(if coverage.is_enabled() {
            rule + 10
        } else {
            rule
        })
    );
}

/// Report the named soundness holes on stderr — a coverage cell is never the
/// only place a hole appears.
pub(crate) fn report_unsupported(name: &str, coverage: Option<&ModelCoverage>) {
    if let Some(cov) = coverage {
        if !cov.unsupported_layers.is_empty() {
            eprintln!(
                "  {name}: ny conformance — {} op(s) outside sound coverage: {}",
                cov.unsupported,
                cov.unsupported_layers.join(", ")
            );
        }
    }
}
