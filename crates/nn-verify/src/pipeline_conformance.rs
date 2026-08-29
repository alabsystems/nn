// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Op soundness-coverage as a pipeline stage — the
//! [`ny_conformance`](crate::ny_conformance) adapter wired into
//! verify-and-record.
//!
//! # What is wired
//!
//! [`OpCoverageMode`] turns ny's conformance gate on for a pipeline run.
//! [`crate::pipeline::verify_and_record_full_with_coverage`] (scalar) and
//! [`crate::pipeline::tensor::verify_tensor_and_record_with_coverage`]
//! (tensor / model-level) classify every operation of the translated
//! `GraphNetwork`, attach the tally to the pipeline result, and — through
//! [`record_op_coverage`] — write a one-line summary onto the status entry.
//! That is what makes an `Unsupported`-class op visible in
//! `nn_verify_status_*.json` rather than only discoverable by reading a refusal.
//!
//! # What stays opt-in
//!
//! Everything. [`OpCoverageMode::Off`] is the default and is what every
//! historical entry point passes, so an unchanged caller sees byte-identical
//! behavior. [`OpCoverageMode::Gate`] is the fail-closed escalation: it refuses
//! before propagation and before any status write.
//!
//! The multi-variable scalar path
//! ([`crate::pipeline::verify_and_record_full_multi_with_config`]) is
//! deliberately not wired; callers wanting coverage there translate with
//! [`kernel_to_graph_multi`](crate::graph::kernel_to_graph_multi) and call
//! [`op_coverage_for_graph`] themselves.

use crate::error::VerifyError;
use crate::ny_conformance::{model_op_coverage, ModelCoverage};
use crate::status::VerifyStatus;

/// Opt-in control for the [`ny_conformance`](crate::ny_conformance) op
/// soundness-coverage stage.
///
/// The coverage stage classifies every operation of the translated
/// `GraphNetwork` through ny's canonical conformance gate. It is **off by
/// default** everywhere: the historical entry points
/// ([`verify_and_record_full`](crate::pipeline::verify_and_record_full),
/// [`verify_and_record_full_with_config`](crate::pipeline::verify_and_record_full_with_config),
/// [`verify_tensor_and_record`](crate::pipeline::verify_tensor_and_record),
/// [`verify_tensor_and_record_with_config`](crate::pipeline::verify_tensor_and_record_with_config))
/// pass [`Off`](Self::Off) and behave exactly as they did before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OpCoverageMode {
    /// Do not run the conformance gate (historical behavior).
    #[default]
    Off,
    /// Run the gate, attach the tally to the pipeline result, and record a
    /// one-line summary on the status entry. Never changes the verdict.
    Report,
    /// As [`Report`](Self::Report), but additionally **refuse** the run when any
    /// operation is `Unsupported`-class.
    ///
    /// The refusal happens *before* propagation and before anything is written
    /// to the status file, so a model with a silent soundness hole cannot leave
    /// a recorded verdict behind (fail-closed).
    Gate,
}

impl OpCoverageMode {
    /// Whether this mode runs the conformance gate at all.
    #[must_use]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Render a one-line summary of an op soundness-coverage tally.
///
/// The `Unsupported` bucket is always spelled out (with the offending layer
/// names) so a soundness hole is legible in a status file, a log line, or a
/// table cell without further lookups.
#[must_use]
pub fn op_coverage_summary(coverage: &ModelCoverage) -> String {
    let head = format!(
        "ny conformance: {} op(s) — exact {}, sound {}, loose {}, unsupported {}",
        coverage.total(),
        coverage.exact,
        coverage.sound,
        coverage.loose,
        coverage.unsupported,
    );
    if coverage.unsupported_layers.is_empty() {
        head
    } else {
        format!("{head} [{}]", coverage.unsupported_layers.join(", "))
    }
}

/// Run the conformance stage over a translated graph, honoring `mode`.
///
/// Returns `Ok(None)` when the stage is off. In [`OpCoverageMode::Gate`] an
/// `Unsupported`-class op is a hard refusal — reported as
/// [`VerifyError::UnsupportedOp`] so it fails closed rather than recording a
/// verdict whose soundness the gate could not vouch for.
///
/// Public so callers on paths the pipeline does not wire (the multi-variable
/// scalar path, ad-hoc graphs) can run the same stage on their own graph.
///
/// # Errors
///
/// [`VerifyError::UnsupportedOp`] in [`OpCoverageMode::Gate`] when any op is
/// `Unsupported`-class.
pub fn op_coverage_for_graph(
    graph: &ny_propagate::GraphNetwork,
    key: &str,
    mode: OpCoverageMode,
) -> Result<Option<ModelCoverage>, VerifyError> {
    if !mode.is_enabled() {
        return Ok(None);
    }
    let coverage = model_op_coverage(graph);
    if mode == OpCoverageMode::Gate && !coverage.is_fully_verifiable() {
        return Err(VerifyError::UnsupportedOp(format!(
            "'{key}' is not soundly verifiable: {} op(s) outside ny's sound bound-propagation \
             coverage [{}] ({})",
            coverage.unsupported,
            coverage.unsupported_layers.join(", "),
            op_coverage_summary(&coverage),
        )));
    }
    Ok(Some(coverage))
}

/// Attach an op-coverage summary to an already-recorded status entry.
///
/// Writes through the existing `soundness_justification` channel — the only
/// free-text field `KernelStatus` persists — so the coverage tally lands in
/// `nn_verify_status_*.json` next to the verdict it describes. An unrelated
/// justification already on the entry is preserved and the summary appended;
/// a previous coverage summary is replaced rather than accumulated.
///
/// # Errors
///
/// [`VerifyError::InvalidInput`] when no status entry exists for `key` — the
/// coverage summary must attach to a recorded verdict, never float free.
pub fn record_op_coverage(
    status: &mut VerifyStatus,
    key: &str,
    coverage: &ModelCoverage,
) -> Result<(), VerifyError> {
    let summary = op_coverage_summary(coverage);
    let combined = match status
        .kernel(key)
        .and_then(|entry| entry.soundness_justification.as_deref())
    {
        Some(existing) if !existing.is_empty() && !existing.contains("ny conformance:") => {
            format!("{existing} | {summary}")
        }
        _ => summary,
    };
    status.set_soundness_justification(key, &combined)
}
