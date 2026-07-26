// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `ConvertReport` — detailed optimization and verification report from
//! `convert()`.
//!
//! Produced by [`ConvertBuilder::build()`](super::ConvertBuilder) to give callers
//! visibility into what happened during import, compilation, and verification,
//! including which exported-artifact intake path was used and whether the
//! report describes a backend-agnostic converted graph or a compiled Metal
//! artifact.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Provenance for the exported-artifact intake consumed by the convert pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertIntakePath {
    /// Pre-exported `torch.export` JSON + `safetensors` artifacts supplied directly.
    #[default]
    ExportedArtifacts,
    /// Artifacts produced by `nn convert --from-pytorch` via `nn_export.py`.
    CliExportedPytorch,
}

impl ConvertIntakePath {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ExportedArtifacts => "exported artifacts",
            Self::CliExportedPytorch => "CLI-exported PyTorch",
        }
    }
}

impl fmt::Display for ConvertIntakePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What kind of artifact the pipeline produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertArtifactKind {
    /// Backend-agnostic converted graph representation.
    #[default]
    BackendAgnosticConvertedGraph,
    /// Compiled Metal model artifact ready for GPU execution.
    CompiledMetalArtifact,
}

impl ConvertArtifactKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BackendAgnosticConvertedGraph => "backend-agnostic converted graph",
            Self::CompiledMetalArtifact => "compiled Metal artifact",
        }
    }
}

impl fmt::Display for ConvertArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Propagation method recorded for the current composition-bounds entry.
///
/// This mirrors the public verifier concepts used by `nn-verify` while
/// keeping `ConvertReport` available even when the `verify` feature is off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[non_exhaustive]
pub enum ConvertCompositionMethod {
    /// Interval Bound Propagation.
    Ibp,
    /// Base CROWN linear relaxation.
    Crown,
    /// Alpha-CROWN linear relaxation.
    AlphaCrown,
    /// Beta-CROWN branch-and-bound refinement.
    BetaCrown,
    /// Closed-form analytical bounds.
    Analytical,
    /// Mixed IBP/CROWN propagation.
    #[serde(rename = "mixed_IBP_CROWN")]
    MixedIbpCrown,
}

impl ConvertCompositionMethod {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ibp => "IBP",
            Self::Crown => "CROWN",
            Self::AlphaCrown => "AlphaCrown",
            Self::BetaCrown => "BetaCrown",
            Self::Analytical => "Analytical",
            Self::MixedIbpCrown => "mixed_IBP_CROWN",
        }
    }

    #[cfg(feature = "verify")]
    #[must_use]
    pub(crate) const fn from_verify_method(method: nn_verify::PropMethod) -> Option<Self> {
        match method {
            nn_verify::PropMethod::Ibp => Some(Self::Ibp),
            nn_verify::PropMethod::Crown => Some(Self::Crown),
            nn_verify::PropMethod::AlphaCrown => Some(Self::AlphaCrown),
            nn_verify::PropMethod::BetaCrown => Some(Self::BetaCrown),
            nn_verify::PropMethod::Analytical => Some(Self::Analytical),
            nn_verify::PropMethod::MixedIbpCrown => Some(Self::MixedIbpCrown),
            _ => None,
        }
    }
}

impl fmt::Display for ConvertCompositionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Soundness mode recorded for the current composition-bounds entry.
///
/// This is only populated for the current `check_composition_bounds()` path
/// when verification runs and the translated graph can be classified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertSoundnessMode {
    /// No known heuristic/unsound switches were used.
    Sound,
    /// At least one heuristic/approximation weakens proof semantics.
    Heuristic,
}

impl ConvertSoundnessMode {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sound => "sound",
            Self::Heuristic => "heuristic",
        }
    }

    #[cfg(feature = "verify")]
    #[must_use]
    pub(crate) const fn from_verify_soundness_mode(
        mode: nn_verify::VerificationSoundnessMode,
    ) -> Option<Self> {
        match mode {
            nn_verify::VerificationSoundnessMode::Sound => Some(Self::Sound),
            nn_verify::VerificationSoundnessMode::Heuristic => Some(Self::Heuristic),
        }
    }
}

impl fmt::Display for ConvertSoundnessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Proof-strength classification recorded for the current composition-bounds entry.
///
/// This mirrors the verifier's public proof-strength classes, but it only
/// describes the current composition-bounds run that produced the report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertProofStrength {
    /// Sound CROWN-family proof.
    SoundCrown,
    /// Sound IBP-only proof.
    SoundIbp,
    /// Heuristic proof with non-vacuous width.
    Heuristic,
    /// Bounds are too wide to be practically useful.
    Vacuous,
    /// Sound mixed IBP/CROWN proof.
    SoundMixed,
}

impl ConvertProofStrength {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SoundCrown => "sound_crown",
            Self::SoundIbp => "sound_ibp",
            Self::Heuristic => "heuristic",
            Self::Vacuous => "vacuous",
            Self::SoundMixed => "sound_mixed",
        }
    }

    #[cfg(feature = "verify")]
    #[must_use]
    pub(crate) const fn from_verify_proof_strength(
        strength: nn_verify::status::ProofStrength,
    ) -> Option<Self> {
        match strength {
            nn_verify::status::ProofStrength::SoundCrown => Some(Self::SoundCrown),
            nn_verify::status::ProofStrength::SoundIbp => Some(Self::SoundIbp),
            nn_verify::status::ProofStrength::Heuristic => Some(Self::Heuristic),
            nn_verify::status::ProofStrength::Vacuous => Some(Self::Vacuous),
            nn_verify::status::ProofStrength::SoundMixed => Some(Self::SoundMixed),
            _ => None,
        }
    }
}

impl fmt::Display for ConvertProofStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Detailed report from the `convert()` pipeline.
///
/// Captures metrics from each phase (import, compile, verify) so callers can
/// inspect optimization effectiveness and verification coverage without
/// re-running the pipeline.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct ConvertReport {
    /// Provenance for how the exported-artifact intake was obtained.
    pub intake_path: ConvertIntakePath,
    /// Whether the result is still backend-agnostic or compiled for Metal.
    pub artifact_kind: ConvertArtifactKind,
    /// Total graph nodes imported from the torch.export JSON.
    pub total_ops_imported: usize,
    /// Number of user inputs (runtime tensors, not parameters/buffers).
    pub num_user_inputs: usize,
    /// Number of parameters + buffers loaded from safetensors.
    pub num_weights_loaded: usize,

    // -- Op mapping stats --
    /// Total ops in the input graph (all aten op nodes, excluding
    /// Input/Constant/getitem placeholders).
    pub op_count: usize,
    /// Successfully mapped ops: `(aten_target, count)`, sorted descending by count.
    pub mapped_ops: Vec<(String, usize)>,
    /// Unsupported/unmapped ops: `(aten_target, count)`, sorted descending by count.
    pub unmapped_ops: Vec<(String, usize)>,

    // -- Compilation stats --
    /// Dispatch count after compilation (IR + NativeOp steps).
    pub dispatch_count: usize,
    /// Dispatch count before fusion/peephole (estimated from graph node count).
    pub dispatch_count_before_fusion: usize,
    /// Peephole optimization statistics (NativeOp fusions).
    pub peephole_stats: PeepholeReport,
    /// Elementwise chain fusion statistics.
    pub fusion_stats: FusionReport,
    /// Total compiled steps (dispatches + passthroughs + inputs + etc.).
    pub total_steps: usize,
    /// Actual Metal kernel launches (after plan expansion).
    pub metal_dispatches: usize,
    /// Ops eliminated by fusion (elementwise chain fusion + peephole).
    pub fusion_count: usize,
    /// Number of NativeOp fused kernels created.
    pub native_op_count: usize,
    /// Compilation wall clock time in milliseconds.
    pub compile_time_ms: u64,

    // -- Performance estimate --
    /// Estimated real-time factor (RTF), if available.
    ///
    /// RTF < 1.0 means faster than real-time. Estimated from Metal dispatch
    /// count using a linear model calibrated against Kokoro benchmarks
    /// (~0.0014 RTF per dispatch on M4 Max). `None` when the estimate is
    /// not meaningful (e.g., no Metal compilation or zero dispatches).
    pub estimated_rtf: Option<f32>,

    // -- Verification --
    /// Verification coverage summary.
    pub verification: VerificationCoverage,
}

impl ConvertReport {
    /// Create a new empty report (all zeros/empty).
    pub(crate) fn new() -> Self {
        Self {
            intake_path: ConvertIntakePath::default(),
            artifact_kind: ConvertArtifactKind::default(),
            total_ops_imported: 0,
            num_user_inputs: 0,
            num_weights_loaded: 0,
            op_count: 0,
            mapped_ops: Vec::new(),
            unmapped_ops: Vec::new(),
            dispatch_count: 0,
            dispatch_count_before_fusion: 0,
            peephole_stats: PeepholeReport::default(),
            fusion_stats: FusionReport::default(),
            total_steps: 0,
            metal_dispatches: 0,
            fusion_count: 0,
            native_op_count: 0,
            compile_time_ms: 0,
            estimated_rtf: None,
            verification: VerificationCoverage::default(),
        }
    }

    /// Estimate RTF from dispatch count.
    ///
    /// Linear model calibrated against Kokoro production benchmarks on M4 Max:
    /// ~186 Metal dispatches produce RTF ~0.281. The per-dispatch overhead is
    /// approximately 0.0015 RTF/dispatch. This is a rough estimate -- actual
    /// RTF depends on kernel complexity, tensor sizes, and hardware.
    pub(crate) fn estimate_rtf(&mut self) {
        if self.metal_dispatches > 0 {
            let rtf = self.metal_dispatches as f32 * 0.0015 + 0.001;
            self.estimated_rtf = Some(rtf);
        }
    }

    /// Returns the dispatch reduction percentage (0-100).
    ///
    /// Returns `None` if `dispatch_count_before_fusion` is zero.
    #[must_use]
    pub fn dispatch_reduction_pct(&self) -> Option<f32> {
        if self.dispatch_count_before_fusion == 0 {
            return None;
        }
        let saved = self
            .dispatch_count_before_fusion
            .saturating_sub(self.dispatch_count);
        Some((saved as f32 / self.dispatch_count_before_fusion as f32) * 100.0)
    }

    /// Total number of successfully mapped op instances.
    #[must_use]
    pub fn mapped_ops_count(&self) -> usize {
        self.mapped_ops.iter().map(|(_, c)| c).sum()
    }

    /// Percentage of ops successfully mapped (0-100).
    ///
    /// Returns `None` if `op_count` is zero.
    #[must_use]
    pub fn mapped_pct(&self) -> Option<f32> {
        if self.op_count == 0 {
            return None;
        }
        Some((self.mapped_ops_count() as f32 / self.op_count as f32) * 100.0)
    }

    /// Print a human-readable summary to stderr.
    ///
    /// Convenience wrapper around the `Display` implementation.
    /// Equivalent to `eprintln!("{}", report)`.
    pub fn print(&self) {
        eprint!("{self}");
    }

    /// Serialize this report as pretty-printed JSON.
    ///
    /// Uses `serde_json::to_string_pretty` for human-readable output suitable
    /// for piping to `jq`, saving to files, or ingesting into dashboards. The
    /// output includes report provenance via `intake_path` and `artifact_kind`.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ConvertReport serialization is infallible")
    }

    /// Returns a compact provenance summary for this report.
    ///
    /// This describes how the exported-artifact intake was obtained and what
    /// kind of artifact the current report covers. It does not imply raw
    /// PyTorch ingestion or a proof-powered compiler path.
    #[must_use]
    pub fn provenance_summary(&self) -> String {
        format!("{} -> {}", self.intake_path, self.artifact_kind)
    }

    /// Returns a short readiness note that matches the recorded artifact kind.
    #[must_use]
    pub const fn artifact_readiness_note(&self) -> &'static str {
        match self.artifact_kind {
            ConvertArtifactKind::BackendAgnosticConvertedGraph => {
                "Backend-agnostic converted graph recorded; backend-specific compilation is not recorded in this report."
            }
            ConvertArtifactKind::CompiledMetalArtifact => {
                "Compiled Metal artifact ready for GPU execution."
            }
        }
    }

    /// Format a Markdown summary table of key metrics.
    ///
    /// Returns a compact table suitable for embedding in commit messages,
    /// issue comments, or CI reports. The table starts with provenance rows so
    /// downstream readers can attribute the report to the intake path and
    /// artifact kind that produced it.
    #[must_use]
    pub fn summary_table(&self) -> String {
        let mut lines = Vec::new();
        lines.push("| Metric | Value |".to_string());
        lines.push("|--------|-------|".to_string());
        lines.push(format!("| Provenance | {} |", self.provenance_summary()));
        lines.push(format!("| Intake path | {} |", self.intake_path));
        lines.push(format!("| Artifact kind | {} |", self.artifact_kind));
        lines.push(format!("| Input ops | {} |", self.op_count));
        if let Some(pct) = self.mapped_pct() {
            lines.push(format!(
                "| Mapped ops | {} ({:.1}%) |",
                self.mapped_ops_count(),
                pct
            ));
        } else {
            lines.push(format!("| Mapped ops | {} |", self.mapped_ops_count()));
        }
        let unmapped_total: usize = self.unmapped_ops.iter().map(|(_, c)| c).sum();
        if unmapped_total > 0 {
            lines.push(format!("| Unmapped ops | {unmapped_total} |"));
        }
        lines.push(format!("| Dispatch count | {} |", self.dispatch_count));
        if let Some(pct) = self.dispatch_reduction_pct() {
            lines.push(format!("| Dispatch reduction | {pct:.0}% |"));
        }
        lines.push(format!("| Metal dispatches | {} |", self.metal_dispatches));
        lines.push(format!("| Total steps | {} |", self.total_steps));
        if self.fusion_count > 0 {
            lines.push(format!("| Fused ops | {} |", self.fusion_count));
            lines.push(format!("| NativeOps | {} |", self.native_op_count));
        }
        if self.compile_time_ms > 0 {
            lines.push(format!("| Compile time | {}ms |", self.compile_time_ms));
        }
        if let Some(rtf) = self.estimated_rtf {
            lines.push(format!("| Estimated RTF | {rtf:.3} |"));
        }
        if let Some(kani) = self.verification.kani_harnesses_applicable {
            lines.push(format!("| Kani harnesses | {kani} |"));
        }
        if self.verification.gamma_crown_layers_total > 0 {
            lines.push(format!(
                "| NY coverage | {}/{} ({:.0}%) |",
                self.verification.gamma_crown_layers_covered,
                self.verification.gamma_crown_layers_total,
                self.verification.gamma_crown_coverage_pct()
            ));
        }
        if let Some(method) = self.verification.composition_method {
            lines.push(format!("| Composition method | {method} |"));
        }
        if let Some(soundness) = self.verification.composition_soundness_mode {
            lines.push(format!("| Composition soundness | {soundness} |"));
        }
        if let Some(strength) = self.verification.composition_proof_strength {
            lines.push(format!("| Composition proof strength | {strength} |"));
        }
        lines.join("\n")
    }
}

impl fmt::Display for ConvertReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Conversion complete:")?;
        writeln!(f, "  Intake path:   {}", self.intake_path)?;
        writeln!(f, "  Artifact kind: {}", self.artifact_kind)?;
        writeln!(f)?;

        // Op mapping stats.
        if self.op_count > 0 {
            let mapped = self.mapped_ops_count();
            writeln!(f, "  Input ops:     {}", self.op_count)?;
            if let Some(pct) = self.mapped_pct() {
                writeln!(f, "  Mapped:        {mapped} ({pct:.1}%)")?;
            } else {
                writeln!(f, "  Mapped:        {mapped}")?;
            }
            let unmapped_total: usize = self.unmapped_ops.iter().map(|(_, c)| c).sum();
            if unmapped_total > 0 {
                let detail: Vec<String> = self
                    .unmapped_ops
                    .iter()
                    .map(|(name, count)| format!("{name} x{count}"))
                    .collect();
                writeln!(
                    f,
                    "  Unsupported:   {} ({})",
                    unmapped_total,
                    detail.join(", ")
                )?;
            }
        }

        // Fusion summary.
        if self.fusion_count > 0 || self.native_op_count > 0 {
            writeln!(
                f,
                "  Fused:         {} ops -> {} NativeOps",
                self.fusion_count, self.native_op_count
            )?;
        }

        // Compile time.
        if self.compile_time_ms > 0 {
            writeln!(f, "  Compile time:  {}ms", self.compile_time_ms)?;
        }

        writeln!(f)?;

        // Detailed import section (backwards-compatible).
        writeln!(
            f,
            "Imported: {} ops from torch.export graph",
            self.total_ops_imported
        )?;
        writeln!(
            f,
            "  User inputs: {}, Weights loaded: {}",
            self.num_user_inputs, self.num_weights_loaded
        )?;
        writeln!(f)?;

        // Optimization section.
        writeln!(f, "Optimization:")?;
        if let Some(pct) = self.dispatch_reduction_pct() {
            writeln!(
                f,
                "  Dispatch count: {} -> {} ({:.0}% reduction)",
                self.dispatch_count_before_fusion, self.dispatch_count, pct
            )?;
        } else {
            writeln!(f, "  Dispatch count: {}", self.dispatch_count)?;
        }
        writeln!(f, "  Metal kernel launches: {}", self.metal_dispatches)?;
        writeln!(f, "  Total compiled steps: {}", self.total_steps)?;

        // Peephole stats.
        if self.peephole_stats.native_ops > 0 {
            writeln!(
                f,
                "  Peephole: {} NativeOps ({} Metal dispatches)",
                self.peephole_stats.native_ops, self.peephole_stats.native_dispatches
            )?;
            for (name, count) in &self.peephole_stats.by_variant {
                writeln!(f, "    {name} ({count}x)")?;
            }
        }

        // Fusion stats.
        if self.fusion_stats.fused_chains > 0 {
            writeln!(
                f,
                "  Elementwise chains fused: {} ({} ops, {} dispatches saved)",
                self.fusion_stats.fused_chains,
                self.fusion_stats.fused_ops,
                self.fusion_stats.dispatches_saved
            )?;
        }

        // Estimated RTF.
        if let Some(rtf) = self.estimated_rtf {
            writeln!(f, "  Estimated RTF: {rtf:.3}")?;
        }
        writeln!(f)?;

        // Verification section.
        writeln!(f, "Verification:")?;
        if let Some(kani) = self.verification.kani_harnesses_applicable {
            writeln!(f, "  Kani: {kani} harnesses applicable")?;
        } else {
            writeln!(f, "  Kani: not checked")?;
        }
        if self.verification.gamma_crown_layers_total > 0 {
            writeln!(
                f,
                "  NY: {}/{} layers covered ({:.0}%)",
                self.verification.gamma_crown_layers_covered,
                self.verification.gamma_crown_layers_total,
                self.verification.gamma_crown_coverage_pct()
            )?;
        }
        if self.verification.composition_bounds_ok {
            writeln!(f, "  Composition bounds: propagated")?;
            if let Some(method) = self.verification.composition_method {
                writeln!(f, "    Method: {method}")?;
            }
            if let Some(soundness) = self.verification.composition_soundness_mode {
                writeln!(f, "    Soundness: {soundness}")?;
            }
            if let Some(strength) = self.verification.composition_proof_strength {
                writeln!(f, "    Proof strength: {strength}")?;
            }
            if let Some(w) = self.verification.composition_bound_width {
                writeln!(f, "    Max output bound width: {w:.4}")?;
            }
        }
        if let Some(passed) = self.verification.reference_parity_passed {
            writeln!(
                f,
                "  Reference parity: {}",
                if passed { "PASSED" } else { "FAILED" }
            )?;
        }
        writeln!(f)?;
        writeln!(f, "{}", self.artifact_readiness_note())?;
        Ok(())
    }
}

/// Peephole optimization report (mirrors `PeepholeStats` from nn-dsl).
#[derive(Clone, Debug, Default, Serialize)]
pub struct PeepholeReport {
    /// Number of NativeOp steps.
    pub native_ops: usize,
    /// Total Metal dispatches from NativeOps.
    pub native_dispatches: usize,
    /// Number of IdentityPassthrough steps (fusion placeholders).
    pub passthrough_count: usize,
    /// Per-variant breakdown: variant name -> count (sorted descending).
    pub by_variant: Vec<(String, usize)>,
}

/// Elementwise chain fusion report (mirrors `FusionStats` from nn-dsl).
#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionReport {
    /// Number of fused dispatch steps.
    pub fused_chains: usize,
    /// Total ops absorbed into fused chains.
    pub fused_ops: usize,
    /// Dispatches eliminated by fusion.
    pub dispatches_saved: usize,
}

/// Verification coverage summary.
#[derive(Clone, Debug, Default, Serialize)]
#[non_exhaustive]
pub struct VerificationCoverage {
    /// Number of Kani harnesses applicable to this model's ops.
    /// `None` if Kani was not checked.
    pub kani_harnesses_applicable: Option<usize>,
    /// NY layers covered (translated successfully).
    pub gamma_crown_layers_covered: usize,
    /// NY layers total (in the model graph).
    pub gamma_crown_layers_total: usize,
    /// Whether IBP composition bounds propagated successfully.
    pub composition_bounds_ok: bool,
    /// Max output bound width from IBP, if available.
    pub composition_bound_width: Option<f32>,
    /// Propagation method recorded for the current composition-bounds result.
    ///
    /// Today this is only populated from the current `check_composition_bounds()`
    /// path. A missing value means that path did not run, failed before
    /// classification, or did not surface a method tag.
    pub composition_method: Option<ConvertCompositionMethod>,
    /// Soundness mode for the current composition-bounds result.
    ///
    /// This does not summarize the whole convert pipeline. It reflects only the
    /// composition-bounds run that produced `composition_bound_width`.
    pub composition_soundness_mode: Option<ConvertSoundnessMode>,
    /// Proof-strength classification for the current composition-bounds result.
    ///
    /// This is currently derived from the composition-bounds run when the
    /// verifier can classify it. It does not imply inline Kani or a complete
    /// end-to-end proof-powered compiler certificate.
    pub composition_proof_strength: Option<ConvertProofStrength>,
    /// Whether reference parity check passed. `None` if not run.
    pub reference_parity_passed: Option<bool>,
}

impl VerificationCoverage {
    /// Returns NY coverage percentage (0-100).
    #[must_use]
    pub fn gamma_crown_coverage_pct(&self) -> f32 {
        if self.gamma_crown_layers_total == 0 {
            return 0.0;
        }
        (self.gamma_crown_layers_covered as f32 / self.gamma_crown_layers_total as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_convert_report_display() {
        let mut report = ConvertReport::new();
        report.intake_path = ConvertIntakePath::CliExportedPytorch;
        report.artifact_kind = ConvertArtifactKind::CompiledMetalArtifact;
        report.total_ops_imported = 847;
        report.num_user_inputs = 3;
        report.num_weights_loaded = 120;
        report.op_count = 42;
        report.mapped_ops = vec![
            ("torch.ops.aten.linear.default".to_string(), 20),
            ("torch.ops.aten.relu.default".to_string(), 18),
        ];
        report.unmapped_ops = vec![
            ("aten::complex_op".to_string(), 1),
            ("aten::custom".to_string(), 1),
        ];
        report.dispatch_count_before_fusion = 300;
        report.dispatch_count = 150;
        report.metal_dispatches = 186;
        report.total_steps = 220;
        report.fusion_count = 12;
        report.native_op_count = 4;
        report.compile_time_ms = 123;
        report.estimated_rtf = Some(0.280);
        report.peephole_stats = PeepholeReport {
            native_ops: 6,
            native_dispatches: 18,
            passthrough_count: 12,
            by_variant: vec![
                ("NormActivConv1d".to_string(), 4),
                ("LstmSequence".to_string(), 2),
            ],
        };
        report.fusion_stats = FusionReport {
            fused_chains: 18,
            fused_ops: 54,
            dispatches_saved: 36,
        };
        report.verification.kani_harnesses_applicable = Some(754);
        report.verification.gamma_crown_layers_covered = 45;
        report.verification.gamma_crown_layers_total = 52;
        report.verification.composition_bounds_ok = true;
        report.verification.composition_bound_width = Some(12.5);
        report.verification.composition_method = Some(ConvertCompositionMethod::Ibp);
        report.verification.composition_soundness_mode = Some(ConvertSoundnessMode::Sound);
        report.verification.composition_proof_strength = Some(ConvertProofStrength::SoundIbp);

        let text = format!("{report}");
        assert!(
            text.contains("Intake path:   CLI-exported PyTorch"),
            "should show intake provenance"
        );
        assert!(
            text.contains("Artifact kind: compiled Metal artifact"),
            "should show artifact kind"
        );
        // New structured op stats.
        assert!(
            text.contains("Input ops:     42"),
            "should show op count: {text}"
        );
        assert!(
            text.contains("Mapped:        38"),
            "should show mapped count"
        );
        assert!(text.contains("90.5%"), "should show mapped percentage");
        assert!(
            text.contains("Unsupported:   2"),
            "should show unsupported count"
        );
        assert!(
            text.contains("aten::complex_op x1"),
            "should show unsupported op names"
        );
        assert!(
            text.contains("12 ops -> 4 NativeOps"),
            "should show fusion summary"
        );
        assert!(
            text.contains("Compile time:  123ms"),
            "should show compile time"
        );
        // Legacy sections still present.
        assert!(text.contains("847 ops"), "should show imported ops");
        assert!(
            text.contains("300 -> 150"),
            "should show dispatch reduction"
        );
        assert!(text.contains("50% reduction"), "should show reduction %");
        assert!(text.contains("754 harnesses"), "should show Kani count");
        assert!(text.contains("45/52"), "should show NY coverage");
        assert!(text.contains("87%"), "should show coverage %");
        assert!(
            text.contains("Method: IBP"),
            "should show composition method"
        );
        assert!(
            text.contains("Soundness: sound"),
            "should show composition soundness"
        );
        assert!(
            text.contains("Proof strength: sound_ibp"),
            "should show composition proof strength"
        );
        assert!(
            text.contains("NormActivConv1d (4x)"),
            "should show peephole variants"
        );
        assert!(text.contains("6 NativeOps"), "should show native ops");
        assert!(
            text.contains("chains fused: 18"),
            "should show fused chains"
        );
        assert!(
            text.contains("Estimated RTF: 0.280"),
            "should show estimated RTF"
        );
        assert!(
            text.contains("Compiled Metal artifact ready for GPU execution."),
            "should show a readiness note that matches the artifact kind"
        );
    }

    #[test]
    fn test_dispatch_reduction_pct() {
        let mut report = ConvertReport::new();
        report.dispatch_count_before_fusion = 100;
        report.dispatch_count = 40;
        let pct = report.dispatch_reduction_pct().unwrap();
        assert!((pct - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_dispatch_reduction_pct_zero_before() {
        let report = ConvertReport::new();
        assert!(report.dispatch_reduction_pct().is_none());
    }

    #[test]
    fn test_gamma_crown_coverage_pct() {
        let vc = VerificationCoverage {
            gamma_crown_layers_covered: 45,
            gamma_crown_layers_total: 52,
            ..Default::default()
        };
        let pct = vc.gamma_crown_coverage_pct();
        assert!((pct - 86.538).abs() < 0.1);
    }

    #[test]
    fn test_gamma_crown_coverage_pct_zero() {
        let vc = VerificationCoverage::default();
        assert!((vc.gamma_crown_coverage_pct()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_convert_report_display_minimal() {
        let report = ConvertReport::new();
        let text = format!("{report}");
        assert!(text.contains("Conversion complete:"));
        assert!(text.contains("Intake path:   exported artifacts"));
        assert!(text.contains("Artifact kind: backend-agnostic converted graph"));
        assert!(text.contains(
            "Backend-agnostic converted graph recorded; backend-specific compilation is not recorded in this report."
        ));
    }

    #[test]
    fn test_convert_report_print() {
        let report = ConvertReport::new();
        // print() should not panic — it writes to stderr.
        report.print();
    }

    #[test]
    fn test_estimate_rtf() {
        let mut report = ConvertReport::new();
        report.metal_dispatches = 186;
        report.estimate_rtf();
        let rtf = report.estimated_rtf.unwrap();
        // 186 * 0.0015 + 0.001 = 0.28
        assert!((rtf - 0.28).abs() < 0.01, "expected ~0.28, got {rtf}");
    }

    #[test]
    fn test_estimate_rtf_zero_dispatches() {
        let mut report = ConvertReport::new();
        report.estimate_rtf();
        assert!(report.estimated_rtf.is_none());
    }

    #[test]
    fn test_mapped_ops_count() {
        let mut report = ConvertReport::new();
        report.mapped_ops = vec![("relu".to_string(), 10), ("linear".to_string(), 5)];
        assert_eq!(report.mapped_ops_count(), 15);
    }

    #[test]
    fn test_mapped_ops_count_empty() {
        let report = ConvertReport::new();
        assert_eq!(report.mapped_ops_count(), 0);
    }

    #[test]
    fn test_mapped_pct() {
        let mut report = ConvertReport::new();
        report.op_count = 100;
        report.mapped_ops = vec![("relu".to_string(), 80), ("linear".to_string(), 15)];
        let pct = report.mapped_pct().unwrap();
        assert!((pct - 95.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_pct_zero_ops() {
        let report = ConvertReport::new();
        assert!(report.mapped_pct().is_none());
    }

    #[test]
    fn test_new_fields_initialized() {
        let report = ConvertReport::new();
        assert_eq!(report.intake_path, ConvertIntakePath::ExportedArtifacts);
        assert_eq!(
            report.artifact_kind,
            ConvertArtifactKind::BackendAgnosticConvertedGraph
        );
        assert_eq!(report.op_count, 0);
        assert!(report.mapped_ops.is_empty());
        assert!(report.unmapped_ops.is_empty());
        assert_eq!(report.fusion_count, 0);
        assert_eq!(report.native_op_count, 0);
        assert_eq!(report.compile_time_ms, 0);
        assert!(report.verification.composition_method.is_none());
        assert!(report.verification.composition_soundness_mode.is_none());
        assert!(report.verification.composition_proof_strength.is_none());
    }

    #[test]
    fn test_json_round_trip() {
        let mut report = ConvertReport::new();
        report.intake_path = ConvertIntakePath::CliExportedPytorch;
        report.artifact_kind = ConvertArtifactKind::CompiledMetalArtifact;
        report.total_ops_imported = 100;
        report.num_user_inputs = 2;
        report.num_weights_loaded = 50;
        report.op_count = 42;
        report.mapped_ops = vec![
            ("aten.linear".to_string(), 20),
            ("aten.relu".to_string(), 18),
        ];
        report.unmapped_ops = vec![("aten.custom".to_string(), 4)];
        report.dispatch_count = 150;
        report.dispatch_count_before_fusion = 300;
        report.metal_dispatches = 186;
        report.total_steps = 220;
        report.fusion_count = 12;
        report.native_op_count = 4;
        report.compile_time_ms = 123;
        report.estimated_rtf = Some(0.280);
        report.peephole_stats = PeepholeReport {
            native_ops: 6,
            native_dispatches: 18,
            passthrough_count: 12,
            by_variant: vec![("NormActivConv1d".to_string(), 4)],
        };
        report.fusion_stats = FusionReport {
            fused_chains: 18,
            fused_ops: 54,
            dispatches_saved: 36,
        };
        report.verification.kani_harnesses_applicable = Some(754);
        report.verification.gamma_crown_layers_covered = 45;
        report.verification.gamma_crown_layers_total = 52;
        report.verification.composition_bounds_ok = true;
        report.verification.composition_bound_width = Some(12.5);
        report.verification.composition_method = Some(ConvertCompositionMethod::Ibp);
        report.verification.composition_soundness_mode = Some(ConvertSoundnessMode::Sound);
        report.verification.composition_proof_strength = Some(ConvertProofStrength::SoundIbp);
        report.verification.reference_parity_passed = Some(true);

        // Serialize to JSON.
        let json = report.to_json();

        // Parse back as serde_json::Value and verify key fields survived.
        let val: serde_json::Value =
            serde_json::from_str(&json).expect("to_json output must be valid JSON");
        assert_eq!(val["intake_path"], "cli_exported_pytorch");
        assert_eq!(val["artifact_kind"], "compiled_metal_artifact");
        assert_eq!(val["total_ops_imported"], 100);
        assert_eq!(val["num_user_inputs"], 2);
        assert_eq!(val["op_count"], 42);
        assert_eq!(val["dispatch_count"], 150);
        assert_eq!(val["metal_dispatches"], 186);
        assert_eq!(val["fusion_count"], 12);
        assert_eq!(val["compile_time_ms"], 123);

        // Check nested structures.
        assert_eq!(val["peephole_stats"]["native_ops"], 6);
        assert_eq!(val["fusion_stats"]["fused_chains"], 18);
        assert_eq!(val["verification"]["kani_harnesses_applicable"], 754);
        assert_eq!(val["verification"]["gamma_crown_layers_covered"], 45);
        assert_eq!(val["verification"]["composition_bounds_ok"], true);
        assert_eq!(val["verification"]["composition_method"], "IBP");
        assert_eq!(val["verification"]["composition_soundness_mode"], "sound");
        assert_eq!(
            val["verification"]["composition_proof_strength"],
            "sound_ibp"
        );
        assert_eq!(val["verification"]["reference_parity_passed"], true);

        // Check mapped_ops array of tuples.
        let mapped = val["mapped_ops"].as_array().unwrap();
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0][0], "aten.linear");
        assert_eq!(mapped[0][1], 20);

        // estimated_rtf present.
        let rtf_val = val["estimated_rtf"].as_f64().unwrap();
        assert!((rtf_val - 0.280).abs() < 0.001);
    }

    #[test]
    fn test_json_minimal_report() {
        let report = ConvertReport::new();
        let json = report.to_json();
        let val: serde_json::Value =
            serde_json::from_str(&json).expect("minimal report must serialize");
        assert_eq!(val["intake_path"], "exported_artifacts");
        assert_eq!(val["artifact_kind"], "backend_agnostic_converted_graph");
        assert_eq!(val["total_ops_imported"], 0);
        assert!(val["estimated_rtf"].is_null());
        assert!(val["mapped_ops"].as_array().unwrap().is_empty());
        assert!(val["verification"]["composition_method"].is_null());
        assert!(val["verification"]["composition_soundness_mode"].is_null());
        assert!(val["verification"]["composition_proof_strength"].is_null());
    }

    #[test]
    fn test_provenance_summary_and_readiness_note() {
        let mut report = ConvertReport::new();
        assert_eq!(
            report.provenance_summary(),
            "exported artifacts -> backend-agnostic converted graph"
        );
        assert_eq!(
            report.artifact_readiness_note(),
            "Backend-agnostic converted graph recorded; backend-specific compilation is not recorded in this report."
        );

        report.intake_path = ConvertIntakePath::CliExportedPytorch;
        report.artifact_kind = ConvertArtifactKind::CompiledMetalArtifact;
        assert_eq!(
            report.provenance_summary(),
            "CLI-exported PyTorch -> compiled Metal artifact"
        );
        assert_eq!(
            report.artifact_readiness_note(),
            "Compiled Metal artifact ready for GPU execution."
        );
    }

    #[test]
    fn test_summary_table() {
        let mut report = ConvertReport::new();
        report.intake_path = ConvertIntakePath::CliExportedPytorch;
        report.artifact_kind = ConvertArtifactKind::CompiledMetalArtifact;
        report.op_count = 42;
        report.mapped_ops = vec![("aten.linear".to_string(), 38)];
        report.dispatch_count = 150;
        report.dispatch_count_before_fusion = 300;
        report.metal_dispatches = 186;
        report.total_steps = 220;
        report.fusion_count = 12;
        report.native_op_count = 4;
        report.compile_time_ms = 123;
        report.estimated_rtf = Some(0.280);
        report.verification.kani_harnesses_applicable = Some(754);
        report.verification.gamma_crown_layers_covered = 45;
        report.verification.gamma_crown_layers_total = 52;
        report.verification.composition_method = Some(ConvertCompositionMethod::Ibp);
        report.verification.composition_soundness_mode = Some(ConvertSoundnessMode::Sound);
        report.verification.composition_proof_strength = Some(ConvertProofStrength::SoundIbp);

        let table = report.summary_table();
        assert!(table.contains("| Metric | Value |"), "header row");
        assert!(
            table.contains("| Provenance | CLI-exported PyTorch -> compiled Metal artifact |"),
            "combined provenance"
        );
        assert!(
            table.contains("| Intake path | CLI-exported PyTorch |"),
            "intake provenance"
        );
        assert!(
            table.contains("| Artifact kind | compiled Metal artifact |"),
            "artifact kind"
        );
        assert!(table.contains("| Input ops | 42 |"), "input ops");
        assert!(table.contains("90.5%"), "mapped percentage");
        assert!(table.contains("| Dispatch count | 150 |"), "dispatch count");
        assert!(table.contains("| Dispatch reduction | 50% |"), "reduction");
        assert!(table.contains("| Metal dispatches | 186 |"), "metal");
        assert!(table.contains("| Fused ops | 12 |"), "fused ops");
        assert!(table.contains("| Compile time | 123ms |"), "compile time");
        assert!(table.contains("| Estimated RTF | 0.280 |"), "rtf");
        assert!(table.contains("| Kani harnesses | 754 |"), "kani");
        assert!(table.contains("45/52"), "NY");
        assert!(
            table.contains("| Composition method | IBP |"),
            "composition method"
        );
        assert!(
            table.contains("| Composition soundness | sound |"),
            "composition soundness"
        );
        assert!(
            table.contains("| Composition proof strength | sound_ibp |"),
            "composition proof strength"
        );
    }
}
