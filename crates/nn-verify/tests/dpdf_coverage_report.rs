// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! dpdf verification coverage dashboard.
//!
//! Generates a comprehensive report of dpdf model verification coverage
//! across all three verification layers:
//! 1. **NY compose tests** — per-model counts from `compose_dpdf_*.rs`
//! 2. **Kani harnesses** — per-file counts from `kani_dpdf_*.rs` in nn-models
//! 3. **ay SMT proofs** — proof count from split `ay_dpdf_*_proofs.rs` helpers
//!
//! Also reads `nn_verify_status_dpdf.json` for soundness breakdown and
//! checks certification property coverage from `dpdf_certify.rs`.
//!
//! Run: `cargo test -p nn-verify --test dpdf_coverage_report -- --nocapture`
//!
//! Part of #3919.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nn_verify::dpdf_certify::{DpdfCertificate, DpdfProperty, PropertyStatus};

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Workspace root (two levels up from CARGO_MANIFEST_DIR = crates/nn-verify).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Returns `true` (and prints a skip notice) when the dpdf status artifact is
/// absent. `nn_verify_status_dpdf.json` is a generated, gitignored artifact
/// produced by `cargo run --example verify_all`; a bare `cargo test`/`nextest`
/// run does not create it. Mirrors the skip-when-absent convention already used
/// by `gap_detector_tests::test_detect_gaps_real_status_file` so these
/// integration tests skip cleanly instead of hard-failing on a missing artifact
/// (without weakening the threshold assertions when the artifact IS present).
fn dpdf_status_missing() -> bool {
    let path = workspace_root().join("nn_verify_status_dpdf.json");
    if path.exists() {
        return false;
    }
    eprintln!(
        "Skipping: nn_verify_status_dpdf.json not found at {}. \
         Run `cargo run -p nn-verify --example verify_all` to generate it.",
        path.display()
    );
    true
}

/// Path to the nn-verify test helpers directory.
fn verify_helpers_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/helpers")
}

/// Path to the nn-models src directory.
fn models_src_dir() -> PathBuf {
    workspace_root().join("crates/nn-models/src")
}

// ---------------------------------------------------------------------------
// Counting helpers
// ---------------------------------------------------------------------------

/// Count occurrences of `pattern` in a file. Returns 0 if the file cannot be read.
fn count_pattern_in_file(path: &Path, pattern: &str) -> usize {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .matches(pattern)
        .count()
}

/// Model name mapping from helper filename to display name.
fn dpdf_compose_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("compose_dpdf_granite_docling.rs", "Granite-Docling"),
        ("compose_dpdf_doclayout_yolo.rs", "DocLayout-YOLO"),
        ("compose_dpdf_glm_ocr.rs", "GLM-OCR"),
        ("compose_dpdf_table_transformer.rs", "Table Transformer"),
        ("compose_dpdf_qwen3_vl.rs", "Qwen3-VL"),
        ("compose_dpdf_paddle_ocr.rs", "PaddleOCR"),
        ("compose_dpdf_firered_ocr.rs", "FireRed-OCR"),
    ]
}

/// Kani harness files in nn-models for dpdf.
fn dpdf_kani_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("kani_dpdf_model_proofs.rs", "Model builders"),
        ("kani_table_transformer_glm_ocr_proofs.rs", "Table+GLM-OCR"),
        ("kani_dpdf_pipeline_paddle_ocr_proofs.rs", "Pipeline+Paddle"),
        (
            "kani_dpdf_postprocess_table_structure_proofs.rs",
            "Postprocess+Table",
        ),
        ("kani_convert_dpdf_proofs.rs", "Weight conversion"),
        ("kani_dpdf_new_modules_proofs.rs", "New modules"),
    ]
}

/// Count #[test] attributes per compose helper file.
fn count_compose_tests() -> BTreeMap<String, usize> {
    let helpers_dir = verify_helpers_dir();
    let mut counts = BTreeMap::new();
    for (filename, display_name) in dpdf_compose_files() {
        let path = helpers_dir.join(filename);
        let count = count_pattern_in_file(&path, "#[test]");
        counts.insert(display_name.to_string(), count);
    }
    counts
}

/// Count #[kani::proof] attributes per Kani harness file.
fn count_kani_harnesses() -> BTreeMap<String, usize> {
    let src_dir = models_src_dir();
    let mut counts = BTreeMap::new();
    for (filename, display_name) in dpdf_kani_files() {
        let path = src_dir.join(filename);
        let count = count_pattern_in_file(&path, "#[kani::proof]");
        counts.insert(display_name.to_string(), count);
    }
    counts
}

/// Count #[test] attributes in split ay_dpdf_*_proofs.rs helpers.
fn count_ay_proofs() -> usize {
    let helpers = verify_helpers_dir();
    std::fs::read_dir(&helpers)
        .unwrap_or_else(|e| panic!("read helpers dir {}: {e}", helpers.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("ay_dpdf_") && name.ends_with("_proofs.rs"))
        })
        .map(|path| count_pattern_in_file(&path, "#[test]"))
        .sum()
}

// ---------------------------------------------------------------------------
// Status JSON parsing (minimal, serde_json)
// ---------------------------------------------------------------------------

/// Minimal representation of a kernel entry in the status JSON.
#[derive(Debug, serde::Deserialize)]
struct StatusKernel {
    soundness_mode: String,
    proof_strength: String,
    #[serde(default)]
    stale: bool,
}

/// Top-level shape of `nn_verify_status_dpdf.json`.
#[derive(Debug, serde::Deserialize)]
struct StatusFile {
    kernels: std::collections::HashMap<String, StatusKernel>,
}

/// Soundness breakdown from the status file.
#[derive(Debug, Default)]
struct SoundnessBreakdown {
    total: usize,
    sound: usize,
    heuristic: usize,
    stale: usize,
    soundness_modes: BTreeMap<String, usize>,
}

fn load_status_breakdown() -> SoundnessBreakdown {
    let path = workspace_root().join("nn_verify_status_dpdf.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  WARNING: Cannot read {}: {e}", path.display());
            return SoundnessBreakdown::default();
        }
    };
    let status: StatusFile = match serde_json::from_str(&raw) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  WARNING: Cannot parse {}: {e}", path.display());
            return SoundnessBreakdown::default();
        }
    };

    let mut bd = SoundnessBreakdown::default();
    for entry in status.kernels.values() {
        bd.total += 1;
        if entry.stale {
            bd.stale += 1;
            continue;
        }
        match entry.proof_strength.as_str() {
            "sound" => bd.sound += 1,
            "heuristic" => bd.heuristic += 1,
            _ => {}
        }
        *bd.soundness_modes
            .entry(entry.soundness_mode.clone())
            .or_insert(0) += 1;
    }
    bd
}

// ---------------------------------------------------------------------------
// Report printing
// ---------------------------------------------------------------------------

fn print_report(
    compose_counts: &BTreeMap<String, usize>,
    kani_counts: &BTreeMap<String, usize>,
    ay_count: usize,
    breakdown: &SoundnessBreakdown,
    cert: &DpdfCertificate,
) {
    let compose_total: usize = compose_counts.values().sum();
    let kani_total: usize = kani_counts.values().sum();

    eprintln!();
    eprintln!("================================================================");
    eprintln!("        dpdf VERIFICATION COVERAGE DASHBOARD");
    eprintln!("        Part of #3919");
    eprintln!("================================================================");

    // -- Compose tests --
    eprintln!();
    eprintln!("  -- NY compose tests --");
    eprintln!("  {:<20} {:>6}", "Model", "Tests");
    eprintln!("  {:<20} {:>6}", "--------------------", "------");
    for (name, count) in compose_counts {
        eprintln!("  {name:<20} {count:>6}");
    }
    eprintln!("  {:<20} {:>6}", "--------------------", "------");
    eprintln!("  {:<20} {:>6}", "TOTAL", compose_total);

    // -- Kani harnesses --
    eprintln!();
    eprintln!("  -- Kani harnesses (nn-models) --");
    eprintln!("  {:<20} {:>6}", "File", "Proofs");
    eprintln!("  {:<20} {:>6}", "--------------------", "------");
    for (name, count) in kani_counts {
        eprintln!("  {name:<20} {count:>6}");
    }
    eprintln!("  {:<20} {:>6}", "--------------------", "------");
    eprintln!("  {:<20} {:>6}", "TOTAL", kani_total);

    // -- ay SMT proofs --
    eprintln!();
    eprintln!("  -- ay SMT proofs --");
    eprintln!("  ay_dpdf_*_proofs.rs:  {ay_count} proofs");

    // -- Status file breakdown --
    eprintln!();
    eprintln!("  -- nn_verify_status_dpdf.json --");
    let active = breakdown.total - breakdown.stale;
    eprintln!("  Total entries:    {}", breakdown.total);
    eprintln!("  Active (non-stale): {active}");
    eprintln!("  Sound:            {}", breakdown.sound);
    eprintln!("  Heuristic:        {}", breakdown.heuristic);
    eprintln!("  Stale:            {}", breakdown.stale);
    if active > 0 {
        let sound_pct = 100.0 * breakdown.sound as f64 / active as f64;
        eprintln!("  Sound rate:       {sound_pct:.1}%");
    }
    eprintln!("  Soundness modes:");
    for (mode, count) in &breakdown.soundness_modes {
        eprintln!("    {mode:<20} {count}");
    }

    // -- Certification properties (P1-P8) --
    eprintln!();
    eprintln!("  -- Certification properties (P1-P8) --");
    let (proven, heuristic, unverified, na) = cert.status_counts();
    for (prop, status, evidence) in &cert.properties {
        eprintln!(
            "  P{}: [{:<10}] {}",
            prop.number(),
            status.to_string(),
            prop.name()
        );
        eprintln!("       Evidence: {evidence}");
    }
    eprintln!();
    eprintln!(
        "  Proven: {proven}  Heuristic: {heuristic}  Unverified: {unverified}  N/A: {na}"
    );
    eprintln!("  Deployment ready: {}", cert.is_deployment_ready());

    // -- Summary --
    eprintln!();
    eprintln!("  ================ SUMMARY ================");
    eprintln!(
        "  Compose tests:    {compose_total:>4}  (threshold: >= 100)"
    );
    eprintln!("  Kani harnesses:   {kani_total:>4}  (threshold: >= 50)");
    eprintln!("  ay proofs:        {ay_count:>4}  (threshold: >= 5)");
    eprintln!(
        "  Status entries:   {:>4}  ({} sound / {} heuristic)",
        active, breakdown.sound, breakdown.heuristic
    );
    eprintln!(
        "  Cert properties:  {:>4} proven / {} total",
        proven,
        DpdfProperty::ALL.len()
    );
    eprintln!("  ==========================================");
    eprintln!();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full dpdf coverage dashboard with threshold assertions.
#[test]
fn dpdf_coverage_dashboard() {
    if dpdf_status_missing() {
        return;
    }
    let compose_counts = count_compose_tests();
    let kani_counts = count_kani_harnesses();
    let ay_count = count_ay_proofs();
    let breakdown = load_status_breakdown();

    // Generate certificate from status file.
    let cert = DpdfCertificate::generate(&workspace_root())
        .unwrap_or_else(|e| panic!("Failed to generate dpdf certificate: {e}"));

    // Print the full report.
    print_report(&compose_counts, &kani_counts, ay_count, &breakdown, &cert);

    // === Threshold assertions ===

    let compose_total: usize = compose_counts.values().sum();
    let kani_total: usize = kani_counts.values().sum();

    // Gate 1: >= 100 total compose tests
    assert!(
        compose_total >= 100,
        "dpdf compose tests {compose_total} below minimum threshold 100. \
         Check that all compose_dpdf_*.rs helper files exist and contain tests."
    );

    // Gate 2: >= 50 total Kani harnesses
    assert!(
        kani_total >= 50,
        "dpdf Kani harnesses {kani_total} below minimum threshold 50. \
         Check that all kani_dpdf_*.rs files exist in nn-models/src/."
    );

    // Gate 3: >= 5 ay proofs
    assert!(
        ay_count >= 5,
        "dpdf ay proofs {ay_count} below minimum threshold 5. \
         Check ay_dpdf_*_proofs.rs in nn-verify/tests/helpers/."
    );

    // Gate 4: every compose helper file contributes at least 1 test
    for (name, count) in &compose_counts {
        assert!(
            *count > 0,
            "Model {name} has 0 compose tests — helper file may be missing or empty."
        );
    }

    // Gate 5: every Kani file contributes at least 1 harness
    for (name, count) in &kani_counts {
        assert!(
            *count > 0,
            "Kani file for {name} has 0 harnesses — file may be missing or empty."
        );
    }

    // Gate 6: status file has entries and sound count is positive
    assert!(
        breakdown.total > 0,
        "nn_verify_status_dpdf.json has 0 entries — file may be missing or empty."
    );
    assert!(
        breakdown.sound > 0,
        "nn_verify_status_dpdf.json has 0 sound entries — verification may have regressed."
    );
}

/// Per-model compose test counts are non-zero and balanced.
#[test]
fn dpdf_per_model_compose_coverage() {
    let counts = count_compose_tests();
    let total: usize = counts.values().sum();

    // Every model family should have at least 10 compose tests.
    let min_per_model = 10;
    for (name, count) in &counts {
        assert!(
            *count >= min_per_model,
            "Model {name} has only {count} compose tests, expected >= {min_per_model}."
        );
    }

    // Verify we have all 7 model families.
    assert_eq!(
        counts.len(),
        7,
        "Expected 7 dpdf model families, found {}. \
         Models: {:?}",
        counts.len(),
        counts.keys().collect::<Vec<_>>()
    );

    eprintln!(
        "\n  dpdf per-model compose coverage: {total} total across {} models\n",
        counts.len()
    );
}

/// Certification property coverage: at least P1-P4 should have evidence.
#[test]
fn dpdf_certification_property_coverage() {
    if dpdf_status_missing() {
        return;
    }
    let cert = DpdfCertificate::generate(&workspace_root())
        .unwrap_or_else(|e| panic!("Failed to generate dpdf certificate: {e}"));

    // P1-P4 should all be at least Heuristic (not Unverified).
    for (prop, status, evidence) in &cert.properties {
        if prop.number() <= 4 {
            assert!(
                !matches!(status, PropertyStatus::Unverified),
                "P{} ({}) is Unverified — expected at least Heuristic. Evidence: {}",
                prop.number(),
                prop.name(),
                evidence
            );
        }
    }

    // At least 4 properties should be Proven.
    let (proven, _heuristic, _unverified, _na) = cert.status_counts();
    assert!(
        proven >= 4,
        "Only {proven} properties are Proven, expected >= 4."
    );
}
