// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive dispatch count audit: every step in every segment.
//!
//! Builds a miniaturized `CompiledKokoro`, synthesizes to compile all
//! segments, then enumerates every compiled step with its type, detail,
//! and estimated Metal dispatch count. Also runs fusion gap analysis
//! per segment and prints a blocker distribution.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all kokoro_dispatch_audit -- --nocapture`
//!
//! Part of #4252.

use std::collections::BTreeMap;

/// Full dispatch audit: per-segment, per-step breakdown.
///
/// Enumerates every compiled step in every Kokoro segment with:
///   - Step index within the segment
///   - Step type (Dispatch, NativeOp, Passthrough, etc.)
///   - Detail (kernel name, NativeOp variant, op name)
///   - Estimated Metal kernel launches for that step
///
/// Prints summary tables for human analysis and asserts structural
/// invariants (total dispatches match, no unknown step types).
///
/// Part of #4252.
#[test]
fn dispatch_audit_full_breakdown() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments (cold path JIT).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // -- Per-segment step audit --
    let audit = kokoro.per_segment_step_audit();

    let mut grand_total_dispatches = 0usize;
    let mut grand_total_metal = 0usize;
    let mut grand_total_steps = 0usize;
    let mut global_type_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut global_native_op_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut global_ir_kernel_counts: BTreeMap<String, usize> = BTreeMap::new();

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  KOKORO DISPATCH AUDIT -- Full Per-Step Breakdown (#4252)");
    eprintln!("{}\n", "=".repeat(80));

    for (seg_name, steps, dispatches, metal_dispatches) in &audit {
        grand_total_dispatches += dispatches;
        grand_total_metal += metal_dispatches;
        grand_total_steps += steps.len();

        eprintln!(
            "--- [{seg_name}] {dispatches} dispatches, \
             {metal_dispatches} Metal launches, {} total steps ---",
            steps.len(),
        );
        eprintln!(
            "  {:>4}  {:<14}  {:>5}  Detail",
            "Step", "Type", "Metal"
        );
        eprintln!("  {}", "-".repeat(70));

        for (idx, step_type, detail, metal) in steps {
            let metal_str = if *metal > 0 {
                format!("{metal}")
            } else {
                "-".to_string()
            };
            eprintln!(
                "  {idx:>4}  {step_type:<14}  {metal_str:>5}  {detail}",
            );

            // Accumulate global counts.
            *global_type_counts.entry(step_type).or_insert(0) += 1;
            if *step_type == "NativeOp" {
                *global_native_op_counts.entry(detail.clone()).or_insert(0) += 1;
            }
            if *step_type == "Dispatch" {
                *global_ir_kernel_counts.entry(detail.clone()).or_insert(0) += 1;
            }
        }
        eprintln!();
    }

    // -- Summary table: dispatches by segment --
    let ds = kokoro.dispatch_summary();
    eprintln!("--- Dispatch Summary by Segment ---");
    eprintln!(
        "  {:>14}  {:>10}  {:>12}",
        "Segment", "Dispatches", "Metal Est."
    );
    eprintln!("  {}", "-".repeat(40));

    let segment_summary = [
        ("plbert", ds.plbert),
        ("text_encoder", ds.text_encoder),
        ("prosody", ds.prosody),
        ("f0_energy", ds.f0_energy),
        ("generator", ds.generator),
        ("regulate", ds.regulate),
        ("sinegen_pre", ds.sinegen_pre),
        ("sinegen_post", ds.sinegen_post),
    ];
    for (name, count) in &segment_summary {
        // Find metal count from audit data.
        let metal = audit
            .iter()
            .find(|(n, _, _, _)| *n == *name)
            .map(|(_, _, _, m)| *m)
            .unwrap_or(0);
        eprintln!("  {name:>14}  {count:>10}  {metal:>12}");
    }
    eprintln!("  {}", "-".repeat(40));
    eprintln!(
        "  {:>14}  {:>10}  {:>12}",
        "TOTAL", grand_total_dispatches, grand_total_metal,
    );
    eprintln!();

    // -- Step type distribution --
    eprintln!("--- Step Type Distribution (all segments) ---");
    for (step_type, count) in &global_type_counts {
        eprintln!("  {step_type:<18} {count:>5}");
    }
    eprintln!("  {:<18} {:>5}", "TOTAL", grand_total_steps);
    eprintln!();

    // -- NativeOp variant distribution --
    if !global_native_op_counts.is_empty() {
        eprintln!("--- NativeOp Variant Distribution ---");
        let mut sorted: Vec<_> = global_native_op_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (variant, count) in &sorted {
            eprintln!("  {variant:<30} {count:>5}");
        }
        let total_native: usize = global_native_op_counts.values().sum();
        eprintln!("  {:<30} {:>5}", "TOTAL NativeOps", total_native);
        eprintln!();
    }

    // -- IR kernel name distribution --
    if !global_ir_kernel_counts.is_empty() {
        eprintln!("--- IR Kernel Distribution ---");
        let mut sorted: Vec<_> = global_ir_kernel_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (kernel, count) in &sorted {
            eprintln!("  {kernel:<40} {count:>5}");
        }
        let total_ir: usize = global_ir_kernel_counts.values().sum();
        eprintln!("  {:<40} {:>5}", "TOTAL IR Dispatches", total_ir);
        eprintln!();
    }

    // -- Fusion gap analysis per segment --
    eprintln!("--- Fusion Gap Analysis (per segment) ---");
    let gap_results = kokoro
        .segment_gap_analysis(&input_ids, &style, 1.0, &cache)
        .unwrap();

    let mut total_gaps = 0usize;
    let mut global_blocker_counts: BTreeMap<String, usize> = BTreeMap::new();
    for seg in &gap_results {
        eprintln!(
            "  [{:>14}] dispatches={:>3} theoretical_min={:>3} gaps={:>3}",
            seg.segment_name,
            seg.dispatch_count,
            seg.theoretical_minimum,
            seg.gap_analysis.gaps.len(),
        );
        total_gaps += seg.gap_analysis.gaps.len();
        for (blocker, count) in seg.gap_analysis.blocker_counts() {
            *global_blocker_counts.entry(blocker).or_insert(0) += count;
        }
    }
    eprintln!();

    // -- Global fusion blocker distribution --
    if !global_blocker_counts.is_empty() {
        eprintln!("--- Fusion Blocker Distribution (all segments) ---");
        let mut sorted: Vec<_> = global_blocker_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (blocker, count) in &sorted {
            eprintln!("  {blocker:<24} {count:>5}");
        }
        eprintln!("  {:<24} {:>5}", "TOTAL gaps", total_gaps);
    }
    eprintln!();
    eprintln!("{}\n", "=".repeat(80));

    // -- Assertions --

    // Total dispatches from step audit must match dispatch_summary().
    let ds_total = ds.total();
    assert_eq!(
        grand_total_dispatches, ds_total,
        "Step audit dispatches ({grand_total_dispatches}) != \
         dispatch_summary total ({ds_total}). Accounting mismatch.",
    );

    // Must have compiled at least 5 segments (plbert, text, prosody, f0, generator).
    assert!(
        audit.len() >= 5,
        "Expected at least 5 compiled segments, got {}",
        audit.len(),
    );

    // No step should be "Unknown" type.
    let unknown_count = global_type_counts.get("Unknown").copied().unwrap_or(0);
    assert_eq!(
        unknown_count, 0,
        "Found {unknown_count} steps with Unknown type -- \
         new CompiledStep variant not handled in per_segment_step_audit()",
    );

    // Dispatch + NativeOp count must match grand_total_dispatches.
    let dispatch_steps = global_type_counts.get("Dispatch").copied().unwrap_or(0);
    let native_steps = global_type_counts.get("NativeOp").copied().unwrap_or(0);
    assert_eq!(
        dispatch_steps + native_steps,
        grand_total_dispatches,
        "Dispatch ({dispatch_steps}) + NativeOp ({native_steps}) != \
         total dispatches ({grand_total_dispatches})",
    );

    // Metal dispatch estimate should be > 0 (sanity).
    assert!(
        grand_total_metal > 0,
        "Total estimated Metal dispatches is 0 -- something is wrong",
    );

    // Cross-check against total_dispatches() and total_metal_dispatches().
    assert_eq!(
        grand_total_dispatches,
        kokoro.total_dispatches(),
        "Step audit total ({grand_total_dispatches}) != \
         total_dispatches() ({})",
        kokoro.total_dispatches(),
    );
    assert_eq!(
        grand_total_metal,
        kokoro.total_metal_dispatches(),
        "Step audit metal ({grand_total_metal}) != \
         total_metal_dispatches() ({})",
        kokoro.total_metal_dispatches(),
    );
}
