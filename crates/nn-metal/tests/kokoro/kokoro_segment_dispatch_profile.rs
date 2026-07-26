// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-segment dispatch count profiling for Kokoro optimization.
//!
//! Prints an actionable report showing WHERE dispatch counts come from
//! so developers know what to optimize next. Covers:
//!
//!   1. Per-segment dispatch breakdown (NativeOp vs IR Dispatch vs zero-cost)
//!   2. Top-10 heaviest kernels by estimated Metal launches
//!   3. IdentityPassthrough / zero-cost step counts (fused-away steps)
//!   4. Adjacent Dispatch pair fusion opportunities
//!
//! This is a diagnostic test -- it prints a report, not a hard gate.
//! Run: `cargo test -p nn-metal --test kokoro_all kokoro_segment_dispatch_profile -- --nocapture`

use std::collections::BTreeMap;

/// Detailed per-segment dispatch profiling report.
///
/// Builds a miniaturized Kokoro, compiles all 8 segments, then walks
/// every compiled step to produce an actionable optimization report.
#[test]
fn segment_dispatch_profile() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments (cold-path JIT).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Collect per-segment step audit data.
    let audit = kokoro.per_segment_step_audit();
    let ds = kokoro.dispatch_summary();

    // -- Segment name → dispatch summary field mapping --
    let ds_fields = [
        ("plbert", ds.plbert),
        ("text_encoder", ds.text_encoder),
        ("prosody", ds.prosody),
        ("f0_energy", ds.f0_energy),
        ("generator", ds.generator),
        ("regulate", ds.regulate),
        ("sinegen_pre", ds.sinegen_pre),
        ("sinegen_post", ds.sinegen_post),
    ];
    let ds_map: BTreeMap<&str, usize> = ds_fields.iter().copied().collect();

    // =====================================================================
    // Section 1: Per-segment dispatch breakdown
    // =====================================================================
    eprintln!("\n{}", "=".repeat(90));
    eprintln!("  KOKORO SEGMENT DISPATCH PROFILE -- Optimization Target Report");
    eprintln!("{}\n", "=".repeat(90));

    eprintln!("--- 1. Per-Segment Dispatch Breakdown ---\n");
    eprintln!(
        "  {:<14} {:>6} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "Segment", "Total", "NativeOp", "IR Disp", "Metal Est", "ZeroCost", "% of All"
    );
    eprintln!("  {}", "-".repeat(78));

    let mut grand_total = 0usize;
    let mut grand_native = 0usize;
    let mut grand_ir = 0usize;
    let mut grand_metal = 0usize;
    let mut grand_zero_cost = 0usize;
    let mut grand_steps = 0usize;

    // Collect per-kernel metal counts for top-10 and fusion analysis.
    let mut kernel_metal_counts: BTreeMap<String, usize> = BTreeMap::new();
    // Collect per-segment adjacent dispatch pairs for fusion analysis.
    let mut adjacent_dispatch_pairs: Vec<(String, String, String)> = Vec::new();

    for (seg_name, steps, dispatches, metal_dispatches) in &audit {
        let mut native_count = 0usize;
        let mut ir_count = 0usize;
        let mut zero_cost_count = 0usize;

        // Track previous dispatch-like step for fusion opportunity detection.
        let mut prev_dispatch: Option<(&str, &str)> = None; // (step_type, detail)

        for (_, step_type, detail, metal) in steps {
            match *step_type {
                "NativeOp" => {
                    native_count += 1;
                    *kernel_metal_counts
                        .entry(format!("[NativeOp] {detail}"))
                        .or_insert(0) += metal;
                    // Check if previous step was also a dispatch-like step.
                    if let Some((prev_type, prev_detail)) = prev_dispatch {
                        adjacent_dispatch_pairs.push((
                            seg_name.clone(),
                            format!("[{prev_type}] {prev_detail}"),
                            format!("[NativeOp] {detail}"),
                        ));
                    }
                    prev_dispatch = Some(("NativeOp", detail));
                }
                "Dispatch" => {
                    ir_count += 1;
                    *kernel_metal_counts
                        .entry(format!("[IR] {detail}"))
                        .or_insert(0) += metal;
                    if let Some((prev_type, prev_detail)) = prev_dispatch {
                        adjacent_dispatch_pairs.push((
                            seg_name.clone(),
                            format!("[{prev_type}] {prev_detail}"),
                            format!("[IR] {detail}"),
                        ));
                    }
                    prev_dispatch = Some(("Dispatch", detail));
                }
                "RuntimeOp" => {
                    *kernel_metal_counts
                        .entry(format!("[RuntimeOp] {detail}"))
                        .or_insert(0) += metal;
                    if let Some((prev_type, prev_detail)) = prev_dispatch {
                        adjacent_dispatch_pairs.push((
                            seg_name.clone(),
                            format!("[{prev_type}] {prev_detail}"),
                            format!("[RuntimeOp] {detail}"),
                        ));
                    }
                    prev_dispatch = Some(("RuntimeOp", detail));
                }
                _ => {
                    // Passthrough, NarrowView, InputForward, IdentityPass,
                    // ConstantValue -- zero-cost (no GPU work).
                    zero_cost_count += 1;
                    prev_dispatch = None; // Break adjacency chain.
                }
            }
        }

        let pct = if ds.total() > 0 {
            *dispatches as f64 / ds.total() as f64 * 100.0
        } else {
            0.0
        };

        eprintln!(
            "  {seg_name:<14} {dispatches:>6} {native_count:>8} {ir_count:>10} {metal_dispatches:>10} {zero_cost_count:>10} {pct:>7.1}%",
        );

        grand_total += dispatches;
        grand_native += native_count;
        grand_ir += ir_count;
        grand_metal += metal_dispatches;
        grand_zero_cost += zero_cost_count;
        grand_steps += steps.len();
    }

    eprintln!("  {}", "-".repeat(78));
    eprintln!(
        "  {:<14} {:>6} {:>8} {:>10} {:>10} {:>10} {:>7.1}%",
        "TOTAL", grand_total, grand_native, grand_ir, grand_metal, grand_zero_cost, 100.0,
    );
    eprintln!(
        "\n  Total compiled steps: {grand_steps}  (dispatch-like: {grand_total}, zero-cost: {grand_zero_cost})",
    );
    eprintln!(
        "  Dispatch target: < 60  |  Current: {}  |  Gap: {}",
        grand_total,
        grand_total.saturating_sub(60),
    );
    eprintln!();

    // =====================================================================
    // Section 2: Top-10 heaviest kernels by Metal launch count
    // =====================================================================
    eprintln!("--- 2. Top-10 Heaviest Kernels (by estimated Metal launches) ---\n");
    let mut sorted_kernels: Vec<_> = kernel_metal_counts.iter().collect();
    sorted_kernels.sort_by(|a, b| b.1.cmp(a.1));

    eprintln!("  {:>4}  {:<50} {:>8}", "Rank", "Kernel", "Metal");
    eprintln!("  {}", "-".repeat(66));
    for (i, (kernel, metal)) in sorted_kernels.iter().take(10).enumerate() {
        eprintln!("  {:>4}  {:<50} {:>8}", i + 1, kernel, metal);
    }
    if sorted_kernels.len() > 10 {
        let rest_metal: usize = sorted_kernels.iter().skip(10).map(|(_, m)| **m).sum();
        eprintln!(
            "  {:>4}  {:<50} {:>8}",
            "",
            format!("... ({} more kernels)", sorted_kernels.len() - 10),
            rest_metal,
        );
    }
    eprintln!("  {}", "-".repeat(66));
    eprintln!(
        "  {:>4}  {:<50} {:>8}",
        "", "TOTAL Metal launches", grand_metal,
    );
    eprintln!();

    // =====================================================================
    // Section 3: Zero-cost step breakdown (fused-away / passthrough)
    // =====================================================================
    eprintln!("--- 3. Zero-Cost Steps (IdentityPassthrough, NarrowView, etc.) ---\n");

    let mut type_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, steps, _, _) in &audit {
        for (_, step_type, _, _) in steps {
            *type_counts.entry(step_type).or_insert(0) += 1;
        }
    }

    let zero_cost_types = [
        "IdentityPass",
        "Passthrough",
        "NarrowView",
        "InputForward",
        "ConstantValue",
    ];

    eprintln!("  {:<20} {:>6}", "Step Type", "Count");
    eprintln!("  {}", "-".repeat(28));
    let mut total_zero = 0usize;
    for t in &zero_cost_types {
        let count = type_counts.get(t).copied().unwrap_or(0);
        if count > 0 {
            eprintln!("  {t:<20} {count:>6}");
        }
        total_zero += count;
    }
    eprintln!("  {}", "-".repeat(28));
    eprintln!("  {:<20} {:>6}", "Total zero-cost", total_zero);
    eprintln!(
        "  {:<20} {:>6}",
        "Total dispatch-like",
        grand_steps - total_zero,
    );
    eprintln!(
        "\n  Efficiency: {:.1}% of compiled steps are zero-cost (fused away)",
        if grand_steps > 0 {
            total_zero as f64 / grand_steps as f64 * 100.0
        } else {
            0.0
        },
    );
    eprintln!();

    // =====================================================================
    // Section 4: Adjacent dispatch pair fusion opportunities
    // =====================================================================
    eprintln!("--- 4. Fusion Opportunities: Adjacent Dispatch Pairs ---\n");
    eprintln!(
        "  Adjacent dispatch pairs that could potentially be fused into\n  \
         a single kernel launch. Sorted by frequency.\n"
    );

    // Deduplicate pairs and count frequency.
    let mut pair_freq: BTreeMap<(String, String), (usize, Vec<String>)> = BTreeMap::new();
    for (seg, a, b) in &adjacent_dispatch_pairs {
        let entry = pair_freq
            .entry((a.clone(), b.clone()))
            .or_insert_with(|| (0, Vec::new()));
        entry.0 += 1;
        if !entry.1.contains(seg) {
            entry.1.push(seg.clone());
        }
    }

    let mut sorted_pairs: Vec<_> = pair_freq.into_iter().collect();
    sorted_pairs.sort_by_key(|x| std::cmp::Reverse(x.1 .0));

    eprintln!(
        "  {:>4}  {:<35} -> {:<35} {:>5}  Segments",
        "Freq", "Step A", "Step B", "Segs"
    );
    eprintln!("  {}", "-".repeat(100));

    let display_limit = 15;
    for ((a, b), (freq, segs)) in sorted_pairs.iter().take(display_limit) {
        let seg_list = segs.join(", ");
        // Truncate kernel names for display.
        let a_short = if a.len() > 35 { &a[..35] } else { a };
        let b_short = if b.len() > 35 { &b[..35] } else { b };
        eprintln!(
            "  {:>4}  {:<35} -> {:<35} {:>5}  {}",
            freq,
            a_short,
            b_short,
            segs.len(),
            seg_list,
        );
    }
    if sorted_pairs.len() > display_limit {
        eprintln!(
            "  ... ({} more pairs not shown)",
            sorted_pairs.len() - display_limit,
        );
    }
    eprintln!(
        "\n  Total adjacent dispatch pairs: {}  (unique patterns: {})",
        adjacent_dispatch_pairs.len(),
        sorted_pairs.len(),
    );
    eprintln!();

    // =====================================================================
    // Section 5: Per-segment NativeOp vs IR Dispatch ratio
    // =====================================================================
    eprintln!("--- 5. Per-Segment NativeOp vs IR Dispatch Ratio ---\n");
    eprintln!(
        "  {:<14} {:>8} {:>10} {:>10}",
        "Segment", "NativeOp", "IR Disp", "Ratio"
    );
    eprintln!("  {}", "-".repeat(46));

    for (seg_name, steps, _, _) in &audit {
        let mut native = 0usize;
        let mut ir = 0usize;
        for (_, step_type, _, _) in steps {
            match *step_type {
                "NativeOp" => native += 1,
                "Dispatch" => ir += 1,
                _ => {}
            }
        }
        let ratio_str = if native + ir > 0 {
            format!(
                "{:.0}% / {:.0}%",
                native as f64 / (native + ir) as f64 * 100.0,
                ir as f64 / (native + ir) as f64 * 100.0,
            )
        } else {
            "- / -".to_string()
        };
        eprintln!(
            "  {seg_name:<14} {native:>8} {ir:>10} {ratio_str:>10}",
        );
    }
    eprintln!();

    // =====================================================================
    // Summary: actionable optimization targets
    // =====================================================================
    eprintln!("--- OPTIMIZATION TARGETS ---\n");

    // Identify the heaviest segment.
    let mut seg_dispatch_list: Vec<(&str, usize)> =
        ds_map.iter().map(|(name, count)| (*name, *count)).collect();
    seg_dispatch_list.sort_by_key(|x| std::cmp::Reverse(x.1));

    eprintln!("  Heaviest segments (by logical dispatch count):");
    for (name, count) in seg_dispatch_list.iter().take(3) {
        let pct = if ds.total() > 0 {
            *count as f64 / ds.total() as f64 * 100.0
        } else {
            0.0
        };
        eprintln!("    {name}: {count} dispatches ({pct:.1}% of total)");
    }
    eprintln!();

    // Heaviest single kernel.
    if let Some((kernel, metal)) = sorted_kernels.first() {
        eprintln!(
            "  Heaviest kernel: {} ({} Metal launches, {:.1}% of all Metal)",
            kernel,
            metal,
            if grand_metal > 0 {
                **metal as f64 / grand_metal as f64 * 100.0
            } else {
                0.0
            },
        );
    }

    // Most frequent fusion pair.
    if let Some(((a, b), (freq, _))) = sorted_pairs.first() {
        eprintln!(
            "  Most frequent fusion opportunity: {a} -> {b} ({freq} occurrences)",
        );
    }

    eprintln!(
        "\n  To reach dispatch target (<60), need to eliminate {} dispatches.",
        grand_total.saturating_sub(60),
    );
    eprintln!("  Focus areas:");
    eprintln!("    1. Fuse adjacent kernel pairs in heaviest segments");
    eprintln!("    2. Replace IR Dispatch steps with fused NativeOps");
    eprintln!("    3. Partition compiler: merge compatible dispatch groups");
    eprintln!("{}\n", "=".repeat(90));

    // -- Assertions (structural sanity, not hard gates) --

    // Verify we got all 8 segments.
    assert!(
        audit.len() >= 5,
        "Expected at least 5 compiled segments, got {}",
        audit.len(),
    );

    // Verify dispatch total matches dispatch_summary.
    assert_eq!(
        grand_total,
        ds.total(),
        "Step audit dispatches ({grand_total}) != dispatch_summary total ({})",
        ds.total(),
    );

    // NativeOp + IR should equal total dispatches.
    assert_eq!(
        grand_native + grand_ir,
        grand_total,
        "NativeOp ({grand_native}) + IR ({grand_ir}) != total ({grand_total})",
    );

    // Metal estimate should be > 0.
    assert!(
        grand_metal > 0,
        "Total estimated Metal launches is 0 -- something is wrong",
    );
}
