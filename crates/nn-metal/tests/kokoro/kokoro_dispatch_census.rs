// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch census tests: structured per-segment dispatch breakdown.
//!
//! Uses the `dispatch_census()` API to obtain a categorized count of
//! all compiled steps, then asserts per-segment dispatch limits.
//! Also prints an optimization-focused report showing fusion candidates.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all kokoro_dispatch_census -- --nocapture`
//!
//! Part of #4264.

/// Dispatch census: structured breakdown with per-segment limits.
///
/// Tests that `dispatch_census()` returns consistent data and that
/// per-segment dispatch counts stay within bounds. Prints an optimization
/// report identifying the heaviest segments and fusion candidates.
#[test]
fn dispatch_census_structured() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments (cold path JIT).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    let census = kokoro.dispatch_census();
    let ds = kokoro.dispatch_summary();

    // Print the full census report.
    eprintln!("\n{census}");

    // ===== Structural invariants =====

    // Census total must match dispatch_summary total.
    assert_eq!(
        census.total_dispatches,
        ds.total(),
        "Census total ({}) != dispatch_summary total ({})",
        census.total_dispatches,
        ds.total(),
    );

    // All 8 segments should be represented.
    assert!(
        census.segments.len() >= 5,
        "Expected at least 5 segments, got {}",
        census.segments.len(),
    );

    // Per-segment dispatch + zero_cost + runtime_ops should equal total_steps
    // for each segment.
    for seg in &census.segments {
        let native_count: usize = seg.native_ops.iter().map(|(_, c)| c).sum();
        let ir_count: usize = seg.ir_dispatches.iter().map(|(_, c)| c).sum();
        let dispatch_count = native_count + ir_count + seg.runtime_ops;
        assert_eq!(
            seg.dispatches, dispatch_count,
            "[{}] dispatches ({}) != native({}) + ir({}) + runtime({})",
            seg.name, seg.dispatches, native_count, ir_count, seg.runtime_ops,
        );
        assert_eq!(
            seg.total_steps,
            dispatch_count + seg.zero_cost,
            "[{}] total_steps ({}) != dispatches({}) + zero_cost({})",
            seg.name,
            seg.total_steps,
            dispatch_count,
            seg.zero_cost,
        );
    }

    // ===== Per-segment dispatch limits =====
    // These limits reflect the current optimized state. Tighten as we reduce.

    let seg_limits = [
        ("plbert", 20),
        ("text_encoder", 20),
        ("prosody", 25),
        ("f0_energy", 40),
        ("generator", 55),
        ("regulate", 10),
        ("sinegen_pre", 15),
        ("sinegen_post", 15),
    ];

    for (name, limit) in &seg_limits {
        if let Some(seg) = census.segments.iter().find(|s| s.name == *name) {
            assert!(
                seg.dispatches <= *limit,
                "[{name}] dispatch count ({}) exceeds limit ({limit}). \
                 NativeOps: {:?}, IR: {:?}",
                seg.dispatches,
                seg.native_ops,
                seg.ir_dispatches,
            );
        }
    }

    // ===== Total dispatch limit =====
    // Current: ~153. Gate at 155 (matches gate_dispatch_count).
    assert!(
        census.total_dispatches < 160,
        "Total dispatches ({}) >= 160. Gap to target: {}",
        census.total_dispatches,
        census.gap_to_target(60),
    );

    // ===== Metal dispatch estimates =====
    // Shape-aware Conv1dGemm K=3 direct path reduces Metal estimates (#4264).
    // Each K=3 Conv1dGemm saves 1 Metal dispatch (direct vs im2col+GEMM).
    let total_metal = census.total_metal_dispatches;
    eprintln!(
        "\n  Total Metal dispatches: {total_metal} (vs logical: {})",
        census.total_dispatches,
    );
    assert!(
        total_metal < 250,
        "Total Metal dispatches ({total_metal}) >= 250. \
         Expected reduction from Conv1dGemm K=3 shape-aware estimation.",
    );

    // ===== Report: heaviest segments and fusion candidates =====
    eprintln!("\n=== DISPATCH REDUCTION ANALYSIS ===\n");

    let heaviest = census.heaviest_segments();
    eprintln!("  Heaviest segments:");
    for seg in heaviest.iter().take(3) {
        eprintln!(
            "    {}: {} dispatches ({} Metal)",
            seg.name, seg.dispatches, seg.metal_dispatches
        );
        for (variant, count) in &seg.native_ops {
            if *count > 1 {
                eprintln!("      NativeOp: {variant} x{count}");
            }
        }
    }

    eprintln!(
        "\n  Total fusion candidates: {}",
        census.total_fusion_candidates()
    );
    eprintln!(
        "  Gap to target (<60): {} dispatches to eliminate",
        census.gap_to_target(60)
    );
    eprintln!();
}

/// Dispatch census consistency: census totals match other diagnostic methods.
#[test]
fn dispatch_census_consistency() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    let census = kokoro.dispatch_census();
    let total_dispatches = kokoro.total_dispatches();
    let metal_dispatches = kokoro.total_metal_dispatches();

    assert_eq!(
        census.total_dispatches, total_dispatches,
        "Census total_dispatches ({}) != total_dispatches() ({})",
        census.total_dispatches, total_dispatches,
    );
    assert_eq!(
        census.total_metal_dispatches, metal_dispatches,
        "Census total_metal ({}) != total_metal_dispatches() ({})",
        census.total_metal_dispatches, metal_dispatches,
    );
}

/// Dispatch census: generator segment breakdown for optimization targeting.
#[test]
fn dispatch_census_generator_detail() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    let census = kokoro.dispatch_census();

    if let Some(generator_seg) = census.segments.iter().find(|s| s.name == "generator") {
        eprintln!("\n=== GENERATOR SEGMENT DETAIL ===");
        eprintln!("  Dispatches: {}", generator_seg.dispatches);
        eprintln!("  Metal dispatches: {}", generator_seg.metal_dispatches);
        eprintln!("  NativeOps:");
        for (variant, count) in &generator_seg.native_ops {
            eprintln!("    {variant}: {count}");
        }
        eprintln!("  IR Dispatches:");
        for (kernel, count) in &generator_seg.ir_dispatches {
            eprintln!("    {kernel}: {count}");
        }
        eprintln!(
            "  Fusion candidates: {}",
            generator_seg.fusion_candidates.len()
        );
        for (a, b) in &generator_seg.fusion_candidates {
            eprintln!("    {a} -> {b}");
        }
        eprintln!();

        // Generator should have FusedResBlock as the dominant NativeOp.
        let resblock_count: usize = generator_seg
            .native_ops
            .iter()
            .filter(|(v, _)| v == "FusedResBlock")
            .map(|(_, c)| *c)
            .sum();
        assert!(
            resblock_count > 0,
            "Generator should have at least 1 FusedResBlock, found 0",
        );

        // BatchedStyleProjection should exist (pass 4 output).
        let bsp_count: usize = generator_seg
            .native_ops
            .iter()
            .filter(|(v, _)| v == "BatchedStyleProjection")
            .map(|(_, c)| *c)
            .sum();
        assert!(
            bsp_count > 0,
            "Generator should have BatchedStyleProjection, found 0",
        );
    }
}
