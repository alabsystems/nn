// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generator segment per-dispatch-type census.
//!
//! Walks every compiled step of the generator segment, categorizes each
//! dispatch by operation type, reports shapes, and identifies consecutive
//! dispatch pairs (fusion candidates). This data drives which NativeOp
//! fusions to implement next.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all kokoro_generator_census -- --nocapture`
//!
//! Part of #4264.

use std::collections::BTreeMap;

/// Generator dispatch census: every step categorized with shapes.
///
/// Walks the generator segment's compiled plan step by step:
/// - Categorizes each dispatch (NativeOp, IR kernel, etc.)
/// - Extracts input/output shapes
/// - Groups by category with counts and percentages
/// - Detects consecutive dispatch pairs for fusion targeting
///
/// Part of #4264.
#[test]
fn generator_dispatch_census() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Get the generator segment's compiled steps directly.
    let audit = kokoro.per_segment_step_audit();
    let generator_audit = audit
        .iter()
        .find(|(name, _, _, _)| name == "generator")
        .expect("generator segment must be compiled");

    let (_seg_name, step_infos, dispatches, metal_dispatches) = generator_audit;

    // Access the actual CompiledStep objects via dispatch_census for shape data.
    let census = kokoro.dispatch_census();
    let gen_census = census
        .segments
        .iter()
        .find(|s| s.name == "generator")
        .expect("generator in census");

    eprintln!("\n========================================================================");
    eprintln!(
        "GENERATOR DISPATCH CENSUS ({dispatches} dispatches, {metal_dispatches} Metal launches)"
    );
    eprintln!("========================================================================\n");

    // === Section 1: Per-step listing ===
    eprintln!(
        "--- Per-Step Listing ({} total steps, {} dispatches) ---\n",
        step_infos.len(),
        dispatches
    );
    eprintln!(
        "  {:<5} {:<16} {:<30} {:>6}",
        "Step", "Type", "Detail", "Metal"
    );
    eprintln!("  {:-<5} {:-<16} {:-<30} {:->6}", "", "", "", "");

    for (idx, step_type, detail, metal) in step_infos {
        let marker = match *step_type {
            "NativeOp" | "Dispatch" | "RuntimeOp" => "*",
            _ => " ",
        };
        eprintln!(
            " {marker}{idx:<5} {step_type:<16} {detail:<30} {metal:>6}",
        );
    }

    // === Section 2: Grouped category counts ===
    // Re-derive categories from step_infos (which is already available).
    let mut category_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut category_metal: BTreeMap<String, usize> = BTreeMap::new();
    let mut dispatch_steps: Vec<(&str, String)> = Vec::new(); // (category, detail) for consecutive analysis

    for (_, step_type, detail, metal) in step_infos {
        match *step_type {
            "NativeOp" => {
                let cat = detail.clone();
                *category_counts.entry(cat.clone()).or_insert(0) += 1;
                *category_metal.entry(cat.clone()).or_insert(0) += *metal;
                dispatch_steps.push(("NativeOp", detail.clone()));
            }
            "Dispatch" => {
                let cat = format!("[IR] {detail}");
                *category_counts.entry(cat.clone()).or_insert(0) += 1;
                *category_metal.entry(cat.clone()).or_insert(0) += *metal;
                dispatch_steps.push(("Dispatch", detail.clone()));
            }
            "RuntimeOp" => {
                let cat = "[RuntimeOp]".to_string();
                *category_counts.entry(cat.clone()).or_insert(0) += 1;
                *category_metal.entry(cat.clone()).or_insert(0) += *metal;
                dispatch_steps.push(("RuntimeOp", detail.clone()));
            }
            _ => {
                // Zero-cost step: no dispatch, breaks consecutive chain.
            }
        }
    }

    let total_dispatch: usize = category_counts.values().sum();
    let total_metal_sum: usize = category_metal.values().sum();

    eprintln!(
        "\n--- Category Summary ({total_dispatch} dispatches, {total_metal_sum} Metal) ---\n"
    );
    eprintln!(
        "  {:<35} {:>5} {:>7} {:>7}",
        "Category", "Count", "  %  ", "Metal"
    );
    eprintln!("  {:-<35} {:->5} {:->7} {:->7}", "", "", "", "");

    // Sort by count descending.
    let mut sorted_cats: Vec<(String, usize)> = category_counts.into_iter().collect();
    sorted_cats.sort_by_key(|x| std::cmp::Reverse(x.1));

    for (cat, count) in &sorted_cats {
        let pct = *count as f64 / total_dispatch.max(1) as f64 * 100.0;
        let metal = category_metal.get(cat).copied().unwrap_or(0);
        let already_fused = cat.contains("Fused")
            || cat.contains("Batched")
            || cat.contains("NormActiv")
            || cat == "FusedResBlock"
            || cat == "FusedResBlockChain";
        let tag = if already_fused { " (fused)" } else { "" };
        eprintln!(
            "  {cat:<35} {count:>5} {pct:>6.1}% {metal:>7}{tag}",
        );
    }

    // === Section 3: Consecutive dispatch pairs (fusion candidates) ===
    // Walk dispatch_steps and count consecutive (A, B) pairs.
    let mut pair_counts: BTreeMap<(String, String), usize> = BTreeMap::new();

    // We need to walk ALL steps (including zero-cost) to detect true adjacency.
    // Rebuild from step_infos with zero-cost steps as breaks.
    let mut prev_dispatch: Option<String> = None;
    for (_, step_type, detail, _) in step_infos {
        match *step_type {
            "NativeOp" | "Dispatch" | "RuntimeOp" => {
                let label = if *step_type == "NativeOp" {
                    detail.clone()
                } else if *step_type == "Dispatch" {
                    format!("[IR]{detail}")
                } else {
                    "[RuntimeOp]".to_string()
                };
                if let Some(ref prev) = prev_dispatch {
                    *pair_counts
                        .entry((prev.clone(), label.clone()))
                        .or_insert(0) += 1;
                }
                prev_dispatch = Some(label);
            }
            _ => {
                // Zero-cost step does NOT break the chain for consecutive analysis.
                // Zero-cost steps (passthrough, narrow, identity, constant) don't
                // issue GPU dispatches, so adjacent dispatch steps across them are
                // still consecutive from the GPU's perspective.
            }
        }
    }

    // Sort pairs by count descending.
    let mut sorted_pairs: Vec<((String, String), usize)> = pair_counts.into_iter().collect();
    sorted_pairs.sort_by_key(|x| std::cmp::Reverse(x.1));

    eprintln!("\n--- Consecutive Dispatch Pairs (fusion candidates) ---\n");
    eprintln!(
        "  {:<30} {:<30} {:>5} {:>8}",
        "Step A", "Step B", "Count", "Saveable"
    );
    eprintln!("  {:-<30} {:-<30} {:->5} {:->8}", "", "", "", "");

    let mut total_saveable = 0usize;
    for ((a, b), count) in &sorted_pairs {
        // Each pair fusion saves 1 dispatch per instance (2 -> 1).
        let saveable = *count;
        total_saveable += saveable;
        eprintln!(
            "  {:<30} {:<30} {:>5} {:>8}",
            truncate(a, 30),
            truncate(b, 30),
            count,
            saveable,
        );
    }

    eprintln!(
        "\n  Total consecutive pairs: {}, potential dispatches saveable: {}",
        sorted_pairs.len(),
        total_saveable,
    );

    // === Section 4: NativeOp-specific detail from census ===
    eprintln!("\n--- NativeOp Variants ---\n");
    for (variant, count) in &gen_census.native_ops {
        eprintln!("  {variant}: {count}");
    }
    eprintln!("\n--- IR Dispatch Kernels ---\n");
    for (kernel, count) in &gen_census.ir_dispatches {
        eprintln!("  {kernel}: {count}");
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Generator dispatches:    {dispatches}");
    eprintln!("  Generator Metal:         {metal_dispatches}");
    eprintln!("  Zero-cost steps:         {}", gen_census.zero_cost);
    eprintln!("  Total steps:             {}", gen_census.total_steps);
    eprintln!(
        "  NativeOp count:          {}",
        gen_census.native_ops.iter().map(|(_, c)| c).sum::<usize>()
    );
    eprintln!(
        "  IR Dispatch count:       {}",
        gen_census
            .ir_dispatches
            .iter()
            .map(|(_, c)| c)
            .sum::<usize>()
    );
    eprintln!("  RuntimeOp count:         {}", gen_census.runtime_ops);
    eprintln!(
        "  Fusion candidate pairs:  {}",
        gen_census.fusion_candidates.len()
    );
    eprintln!("\n========================================================================\n");

    // === Assertions ===

    // Generator must have at least 1 dispatch.
    assert!(
        *dispatches > 0,
        "Generator segment has 0 dispatches -- trace compilation may have failed",
    );

    // Category total must match dispatch count.
    assert_eq!(
        total_dispatch, *dispatches,
        "Category total ({total_dispatch}) != generator dispatches ({dispatches})",
    );

    // Generator should have FusedResBlock or FusedResBlockChain as dominant ops.
    let has_resblock = sorted_cats.iter().any(|(cat, count)| {
        (cat == "FusedResBlock"
            || cat == "FusedResBlockChain"
            || cat == "FusedConv1dSnakeNormResBlock")
            && *count > 0
    });
    assert!(
        has_resblock,
        "Generator should have at least one FusedResBlock/Chain variant",
    );

    // Generator should have BatchedStyleProjection.
    let has_bsp = sorted_cats
        .iter()
        .any(|(cat, _)| cat == "BatchedStyleProjection");
    assert!(
        has_bsp,
        "Generator should have BatchedStyleProjection (pass 4 output)",
    );
}

/// Production generator census: requires KOKORO_WEIGHTS for D=512 model.
///
/// Same analysis as `generator_dispatch_census` but on the full production
/// model. This is the ground truth for dispatch reduction targeting.
///
/// Part of #4264.
#[test]
fn generator_dispatch_census_production() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "generator_dispatch_census_production skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("load Kokoro weights")
    };

    // 15 phoneme tokens to get a representative generator shape.
    let token_ids: Vec<i64> = (0..15).collect();
    let input_ids =
        nn_core::dyn_tensor::DynTensor::from_vec_i64(token_ids, &[1, 15], &nn_core::Device::Cpu)
            .unwrap();
    let style = nn_core::dyn_tensor::DynTensor::full(
        &[1, 256],
        0.01,
        nn_core::DType::F32,
        &nn_core::Device::Cpu,
    )
    .unwrap();

    // Warmup to compile all segments.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    let audit = kokoro.per_segment_step_audit();
    let generator_audit = audit
        .iter()
        .find(|(name, _, _, _)| name == "generator")
        .expect("generator segment must be compiled");

    let (_, step_infos, dispatches, metal_dispatches) = generator_audit;

    let census = kokoro.dispatch_census();
    let gen_census = census
        .segments
        .iter()
        .find(|s| s.name == "generator")
        .expect("generator in census");

    eprintln!("\n========================================================================");
    eprintln!(
        "PRODUCTION GENERATOR CENSUS (D=512, {dispatches} dispatches, {metal_dispatches} Metal)",
    );
    eprintln!("========================================================================\n");

    // Per-step listing.
    eprintln!(
        "--- Per-Step Listing ({} steps, {} dispatches) ---\n",
        step_infos.len(),
        dispatches,
    );
    eprintln!(
        "  {:<5} {:<16} {:<40} {:>6}",
        "Step", "Type", "Detail", "Metal"
    );
    eprintln!("  {:-<5} {:-<16} {:-<40} {:->6}", "", "", "", "");

    for (idx, step_type, detail, metal) in step_infos {
        let marker = match *step_type {
            "NativeOp" | "Dispatch" | "RuntimeOp" => "*",
            _ => " ",
        };
        eprintln!(
            " {marker}{:<5} {:<16} {:<40} {:>6}",
            idx,
            step_type,
            truncate(detail, 40),
            metal,
        );
    }

    // Category summary.
    let mut category_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut category_metal: BTreeMap<String, usize> = BTreeMap::new();
    let mut prev_dispatch: Option<String> = None;
    let mut pair_counts: BTreeMap<(String, String), usize> = BTreeMap::new();

    for (_, step_type, detail, metal) in step_infos {
        match *step_type {
            "NativeOp" => {
                *category_counts.entry(detail.clone()).or_insert(0) += 1;
                *category_metal.entry(detail.clone()).or_insert(0) += *metal;
                if let Some(ref prev) = prev_dispatch {
                    *pair_counts
                        .entry((prev.clone(), detail.clone()))
                        .or_insert(0) += 1;
                }
                prev_dispatch = Some(detail.clone());
            }
            "Dispatch" => {
                let cat = format!("[IR]{detail}");
                *category_counts.entry(cat.clone()).or_insert(0) += 1;
                *category_metal.entry(cat.clone()).or_insert(0) += *metal;
                if let Some(ref prev) = prev_dispatch {
                    *pair_counts.entry((prev.clone(), cat.clone())).or_insert(0) += 1;
                }
                prev_dispatch = Some(cat);
            }
            "RuntimeOp" => {
                let cat = "[RuntimeOp]".to_string();
                *category_counts.entry(cat.clone()).or_insert(0) += 1;
                *category_metal.entry(cat.clone()).or_insert(0) += *metal;
                if let Some(ref prev) = prev_dispatch {
                    *pair_counts.entry((prev.clone(), cat.clone())).or_insert(0) += 1;
                }
                prev_dispatch = Some(cat);
            }
            _ => {}
        }
    }

    let total_dispatch: usize = category_counts.values().sum();
    let total_metal_sum: usize = category_metal.values().sum();

    eprintln!(
        "\n--- Category Summary ({total_dispatch} dispatches, {total_metal_sum} Metal) ---\n"
    );
    eprintln!(
        "  {:<35} {:>5} {:>7} {:>7}",
        "Category", "Count", "  %  ", "Metal",
    );
    eprintln!("  {:-<35} {:->5} {:->7} {:->7}", "", "", "", "");

    let mut sorted_cats: Vec<(String, usize)> = category_counts.into_iter().collect();
    sorted_cats.sort_by_key(|x| std::cmp::Reverse(x.1));

    for (cat, count) in &sorted_cats {
        let pct = *count as f64 / total_dispatch.max(1) as f64 * 100.0;
        let metal = category_metal.get(cat).copied().unwrap_or(0);
        eprintln!(
            "  {:<35} {:>5} {:>6.1}% {:>7}",
            truncate(cat, 35),
            count,
            pct,
            metal,
        );
    }

    // Consecutive pairs.
    let mut sorted_pairs: Vec<((String, String), usize)> = pair_counts.into_iter().collect();
    sorted_pairs.sort_by_key(|x| std::cmp::Reverse(x.1));

    eprintln!("\n--- Consecutive Dispatch Pairs ---\n");
    eprintln!("  {:<30} {:<30} {:>5}", "Step A", "Step B", "Count");
    eprintln!("  {:-<30} {:-<30} {:->5}", "", "", "");

    for ((a, b), count) in sorted_pairs.iter().take(20) {
        eprintln!(
            "  {:<30} {:<30} {:>5}",
            truncate(a, 30),
            truncate(b, 30),
            count,
        );
    }

    eprintln!("\n--- NativeOp Variants ---");
    for (variant, count) in &gen_census.native_ops {
        eprintln!("  {variant}: {count}");
    }
    eprintln!("\n--- IR Dispatch Kernels ---");
    for (kernel, count) in &gen_census.ir_dispatches {
        eprintln!("  {kernel}: {count}");
    }

    eprintln!("\n--- Production Summary ---");
    eprintln!("  Generator dispatches:    {dispatches}");
    eprintln!("  Generator Metal:         {metal_dispatches}");
    eprintln!("  Zero-cost steps:         {}", gen_census.zero_cost);
    eprintln!("  Total steps:             {}", gen_census.total_steps);
    eprintln!("  Fusion candidate pairs:  {}", sorted_pairs.len());
    eprintln!();
}

/// Truncate a string to a maximum length, appending ".." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}..", &s[..max_len.saturating_sub(2)])
    }
}
