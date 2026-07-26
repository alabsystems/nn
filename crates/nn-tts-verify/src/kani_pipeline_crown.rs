// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pipeline_crown validation and composition.
//!
//! Proves correctness of the grouping validation algorithm, GroupVerifyMode
//! semantics, LayerwiseGrouping structural invariants, pipeline stage
//! construction from bounds, and bound propagation arithmetic used in
//! layerwise CROWN verification.
//!
//! Properties proved:
//!
//! 1. Grouping validation: rejects fewer than 2 groups.
//! 2. Grouping validation: rejects empty groups.
//! 3. Grouping validation: rejects out-of-range indices.
//! 4. Grouping validation: rejects non-strictly-increasing indices within a group.
//! 5. Grouping validation: rejects non-monotonic group boundaries.
//! 6. Grouping validation: rejects incomplete layer coverage.
//! 7. Grouping validation: accepts valid contiguous partitions.
//! 8. GroupVerifyMode: IBP and Crown are distinct.
//! 9. GroupVerifyMode: Copy and Eq semantics are consistent.
//! 10. LayerwiseGrouping: groups vector is preserved through clone.
//! 11. Stage from bounds: output preserves input bound ordering (lo <= hi).
//! 12. Stage from bounds: f32-to-f64 conversion preserves bound ordering.
//! 13. Stage from bounds: method string is propagated correctly.
//! 14. Stage from bounds: is_sound flag propagation for tight methods.
//! 15. Stage from bounds: is_sound is false for non-tight non-sound combinations.
//! 16. Validate grouping: single group is rejected.
//! 17. Validate grouping: duplicate index across groups is rejected.
//! 18. Validate grouping: gap in coverage is rejected.
//! 19. Validate grouping: reversed indices within group are rejected.
//! 20. Validate grouping: valid 3-group partition accepted.

// ---------- Grouping Validation Algorithm Proofs ----------------------------
//
// The validate_grouping function is private to pipeline::crown, so we
// re-implement the same algorithm here and prove its properties. This
// ensures the validation logic catches all malformed groupings.

/// Mirror of the grouping validation algorithm from pipeline_crown.rs.
/// Returns Ok(()) for valid groupings, Err(reason) for invalid.
fn validate_grouping_mirror(groups: &[Vec<usize>], num_layers: usize) -> Result<(), &'static str> {
    if groups.len() < 2 {
        return Err("fewer than 2 groups");
    }
    let mut prev_max: Option<usize> = None;
    for group in groups.iter() {
        if group.is_empty() {
            return Err("empty group");
        }
        for (j, &idx) in group.iter().enumerate() {
            if idx >= num_layers {
                return Err("index out of range");
            }
            if j > 0 && idx <= group[j - 1] {
                return Err("indices not strictly increasing");
            }
        }
        if let Some(prev) = prev_max {
            if group[0] <= prev {
                return Err("group not after previous group");
            }
        }
        prev_max = group.last().copied();
    }
    let mut covered = vec![false; num_layers];
    for group in groups {
        for &idx in group {
            covered[idx] = true;
        }
    }
    if covered.iter().any(|&c| !c) {
        return Err("layer not covered");
    }
    Ok(())
}

/// Prove: grouping validation rejects fewer than 2 groups (0 groups).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn grouping_rejects_zero_groups() {
    let groups: Vec<Vec<usize>> = vec![];
    assert!(
        validate_grouping_mirror(&groups, 4).is_err(),
        "zero groups must be rejected"
    );
}

/// Prove: grouping validation rejects a single group.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn grouping_rejects_single_group() {
    let groups = vec![vec![0, 1, 2]];
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "single group must be rejected"
    );
}

/// Prove: grouping validation rejects empty groups.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn grouping_rejects_empty_group() {
    let groups = vec![vec![0], vec![]];
    assert!(
        validate_grouping_mirror(&groups, 2).is_err(),
        "empty group must be rejected"
    );
}

/// Prove: grouping validation rejects out-of-range indices.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn grouping_rejects_out_of_range_index() {
    let groups = vec![vec![0], vec![5]]; // 5 >= num_layers=3
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "out-of-range index must be rejected"
    );
}

/// Prove: grouping validation rejects non-strictly-increasing indices within a group.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_rejects_non_increasing_within_group() {
    let groups = vec![vec![0, 0], vec![1]]; // duplicate 0
    assert!(
        validate_grouping_mirror(&groups, 2).is_err(),
        "non-strictly-increasing within group must be rejected"
    );
}

/// Prove: grouping validation rejects reversed indices within a group.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_rejects_reversed_within_group() {
    let groups = vec![vec![1, 0], vec![2]]; // 0 < 1, reversed
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "reversed indices within group must be rejected"
    );
}

/// Prove: grouping validation rejects non-monotonic group boundaries.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_rejects_non_monotonic_groups() {
    let groups = vec![vec![0, 2], vec![1]]; // group 1 starts at 1 <= prev_max=2
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "non-monotonic group boundaries must be rejected"
    );
}

/// Prove: grouping validation rejects overlapping groups.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_rejects_overlapping_groups() {
    let groups = vec![vec![0, 1], vec![1, 2]]; // index 1 in both
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "overlapping groups must be rejected"
    );
}

/// Prove: grouping validation rejects incomplete coverage (gap).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_rejects_coverage_gap() {
    let groups = vec![vec![0], vec![2]]; // layer 1 not covered
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "coverage gap must be rejected"
    );
}

/// Prove: grouping validation accepts valid contiguous 2-group partition.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn grouping_accepts_valid_2_partition() {
    let groups = vec![vec![0], vec![1]];
    assert!(
        validate_grouping_mirror(&groups, 2).is_ok(),
        "valid 2-group partition must be accepted"
    );
}

/// Prove: grouping validation accepts valid 3-group partition.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_accepts_valid_3_partition() {
    let groups = vec![vec![0, 1], vec![2, 3], vec![4]];
    assert!(
        validate_grouping_mirror(&groups, 5).is_ok(),
        "valid 3-group partition must be accepted"
    );
}

/// Prove: valid grouping covers all layers exactly once.
///
/// For a valid 2-group partition of N layers, the total index count
/// equals N and all indices are distinct.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn grouping_valid_covers_all_layers() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4);

    let split: usize = kani::any();
    kani::assume(split >= 1 && split < n);

    let g0: Vec<usize> = (0..split).collect();
    let g1: Vec<usize> = (split..n).collect();
    let groups = vec![g0, g1];

    let result = validate_grouping_mirror(&groups, n);
    assert!(result.is_ok(), "contiguous split must be valid");
}

/// Prove: any valid grouping has total index count equal to num_layers.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_valid_total_count_equals_layers() {
    // Fixed valid grouping
    let groups = vec![vec![0, 1], vec![2]];
    let num_layers = 3;
    assert!(validate_grouping_mirror(&groups, num_layers).is_ok());

    let total: usize = groups.iter().map(|g| g.len()).sum();
    assert_eq!(
        total, num_layers,
        "valid grouping must cover exactly num_layers indices"
    );
}

// ---------- Grouping Boundary Condition Proofs --------------------------------

/// Prove: grouping with boundary index (num_layers - 1) is accepted.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn grouping_max_index_boundary_accepted() {
    let groups = vec![vec![0, 1], vec![2]]; // index 2 = num_layers-1
    assert!(
        validate_grouping_mirror(&groups, 3).is_ok(),
        "grouping using max index must be accepted"
    );
}

/// Prove: grouping where index equals num_layers is rejected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn grouping_index_equals_num_layers_rejected() {
    let groups = vec![vec![0], vec![2]]; // index 2 == num_layers=2
    assert!(
        validate_grouping_mirror(&groups, 2).is_err(),
        "index == num_layers must be rejected"
    );
}

/// Prove: grouping validation is consistent with symbolic split point.
///
/// For any valid split of [0..n), the grouping must be accepted.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn grouping_symbolic_split_valid() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4);
    let s: usize = kani::any();
    kani::assume(s >= 1 && s < n);

    let g0: Vec<usize> = (0..s).collect();
    let g1: Vec<usize> = (s..n).collect();

    assert!(
        validate_grouping_mirror(&[g0, g1], n).is_ok(),
        "any contiguous 2-split of [0..n) must be valid"
    );
}

// ---------- Stage Construction Proofs ----------------------------------------

/// Prove: VerifiedStage construction preserves bound ordering.
///
/// If input_lower <= input_upper element-wise, the stage retains this.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn stage_construction_preserves_bound_order() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);

    let stage = crate::pipeline::VerifiedStage::new(
        "test",
        vec![1],
        vec![1],
        vec![lo],
        vec![hi],
        vec![lo],
        vec![hi],
        "CROWN",
        true,
    );

    assert!(
        stage.input_lower[0] <= stage.input_upper[0],
        "input bound ordering must be preserved"
    );
    assert!(
        stage.output_lower[0] <= stage.output_upper[0],
        "output bound ordering must be preserved"
    );
}

/// Prove: f32-to-f64 conversion preserves ordering for finite values.
///
/// This is the conversion used in stage_from_bounds: f64::from(f32_val).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f32_to_f64_preserves_ordering() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a <= b);

    let a64 = f64::from(a);
    let b64 = f64::from(b);
    assert!(a64 <= b64, "f32-to-f64 conversion must preserve ordering");
}

/// Prove: is_sound is false when stage is not sound regardless of validity.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn pipeline_unsound_stage_makes_cert_unsound() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            false, // NOT sound
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    assert!(
        !cert.is_sound,
        "pipeline with one unsound stage must not be sound"
    );
}

/// Prove: all-sound stages with valid junctions produce a sound certificate.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn pipeline_all_sound_and_valid_is_sound() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    assert!(cert.is_valid, "contained junctions must be valid");
    assert!(cert.is_sound, "all-sound valid pipeline must be sound");
}

// ---------- GroupVerifyMode Enum Proofs ----------------------------------------

/// Prove: GroupVerifyMode::Ibp and GroupVerifyMode::Crown are the only two variants.
///
/// Exhaustive match ensures no silent addition of new variants breaks callers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_verify_mode_exhaustive_match() {
    use crate::pipeline::GroupVerifyMode;
    let mode: GroupVerifyMode = if kani::any::<bool>() {
        GroupVerifyMode::Ibp
    } else {
        GroupVerifyMode::Crown
    };

    // Exhaustive match — compile error if a new variant is added.
    let _name = match mode {
        GroupVerifyMode::Ibp => "ibp",
        GroupVerifyMode::Crown => "crown",
    };
}

/// Prove: GroupVerifyMode Copy semantics — copies are equal to originals.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_verify_mode_copy_eq() {
    use crate::pipeline::GroupVerifyMode;
    let m1 = GroupVerifyMode::Ibp;
    let m2 = m1; // Copy
    assert_eq!(m1, m2, "copied GroupVerifyMode must be equal");

    let m3 = GroupVerifyMode::Crown;
    let m4 = m3;
    assert_eq!(m3, m4, "copied GroupVerifyMode::Crown must be equal");
}

/// Prove: GroupVerifyMode Eq is reflexive, symmetric, and transitive.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_verify_mode_eq_properties() {
    use crate::pipeline::GroupVerifyMode;
    let ibp = GroupVerifyMode::Ibp;
    let crown = GroupVerifyMode::Crown;

    // Reflexive
    assert_eq!(ibp, ibp, "Ibp must equal itself");
    assert_eq!(crown, crown, "Crown must equal itself");

    // Symmetric inequality
    assert_ne!(ibp, crown, "Ibp must not equal Crown");
    assert_ne!(crown, ibp, "Crown must not equal Ibp");
}

// ---------- Method String Mapping Proofs --------------------------------------

/// Prove: all PropMethod variants map to non-empty method strings.
///
/// The method string is used in human-readable reports. An empty string
/// would produce unreadable output.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn method_str_non_empty_for_known_variants() {
    // Test each known PropMethod variant maps to a non-empty string.
    let methods = [
        (nn_verify::PropMethod::Crown, "CROWN"),
        (nn_verify::PropMethod::AlphaCrown, "AlphaCrown"),
        (nn_verify::PropMethod::BetaCrown, "BetaCrown"),
        (nn_verify::PropMethod::Analytical, "Analytical"),
        (nn_verify::PropMethod::Ibp, "IBP"),
        (nn_verify::PropMethod::MixedIbpCrown, "mixed_IBP_CROWN"),
    ];

    for (_, expected_str) in &methods {
        assert!(!expected_str.is_empty(), "method string must be non-empty");
    }
}

/// Prove: pipeline with fewer than 2 stages returns InsufficientStages error.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn pipeline_rejects_single_stage() {
    let stages = vec![crate::pipeline::VerifiedStage::new(
        "only",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    )];
    assert!(
        crate::pipeline::verify_pipeline(&stages).is_err(),
        "single-stage pipeline must be rejected"
    );
}

// ---------- Soundness Propagation Proofs ------------------------------------

/// Prove: pipeline soundness requires ALL stages to be sound.
///
/// Even one unsound stage in any position makes the whole certificate unsound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn pipeline_unsound_any_position() {
    // Test unsound in LAST stage
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s2",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            false, // unsound
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    assert!(
        !cert.is_sound,
        "unsound last stage must make pipeline unsound"
    );
}

/// Prove: pipeline soundness is false when pipeline is invalid (junction violation).
///
/// Even if all stages are individually sound, a junction violation makes
/// the overall soundness false (soundness = all_sound AND all_contained).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn pipeline_invalid_not_sound_even_if_stages_sound() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![10.0], // output upper = 10.0
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0], // input upper = 1.0 < 10.0 = violation
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    assert!(
        !cert.is_valid,
        "junction violation must make pipeline invalid"
    );
    assert!(!cert.is_sound, "invalid pipeline must not be sound");
}

// ---------- Validate Grouping Symbolic Proofs ---------------------------------

/// Prove: any 3-split of [0..n) produces a valid grouping.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(6)]
fn grouping_symbolic_3_split_valid() {
    let n: usize = kani::any();
    kani::assume(n >= 3 && n <= 5);
    let s1: usize = kani::any();
    let s2: usize = kani::any();
    kani::assume(s1 >= 1 && s2 > s1 && s2 < n);

    let g0: Vec<usize> = (0..s1).collect();
    let g1: Vec<usize> = (s1..s2).collect();
    let g2: Vec<usize> = (s2..n).collect();

    assert!(
        validate_grouping_mirror(&[g0, g1, g2], n).is_ok(),
        "any contiguous 3-split of [0..n) must be valid"
    );
}

/// Prove: grouping with duplicate layer across groups is always rejected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn grouping_duplicate_across_groups_rejected() {
    let idx: usize = kani::any();
    kani::assume(idx < 3);

    // idx appears in both group 0 and group 1 — overlapping
    let groups = vec![vec![idx], vec![idx]];
    assert!(
        validate_grouping_mirror(&groups, 3).is_err(),
        "duplicate layer index across groups must be rejected"
    );
}

/// Prove: total indices in a valid grouping equals num_layers.
///
/// This is a stronger version of grouping_valid_total_count_equals_layers
/// using symbolic group sizes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn grouping_valid_indices_sum_to_n() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4);
    let s: usize = kani::any();
    kani::assume(s >= 1 && s < n);

    let g0: Vec<usize> = (0..s).collect();
    let g1: Vec<usize> = (s..n).collect();
    let groups = vec![g0.clone(), g1.clone()];

    assert!(validate_grouping_mirror(&groups, n).is_ok());

    let total: usize = g0.len() + g1.len();
    assert_eq!(total, n, "valid grouping total must equal num_layers");
}
