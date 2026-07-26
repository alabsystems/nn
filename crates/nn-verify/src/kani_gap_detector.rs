// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `gap_detector.rs` (#3750).
//!
//! Proves properties of:
//! - `method_is_crown`: CROWN-family string classification
//! - `classify_entry`: Per-stage verification classification logic
//! - `count_gaps_and_vacuous`: Gap report counting invariants
//! - `has_any_bounds`: StageGapResult predicate consistency

#[cfg(kani)]
mod proofs {
    use crate::gap_detector::{
        classify_entry, count_gaps_and_vacuous, kokoro_pipeline_stages, method_is_crown,
        PipelineStage, StageGapResult,
    };

    // ========================================================================
    // method_is_crown proofs
    // ========================================================================

    /// All 7 known CROWN method strings return true (exhaustive positive cases).
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_all_known_variants_return_true() {
        assert!(method_is_crown("CROWN"));
        assert!(method_is_crown("ALPHACROWN"));
        assert!(method_is_crown("ALPHA-CROWN"));
        assert!(method_is_crown("BETACROWN"));
        assert!(method_is_crown("BETA-CROWN"));
        assert!(method_is_crown("MIXED_IBP_CROWN"));
        assert!(method_is_crown("MIXED-IBP-CROWN"));
    }

    /// Case-insensitive: lowercase versions of all 7 known CROWN strings return true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_case_insensitive_lowercase() {
        assert!(method_is_crown("crown"));
        assert!(method_is_crown("alphacrown"));
        assert!(method_is_crown("alpha-crown"));
        assert!(method_is_crown("betacrown"));
        assert!(method_is_crown("beta-crown"));
        assert!(method_is_crown("mixed_ibp_crown"));
        assert!(method_is_crown("mixed-ibp-crown"));
    }

    /// Case-insensitive: mixed-case versions return true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_case_insensitive_mixed() {
        assert!(method_is_crown("Crown"));
        assert!(method_is_crown("AlphaCrown"));
        assert!(method_is_crown("Alpha-Crown"));
        assert!(method_is_crown("BetaCrown"));
        assert!(method_is_crown("Beta-Crown"));
        assert!(method_is_crown("Mixed_Ibp_Crown"));
        assert!(method_is_crown("Mixed-Ibp-Crown"));
    }

    /// Empty string returns false.
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_empty_string_is_false() {
        assert!(!method_is_crown(""));
    }

    /// Whitespace-only string returns false.
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_whitespace_only_is_false() {
        assert!(!method_is_crown("   "));
        assert!(!method_is_crown("\t"));
        assert!(!method_is_crown(" \t "));
    }

    /// Leading/trailing whitespace does not change the result — known strings
    /// with spaces still return true (trimming works).
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_whitespace_trimming() {
        assert!(method_is_crown("  CROWN  "));
        assert!(method_is_crown(" ALPHACROWN "));
        assert!(method_is_crown("\tALPHA-CROWN\t"));
        assert!(method_is_crown("  BETACROWN"));
        assert!(method_is_crown("BETA-CROWN  "));
        assert!(method_is_crown("  MIXED_IBP_CROWN  "));
        assert!(method_is_crown("  MIXED-IBP-CROWN  "));
    }

    /// Strings that are NOT any known CROWN variant return false.
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_non_crown_methods_are_false() {
        assert!(!method_is_crown("IBP"));
        assert!(!method_is_crown("ANALYTICAL"));
        assert!(!method_is_crown("ibp"));
        assert!(!method_is_crown("analytical"));
        assert!(!method_is_crown("CROWN_IBP"));
        assert!(!method_is_crown("GAMMA-CROWN"));
        assert!(!method_is_crown("ALPHABETACROWN"));
        assert!(!method_is_crown("NOTCROWN"));
        assert!(!method_is_crown("verified"));
        assert!(!method_is_crown("bounds_computed"));
    }

    /// Partial matches do NOT return true — "CROW" and "CROWNS" are not CROWN.
    #[kani::unwind(1)]
    #[kani::proof]
    fn method_is_crown_partial_matches_are_false() {
        assert!(!method_is_crown("CROW"));
        assert!(!method_is_crown("CROWNS"));
        assert!(!method_is_crown("ALPHA"));
        assert!(!method_is_crown("BETA"));
        assert!(!method_is_crown("MIXED"));
        assert!(!method_is_crown("MIXED_IBP"));
        assert!(!method_is_crown("IBP_CROWN"));
    }

    // ========================================================================
    // classify_entry proofs
    // ========================================================================

    /// Both entries invalid => has_any_bounds is false (this is a gap).
    /// Also: has_ibp, has_crown, has_analytical must all be false.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_no_valid_entries_is_gap() {
        let width: Option<f64> = if kani::any() { Some(500.0) } else { None };
        let ps: Option<&str> = if kani::any() { Some("sound") } else { None };

        let (has_ibp, has_crown, has_analytical, _is_vacuous, has_any_bounds) =
            classify_entry(false, false, "IBP", "CROWN", width, ps);

        // No valid entries => no bounds of any kind
        assert!(!has_any_bounds);
        assert!(!has_ibp);
        assert!(!has_crown);
        assert!(!has_analytical);
    }

    /// has_any_bounds == false always implies all three bound types are false.
    ///
    /// This is the key gap invariant: if classify_entry says "no bounds at all",
    /// then none of {IBP, CROWN, ANALYTICAL} can be individually true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_no_bounds_implies_no_individual_bounds() {
        let primary_valid: bool = kani::any();
        let crown_valid: bool = kani::any();

        // Pick one of the known method strings to keep Kani tractable
        let pm_idx: u8 = kani::any();
        kani::assume(pm_idx < 5);
        let primary_method = match pm_idx {
            0 => "",
            1 => "IBP",
            2 => "CROWN",
            3 => "ANALYTICAL",
            _ => "ALPHACROWN",
        };

        let cm_idx: u8 = kani::any();
        kani::assume(cm_idx < 5);
        let crown_method = match cm_idx {
            0 => "",
            1 => "IBP",
            2 => "CROWN",
            3 => "ANALYTICAL",
            _ => "BETACROWN",
        };

        let (has_ibp, has_crown, has_analytical, _is_vacuous, has_any_bounds) = classify_entry(
            primary_valid,
            crown_valid,
            primary_method,
            crown_method,
            None,
            None,
        );

        if !has_any_bounds {
            assert!(!has_ibp);
            assert!(!has_crown);
            assert!(!has_analytical);
        }
    }

    /// Primary valid with IBP method => has_ibp is true and has_any_bounds is true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_primary_ibp_has_ibp_bounds() {
        let (has_ibp, _has_crown, _has_analytical, _is_vacuous, has_any_bounds) =
            classify_entry(true, false, "IBP", "", None, None);

        assert!(has_ibp);
        assert!(has_any_bounds);
    }

    /// Primary valid with empty method (legacy) => has_ibp is true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_primary_empty_method_has_ibp_bounds() {
        let (has_ibp, _has_crown, _has_analytical, _is_vacuous, has_any_bounds) =
            classify_entry(true, false, "", "", None, None);

        assert!(has_ibp);
        assert!(has_any_bounds);
    }

    /// Crown entry valid with CROWN method => has_crown is true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_crown_entry_has_crown_bounds() {
        let (_, has_crown, _, _, has_any_bounds) =
            classify_entry(false, true, "", "CROWN", None, None);

        assert!(has_crown);
        assert!(has_any_bounds);
    }

    /// Primary valid with CROWN method (no separate _crown entry) => has_crown is true.
    /// This is the iSTFT case.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_primary_crown_method_has_crown_bounds() {
        let (_, has_crown, _, _, has_any_bounds) =
            classify_entry(true, false, "CROWN", "", None, None);

        assert!(has_crown);
        assert!(has_any_bounds);
    }

    /// ANALYTICAL method on primary => has_analytical is true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_analytical_bounds() {
        let (has_ibp, has_crown, has_analytical, _, has_any_bounds) =
            classify_entry(true, false, "ANALYTICAL", "", None, None);

        assert!(has_analytical);
        assert!(has_any_bounds);
        assert!(!has_ibp);
        assert!(!has_crown);
    }

    /// proof_strength == "vacuous" => is_vacuous is true, regardless of width.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_vacuous_label_forces_vacuous() {
        let primary_valid: bool = kani::any();
        let crown_valid: bool = kani::any();

        // Width could be small or absent — vacuous label overrides
        let (_, _, _, is_vacuous, _) = classify_entry(
            primary_valid,
            crown_valid,
            "IBP",
            "",
            Some(1.0),
            Some("vacuous"),
        );
        assert!(is_vacuous);

        let (_, _, _, is_vacuous2, _) =
            classify_entry(primary_valid, crown_valid, "IBP", "", None, Some("vacuous"));
        assert!(is_vacuous2);
    }

    /// Width > VACUOUS_WIDTH_THRESHOLD (1000.0) => is_vacuous is true.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_wide_bounds_are_vacuous() {
        let (_, _, _, is_vacuous, _) =
            classify_entry(true, false, "IBP", "", Some(1001.0), Some("sound"));
        assert!(is_vacuous);

        let (_, _, _, is_vacuous2, _) = classify_entry(true, false, "IBP", "", Some(5000.0), None);
        assert!(is_vacuous2);
    }

    /// Width <= 1000.0 and proof_strength != "vacuous" => NOT vacuous.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_narrow_non_vacuous() {
        let ps_idx: u8 = kani::any();
        kani::assume(ps_idx < 3);
        let ps = match ps_idx {
            0 => Some("sound"),
            1 => Some("heuristic"),
            _ => None,
        };

        let (_, _, _, is_vacuous, _) = classify_entry(true, false, "IBP", "", Some(999.0), ps);
        assert!(!is_vacuous);
    }

    /// No width and no vacuous label => NOT vacuous.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_no_width_no_vacuous_label_not_vacuous() {
        let primary_valid: bool = kani::any();
        let crown_valid: bool = kani::any();

        let ps_idx: u8 = kani::any();
        kani::assume(ps_idx < 3);
        let ps = match ps_idx {
            0 => Some("sound"),
            1 => Some("heuristic"),
            _ => None,
        };

        let (_, _, _, is_vacuous, _) =
            classify_entry(primary_valid, crown_valid, "IBP", "", None, ps);
        assert!(!is_vacuous);
    }

    /// IBP method in a _crown suffix entry but that entry is valid => has_ibp is true
    /// but has_crown is false. This is the F0EnergyPredictor case.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_ibp_in_crown_slot_is_ibp_not_crown() {
        let (has_ibp, has_crown, _, _, _) = classify_entry(true, true, "IBP", "IBP", None, None);

        assert!(has_ibp);
        assert!(!has_crown);
    }

    /// Key invariant: has_any_bounds is equivalent to (primary_valid || crown_valid).
    /// It does NOT depend on method strings.
    #[kani::unwind(1)]
    #[kani::proof]
    fn classify_entry_has_any_bounds_equals_any_valid() {
        let primary_valid: bool = kani::any();
        let crown_valid: bool = kani::any();

        let pm_idx: u8 = kani::any();
        kani::assume(pm_idx < 4);
        let primary_method = match pm_idx {
            0 => "",
            1 => "IBP",
            2 => "CROWN",
            _ => "ANALYTICAL",
        };

        let cm_idx: u8 = kani::any();
        kani::assume(cm_idx < 4);
        let crown_method = match cm_idx {
            0 => "",
            1 => "IBP",
            2 => "CROWN",
            _ => "ANALYTICAL",
        };

        let (_, _, _, _, has_any_bounds) = classify_entry(
            primary_valid,
            crown_valid,
            primary_method,
            crown_method,
            None,
            None,
        );

        assert_eq!(has_any_bounds, primary_valid || crown_valid);
    }

    // ========================================================================
    // count_gaps_and_vacuous proofs
    // ========================================================================

    /// Helper: build a minimal StageGapResult for counting tests.
    fn stub_result(
        has_ibp: bool,
        has_crown: bool,
        has_analytical: bool,
        is_vacuous: bool,
    ) -> StageGapResult {
        StageGapResult {
            stage: PipelineStage {
                name: "test",
                status_key: "test_key",
                is_compiled_segment: false,
                is_bridge: false,
                source_file: "test.rs",
                cpu_bridges: &[],
            },
            has_ibp_bounds: has_ibp,
            has_crown_bounds: has_crown,
            has_analytical_bounds: has_analytical,
            is_vacuous,
            bound_width: None,
            proof_strength: None,
            soundness_mode: None,
            has_constructive_certificate: false,
        }
    }

    /// Empty slice => zero gaps, zero vacuous.
    #[kani::unwind(8)]
    #[kani::proof]
    fn count_gaps_empty_slice() {
        let results: Vec<StageGapResult> = vec![];
        let (gaps, vacuous) = count_gaps_and_vacuous(&results);
        assert_eq!(gaps, 0);
        assert_eq!(vacuous, 0);
    }

    /// Single entry with no bounds => 1 gap, 0 vacuous.
    #[kani::unwind(8)]
    #[kani::proof]
    fn count_gaps_single_gap_entry() {
        let results = vec![stub_result(false, false, false, false)];
        let (gaps, vacuous) = count_gaps_and_vacuous(&results);
        assert_eq!(gaps, 1);
        assert_eq!(vacuous, 0);
    }

    /// Single entry with bounds and vacuous => 0 gaps, 1 vacuous.
    #[kani::unwind(8)]
    #[kani::proof]
    fn count_gaps_single_vacuous_entry() {
        let results = vec![stub_result(true, false, false, true)];
        let (gaps, vacuous) = count_gaps_and_vacuous(&results);
        assert_eq!(gaps, 0);
        assert_eq!(vacuous, 1);
    }

    /// For a symbolic 3-element array: total_gaps == count where !has_any_bounds,
    /// vacuous_count == count where is_vacuous.
    #[kani::unwind(8)]
    #[kani::proof]
    fn count_gaps_three_entries_matches_predicates() {
        let has_ibp: [bool; 3] = [kani::any(), kani::any(), kani::any()];
        let has_crown: [bool; 3] = [kani::any(), kani::any(), kani::any()];
        let has_analytical: [bool; 3] = [kani::any(), kani::any(), kani::any()];
        let is_vacuous: [bool; 3] = [kani::any(), kani::any(), kani::any()];

        let results: Vec<StageGapResult> = (0..3)
            .map(|i| stub_result(has_ibp[i], has_crown[i], has_analytical[i], is_vacuous[i]))
            .collect();

        let (total_gaps, vacuous_count) = count_gaps_and_vacuous(&results);

        // Manually count expected values
        let mut expected_gaps = 0usize;
        let mut expected_vacuous = 0usize;
        for i in 0..3 {
            if !results[i].has_any_bounds() {
                expected_gaps += 1;
            }
            if results[i].is_vacuous {
                expected_vacuous += 1;
            }
        }

        assert_eq!(total_gaps, expected_gaps);
        assert_eq!(vacuous_count, expected_vacuous);
    }

    /// has_any_bounds() == (has_ibp || has_crown || has_analytical) for a symbolic entry.
    #[kani::unwind(8)]
    #[kani::proof]
    fn has_any_bounds_predicate_matches_disjunction() {
        let has_ibp: bool = kani::any();
        let has_crown: bool = kani::any();
        let has_analytical: bool = kani::any();

        let result = stub_result(has_ibp, has_crown, has_analytical, false);
        assert_eq!(
            result.has_any_bounds(),
            has_ibp || has_crown || has_analytical
        );
    }

    // ========================================================================
    // Pipeline stage registry structural proofs
    // ========================================================================

    /// All pipeline stages have non-empty names and status_keys.
    #[kani::unwind(8)]
    #[kani::proof]
    fn pipeline_stages_all_have_names_and_keys() {
        let stages = kokoro_pipeline_stages();
        for stage in &stages {
            assert!(!stage.name.is_empty());
            assert!(!stage.status_key.is_empty());
            assert!(!stage.source_file.is_empty());
        }
    }

    /// No stage is both compiled_segment and bridge simultaneously.
    #[kani::unwind(8)]
    #[kani::proof]
    fn pipeline_stages_no_dual_classification() {
        let stages = kokoro_pipeline_stages();
        for stage in &stages {
            assert!(!(stage.is_compiled_segment && stage.is_bridge));
        }
    }
}
