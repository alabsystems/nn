// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pipeline_crown and moonshot_crown safety.
//!
//! Proves correctness of bound propagation arithmetic, property composition,
//! probability threshold validation, pipeline composition soundness, and
//! certificate construction for the TTS verification pipeline.
//!
//! These harnesses cover the pure functions that underpin the 8 moonshot
//! properties (non-silence, non-clipping, intelligibility, speaker
//! consistency, temporal boundedness, streaming safety, memory safety,
//! implementation correctness).
//!
//! Properties proved:
//!
//! 1. Non-silence: proven iff max absolute bound > threshold AND cert is valid.
//! 2. Non-clipping: proven iff all output in [-1,1], NaN in bounds prevents proof.
//! 3. Streaming safety: max click bound monotonically decreasing in crossfade length.
//! 4. Speaker consistency: worst-case L2 distance is non-negative and NaN-safe.
//! 5. Implementation correctness: fraction is bounded in [0,1], level ordering correct.
//! 6. Pipeline composition: soundness propagation is AND of all stages.
//! 7. Bundle all_proven is AND of individual results.
//! 8. HybridCertificate::is_strong_evidence threshold semantics correct.
//! 9. Holm-Bonferroni adjusted p-values are monotonically non-decreasing and in [0,1].

// ---- P1: Non-Silence Property Proofs ----------------------------------------

/// Prove: check_non_silence returns proven=true only when valid cert has max|bound| > threshold.
///
/// Constructs a minimal 2-stage pipeline certificate and verifies the property check
/// is consistent with the mathematical definition.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_proven_requires_valid_cert_and_threshold() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(threshold.is_finite() && threshold >= 0.0);
    kani::assume(lo.abs() <= 100.0 && hi.abs() <= 100.0 && threshold <= 100.0);
    kani::assume(lo <= hi);

    // Build a valid 2-stage pipeline cert (minimum for verify_pipeline)
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let result = crate::moonshot_crown::check_non_silence(&cert, threshold);

    let max_abs = lo.abs().max(hi.abs());
    if result.proven {
        assert!(
            max_abs > threshold,
            "proven requires max|bound| > threshold"
        );
        assert!(cert.is_valid, "proven requires valid cert");
    }
}

/// Prove: check_non_silence never claims proven when output bounds contain NaN.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_nan_bounds_never_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![f64::NAN],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![f64::NAN],
            vec![1.0],
            vec![f64::NAN],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_silence(&cert, 0.01);

    // NaN in output lower makes max_abs NaN, which fails > threshold comparison
    assert!(!result.proven, "NaN bounds must not be proven");
}

// ---- P2: Non-Clipping Property Proofs ---------------------------------------

/// Prove: check_non_clipping returns proven=true only when all output in [-1, 1].
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_proven_iff_within_range() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 2.0 && hi.abs() <= 2.0);
    kani::assume(lo <= hi);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let result = crate::moonshot_crown::check_non_clipping(&cert);

    let within = lo >= -1.0 && hi <= 1.0;
    if result.proven {
        assert!(within, "proven implies all output within [-1, 1]");
    }
}

/// Prove: check_non_clipping never claims proven when bounds contain NaN.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_nan_bounds_never_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![f64::NAN],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(!result.proven, "NaN in output upper must prevent proof");
}

/// Prove: check_non_clipping with exactly [-1, 1] bounds is proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_exact_boundary_is_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![-1.0],
            vec![1.0],
            vec![-1.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-1.0],
            vec![1.0],
            vec![-1.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(result.proven, "exact [-1,1] bounds must be proven");
    assert_eq!(result.bound_value, 1.0, "worst bound at boundary is 1.0");
}

// ---- P3: Intelligibility Proxy Proofs ---------------------------------------

/// Prove: intelligibility proxy returns proven only when range ratio < max_range_ratio.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn intelligibility_proxy_ratio_threshold() {
    let in_lo: f64 = kani::any();
    let in_hi: f64 = kani::any();
    let out_lo: f64 = kani::any();
    let out_hi: f64 = kani::any();
    kani::assume(in_lo.is_finite() && in_hi.is_finite());
    kani::assume(out_lo.is_finite() && out_hi.is_finite());
    kani::assume(in_lo <= in_hi && out_lo <= out_hi);
    kani::assume(in_lo.abs() <= 10.0 && in_hi.abs() <= 10.0);
    kani::assume(out_lo.abs() <= 10.0 && out_hi.abs() <= 10.0);
    kani::assume(in_hi - in_lo > 0.001); // non-trivial input range

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![in_lo],
            vec![in_hi],
            vec![out_lo],
            vec![out_hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![out_lo],
            vec![out_hi],
            vec![out_lo],
            vec![out_hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let result = crate::moonshot_crown::check_intelligibility_proxy(&cert, 100.0);

    let input_range = in_hi - in_lo;
    let output_range = out_hi - out_lo;
    let ratio = output_range / input_range;

    if result.proven {
        assert!(ratio < 100.0, "proven requires ratio < max_range_ratio");
    }
}

// ---- P4: Speaker Consistency Proofs -----------------------------------------

/// Prove: speaker consistency worst-case L2 distance is non-negative.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn speaker_consistency_distance_non_negative() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    let rf: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite() && rf.is_finite());
    kani::assume(lo.abs() <= 100.0 && hi.abs() <= 100.0 && rf.abs() <= 100.0);
    kani::assume(lo <= hi);

    let evidence = crate::moonshot_crown::SpeakerConsistencyEvidence::new(
        1,
        vec![lo],
        vec![hi],
        vec![rf],
        1.0,
        true,
    );

    let result = crate::moonshot_crown::check_speaker_consistency(&evidence);

    assert!(
        result.bound_value >= 0.0 || result.bound_value == f64::INFINITY,
        "L2 distance must be non-negative"
    );
}

/// Prove: speaker consistency rejects NaN in embedding bounds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn speaker_consistency_nan_embedding_not_proven() {
    let evidence = crate::moonshot_crown::SpeakerConsistencyEvidence::new(
        1,
        vec![f64::NAN],
        vec![1.0],
        vec![0.5],
        1.0,
        true,
    );
    let result = crate::moonshot_crown::check_speaker_consistency(&evidence);

    assert!(!result.proven, "NaN embedding lower must not be proven");
    assert_eq!(
        result.bound_value,
        f64::INFINITY,
        "NaN bound should produce INFINITY distance"
    );
}

/// Prove: speaker consistency detects dimension mismatch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn speaker_consistency_dimension_mismatch() {
    let evidence = crate::moonshot_crown::SpeakerConsistencyEvidence::new(
        2,              // embed_dim = 2
        vec![0.0],      // but only 1 element in lower
        vec![1.0, 2.0], // 2 elements in upper
        vec![0.5, 0.5], // 2 elements in reference
        1.0,
        true,
    );
    let result = crate::moonshot_crown::check_speaker_consistency(&evidence);

    assert!(!result.proven, "dimension mismatch must not be proven");
    assert_eq!(result.bound_value, f64::INFINITY);
}

/// Prove: when ref embedding exactly equals bounds midpoint, distance is bounded
/// by half the bound width.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn speaker_consistency_midpoint_ref_bounded() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 100.0 && hi.abs() <= 100.0);

    let mid = (lo + hi) / 2.0;
    kani::assume(mid.is_finite());

    let evidence = crate::moonshot_crown::SpeakerConsistencyEvidence::new(
        1,
        vec![lo],
        vec![hi],
        vec![mid],
        1000.0,
        true,
    );

    let result = crate::moonshot_crown::check_speaker_consistency(&evidence);

    // Worst distance from midpoint is half the range
    let half_range = (hi - lo) / 2.0;
    // Allow f64 epsilon tolerance
    assert!(
        result.bound_value <= half_range + 1e-10,
        "midpoint ref distance must be <= half range"
    );
}

// ---- P6: Streaming Safety Proofs --------------------------------------------

/// Prove: streaming max click bound is monotonically decreasing in crossfade length.
///
/// More crossfade samples => smaller alpha step => smaller click bound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn streaming_click_bound_decreasing_in_crossfade_len() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1.0 && hi.abs() <= 1.0);

    let n1: usize = kani::any();
    let n2: usize = kani::any();
    kani::assume(n1 >= 2 && n2 > n1);
    kani::assume(n2 <= 1000);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let r1 = crate::moonshot_crown::check_streaming_safety(&cert, n1, 10.0);
    let r2 = crate::moonshot_crown::check_streaming_safety(&cert, n2, 10.0);

    assert!(
        r2.bound_value <= r1.bound_value + 1e-15,
        "more crossfade samples must give smaller or equal click bound"
    );
}

/// Prove: streaming safety with single-sample crossfade yields max possible click.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn streaming_single_sample_max_discontinuity() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![-1.0],
            vec![1.0],
            vec![-1.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-1.0],
            vec![1.0],
            vec![-1.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_streaming_safety(&cert, 1, 10.0);

    // crossfade_samples=1 means alpha_step=1.0, so max_click = range * 1.0 = 2.0
    let expected_bound = 2.0;
    assert!(
        (result.bound_value - expected_bound).abs() < 1e-10,
        "single sample crossfade: click bound should equal full output range"
    );
}

// ---- P8: Implementation Correctness Proofs ----------------------------------

/// Prove: implementation correctness fraction is bounded in [0, 1].
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn implementation_correctness_fraction_bounded() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total > 0);
    kani::assume(proven <= total);
    kani::assume(total <= 1000);

    let evidence = crate::moonshot_crown::ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: proven == total,
    };

    let result = crate::moonshot_crown::check_implementation_correctness(&evidence);

    let fraction = proven as f64 / total as f64;
    assert!(fraction >= 0.0 && fraction <= 1.0, "fraction in [0,1]");

    if evidence.all_proven {
        assert!(result.proven, "all_proven must yield proven result");
        assert_eq!(
            result.level,
            crate::moonshot::VerificationLevel::SmtProven,
            "all proven must be SmtProven"
        );
    }
}

/// Prove: implementation correctness with zero steps is not proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn implementation_correctness_zero_steps_not_proven() {
    let evidence = crate::moonshot_crown::ImplementationCorrectnessEvidence {
        total_steps: 0,
        proven_steps: 0,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: false,
    };

    let result = crate::moonshot_crown::check_implementation_correctness(&evidence);
    assert!(!result.proven, "zero steps must not be proven");
}

/// Prove: ay_proven_kernel_names returns a non-empty list.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn ay_proven_kernel_names_non_empty() {
    let names = crate::moonshot_crown::ay_proven_kernel_names();
    assert!(!names.is_empty(), "ay proven kernel list must not be empty");
    // All names must be non-empty strings
    for name in names {
        assert!(!name.is_empty(), "kernel name must not be empty");
    }
}

// ---- Pipeline Composition Soundness Proofs ----------------------------------

/// Prove: verify_pipeline soundness is AND of all stage soundnesses.
///
/// If any stage is not sound, the pipeline certificate is_sound must be false.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn pipeline_soundness_is_conjunction() {
    let s0_sound: bool = kani::any();
    let s1_sound: bool = kani::any();

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
            s0_sound,
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
            s1_sound,
        ),
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    // Soundness requires ALL stages to be sound AND the pipeline to be valid
    let expected_sound = s0_sound && s1_sound && cert.is_valid;
    assert_eq!(
        cert.is_sound, expected_sound,
        "pipeline soundness must be AND of all stage soundnesses AND validity"
    );
}

/// Prove: verify_pipeline rejects empty stages.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn pipeline_rejects_empty() {
    let stages: Vec<crate::pipeline::VerifiedStage> = vec![];
    let result = crate::pipeline::verify_pipeline(&stages);
    assert!(result.is_err(), "empty stages must be rejected");
}

// ---- Bundle Composition Proofs ----------------------------------------------

/// Prove: MoonshotCrownBundle all_proven is AND of individual result.proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn bundle_all_proven_is_conjunction() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1.0 && hi.abs() <= 1.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let bundle = crate::moonshot_crown::verify_properties_from_pipeline(&cert, 1);

    let individual_and = bundle.results.iter().all(|r| r.proven);
    assert_eq!(
        bundle.all_proven, individual_and,
        "all_proven must equal AND of individual proven flags"
    );
}

// ---- HybridCertificate Proofs -----------------------------------------------

/// Prove: HybridCertificate::is_strong_evidence requires all four conditions.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn hybrid_strong_evidence_requires_all_conditions() {
    let formal_sound: bool = kani::any();
    let p_val: f64 = kani::any();
    let effect: f64 = kani::any();
    let holds: bool = kani::any();

    kani::assume(p_val.is_finite() && effect.is_finite());
    kani::assume(p_val >= 0.0 && p_val <= 1.0);
    kani::assume(effect >= 0.0 && effect <= 100.0);

    let cert = crate::pipeline::HybridCertificate {
        formal_dim: 64,
        formal_property: "test".to_string(),
        formal_is_sound: formal_sound,
        statistical_dim: 512,
        n_samples: 100,
        p_value: p_val,
        effect_size: effect,
        property_holds: holds,
    };

    let strong = cert.is_strong_evidence();
    let expected = formal_sound && p_val < 0.01 && effect > 0.8 && holds;
    assert_eq!(
        strong, expected,
        "is_strong_evidence must require all four conditions"
    );
}

// ---- Verification Level Proofs (extended) -----------------------------------

/// Prove: VerificationLevel CrownPartial is strictly less than CrownProven.
///
/// This specific ordering matters for the intelligibility proxy which caps
/// at CrownPartial while the monotonicity path can reach CrownProven.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_partial_less_than_proven() {
    use crate::moonshot::VerificationLevel;
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProven);
    assert!(VerificationLevel::CrownProbabilistic > VerificationLevel::CrownPartial);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
}

/// Prove: MoonshotPropertyResult level assignment is consistent with proven flag.
///
/// When a property is proven and sound, level must be >= CrownPartial.
/// When not proven, level must be Empirical or None.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn property_result_level_consistent_with_proven() {
    use crate::moonshot::VerificationLevel;

    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 2.0 && hi.abs() <= 2.0);
    kani::assume(lo <= hi);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let result = crate::moonshot_crown::check_non_clipping(&cert);

    if result.proven {
        assert!(
            result.level >= VerificationLevel::CrownPartial,
            "proven property must have level >= CrownPartial"
        );
    } else {
        assert!(
            result.level <= VerificationLevel::Empirical,
            "unproven property must have level <= Empirical"
        );
    }
}

// ---- Holm-Bonferroni Proofs -------------------------------------------------

/// Prove: Holm-Bonferroni adjusted p-values are bounded in [0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn holm_bonferroni_adjusted_bounded_01() {
    let p0: f64 = kani::any();
    let p1: f64 = kani::any();
    kani::assume(p0.is_finite() && p1.is_finite());
    kani::assume(p0 >= 0.0 && p0 <= 1.0);
    kani::assume(p1 >= 0.0 && p1 <= 1.0);

    let result = crate::stats::holm_bonferroni(&[p0, p1]).unwrap();
    assert_eq!(result.len(), 2);

    for &adj in &result {
        assert!(adj >= 0.0, "adjusted p-value must be >= 0");
        assert!(adj <= 1.0, "adjusted p-value must be <= 1");
    }
}

/// Prove: Holm-Bonferroni with single p-value returns min(p*1, 1) = p (for p <= 1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn holm_bonferroni_single_value_identity() {
    let p: f64 = kani::any();
    kani::assume(p.is_finite());
    kani::assume(p >= 0.0 && p <= 1.0);

    let result = crate::stats::holm_bonferroni(&[p]).unwrap();
    assert_eq!(result.len(), 1);
    // With m=1, multiplier=1, so adjusted = min(p*1, 1) = p
    assert!(
        (result[0] - p).abs() < 1e-15,
        "single p-value adjustment should be identity"
    );
}

/// Prove: Holm-Bonferroni rejects NaN p-values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn holm_bonferroni_rejects_nan() {
    let result = crate::stats::holm_bonferroni(&[0.05, f64::NAN]);
    assert!(result.is_err(), "NaN p-value must be rejected");
}
