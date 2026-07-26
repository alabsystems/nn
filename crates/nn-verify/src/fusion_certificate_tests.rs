// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fusion equivalence certificates.

use super::*;

fn sample_verification() -> FusionVerification {
    FusionVerification {
        fused_kernel_name: "adain_snake_fused".to_string(),
        diff_lower: -0.001,
        diff_upper: 0.001,
        max_abs_diff: 0.001,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    }
}

fn sample_bounds() -> Vec<(f32, f32)> {
    vec![
        (-10.0, 10.0), // x
        (-5.0, 5.0),   // mu
        (0.01, 10.0),  // var
        (-2.0, 2.0),   // gamma
        (-2.0, 2.0),   // beta
        (0.5, 2.0),    // alpha
        (1e-6, 1e-3),  // eps
    ]
}

#[test]
fn test_from_verification_populates_fields() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );

    assert_eq!(cert.version, FUSION_CERTIFICATE_VERSION);
    assert_eq!(cert.fused_kernel_name, "adain_snake_fused");
    assert_eq!(
        cert.sequential_names,
        ("adain".to_string(), "snake".to_string())
    );
    assert_eq!(cert.dimension, 512);
    assert_eq!(cert.epsilon, 1e-4);
    assert_eq!(cert.crown_bound, Some(0.001));
    assert_eq!(cert.crown_method, Some(PropMethod::Crown));
    assert!(cert.analytical_bound.is_none());
    assert_eq!(cert.variable_bounds.len(), 7);
}

#[test]
fn test_analytical_bound_compute() {
    let bound = AnalyticalFusionBound::compute(2, 64.0, 2.0).expect("valid bound");
    assert_eq!(bound.differing_op_count, 2);
    assert!((bound.max_magnitude - 64.0).abs() < 1e-10);
    // 64 * 2 * 2^-24 * 2 = 64 * 2 * 5.96e-8 * 2 ≈ 1.53e-5
    assert!(bound.max_abs_diff > 1e-6);
    assert!(bound.max_abs_diff < 1e-4);
    assert!(bound.proves_within_epsilon(1e-4));
}

#[test]
fn test_analytical_bound_zero_ops_rejected() {
    let result = AnalyticalFusionBound::compute(0, 64.0, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_negative_magnitude_rejected() {
    let result = AnalyticalFusionBound::compute(2, -1.0, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_nan_rejected() {
    let result = AnalyticalFusionBound::compute(2, f64::NAN, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_nan_lipschitz_rejected() {
    let result = AnalyticalFusionBound::compute(2, 64.0, f64::NAN);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_negative_lipschitz_rejected() {
    let result = AnalyticalFusionBound::compute(2, 64.0, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_inf_magnitude_rejected() {
    let result = AnalyticalFusionBound::compute(2, f64::INFINITY, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_overflow_rejected() {
    // Very large but finite inputs whose product overflows to Inf
    let result = AnalyticalFusionBound::compute(usize::MAX, f64::MAX, f64::MAX);
    assert!(result.is_err());
}

#[test]
fn test_analytical_bound_zero_magnitude_accepted() {
    let bound = AnalyticalFusionBound::compute(2, 0.0, 2.0).expect("valid");
    assert!((bound.max_abs_diff - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_analytical_bound_zero_lipschitz_accepted() {
    let bound = AnalyticalFusionBound::compute(2, 64.0, 0.0).expect("valid");
    assert!((bound.max_abs_diff - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_known_bounds_adain_snake_within_epsilon() {
    let bound = known_bounds::adain_snake().expect("valid");
    // Should be ~1.53e-5 — well within 1e-4
    assert!(bound.max_abs_diff < 1e-4);
    assert!(bound.proves_within_epsilon(1e-4));
}

#[test]
fn test_known_bounds_layer_norm_gelu_within_epsilon() {
    let bound = known_bounds::layer_norm_gelu().expect("valid");
    // ~1.43e-6 with L=1.2 — well within 1e-4
    assert!(bound.max_abs_diff < 1e-4);
    assert!(bound.proves_within_epsilon(1e-4));
}

#[test]
fn test_known_bounds_rms_norm_silu_mul_within_epsilon() {
    let bound = known_bounds::rms_norm_silu_mul().expect("valid");
    // ~9.44e-6 with L=1.1 — well within 1e-4
    assert!(bound.max_abs_diff < 1e-4);
    assert!(bound.proves_within_epsilon(1e-4));
}

#[test]
fn test_known_bounds_adain_leaky_relu_within_epsilon() {
    let bound = known_bounds::adain_leaky_relu().expect("valid");
    // ~7.63e-6 with L=1.0 — well within 1e-4
    assert!(bound.max_abs_diff < 1e-4);
    assert!(bound.proves_within_epsilon(1e-4));
    // LeakyReLU Lipschitz 1.0 is tighter than Snake Lipschitz 2.0
    let snake = known_bounds::adain_snake().expect("valid");
    assert!(
        bound.max_abs_diff < snake.max_abs_diff,
        "adain_leaky_relu ({}) should be tighter than adain_snake ({})",
        bound.max_abs_diff,
        snake.max_abs_diff,
    );
}

#[test]
fn test_known_bounds_ada_layer_norm_within_epsilon() {
    let bound = known_bounds::ada_layer_norm().expect("valid");
    // 10 * 2 * 2^-24 * 2.0 ≈ 2.38e-6
    assert!(bound.max_abs_diff < 1e-4);
    assert!(bound.proves_within_epsilon(1e-4));
    // AdaLayerNorm uses same magnitude as LayerNorm+GELU but different Lipschitz
    let ln_gelu = known_bounds::layer_norm_gelu().expect("valid");
    assert!(
        (bound.max_abs_diff - ln_gelu.max_abs_diff).abs() > f64::EPSILON,
        "ada_layer_norm ({}) should differ from layer_norm_gelu ({}) due to different Lipschitz",
        bound.max_abs_diff,
        ln_gelu.max_abs_diff,
    );
}

#[test]
fn test_proves_equivalence_crown_only() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    // crown_bound = 0.001, epsilon = 1e-4
    // 0.001 > 1e-4 so CROWN alone does NOT prove equivalence
    assert!(!cert.proves_equivalence());
}

#[test]
fn test_proves_equivalence_analytical_only() {
    let mut v = sample_verification();
    v.max_abs_diff = 1.0; // CROWN bound is too loose
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    )
    .with_analytical_bound(known_bounds::adain_snake().expect("valid"));
    // analytical ~1.5e-5 < 1e-4 => proves equivalence
    assert!(cert.proves_equivalence());
}

#[test]
fn test_proves_equivalence_both_sources() {
    let mut v = sample_verification();
    v.max_abs_diff = 5e-5; // CROWN tight enough
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    )
    .with_analytical_bound(known_bounds::adain_snake().expect("valid"));
    assert!(cert.proves_equivalence());
}

#[test]
fn test_tightest_bound_picks_analytical() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    )
    .with_analytical_bound(known_bounds::adain_snake().expect("valid"));
    let tightest = cert.tightest_bound().expect("has bounds");
    // Analytical (~1.5e-5) should be tighter than CROWN (0.001)
    assert!(tightest < 0.001);
}

#[test]
fn test_validate_valid_certificate() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    )
    .with_analytical_bound(known_bounds::adain_snake().expect("valid"));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_validate_empty_name_rejected() {
    let mut v = sample_verification();
    v.fused_kernel_name = String::new();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    assert!(cert.validate().is_err());
}

#[test]
fn test_validate_nan_epsilon_rejected() {
    let v = sample_verification();
    let mut cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    cert.epsilon = f32::NAN;
    assert!(cert.validate().is_err());
}

#[test]
fn test_validate_inverted_bounds_rejected() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &[(10.0, -10.0)], // inverted
    );
    assert!(cert.validate().is_err());
}

#[test]
fn test_validate_bad_hash_rejected() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    )
    .with_source_hash("not-a-hash".to_string());
    assert!(cert.validate().is_err());
}

#[test]
fn test_serde_roundtrip() {
    let v = sample_verification();
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    )
    .with_analytical_bound(known_bounds::adain_snake().expect("valid"));

    let json = cert.to_json().expect("serialize");
    let cert2: FusionEquivalenceCertificate = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(cert.version, cert2.version);
    assert_eq!(cert.fused_kernel_name, cert2.fused_kernel_name);
    assert_eq!(cert.sequential_names, cert2.sequential_names);
    assert_eq!(cert.dimension, cert2.dimension);
    assert_eq!(cert.epsilon, cert2.epsilon);
    assert_eq!(cert.crown_bound, cert2.crown_bound);
    assert_eq!(
        cert.analytical_bound.as_ref().map(|b| b.max_abs_diff),
        cert2.analytical_bound.as_ref().map(|b| b.max_abs_diff),
    );
    assert_eq!(cert.variable_bounds, cert2.variable_bounds);
}

#[test]
fn test_version_zero_rejected() {
    let v = sample_verification();
    let mut cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    cert.version = 0;
    assert!(cert.validate().is_err());
}

#[test]
fn test_non_finite_crown_bound_rejected() {
    let v = sample_verification();
    let mut cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    cert.crown_bound = Some(f32::INFINITY);
    assert!(cert.validate().is_err());
}

#[test]
fn test_negative_crown_bound_rejected() {
    let v = sample_verification();
    let mut cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    cert.crown_bound = Some(-0.5);
    assert!(cert.validate().is_err());
}

#[test]
fn test_now_iso8601_format() {
    let ts = now_iso8601();
    // Must match YYYY-MM-DDTHH:MM:SSZ pattern
    assert_eq!(ts.len(), 20, "ISO 8601 timestamp wrong length: {ts}");
    assert!(ts.ends_with('Z'));
    assert_eq!(&ts[4..5], "-");
    assert_eq!(&ts[7..8], "-");
    assert_eq!(&ts[10..11], "T");
    assert_eq!(&ts[13..14], ":");
    assert_eq!(&ts[16..17], ":");
}

#[test]
fn test_unix_secs_to_iso8601_known_dates() {
    // 1970-01-01T00:00:00Z (epoch)
    assert_eq!(unix_secs_to_iso8601(0), "1970-01-01T00:00:00Z");
    // 2000-01-01T00:00:00Z
    assert_eq!(unix_secs_to_iso8601(946_684_800), "2000-01-01T00:00:00Z");
    // 2026-03-15T12:00:00Z (20527 days * 86400 + 43200)
    assert_eq!(unix_secs_to_iso8601(1_773_576_000), "2026-03-15T12:00:00Z");
    // 2024-02-29T00:00:00Z (leap year)
    assert_eq!(unix_secs_to_iso8601(1_709_164_800), "2024-02-29T00:00:00Z");
}

#[test]
fn test_validate_rejects_unix_epoch_timestamp() {
    let v = sample_verification();
    let mut cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    // Old format: bare UNIX epoch seconds with Z suffix
    cert.generated_at = "1742000000Z".to_string();
    let err = cert.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ISO 8601"),
        "error should mention ISO 8601: {msg}"
    );
}

#[test]
fn test_negative_analytical_bound_rejected() {
    let v = sample_verification();
    let mut cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &sample_bounds(),
    );
    cert.analytical_bound = Some(AnalyticalFusionBound {
        differing_op_count: 2,
        max_magnitude: 64.0,
        lipschitz_factor: 2.0,
        max_abs_diff: -1.0,
    });
    assert!(cert.validate().is_err());
}
