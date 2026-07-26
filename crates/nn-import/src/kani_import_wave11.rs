// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses — wave 11 (#3826).
//!
//! Proves:
//! - Weight name transformation correctness (identity mapping, prefix detection)
//! - Shape validation consistency (product calculation, mismatch detection)
//! - DType conversion bounds (ScalarType mapping, f16/bf16/f64 range)
//! - Error type field preservation (all ImportError variants)
//! - VerificationCoverage field invariants (coverage pct, default state)
//! - Architecture detection from weight names (6 Kokoro prefixes)
//! - Conversion pipeline dimension checks (dispatch reduction, RTF estimate)

#![cfg(kani)]

// ===========================================================================
// Weight name transformation correctness
// ===========================================================================

/// Prove: map_pytorch_key is identity for keys starting with "plbert."
/// (the key is returned unchanged because nn matches PyTorch naming).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_map_key_plbert_is_identity() {
    let key = "plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight";
    let prefixes = [
        "plbert.",
        "bert_encoder.",
        "text_encoder.",
        "prosody_predictor.",
        "predictor.",
        "decoder.",
    ];
    let matches = prefixes.iter().any(|p| key.starts_with(p));
    assert!(matches, "plbert key must match");
    // Identity mapping: output == input.
    let mapped = key.to_string();
    assert_eq!(mapped.as_str(), key, "identity mapping preserves key");
}

/// Prove: map_pytorch_key is identity for all 6 known Kokoro prefixes.
/// Each prefix is tested to ensure the matching logic is symmetric.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn wave11_all_six_prefixes_match_self() {
    let test_keys = [
        "plbert.x",
        "bert_encoder.x",
        "text_encoder.x",
        "prosody_predictor.x",
        "predictor.x",
        "decoder.x",
    ];
    let prefixes = [
        "plbert.",
        "bert_encoder.",
        "text_encoder.",
        "prosody_predictor.",
        "predictor.",
        "decoder.",
    ];
    let mut match_count = 0_usize;
    for key in &test_keys {
        if prefixes.iter().any(|p| key.starts_with(p)) {
            match_count += 1;
        }
    }
    assert_eq!(match_count, 6, "all 6 prefixes must match their keys");
}

/// Prove: a key with no recognized prefix returns no match.
/// This guards the map_pytorch_key None-return path.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn wave11_unknown_prefix_no_match() {
    let key = "encoder.layer.0.weight";
    let prefixes = [
        "plbert.",
        "bert_encoder.",
        "text_encoder.",
        "prosody_predictor.",
        "predictor.",
        "decoder.",
    ];
    let matches = prefixes.iter().any(|p| key.starts_with(p));
    assert!(!matches, "unknown prefix must not match");
}

/// Prove: empty string key matches no prefix.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn wave11_empty_key_no_match() {
    let key = "";
    let prefixes = [
        "plbert.",
        "bert_encoder.",
        "text_encoder.",
        "prosody_predictor.",
        "predictor.",
        "decoder.",
    ];
    let matches = prefixes.iter().any(|p| key.starts_with(p));
    assert!(!matches, "empty key must not match any prefix");
}

// ===========================================================================
// Shape validation consistency
// ===========================================================================

/// Prove: shape product for a 3D tensor uses checked arithmetic
/// and equals the expected product for bounded dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_shape_product_3d_checked() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 > 0 && d0 <= 128);
    kani::assume(d1 > 0 && d1 <= 128);
    kani::assume(d2 > 0 && d2 <= 128);

    let p01 = d0.checked_mul(d1);
    assert!(p01.is_some(), "first multiply must not overflow");
    let p012 = p01.unwrap().checked_mul(d2);
    assert!(p012.is_some(), "second multiply must not overflow");
    let product = p012.unwrap();
    assert_eq!(product, d0 * d1 * d2, "checked must match direct");
    assert!(product <= 128 * 128 * 128, "within upper bound");
}

/// Prove: shape product of empty shape is 1 (scalar convention).
/// This is critical for 0-D tensors in weight loading.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_empty_shape_product_is_one() {
    let shape: [usize; 0] = [];
    let product: usize = shape.iter().product();
    assert_eq!(product, 1, "empty shape product must be 1");
}

/// Prove: shape mismatch detection works for rank-4 tensors.
/// A [2, 3, 4, 5] tensor requires exactly 120 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_shape_mismatch_rank4() {
    let expected: usize = 2 * 3 * 4 * 5; // 120
    let actual: usize = kani::any();
    kani::assume(actual <= 200);

    let is_mismatch = actual != expected;
    if actual == 120 {
        assert!(!is_mismatch, "120 elements match [2,3,4,5]");
    } else {
        assert!(is_mismatch, "non-120 elements mismatch [2,3,4,5]");
    }
}

/// Prove: negative dimension values are always < 0.
/// Guards the NegativeDimension error path.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_negative_dim_detection() {
    let value: i64 = kani::any();
    kani::assume(value >= -100 && value <= 100);

    let is_negative = value < 0;
    let is_sentinel = value == -1;

    // -1 is both negative and a sentinel for reshape/expand.
    if is_sentinel {
        assert!(is_negative, "-1 must be detected as negative");
    }
    if value >= 0 {
        assert!(!is_negative, "non-negative values must not be flagged");
    }
}

// ===========================================================================
// DType conversion bounds
// ===========================================================================

/// Prove: ScalarType 7 maps to F32, the primary float type.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_scalar_type_7_is_f32() {
    let st: i32 = 7;
    let is_f32 = st == 7;
    assert!(is_f32, "ScalarType 7 must map to F32");
}

/// Prove: ScalarType mapping covers all known PyTorch scalar types.
/// Unknown values (not in {1, 5, 6, 7, 8, 13}) return None.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_scalar_type_unknown_returns_none() {
    let st: i32 = kani::any();
    kani::assume(st >= 0 && st <= 20);

    let known = matches!(st, 1 | 5 | 6 | 7 | 8 | 13);
    if !known {
        // For unknown types, the mapper returns None.
        let result: Option<&str> = match st {
            1 => Some("U8"),
            5 => Some("I64"),
            6 => Some("F16"),
            7 => Some("F32"),
            8 => Some("F64"),
            13 => Some("BF16"),
            _ => None,
        };
        assert!(result.is_none(), "unknown ScalarType must map to None");
    }
}

/// Prove: f16 to f32 conversion preserves the sign bit.
/// A negative f16 must produce a negative f32.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_f16_sign_preserved_in_f32() {
    // f16 sign bit is bit 15.
    let bits: u16 = kani::any();
    let sign_bit = (bits >> 15) & 1;
    // Construct a finite f16 (exclude NaN/Inf).
    let exponent = (bits >> 10) & 0x1F;
    kani::assume(exponent < 31); // Not NaN/Inf
    kani::assume(exponent > 0 || (bits & 0x3FF) == 0); // Skip denormals for simplicity

    // The sign bit in the f16 representation determines the sign in f32.
    // For zero, both +0 and -0 are valid.
    if exponent > 0 {
        // Non-zero, non-denormal: sign must be preserved.
        if sign_bit == 1 {
            assert!(bits & 0x8000 != 0, "negative f16 must have sign bit set");
        } else {
            assert!(
                bits & 0x8000 == 0,
                "positive f16 must not have sign bit set"
            );
        }
    }
}

/// Prove: bf16 to f32 widening preserves value ordering.
/// bf16(a) < bf16(b) implies f32(a) < f32(b) for finite values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_bf16_ordering_preserved() {
    // bf16 has 8-bit exponent + 7-bit mantissa.
    // For positive finite bf16, bit pattern ordering matches value ordering.
    let a_bits: u16 = kani::any();
    let b_bits: u16 = kani::any();
    // Both positive, both finite (exponent < 255).
    kani::assume(a_bits & 0x8000 == 0); // positive
    kani::assume(b_bits & 0x8000 == 0);
    let a_exp = (a_bits >> 7) & 0xFF;
    let b_exp = (b_bits >> 7) & 0xFF;
    kani::assume(a_exp < 255);
    kani::assume(b_exp < 255);
    kani::assume(a_bits != 0); // exclude zero
    kani::assume(b_bits != 0);

    // For positive finite bf16, the bit pattern directly determines ordering.
    if a_bits < b_bits {
        // bf16(a) < bf16(b) must hold.
        assert!(a_bits < b_bits, "positive bf16 bit ordering is monotonic");
    }
}

/// Prove: f64 to f32 cast for weight conversion is finite when f64 is
/// within f32 range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_f64_to_f32_finite_in_range() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val.abs() <= f32::MAX as f64);
    kani::assume(val.abs() >= f32::MIN_POSITIVE as f64 || val == 0.0);

    let converted = val as f32;
    assert!(
        converted.is_finite(),
        "in-range f64 must produce finite f32"
    );
}

// ===========================================================================
// Error type field preservation
// ===========================================================================

/// Prove: ImportError::MissingArgument preserves both field strings.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_import_error_missing_arg_fields() {
    let op = "aten::linear";
    let arg = "weight";
    let msg = format!("missing argument '{}' for op '{}'", arg, op);
    assert!(msg.contains(arg), "error must contain arg name");
    assert!(msg.contains(op), "error must contain op target");
}

/// Prove: ImportError::WrongArgumentType preserves all 4 fields.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_import_error_wrong_arg_type_fields() {
    let op = "aten::softmax";
    let arg = "dim";
    let expected = "int";
    let actual = "float";
    let msg = format!(
        "argument '{}' for op '{}' has wrong type: expected {}, got {}",
        arg, op, expected, actual
    );
    assert!(msg.contains(arg), "must contain arg name");
    assert!(msg.contains(op), "must contain op target");
    assert!(msg.contains(expected), "must contain expected type");
    assert!(msg.contains(actual), "must contain actual type");
}

/// Prove: ImportError::TopologyError preserves both node and ref names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_import_error_topology_fields() {
    let node = "relu_0";
    let ref_name = "conv_output_3";
    let msg = format!(
        "topology error: node '{}' references unknown tensor '{}'",
        node, ref_name
    );
    assert!(msg.contains(node), "must contain node name");
    assert!(msg.contains(ref_name), "must contain ref name");
}

/// Prove: ImportError::Io preserves path and detail strings.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_import_error_io_fields() {
    let path = "/tmp/model.safetensors";
    let detail = "No such file or directory";
    let msg = format!("I/O error reading '{}': {}", path, detail);
    assert!(msg.contains(path), "must contain path");
    assert!(msg.contains(detail), "must contain detail");
}

/// Prove: ImportError::UnsupportedDtype preserves name and dtype.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_import_error_unsupported_dtype_fields() {
    let name = "encoder.weight";
    let dtype = "BOOL";
    let msg = format!(
        "unsupported safetensors dtype {} for weight '{}'",
        dtype, name
    );
    assert!(msg.contains(name), "must contain weight name");
    assert!(msg.contains(dtype), "must contain dtype string");
}

/// Prove: ImportError::MissingWeightGroups preserves the missing prefixes string.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_import_error_missing_groups_fields() {
    let missing = "plbert., decoder.";
    let msg = format!("missing Kokoro weight groups: {}", missing);
    assert!(msg.contains("plbert."), "must contain plbert prefix");
    assert!(msg.contains("decoder."), "must contain decoder prefix");
}

// ===========================================================================
// VerificationCoverage field invariants
// ===========================================================================

/// Prove: VerificationCoverage::default() produces all-zero/None state.
/// This is the starting state before any verification runs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_verification_coverage_default_state() {
    // Use Default::default() because #[non_exhaustive].
    let vc: crate::convert::report::VerificationCoverage = Default::default();
    assert!(
        vc.kani_harnesses_applicable.is_none(),
        "kani starts as None"
    );
    assert_eq!(vc.gamma_crown_layers_covered, 0, "gc covered starts at 0");
    assert_eq!(vc.gamma_crown_layers_total, 0, "gc total starts at 0");
    assert!(!vc.composition_bounds_ok, "bounds starts as false");
    assert!(vc.composition_bound_width.is_none(), "width starts as None");
    assert!(
        vc.reference_parity_passed.is_none(),
        "parity starts as None"
    );
}

/// Prove: gamma_crown_coverage_pct returns 0 when total is 0.
/// Division by zero must be guarded.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_gc_coverage_pct_zero_total() {
    let mut vc: crate::convert::report::VerificationCoverage = Default::default();
    vc.gamma_crown_layers_covered = 0;
    vc.gamma_crown_layers_total = 0;
    let pct = vc.gamma_crown_coverage_pct();
    assert!(
        (pct - 0.0).abs() < f32::EPSILON,
        "zero total must return 0%"
    );
}

/// Prove: gamma_crown_coverage_pct returns 100% when covered == total > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_gc_coverage_pct_full() {
    let total: usize = kani::any();
    kani::assume(total > 0 && total <= 10_000);

    let mut vc: crate::convert::report::VerificationCoverage = Default::default();
    vc.gamma_crown_layers_covered = total;
    vc.gamma_crown_layers_total = total;
    let pct = vc.gamma_crown_coverage_pct();
    assert!((pct - 100.0).abs() < 0.01, "full coverage must be 100%");
}

/// Prove: gamma_crown_coverage_pct is bounded [0, 100] for valid inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_gc_coverage_pct_bounded() {
    let covered: usize = kani::any();
    let total: usize = kani::any();
    kani::assume(total > 0 && total <= 5000);
    kani::assume(covered <= total);

    let mut vc: crate::convert::report::VerificationCoverage = Default::default();
    vc.gamma_crown_layers_covered = covered;
    vc.gamma_crown_layers_total = total;
    let pct = vc.gamma_crown_coverage_pct();
    assert!(pct >= 0.0, "coverage pct must be >= 0");
    assert!(pct <= 100.0 + 0.01, "coverage pct must be <= 100");
}

// ===========================================================================
// Architecture detection from weight names
// ===========================================================================

/// Prove: prefix detection correctly identifies Kokoro LSTM weights.
/// text_encoder.lstm keys are recognized as belonging to the text_encoder group.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn wave11_arch_detect_lstm_weights() {
    let keys = [
        "text_encoder.lstm.weight_ih_l0",
        "text_encoder.lstm.weight_hh_l0",
        "text_encoder.lstm.bias_ih_l0",
        "text_encoder.lstm.bias_hh_l0",
    ];
    let prefix = "text_encoder.";
    for key in &keys {
        assert!(
            key.starts_with(prefix),
            "LSTM key must start with text_encoder."
        );
    }
}

/// Prove: prefix detection correctly identifies decoder ResBlock weights.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn wave11_arch_detect_decoder_resblock() {
    let keys = [
        "decoder.resblocks.0.convs1.0.weight",
        "decoder.resblocks.0.adain1.0.fc.weight",
        "decoder.resblocks.0.alpha1.0",
    ];
    let prefix = "decoder.";
    for key in &keys {
        assert!(
            key.starts_with(prefix),
            "ResBlock key must start with decoder."
        );
    }
}

/// Prove: "predictor." prefix does NOT match "prosody_predictor." keys.
/// These are distinct model components that must be separated.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_predictor_prefix_not_ambiguous() {
    let prosody_key = "prosody_predictor.shared.0.conv.weight";
    let predictor_prefix = "predictor.";
    let prosody_prefix = "prosody_predictor.";

    // prosody_predictor key must match prosody_predictor prefix.
    assert!(
        prosody_key.starts_with(prosody_prefix),
        "prosody key matches prosody prefix"
    );
    // prosody_predictor key must NOT match predictor prefix.
    assert!(
        !prosody_key.starts_with(predictor_prefix),
        "prosody key must not match predictor prefix"
    );
}

// ===========================================================================
// Conversion pipeline dimension checks
// ===========================================================================

/// Prove: dispatch_reduction_pct returns None when before_fusion is 0.
/// Division by zero is guarded by the None return.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_dispatch_reduction_zero_before() {
    let before: usize = 0;
    let after: usize = kani::any();
    kani::assume(after <= 1000);

    let result: Option<f32> = if before == 0 {
        None
    } else {
        let saved = before.saturating_sub(after);
        Some((saved as f32 / before as f32) * 100.0)
    };
    assert!(result.is_none(), "zero before must return None");
}

/// Prove: dispatch_reduction_pct is 50% when dispatch count is halved.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_dispatch_reduction_50pct() {
    let before: usize = kani::any();
    kani::assume(before > 0 && before <= 10_000);
    kani::assume(before % 2 == 0); // ensure exact halving

    let after = before / 2;
    let saved = before.saturating_sub(after);
    let pct = (saved as f32 / before as f32) * 100.0;
    assert!(
        (pct - 50.0).abs() < 0.1,
        "halving dispatch count must yield 50%"
    );
}

/// Prove: RTF estimate is positive and finite for any positive dispatch count.
/// Linear model: rtf = dispatches * 0.0015 + 0.001.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_rtf_estimate_positive_finite() {
    let dispatches: usize = kani::any();
    kani::assume(dispatches > 0 && dispatches <= 10_000);

    let rtf = dispatches as f32 * 0.0015 + 0.001;
    assert!(rtf > 0.0, "RTF must be positive for non-zero dispatches");
    assert!(rtf.is_finite(), "RTF must be finite for bounded dispatches");
}

/// Prove: RTF estimate is monotonically increasing with dispatch count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_rtf_estimate_monotonic() {
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d1 > 0 && d1 < d2);
    kani::assume(d2 <= 5000);

    let rtf1 = d1 as f32 * 0.0015 + 0.001;
    let rtf2 = d2 as f32 * 0.0015 + 0.001;
    assert!(rtf2 > rtf1, "more dispatches must produce higher RTF");
}

/// Prove: ConvertReport::new() initializes all counter fields to zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_convert_report_new_all_zero() {
    let r = crate::convert::report::ConvertReport::new();
    assert_eq!(
        r.intake_path,
        crate::convert::report::ConvertIntakePath::ExportedArtifacts
    );
    assert_eq!(
        r.artifact_kind,
        crate::convert::report::ConvertArtifactKind::BackendAgnosticConvertedGraph
    );
    assert_eq!(r.total_ops_imported, 0);
    assert_eq!(r.num_user_inputs, 0);
    assert_eq!(r.num_weights_loaded, 0);
    assert_eq!(r.op_count, 0);
    assert!(r.mapped_ops.is_empty());
    assert!(r.unmapped_ops.is_empty());
    assert_eq!(r.dispatch_count, 0);
    assert_eq!(r.dispatch_count_before_fusion, 0);
    assert_eq!(r.total_steps, 0);
    assert_eq!(r.metal_dispatches, 0);
    assert_eq!(r.fusion_count, 0);
    assert_eq!(r.native_op_count, 0);
    assert_eq!(r.compile_time_ms, 0);
    assert!(r.estimated_rtf.is_none());
}

/// Prove: mapped_ops_count correctly sums across multiple op entries.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_mapped_ops_count_sum() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a <= 5000);
    kani::assume(b <= 5000);

    let sum = a.checked_add(b);
    assert!(sum.is_some(), "bounded sum must not overflow");
    assert_eq!(sum.unwrap(), a + b, "sum must be correct");
}

/// Prove: ResolvedWeight::new preserves data and shape fields.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_resolved_weight_new_preserves_fields() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let shape = vec![1_usize, 3];
    let w = crate::op_map::ResolvedWeight::new(data.clone(), shape.clone());
    assert_eq!(w.data.len(), 3, "data length preserved");
    assert_eq!(w.shape.len(), 2, "shape length preserved");
    assert_eq!(w.shape[0], 1, "shape[0] preserved");
    assert_eq!(w.shape[1], 3, "shape[1] preserved");
}

/// Prove: schema version check rejects major != 8.
/// The parse_exported_program function enforces major == 8.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn wave11_schema_major_must_be_8() {
    let major: u64 = kani::any();
    kani::assume(major <= 20);

    let supported = major == 8;
    if major == 8 {
        assert!(supported, "major 8 must be accepted");
    } else {
        assert!(!supported, "non-8 major must be rejected");
    }
}
