// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended mixed-precision and dtype tests for nn-core.
//!
//! Covers MixedPrecisionPolicy construction, op-category classification,
//! OpDTypeCategory variants, DType sizing/display/conversion edge cases,
//! and dtype compatibility semantics.

use crate::mixed_precision::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};
use crate::DType;

// =============================================================================
// MixedPrecisionPolicy tests
// =============================================================================

#[test]
fn test_policy_creation() {
    // All three preset constructors produce distinct, valid policies.
    let f32 = MixedPrecisionPolicy::f32_only();
    let apple = MixedPrecisionPolicy::apple_silicon_default();
    let cuda = MixedPrecisionPolicy::cuda_bf16();

    // Each has three float dtypes.
    for p in [&f32, &apple, &cuda] {
        assert!(p.weight_dtype.is_float());
        assert!(p.compute_dtype.is_float());
        assert!(p.accumulate_dtype.is_float());
    }
}

#[test]
fn test_default_op_category_returns_expected_for_known_ops() {
    // Spot-check a few known ops beyond what the base tests cover —
    // ensure each arm in default_op_category is exercised at least once.
    assert_eq!(default_op_category("matmul"), OpDTypeCategory::Compute);
    assert_eq!(default_op_category("softmax"), OpDTypeCategory::Accumulate);
    assert_eq!(default_op_category("relu"), OpDTypeCategory::Inherit);
}

#[test]
fn test_op_dtype_category_variants() {
    // The enum has exactly three meaningful variants. Verify they are distinct
    // and cover the full set via exhaustive match.
    let variants = [
        OpDTypeCategory::Compute,
        OpDTypeCategory::Accumulate,
        OpDTypeCategory::Inherit,
    ];
    for (i, a) in variants.iter().enumerate() {
        for (j, b) in variants.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
    // Verify exhaustive match compiles (pattern coverage check).
    for v in &variants {
        match v {
            OpDTypeCategory::Compute => {}
            OpDTypeCategory::Accumulate => {}
            OpDTypeCategory::Inherit => {}
        }
    }
}

#[test]
fn test_policy_compute_heavy_ops() {
    // matmul and linear are the canonical compute-heavy ops.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compute_ops = ["matmul", "linear", "conv1d", "conv2d", "embedding"];
    for op in &compute_ops {
        let cat = default_op_category(op);
        assert_eq!(cat, OpDTypeCategory::Compute, "op '{op}' should be Compute");
        // On Apple Silicon, compute-heavy ops resolve to F16.
        assert_eq!(policy.dtype_for_op(cat), DType::F16);
    }
}

#[test]
fn test_policy_memory_bound_ops() {
    // Element-wise / normalization ops that are memory-bound inherit from input.
    let inherit_ops = ["relu", "gelu", "silu", "tanh", "sigmoid", "snake"];
    for op in &inherit_ops {
        let cat = default_op_category(op);
        assert_eq!(
            cat,
            OpDTypeCategory::Inherit,
            "op '{op}' should be Inherit (memory-bound / elementwise)"
        );
    }
}

#[test]
fn test_policy_precision_sensitive_ops() {
    // Softmax, loss, and normalization ops require full precision.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let sensitive_ops = [
        "softmax",
        "log_softmax",
        "layer_norm",
        "group_norm",
        "instance_norm",
        "rms_norm",
        "batch_norm",
        "sum",
        "mean",
        "log",
        "pow",
        "cumsum",
    ];
    for op in &sensitive_ops {
        let cat = default_op_category(op);
        assert_eq!(
            cat,
            OpDTypeCategory::Accumulate,
            "op '{op}' should be Accumulate (precision-sensitive)"
        );
        // Precision-sensitive ops always resolve to F32 (accumulate_dtype).
        assert_eq!(policy.dtype_for_op(cat), DType::F32);
    }
}

// =============================================================================
// DType tests
// =============================================================================

#[test]
fn test_dtype_size_bytes() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::U32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::F64.size_bytes(), 8);
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::Bool.size_bytes(), 1);
}

#[test]
fn test_dtype_is_float() {
    let floats = [DType::F32, DType::F16, DType::BF16, DType::F64];
    let non_floats = [DType::U8, DType::U32, DType::I32, DType::I64, DType::Bool];
    for dt in &floats {
        assert!(dt.is_float(), "{dt} should be float");
    }
    for dt in &non_floats {
        assert!(!dt.is_float(), "{dt} should not be float");
    }
}

#[test]
fn test_dtype_display() {
    assert_eq!(format!("{}", DType::F32), "f32");
    assert_eq!(format!("{}", DType::F16), "f16");
    assert_eq!(format!("{}", DType::BF16), "bf16");
    assert_eq!(format!("{}", DType::F64), "f64");
    assert_eq!(format!("{}", DType::I32), "i32");
    assert_eq!(format!("{}", DType::I64), "i64");
    assert_eq!(format!("{}", DType::U32), "u32");
    assert_eq!(format!("{}", DType::U8), "u8");
    assert_eq!(format!("{}", DType::Bool), "bool");
}

#[test]
fn test_dtype_from_str_via_display_roundtrip() {
    // DType does not implement FromStr, so we verify Display produces
    // canonical lowercase strings that could round-trip through a manual parser.
    let all = [
        (DType::F32, "f32"),
        (DType::F16, "f16"),
        (DType::BF16, "bf16"),
        (DType::F64, "f64"),
        (DType::I32, "i32"),
        (DType::I64, "i64"),
        (DType::U32, "u32"),
        (DType::U8, "u8"),
        (DType::Bool, "bool"),
    ];
    for (dt, expected_str) in &all {
        let displayed = format!("{dt}");
        assert_eq!(
            &displayed, expected_str,
            "Display for {dt:?} should be '{expected_str}'"
        );
    }
}

// =============================================================================
// DType conversion edge cases
// =============================================================================

#[test]
fn test_f32_to_f16_precision_loss() {
    use half::f16;
    // 1.0000001 in f32 loses its sub-ulp detail in f16.
    let original: f32 = 1.0000001;
    let as_f16 = f16::from_f32(original);
    let back: f32 = as_f16.to_f32();
    // f16 has ~3.3 decimal digits of precision, so 1.0000001 rounds to 1.0.
    assert_ne!(
        original, back,
        "f16 round-trip should lose precision for 1.0000001"
    );
    assert!(
        (back - 1.0).abs() < 1e-3,
        "f16 round-trip should be close to 1.0"
    );

    // Small subnormal values also lose precision.
    let small: f32 = 1e-7;
    let small_f16 = f16::from_f32(small);
    let small_back = small_f16.to_f32();
    // f16 min subnormal is ~5.96e-8, so 1e-7 may round.
    assert!(
        (small_back - small).abs() <= small.abs(),
        "f16 round-trip should not explode for small values"
    );
}

#[test]
fn test_bf16_range() {
    use half::bf16;
    // bf16 has the same exponent range as f32 (8 exponent bits),
    // but only 7 mantissa bits vs f32's 23.
    let large: f32 = 1.0e38; // well within bf16 max (~3.39e38)
    let as_bf16 = bf16::from_f32(large);
    let back: f32 = as_bf16.to_f32();
    // bf16 can represent the magnitude, but with less precision.
    assert!(back.is_finite(), "bf16 should handle large f32 magnitudes");
    assert!(
        (back - large).abs() / large < 0.01,
        "bf16 should preserve large magnitude within ~1%"
    );

    // bf16 precision is much lower than f32 for typical values.
    let typical: f32 = 1.234_567_9;
    let bf16_val = bf16::from_f32(typical);
    let bf16_back = bf16_val.to_f32();
    // bf16 has ~2.4 decimal digits of precision.
    assert!(
        (bf16_back - typical).abs() < 0.02,
        "bf16 should be within 0.02 of original for typical values"
    );
    // But not exact:
    assert_ne!(
        bf16_back, typical,
        "bf16 should lose precision compared to f32"
    );
}

#[test]
fn test_dtype_compatibility_same_byte_width() {
    // Dtypes with the same byte width can potentially share GPU buffers
    // (subject to same_gpu_byte_width guards). Verify the byte-width groupings.
    let two_byte = [DType::F16, DType::BF16];
    let four_byte = [DType::F32, DType::I32, DType::U32];
    let eight_byte = [DType::F64, DType::I64];
    let one_byte = [DType::U8, DType::Bool];

    for a in &two_byte {
        for b in &two_byte {
            assert_eq!(
                a.size_bytes(),
                b.size_bytes(),
                "{a} and {b} should have same byte width"
            );
        }
    }
    for a in &four_byte {
        for b in &four_byte {
            assert_eq!(
                a.size_bytes(),
                b.size_bytes(),
                "{a} and {b} should have same byte width"
            );
        }
    }
    for a in &eight_byte {
        for b in &eight_byte {
            assert_eq!(
                a.size_bytes(),
                b.size_bytes(),
                "{a} and {b} should have same byte width"
            );
        }
    }
    for a in &one_byte {
        for b in &one_byte {
            assert_eq!(
                a.size_bytes(),
                b.size_bytes(),
                "{a} and {b} should have same byte width"
            );
        }
    }

    // Cross-group: different byte widths.
    assert_ne!(DType::F16.size_bytes(), DType::F32.size_bytes());
    assert_ne!(DType::F32.size_bytes(), DType::F64.size_bytes());
    assert_ne!(DType::U8.size_bytes(), DType::U32.size_bytes());
}

#[test]
fn test_f16_special_values() {
    use half::f16;
    // f16 can represent infinity and NaN.
    let inf = f16::from_f32(f32::INFINITY);
    assert!(inf.to_f32().is_infinite());

    let neg_inf = f16::from_f32(f32::NEG_INFINITY);
    assert!(neg_inf.to_f32().is_infinite());
    assert!(neg_inf.to_f32().is_sign_negative());

    let nan = f16::from_f32(f32::NAN);
    assert!(nan.to_f32().is_nan());

    // f16 max is ~65504
    let f16_max = f16::MAX.to_f32();
    assert!(f16_max > 65000.0);
    assert!(f16_max < 66000.0);
}

#[test]
fn test_bf16_special_values() {
    use half::bf16;
    // bf16 can represent infinity and NaN.
    let inf = bf16::from_f32(f32::INFINITY);
    assert!(inf.to_f32().is_infinite());

    let nan = bf16::from_f32(f32::NAN);
    assert!(nan.to_f32().is_nan());

    // bf16 max is ~3.39e38 (same exponent range as f32).
    let bf16_max = bf16::MAX.to_f32();
    assert!(bf16_max > 3.3e38);
}

#[test]
fn test_dtype_float_int_bool_partition() {
    // Every DType variant is exactly one of: float, int, or bool (neither).
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in &all {
        let is_f = dt.is_float();
        let is_i = dt.is_int();
        // Float and int are mutually exclusive.
        assert!(!(is_f && is_i), "{dt} cannot be both float and int");
        // Bool is the only type that is neither float nor int.
        if !is_f && !is_i {
            assert_eq!(
                *dt,
                DType::Bool,
                "only Bool should be neither float nor int"
            );
        }
    }
}

#[test]
fn test_mixed_precision_policy_dtype_for_op_all_categories() {
    // Test dtype_for_op with every category variant on every policy preset.
    let policies = [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
    ];
    let categories = [
        OpDTypeCategory::Compute,
        OpDTypeCategory::Accumulate,
        OpDTypeCategory::Inherit,
    ];
    for policy in &policies {
        for &cat in &categories {
            let dt = policy.dtype_for_op(cat);
            // Result must always be a float dtype.
            assert!(
                dt.is_float(),
                "dtype_for_op({cat:?}) on {policy:?} returned non-float {dt}"
            );
        }
    }
}

#[test]
fn test_f32_to_f16_overflow_clamps_to_inf() {
    use half::f16;
    // Values above f16 max (~65504) overflow to infinity in f16.
    let big: f32 = 100_000.0;
    let as_f16 = f16::from_f32(big);
    assert!(
        as_f16.to_f32().is_infinite(),
        "f32 value {big} should overflow to infinity in f16"
    );
}

#[test]
fn test_bf16_preserves_sign() {
    use half::bf16;
    let neg: f32 = -42.5;
    let as_bf16 = bf16::from_f32(neg);
    let back = as_bf16.to_f32();
    assert!(back < 0.0, "bf16 should preserve sign");
    assert!((back - neg).abs() < 1.0, "bf16 should be close to original");
}
