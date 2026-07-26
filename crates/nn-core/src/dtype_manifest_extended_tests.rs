// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for DType, MixedPrecisionPolicy, ModelManifest, and Device.
//!
//! Covers:
//! - DType byte_size for all 9 variants with size relationships
//! - DType is_float / is_int exhaustive partition property
//! - DType Display uniqueness (no two variants share a display string)
//! - DType Hash and Eq consistency
//! - MixedPrecisionPolicy op categorization across all presets
//! - OpDTypeCategory all variants with trait coverage
//! - default_op_category for known ops with case sensitivity
//! - ModelManifest construction, field access, and validation edge cases
//! - StaticModelConfig validation error ordering
//! - ManifestValidationError Display and Clone
//! - Device types, predicates, and accelerator classification
//! - Edge cases: same-type DType comparison, mixed precision all-compute pipeline

use crate::device::Device;
use crate::dtype::DType;
use crate::mixed_precision::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};
use crate::model_manifest::{
    assert_divisible, assert_positive, assert_shape_compatible, ManifestValidationError,
    ModelManifest, StaticModelConfig,
};
use std::collections::{HashMap, HashSet};

// ==========================================================================
// Helper manifests
// ==========================================================================

/// Image classifier: rank-4 in ([B, C, H, W]), rank-2 out ([B, Classes]).
struct ImageClassifier;
impl ModelManifest for ImageClassifier {
    const INPUT_RANK: usize = 4;
    const OUTPUT_RANK: usize = 2;
    const WEIGHT_NAMES: &'static [&'static str] = &[
        "backbone.conv1.weight",
        "backbone.conv1.bias",
        "backbone.bn1.weight",
        "backbone.bn1.bias",
        "head.fc.weight",
        "head.fc.bias",
    ];
    const LAYER_COUNT: usize = 20;
    const PARAM_COUNT_BOUND: usize = 25_000_000;
}

/// Scalar model: rank-1 in, rank-1 out, single weight.
struct ScalarModel;
impl ModelManifest for ScalarModel {
    const INPUT_RANK: usize = 1;
    const OUTPUT_RANK: usize = 1;
    const WEIGHT_NAMES: &'static [&'static str] = &["scale"];
    const LAYER_COUNT: usize = 1;
    const PARAM_COUNT_BOUND: usize = 1;
}

/// Zero-layer model (identity).
struct IdentityModel;
impl ModelManifest for IdentityModel {
    const INPUT_RANK: usize = 2;
    const OUTPUT_RANK: usize = 2;
    const WEIGHT_NAMES: &'static [&'static str] = &[];
    const LAYER_COUNT: usize = 0;
    const PARAM_COUNT_BOUND: usize = 0;
}

// ==========================================================================
// 1. DType byte_size — all variants
// ==========================================================================

#[test]
fn test_dtype_byte_size_f32() {
    assert_eq!(DType::F32.size_bytes(), 4);
}

#[test]
fn test_dtype_byte_size_f16() {
    assert_eq!(DType::F16.size_bytes(), 2);
}

#[test]
fn test_dtype_byte_size_bf16() {
    assert_eq!(DType::BF16.size_bytes(), 2);
}

#[test]
fn test_dtype_byte_size_f64() {
    assert_eq!(DType::F64.size_bytes(), 8);
}

#[test]
fn test_dtype_byte_size_i32() {
    assert_eq!(DType::I32.size_bytes(), 4);
}

#[test]
fn test_dtype_byte_size_u8() {
    assert_eq!(DType::U8.size_bytes(), 1);
}

#[test]
fn test_dtype_byte_size_u32() {
    assert_eq!(DType::U32.size_bytes(), 4);
}

#[test]
fn test_dtype_byte_size_i64() {
    assert_eq!(DType::I64.size_bytes(), 8);
}

#[test]
fn test_dtype_byte_size_bool() {
    assert_eq!(DType::Bool.size_bytes(), 1);
}

#[test]
fn test_dtype_byte_size_all_nonzero() {
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
    for dt in all {
        assert!(dt.size_bytes() > 0, "{dt:?} should have nonzero byte size");
    }
}

#[test]
fn test_dtype_byte_size_all_power_of_two() {
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
    for dt in all {
        let sz = dt.size_bytes();
        assert!(sz.is_power_of_two(), "{dt:?} size {sz} is not power of 2");
    }
}

#[test]
fn test_dtype_byte_size_f16_equals_bf16() {
    assert_eq!(DType::F16.size_bytes(), DType::BF16.size_bytes());
}

#[test]
fn test_dtype_byte_size_ordering_float_hierarchy() {
    assert!(DType::F16.size_bytes() < DType::F32.size_bytes());
    assert!(DType::F32.size_bytes() < DType::F64.size_bytes());
}

// ==========================================================================
// 2. DType is_float / is_int exhaustive predicates
// ==========================================================================

#[test]
fn test_dtype_is_float_exhaustive_true() {
    for dt in [DType::F32, DType::F16, DType::BF16, DType::F64] {
        assert!(dt.is_float(), "{dt:?} should be float");
    }
}

#[test]
fn test_dtype_is_float_exhaustive_false() {
    for dt in [DType::I32, DType::I64, DType::U32, DType::U8, DType::Bool] {
        assert!(!dt.is_float(), "{dt:?} should not be float");
    }
}

#[test]
fn test_dtype_is_int_exhaustive_true() {
    for dt in [DType::I32, DType::I64, DType::U32, DType::U8] {
        assert!(dt.is_int(), "{dt:?} should be int");
    }
}

#[test]
fn test_dtype_is_int_exhaustive_false() {
    for dt in [DType::F32, DType::F16, DType::BF16, DType::F64, DType::Bool] {
        assert!(!dt.is_int(), "{dt:?} should not be int");
    }
}

#[test]
fn test_dtype_bool_is_neither_float_nor_int() {
    assert!(!DType::Bool.is_float());
    assert!(!DType::Bool.is_int());
}

#[test]
fn test_dtype_no_variant_is_both_float_and_int() {
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
    for dt in all {
        assert!(
            !(dt.is_float() && dt.is_int()),
            "{dt:?} must not be both float and int"
        );
    }
}

#[test]
fn test_dtype_partition_counts() {
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
    let float_count = all.iter().filter(|d| d.is_float()).count();
    let int_count = all.iter().filter(|d| d.is_int()).count();
    let neither_count = all.iter().filter(|d| !d.is_float() && !d.is_int()).count();
    assert_eq!(float_count, 4, "expected 4 float variants");
    assert_eq!(int_count, 4, "expected 4 int variants");
    assert_eq!(neither_count, 1, "expected 1 neither variant (Bool)");
    assert_eq!(float_count + int_count + neither_count, all.len());
}

// ==========================================================================
// 3. DType Display uniqueness
// ==========================================================================

#[test]
fn test_dtype_display_all_unique() {
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
    let mut seen = HashSet::new();
    for dt in all {
        let s = format!("{dt}");
        assert!(
            seen.insert(s.clone()),
            "duplicate Display string: '{s}' for {dt:?}"
        );
    }
    assert_eq!(seen.len(), 9);
}

#[test]
fn test_dtype_display_matches_expected_strings() {
    let expected: Vec<(DType, &str)> = vec![
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
    for (dt, exp) in expected {
        assert_eq!(format!("{dt}"), exp, "Display mismatch for {dt:?}");
    }
}

#[test]
fn test_dtype_debug_distinct_from_display() {
    // Debug uses uppercase variant names, Display uses lowercase type names.
    assert_ne!(format!("{:?}", DType::F32), format!("{}", DType::F32));
    assert_ne!(format!("{:?}", DType::Bool), format!("{}", DType::Bool));
}

#[test]
fn test_dtype_hash_all_distinct() {
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
    let set: HashSet<DType> = all.iter().copied().collect();
    assert_eq!(set.len(), 9, "all 9 DType variants should hash distinctly");
}

#[test]
fn test_dtype_eq_reflexive() {
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
    for dt in all {
        assert_eq!(dt, dt, "{dt:?} should equal itself");
    }
}

// ==========================================================================
// 4. MixedPrecisionPolicy op categorization
// ==========================================================================

#[test]
fn test_policy_f32_compute_category_returns_f32() {
    let p = MixedPrecisionPolicy::f32_only();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::F32);
}

#[test]
fn test_policy_f32_accumulate_category_returns_f32() {
    let p = MixedPrecisionPolicy::f32_only();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
}

#[test]
fn test_policy_f32_inherit_category_returns_f32() {
    let p = MixedPrecisionPolicy::f32_only();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::F32);
}

#[test]
fn test_policy_apple_compute_is_f16() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::F16);
}

#[test]
fn test_policy_apple_accumulate_is_f32() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
}

#[test]
fn test_policy_cuda_compute_is_bf16() {
    let p = MixedPrecisionPolicy::cuda_bf16();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::BF16);
}

#[test]
fn test_policy_all_presets_accumulate_is_f32() {
    for p in [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
        MixedPrecisionPolicy::default(),
    ] {
        assert_eq!(
            p.accumulate_dtype,
            DType::F32,
            "all presets must have F32 accumulate: {p:?}"
        );
    }
}

#[test]
fn test_policy_pipeline_matmul_then_softmax_apple() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    let matmul_dt = p.dtype_for_op(default_op_category("matmul"));
    let softmax_dt = p.dtype_for_op(default_op_category("softmax"));
    assert_eq!(matmul_dt, DType::F16);
    assert_eq!(softmax_dt, DType::F32);
    // Accumulate dtype should be wider than compute dtype.
    assert!(softmax_dt.size_bytes() >= matmul_dt.size_bytes());
}

// ==========================================================================
// 5. OpDTypeCategory all variants
// ==========================================================================

#[test]
fn test_op_category_compute_clone() {
    let c = OpDTypeCategory::Compute;
    let c2 = c;
    assert_eq!(c, c2);
}

#[test]
fn test_op_category_accumulate_debug() {
    assert_eq!(format!("{:?}", OpDTypeCategory::Accumulate), "Accumulate");
}

#[test]
fn test_op_category_inherit_copy() {
    let i = OpDTypeCategory::Inherit;
    let i2 = i; // Copy trait
    assert_eq!(i, i2);
}

#[test]
fn test_op_category_all_three_distinct() {
    let variants = [
        OpDTypeCategory::Compute,
        OpDTypeCategory::Accumulate,
        OpDTypeCategory::Inherit,
    ];
    for (a, b) in variants.iter().enumerate().flat_map(|(i, a)| {
        variants
            .iter()
            .enumerate()
            .filter(move |(j, _)| *j > i)
            .map(move |(_, b)| (a, b))
    }) {
        assert_ne!(a, b, "{a:?} and {b:?} should be distinct");
    }
}

// ==========================================================================
// 6. default_op_category for known ops
// ==========================================================================

#[test]
fn test_default_op_category_case_sensitive() {
    // "matmul" is Compute, but "Matmul", "MATMUL" are not recognized.
    assert_eq!(default_op_category("matmul"), OpDTypeCategory::Compute);
    assert_eq!(default_op_category("Matmul"), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("MATMUL"), OpDTypeCategory::Inherit);
}

#[test]
fn test_default_op_category_empty_string_is_inherit() {
    assert_eq!(default_op_category(""), OpDTypeCategory::Inherit);
}

#[test]
fn test_default_op_category_whitespace_is_inherit() {
    assert_eq!(default_op_category(" matmul "), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("matmul "), OpDTypeCategory::Inherit);
}

#[test]
fn test_default_op_category_all_fused_ops_are_compute() {
    let fused = [
        "norm_activ_conv1d",
        "fused_res_block",
        "norm_linear",
        "batched_linear_projection",
        "adain_snake",
        "adain_leaky_relu",
    ];
    for op in fused {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Compute,
            "fused op '{op}' should be Compute"
        );
    }
}

#[test]
fn test_default_op_category_all_norm_ops_are_accumulate() {
    let norms = [
        "layer_norm",
        "group_norm",
        "instance_norm",
        "rms_norm",
        "batch_norm",
    ];
    for op in norms {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Accumulate,
            "norm op '{op}' should be Accumulate"
        );
    }
}

#[test]
fn test_default_op_category_reduction_ops_are_accumulate() {
    let reductions = ["sum", "mean", "cumsum"];
    for op in reductions {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Accumulate,
            "reduction op '{op}' should be Accumulate"
        );
    }
}

#[test]
fn test_default_op_category_per_op_count() {
    // Track how many ops land in each category from a comprehensive list.
    let all_ops = [
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
        "linear",
        "embedding",
        "lstm_gates",
        "attention",
        "flash_attention",
        "norm_activ_conv1d",
        "fused_res_block",
        "norm_linear",
        "batched_linear_projection",
        "adain_snake",
        "adain_leaky_relu",
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
        "relu",
        "gelu",
        "silu",
        "tanh",
        "sigmoid",
    ];
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for op in all_ops {
        let cat = format!("{:?}", default_op_category(op));
        *counts
            .entry(if cat == "Compute" {
                "Compute"
            } else if cat == "Accumulate" {
                "Accumulate"
            } else {
                "Inherit"
            })
            .or_default() += 1;
    }
    assert_eq!(counts["Compute"], 16);
    assert_eq!(counts["Accumulate"], 12);
    assert_eq!(counts["Inherit"], 5);
}

// ==========================================================================
// 7. ModelManifest construction and field access
// ==========================================================================

#[test]
fn test_image_classifier_manifest_consts() {
    assert_eq!(ImageClassifier::INPUT_RANK, 4);
    assert_eq!(ImageClassifier::OUTPUT_RANK, 2);
    assert_eq!(ImageClassifier::WEIGHT_NAMES.len(), 6);
    assert_eq!(ImageClassifier::LAYER_COUNT, 20);
    assert_eq!(ImageClassifier::PARAM_COUNT_BOUND, 25_000_000);
}

#[test]
fn test_scalar_model_manifest_minimal() {
    assert_eq!(ScalarModel::INPUT_RANK, 1);
    assert_eq!(ScalarModel::OUTPUT_RANK, 1);
    assert_eq!(ScalarModel::WEIGHT_NAMES.len(), 1);
    assert_eq!(ScalarModel::WEIGHT_NAMES[0], "scale");
    assert_eq!(ScalarModel::LAYER_COUNT, 1);
    assert_eq!(ScalarModel::PARAM_COUNT_BOUND, 1);
}

#[test]
fn test_identity_model_zero_everything() {
    assert_eq!(IdentityModel::WEIGHT_NAMES.len(), 0);
    assert_eq!(IdentityModel::LAYER_COUNT, 0);
    assert_eq!(IdentityModel::PARAM_COUNT_BOUND, 0);
}

#[test]
fn test_manifest_weight_names_are_unique() {
    let names: HashSet<&str> = ImageClassifier::WEIGHT_NAMES.iter().copied().collect();
    assert_eq!(
        names.len(),
        ImageClassifier::WEIGHT_NAMES.len(),
        "weight names should be unique"
    );
}

#[test]
fn test_validate_image_classifier_success() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![1, 1000],
        weight_count: 6,
    };
    cfg.validate::<ImageClassifier>().unwrap();
}

#[test]
fn test_validate_scalar_model_success() {
    let cfg = StaticModelConfig {
        input_dims: vec![10],
        output_dims: vec![10],
        weight_count: 1,
    };
    cfg.validate::<ScalarModel>().unwrap();
}

#[test]
fn test_validate_identity_model_no_weights() {
    let cfg = StaticModelConfig {
        input_dims: vec![8, 64],
        output_dims: vec![8, 64],
        weight_count: 0,
    };
    cfg.validate::<IdentityModel>().unwrap();
}

// ==========================================================================
// 8. ModelManifest validation errors
// ==========================================================================

#[test]
fn test_validate_input_rank_error() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224], // rank 3, expected 4
        output_dims: vec![1, 1000],
        weight_count: 6,
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert_eq!(
        err,
        ManifestValidationError::InputRankMismatch {
            expected: 4,
            actual: 3
        }
    );
}

#[test]
fn test_validate_output_rank_error() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![1, 1000, 1], // rank 3, expected 2
        weight_count: 6,
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert_eq!(
        err,
        ManifestValidationError::OutputRankMismatch {
            expected: 2,
            actual: 3
        }
    );
}

#[test]
fn test_validate_weight_count_error() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![1, 1000],
        weight_count: 3, // expected 6
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert_eq!(
        err,
        ManifestValidationError::WeightCountMismatch {
            expected: 6,
            actual: 3
        }
    );
}

#[test]
fn test_validate_zero_input_dim_error() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 0, 224, 224], // zero at index 1
        output_dims: vec![1, 1000],
        weight_count: 6,
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert_eq!(err, ManifestValidationError::ZeroDimension { index: 1 });
}

#[test]
fn test_validate_zero_output_dim_error_offset() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![0, 1000], // zero at output index 0 -> overall index 4
        weight_count: 6,
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert_eq!(err, ManifestValidationError::ZeroDimension { index: 4 });
}

#[test]
fn test_validate_error_priority_input_before_output() {
    // Both input and output rank are wrong; input rank error takes priority.
    let cfg = StaticModelConfig {
        input_dims: vec![1],        // rank 1, expected 4
        output_dims: vec![1, 2, 3], // rank 3, expected 2
        weight_count: 6,
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert!(matches!(
        err,
        ManifestValidationError::InputRankMismatch { .. }
    ));
}

#[test]
fn test_validate_error_priority_weight_before_zero_dim() {
    // Ranks correct, weight wrong, zero dim present.
    let cfg = StaticModelConfig {
        input_dims: vec![0, 3, 224, 224], // zero dim
        output_dims: vec![1, 1000],
        weight_count: 99, // wrong weight count
    };
    let err = cfg.validate::<ImageClassifier>().unwrap_err();
    assert!(matches!(
        err,
        ManifestValidationError::WeightCountMismatch { .. }
    ));
}

// ==========================================================================
// 9. ManifestValidationError Display and traits
// ==========================================================================

#[test]
fn test_manifest_error_display_input_rank() {
    let err = ManifestValidationError::InputRankMismatch {
        expected: 4,
        actual: 2,
    };
    let s = err.to_string();
    assert!(s.contains("input rank mismatch"));
    assert!(s.contains("expected 4"));
    assert!(s.contains("got 2"));
}

#[test]
fn test_manifest_error_display_output_rank() {
    let err = ManifestValidationError::OutputRankMismatch {
        expected: 2,
        actual: 5,
    };
    let s = err.to_string();
    assert!(s.contains("output rank mismatch"));
    assert!(s.contains("expected 2"));
    assert!(s.contains("got 5"));
}

#[test]
fn test_manifest_error_display_weight_count() {
    let err = ManifestValidationError::WeightCountMismatch {
        expected: 6,
        actual: 0,
    };
    let s = err.to_string();
    assert!(s.contains("weight count mismatch"));
    assert!(s.contains("6 weights"));
}

#[test]
fn test_manifest_error_display_zero_dim() {
    let err = ManifestValidationError::ZeroDimension { index: 3 };
    assert!(err.to_string().contains("dimension 3 is zero"));
}

#[test]
fn test_manifest_error_clone_eq() {
    let e1 = ManifestValidationError::InputRankMismatch {
        expected: 3,
        actual: 1,
    };
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}

#[test]
fn test_manifest_error_ne_different_variants() {
    let e1 = ManifestValidationError::InputRankMismatch {
        expected: 3,
        actual: 1,
    };
    let e2 = ManifestValidationError::OutputRankMismatch {
        expected: 3,
        actual: 1,
    };
    assert_ne!(e1, e2);
}

// ==========================================================================
// 10. StaticModelConfig traits
// ==========================================================================

#[test]
fn test_static_model_config_clone_produces_equal() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![1, 1000],
        weight_count: 6,
    };
    let cfg2 = cfg.clone();
    assert_eq!(cfg, cfg2);
}

#[test]
fn test_static_model_config_ne_different_input() {
    let cfg1 = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![1, 1000],
        weight_count: 6,
    };
    let cfg2 = StaticModelConfig {
        input_dims: vec![2, 3, 224, 224],
        output_dims: vec![1, 1000],
        weight_count: 6,
    };
    assert_ne!(cfg1, cfg2);
}

#[test]
fn test_static_model_config_debug_contains_fields() {
    let cfg = StaticModelConfig {
        input_dims: vec![1],
        output_dims: vec![1],
        weight_count: 0,
    };
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("StaticModelConfig"));
    assert!(dbg.contains("input_dims"));
    assert!(dbg.contains("output_dims"));
    assert!(dbg.contains("weight_count"));
}

// ==========================================================================
// 11. Const assertion helpers
// ==========================================================================

const _DIVISIBLE_768_12: () = assert_divisible(768, 12);
const _SHAPE_COMPAT_512: () = assert_shape_compatible(512, 512);
const _POSITIVE_64: () = assert_positive(64);

#[test]
fn test_assert_divisible_various() {
    let () = assert_divisible(100, 10);
    let () = assert_divisible(0, 5); // 0 divisible by any nonzero
    let () = assert_divisible(1, 1);
}

#[test]
#[should_panic(expected = "divisor must be nonzero")]
fn test_assert_divisible_zero_divisor_panics() {
    let () = assert_divisible(10, 0);
}

#[test]
#[should_panic(expected = "not divisible")]
fn test_assert_divisible_not_divisible_panics() {
    let () = assert_divisible(7, 3);
}

#[test]
fn test_assert_shape_compatible_equal() {
    let () = assert_shape_compatible(256, 256);
}

#[test]
#[should_panic(expected = "does not match")]
fn test_assert_shape_compatible_mismatch_panics() {
    let () = assert_shape_compatible(256, 512);
}

#[test]
fn test_assert_positive_valid() {
    let () = assert_positive(1);
    let () = assert_positive(usize::MAX);
}

#[test]
#[should_panic(expected = "must be nonzero")]
fn test_assert_positive_zero_panics() {
    let () = assert_positive(0);
}

// ==========================================================================
// 12. Device types and predicates
// ==========================================================================

#[test]
fn test_device_default_is_cpu() {
    assert_eq!(Device::default(), Device::Cpu);
}

#[test]
fn test_device_convenience_constructors() {
    assert!(matches!(Device::metal(), Device::Metal { device_id: 0 }));
    assert!(matches!(Device::cuda(), Device::Cuda { device_id: 0 }));
    assert!(matches!(Device::vulkan(), Device::Vulkan { device_id: 0 }));
}

#[test]
fn test_device_is_gpu_for_all_gpu_variants() {
    assert!(Device::metal().is_gpu());
    assert!(Device::Metal { device_id: 5 }.is_gpu());
    assert!(Device::cuda().is_gpu());
    assert!(Device::Cuda { device_id: 3 }.is_gpu());
    assert!(Device::vulkan().is_gpu());
    assert!(Device::Vulkan { device_id: 7 }.is_gpu());
}

#[test]
fn test_device_is_gpu_false_for_non_gpu() {
    assert!(!Device::Cpu.is_gpu());
    assert!(!Device::Ane.is_gpu());
}

#[test]
fn test_device_is_accelerator_includes_ane() {
    assert!(Device::Ane.is_accelerator());
    assert!(Device::metal().is_accelerator());
    assert!(!Device::Cpu.is_accelerator());
}

#[test]
fn test_device_specific_predicates() {
    assert!(Device::Cpu.is_cpu());
    assert!(Device::metal().is_metal());
    assert!(Device::cuda().is_cuda());
    assert!(Device::vulkan().is_vulkan());
    assert!(Device::Ane.is_ane());
}

#[test]
fn test_device_display_all_variants() {
    assert_eq!(format!("{}", Device::Cpu), "CPU");
    assert_eq!(format!("{}", Device::metal()), "Metal(0)");
    assert_eq!(format!("{}", Device::cuda()), "CUDA(0)");
    assert_eq!(format!("{}", Device::vulkan()), "Vulkan(0)");
    assert_eq!(format!("{}", Device::Ane), "ANE");
}

#[test]
fn test_device_display_with_nonzero_id() {
    assert_eq!(format!("{}", Device::Metal { device_id: 3 }), "Metal(3)");
    assert_eq!(format!("{}", Device::Cuda { device_id: 7 }), "CUDA(7)");
    assert_eq!(format!("{}", Device::Vulkan { device_id: 2 }), "Vulkan(2)");
}

#[test]
fn test_device_hash_consistency() {
    let mut set = HashSet::new();
    set.insert(Device::Cpu);
    set.insert(Device::metal());
    set.insert(Device::cuda());
    set.insert(Device::vulkan());
    set.insert(Device::Ane);
    assert_eq!(set.len(), 5);
    // Re-inserting same values should not grow the set.
    set.insert(Device::Cpu);
    set.insert(Device::metal());
    assert_eq!(set.len(), 5);
}

#[test]
fn test_device_different_ids_are_different() {
    assert_ne!(
        Device::Metal { device_id: 0 },
        Device::Metal { device_id: 1 }
    );
    assert_ne!(Device::Cuda { device_id: 0 }, Device::Cuda { device_id: 1 });
    assert_ne!(
        Device::Vulkan { device_id: 0 },
        Device::Vulkan { device_id: 1 }
    );
}

// ==========================================================================
// 13. Edge cases: DType same-type, mixed precision all-compute pipeline
// ==========================================================================

#[test]
fn test_dtype_same_variant_equality() {
    // Verify that comparing the same DType variant is reflexively equal.
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
    for dt in all {
        let dt2 = dt; // Copy
        assert_eq!(dt, dt2);
    }
}

#[test]
fn test_mixed_precision_all_compute_pipeline() {
    // Simulate a pipeline where every op is a Compute op.
    let p = MixedPrecisionPolicy::apple_silicon_default();
    let ops = ["matmul", "linear", "conv1d", "conv2d", "embedding"];
    for op in ops {
        let cat = default_op_category(op);
        assert_eq!(cat, OpDTypeCategory::Compute);
        assert_eq!(p.dtype_for_op(cat), DType::F16);
    }
}

#[test]
fn test_mixed_precision_all_accumulate_pipeline() {
    // Simulate a pipeline where every op is an Accumulate op.
    let p = MixedPrecisionPolicy::cuda_bf16();
    let ops = ["softmax", "layer_norm", "batch_norm", "sum", "mean"];
    for op in ops {
        let cat = default_op_category(op);
        assert_eq!(cat, OpDTypeCategory::Accumulate);
        assert_eq!(p.dtype_for_op(cat), DType::F32);
    }
}

#[test]
fn test_mixed_precision_alternating_pipeline() {
    // Typical transformer pattern: matmul -> softmax -> matmul -> layer_norm
    let p = MixedPrecisionPolicy::apple_silicon_default();
    let pipeline = [
        ("matmul", DType::F16),
        ("softmax", DType::F32),
        ("matmul", DType::F16),
        ("layer_norm", DType::F32),
    ];
    for (op, expected_dt) in pipeline {
        let cat = default_op_category(op);
        assert_eq!(p.dtype_for_op(cat), expected_dt, "op '{op}'");
    }
}

#[test]
fn test_policy_custom_with_integer_dtypes() {
    // A policy can technically use integer dtypes (even if unusual).
    let p = MixedPrecisionPolicy {
        weight_dtype: DType::U8,
        compute_dtype: DType::I32,
        accumulate_dtype: DType::F32,
    };
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::I32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::I32);
    // Weight dtype is stored but not resolved via dtype_for_op.
    assert_eq!(p.weight_dtype, DType::U8);
}
