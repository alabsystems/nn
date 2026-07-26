// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for [`ModelManifest`], [`StaticModelConfig`],
//! [`MixedPrecisionPolicy`], audio primitives, and Device/DType edge cases.
//!
//! These complement the inline tests in their respective modules, focusing on
//! cross-module interactions, boundary conditions, and property-based checks.

use crate::audio::{
    crossfade_blend_into, crossfade_linear_blend, hann_window, hz_to_mel_htk, hz_to_mel_slaney,
    mel_to_hz_htk, mel_to_hz_slaney,
};
use crate::device::Device;
use crate::dtype::DType;
use crate::mixed_precision::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};
use crate::model_manifest::{
    assert_divisible, assert_positive, assert_shape_compatible, ManifestValidationError,
    ModelManifest, StaticModelConfig,
};

// ==========================================================================
// Helper manifests
// ==========================================================================

/// Audio encoder: rank-3 in, rank-3 out, many weights.
struct AudioEncoder;
impl ModelManifest for AudioEncoder {
    const INPUT_RANK: usize = 3;
    const OUTPUT_RANK: usize = 3;
    const WEIGHT_NAMES: &'static [&'static str] = &[
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "norm.weight",
        "norm.bias",
    ];
    const LAYER_COUNT: usize = 12;
    const PARAM_COUNT_BOUND: usize = 5_000_000;
}

/// Classifier: rank-2 in ([B, features]), rank-1 out ([B]).
struct Classifier;
impl ModelManifest for Classifier {
    const INPUT_RANK: usize = 2;
    const OUTPUT_RANK: usize = 1;
    const WEIGHT_NAMES: &'static [&'static str] = &["fc.weight", "fc.bias"];
    const LAYER_COUNT: usize = 1;
    const PARAM_COUNT_BOUND: usize = 1024;
}

/// No-weight model (e.g., pure activation graph).
struct NoWeightModel;
impl ModelManifest for NoWeightModel {
    const INPUT_RANK: usize = 1;
    const OUTPUT_RANK: usize = 1;
    const WEIGHT_NAMES: &'static [&'static str] = &[];
    const LAYER_COUNT: usize = 3;
    const PARAM_COUNT_BOUND: usize = 0;
}

/// Large model with many weights.
struct LargeModel;
impl ModelManifest for LargeModel {
    const INPUT_RANK: usize = 4;
    const OUTPUT_RANK: usize = 4;
    const WEIGHT_NAMES: &'static [&'static str] = &[
        "layer0.w", "layer0.b", "layer1.w", "layer1.b", "layer2.w", "layer2.b", "layer3.w",
        "layer3.b", "layer4.w", "layer4.b",
    ];
    const LAYER_COUNT: usize = 100;
    const PARAM_COUNT_BOUND: usize = 1_000_000_000;
}

// ==========================================================================
// 1. Manifest construction & validation (10+ tests)
// ==========================================================================

#[test]
fn test_audio_encoder_manifest_consts() {
    assert_eq!(AudioEncoder::INPUT_RANK, 3);
    assert_eq!(AudioEncoder::OUTPUT_RANK, 3);
    assert_eq!(AudioEncoder::WEIGHT_NAMES.len(), 6);
    assert_eq!(AudioEncoder::LAYER_COUNT, 12);
    assert_eq!(AudioEncoder::PARAM_COUNT_BOUND, 5_000_000);
}

#[test]
fn test_classifier_manifest_asymmetric_ranks() {
    assert_eq!(Classifier::INPUT_RANK, 2);
    assert_eq!(Classifier::OUTPUT_RANK, 1);
    assert_ne!(Classifier::INPUT_RANK, Classifier::OUTPUT_RANK);
}

#[test]
fn test_no_weight_model_zero_weights() {
    assert_eq!(NoWeightModel::WEIGHT_NAMES.len(), 0);
    assert_eq!(NoWeightModel::PARAM_COUNT_BOUND, 0);
}

#[test]
fn test_large_model_billion_params() {
    assert_eq!(LargeModel::PARAM_COUNT_BOUND, 1_000_000_000);
    assert_eq!(LargeModel::WEIGHT_NAMES.len(), 10);
}

#[test]
fn test_validate_audio_encoder_success() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 1, 16000],
        output_dims: vec![1, 512, 100],
        weight_count: 6,
    };
    cfg.validate::<AudioEncoder>().unwrap();
}

#[test]
fn test_validate_classifier_success() {
    let cfg = StaticModelConfig {
        input_dims: vec![32, 768],
        output_dims: vec![32],
        weight_count: 2,
    };
    cfg.validate::<Classifier>().unwrap();
}

#[test]
fn test_validate_no_weight_model_zero_weights() {
    let cfg = StaticModelConfig {
        input_dims: vec![100],
        output_dims: vec![100],
        weight_count: 0,
    };
    cfg.validate::<NoWeightModel>().unwrap();
}

#[test]
fn test_validate_large_model_rank4() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 3, 224, 224],
        output_dims: vec![1, 3, 224, 224],
        weight_count: 10,
    };
    cfg.validate::<LargeModel>().unwrap();
}

#[test]
fn test_validate_first_error_is_input_rank() {
    // Both input rank and output rank wrong; input rank error comes first.
    let cfg = StaticModelConfig {
        input_dims: vec![1],        // rank 1, expected 2
        output_dims: vec![1, 2, 3], // rank 3, expected 1
        weight_count: 99,           // also wrong
    };
    let err = cfg.validate::<Classifier>().unwrap_err();
    assert!(matches!(
        err,
        ManifestValidationError::InputRankMismatch { .. }
    ));
}

#[test]
fn test_validate_output_rank_error_before_weight() {
    // Input rank is correct, output rank wrong, weight count wrong.
    let cfg = StaticModelConfig {
        input_dims: vec![1, 768],
        output_dims: vec![1, 2], // rank 2, expected 1
        weight_count: 99,
    };
    let err = cfg.validate::<Classifier>().unwrap_err();
    assert!(matches!(
        err,
        ManifestValidationError::OutputRankMismatch { .. }
    ));
}

#[test]
fn test_validate_weight_error_before_zero_dim() {
    // Ranks correct, weight count wrong, AND zero dims present.
    let cfg = StaticModelConfig {
        input_dims: vec![0, 768], // zero dim at index 0
        output_dims: vec![32],
        weight_count: 99, // wrong
    };
    let err = cfg.validate::<Classifier>().unwrap_err();
    assert!(matches!(
        err,
        ManifestValidationError::WeightCountMismatch { .. }
    ));
}

#[test]
fn test_validate_zero_dim_index_in_output_offset() {
    // Confirm that zero-output-dim index is offset by input rank.
    let cfg = StaticModelConfig {
        input_dims: vec![1, 1, 16000],
        output_dims: vec![1, 0, 100], // zero at output index 1 -> overall index 4
        weight_count: 6,
    };
    let err = cfg.validate::<AudioEncoder>().unwrap_err();
    assert_eq!(err, ManifestValidationError::ZeroDimension { index: 4 });
}

#[test]
fn test_static_model_config_clone_eq() {
    let cfg = StaticModelConfig {
        input_dims: vec![1, 2, 3],
        output_dims: vec![4, 5],
        weight_count: 7,
    };
    let cfg2 = cfg.clone();
    assert_eq!(cfg, cfg2);
}

#[test]
fn test_static_model_config_debug() {
    let cfg = StaticModelConfig {
        input_dims: vec![1],
        output_dims: vec![1],
        weight_count: 0,
    };
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("StaticModelConfig"));
    assert!(dbg.contains("input_dims"));
}

#[test]
fn test_error_display_all_variants() {
    let e1 = ManifestValidationError::InputRankMismatch {
        expected: 3,
        actual: 1,
    };
    assert!(e1.to_string().contains("input rank mismatch"));
    assert!(e1.to_string().contains("expected 3"));
    assert!(e1.to_string().contains("got 1"));

    let e2 = ManifestValidationError::OutputRankMismatch {
        expected: 2,
        actual: 4,
    };
    assert!(e2.to_string().contains("output rank mismatch"));

    let e3 = ManifestValidationError::WeightCountMismatch {
        expected: 6,
        actual: 0,
    };
    assert!(e3.to_string().contains("weight count mismatch"));

    let e4 = ManifestValidationError::ZeroDimension { index: 7 };
    assert!(e4.to_string().contains("dimension 7 is zero"));
}

#[test]
fn test_error_clone_eq() {
    let e1 = ManifestValidationError::ZeroDimension { index: 0 };
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}

// -- Const assertion helpers boundary tests -----------------------------------

#[test]
fn test_assert_divisible_large_values() {
    let () = assert_divisible(usize::MAX - (usize::MAX % 7), 7);
    let () = assert_divisible(1_000_000, 1000);
}

#[test]
fn test_assert_shape_compatible_large() {
    let () = assert_shape_compatible(usize::MAX, usize::MAX);
}

#[test]
fn test_assert_positive_one() {
    let () = assert_positive(1);
}

// ==========================================================================
// 2. Mixed precision policy tests (12+ tests)
// ==========================================================================

#[test]
fn test_policy_f32_only_dtype_for_all_categories() {
    let p = MixedPrecisionPolicy::f32_only();
    for cat in [
        OpDTypeCategory::Compute,
        OpDTypeCategory::Accumulate,
        OpDTypeCategory::Inherit,
    ] {
        assert_eq!(p.dtype_for_op(cat), DType::F32);
    }
}

#[test]
fn test_apple_silicon_compute_ops_get_f16() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    let compute_ops = ["matmul", "conv1d", "linear", "embedding", "attention"];
    for op in compute_ops {
        let cat = default_op_category(op);
        assert_eq!(
            p.dtype_for_op(cat),
            DType::F16,
            "op '{op}' should resolve to F16 on Apple Silicon"
        );
    }
}

#[test]
fn test_apple_silicon_accumulate_ops_get_f32() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    let acc_ops = ["softmax", "layer_norm", "rms_norm", "batch_norm", "sum"];
    for op in acc_ops {
        let cat = default_op_category(op);
        assert_eq!(
            p.dtype_for_op(cat),
            DType::F32,
            "op '{op}' should resolve to F32 (accumulate) on Apple Silicon"
        );
    }
}

#[test]
fn test_cuda_bf16_compute_ops_get_bf16() {
    let p = MixedPrecisionPolicy::cuda_bf16();
    let compute_ops = ["matmul", "conv1d", "conv2d", "linear", "flash_attention"];
    for op in compute_ops {
        let cat = default_op_category(op);
        assert_eq!(
            p.dtype_for_op(cat),
            DType::BF16,
            "op '{op}' should resolve to BF16 on CUDA"
        );
    }
}

#[test]
fn test_default_op_category_all_compute_ops_exhaustive() {
    let compute_ops = [
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
    ];
    for op in compute_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Compute,
            "'{op}' should be Compute"
        );
    }
}

#[test]
fn test_default_op_category_all_accumulate_ops_exhaustive() {
    let acc_ops = [
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
    for op in acc_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Accumulate,
            "'{op}' should be Accumulate"
        );
    }
}

#[test]
fn test_default_op_category_inherit_for_activations() {
    let inherit_ops = [
        "relu",
        "gelu",
        "silu",
        "tanh",
        "sigmoid",
        "snake",
        "leaky_relu",
        "elu",
        "swish",
        "mish",
        "add",
        "mul",
        "sub",
        "div",
    ];
    for op in inherit_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Inherit,
            "'{op}' should be Inherit"
        );
    }
}

#[test]
fn test_custom_policy_with_all_different_dtypes() {
    let p = MixedPrecisionPolicy {
        weight_dtype: DType::F64,
        compute_dtype: DType::BF16,
        accumulate_dtype: DType::F32,
    };
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::BF16);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::BF16);
}

#[test]
fn test_policy_ne_different_weight_dtype() {
    let p1 = MixedPrecisionPolicy {
        weight_dtype: DType::F32,
        compute_dtype: DType::F32,
        accumulate_dtype: DType::F32,
    };
    let p2 = MixedPrecisionPolicy {
        weight_dtype: DType::F16,
        compute_dtype: DType::F32,
        accumulate_dtype: DType::F32,
    };
    assert_ne!(p1, p2);
}

#[test]
fn test_policy_ne_different_compute_dtype() {
    let p1 = MixedPrecisionPolicy::f32_only();
    let p2 = MixedPrecisionPolicy {
        weight_dtype: DType::F32,
        compute_dtype: DType::F16,
        accumulate_dtype: DType::F32,
    };
    assert_ne!(p1, p2);
}

#[test]
fn test_policy_ne_different_accumulate_dtype() {
    let p1 = MixedPrecisionPolicy::f32_only();
    let p2 = MixedPrecisionPolicy {
        weight_dtype: DType::F32,
        compute_dtype: DType::F32,
        accumulate_dtype: DType::F16,
    };
    assert_ne!(p1, p2);
}

#[test]
fn test_all_presets_weight_dtype_is_float() {
    let presets = [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
        MixedPrecisionPolicy::default(),
    ];
    for p in &presets {
        assert!(
            p.weight_dtype.is_float(),
            "weight_dtype must be float in {p:?}"
        );
        assert!(
            p.compute_dtype.is_float(),
            "compute_dtype must be float in {p:?}"
        );
        assert!(
            p.accumulate_dtype.is_float(),
            "accumulate_dtype must be float in {p:?}"
        );
    }
}

// ==========================================================================
// 3. Audio module tests (10+ tests)
// ==========================================================================

#[test]
fn test_htk_mel_monotonically_increasing() {
    let freqs = [0.0, 100.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
    let mels: Vec<f64> = freqs.iter().map(|&f| hz_to_mel_htk(f)).collect();
    for i in 1..mels.len() {
        assert!(
            mels[i] > mels[i - 1],
            "mel should increase: mel[{}]={} <= mel[{}]={}",
            i,
            mels[i],
            i - 1,
            mels[i - 1]
        );
    }
}

#[test]
fn test_slaney_mel_monotonically_increasing() {
    let freqs = [0.0, 100.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
    let mels: Vec<f64> = freqs.iter().map(|&f| hz_to_mel_slaney(f)).collect();
    for i in 1..mels.len() {
        assert!(
            mels[i] > mels[i - 1],
            "slaney mel should increase: mel[{}]={} <= mel[{}]={}",
            i,
            mels[i],
            i - 1,
            mels[i - 1]
        );
    }
}

#[test]
fn test_htk_zero_hz_is_zero_mel() {
    assert!((hz_to_mel_htk(0.0)).abs() < 1e-15);
}

#[test]
fn test_slaney_zero_hz_is_zero_mel() {
    assert!((hz_to_mel_slaney(0.0)).abs() < 1e-15);
}

#[test]
fn test_htk_inverse_large_frequencies() {
    // Test very high frequency roundtrip
    let hz = 44100.0;
    let mel = hz_to_mel_htk(hz);
    let back = mel_to_hz_htk(mel);
    assert!(
        (back - hz).abs() < 1e-6,
        "roundtrip failed for hz={hz}: got {back}"
    );
}

#[test]
fn test_slaney_inverse_large_frequencies() {
    let hz = 44100.0;
    let mel = hz_to_mel_slaney(hz);
    let back = mel_to_hz_slaney(mel);
    assert!(
        (back - hz).abs() < 1e-6,
        "roundtrip failed for hz={hz}: got {back}"
    );
}

#[test]
fn test_hann_window_length_one() {
    let w = hann_window(1);
    assert_eq!(w.len(), 1);
    // w[0] = 0.5 * (1 - cos(0)) = 0
    assert!(w[0].abs() < 1e-15);
}

#[test]
fn test_hann_window_length_two() {
    let w = hann_window(2);
    assert_eq!(w.len(), 2);
    // w[0] = 0.5 * (1 - cos(0)) = 0
    // w[1] = 0.5 * (1 - cos(pi)) = 1
    assert!(w[0].abs() < 1e-15);
    assert!((w[1] - 1.0).abs() < 1e-15);
}

#[test]
fn test_hann_window_sum_property() {
    // For periodic Hann window of even length n, sum should be approximately n/2
    let n = 512;
    let w = hann_window(n);
    let sum: f64 = w.iter().sum();
    let expected = n as f64 / 2.0;
    assert!(
        (sum - expected).abs() < 1.0,
        "hann sum should be ~{expected}, got {sum}"
    );
}

#[test]
fn test_crossfade_linear_blend_full_ramp() {
    // tail = [1, 1, 1, 1], head = [0, 0, 0, 0]
    // Result should linearly ramp from 1.0 to 0.0
    let tail = vec![1.0_f32; 4];
    let head = vec![0.0_f32; 4];
    let result = crossfade_linear_blend(&tail, &head, 4);
    assert_eq!(result.len(), 4);
    // alpha at i=0 is 0 => tail only => 1.0
    assert!((result[0] - 1.0).abs() < 1e-6);
    // alpha at i=3 is 1 => head only => 0.0
    assert!((result[3]).abs() < 1e-6);
}

#[test]
fn test_crossfade_blend_into_cf_one() {
    // When cf == 1, result is average
    let mut out = Vec::new();
    crossfade_blend_into(&mut out, &[2.0], &[4.0], 1, 1);
    assert_eq!(out.len(), 1);
    assert!((out[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_crossfade_linear_blend_convex_combination() {
    // Every output sample should be between min(tail, head) and max(tail, head)
    let tail = vec![0.2_f32, 0.4, 0.6, 0.8, 1.0];
    let head = vec![1.0_f32, 0.8, 0.6, 0.4, 0.2];
    let result = crossfade_linear_blend(&tail, &head, 5);
    for (i, &v) in result.iter().enumerate() {
        let lo = tail[i].min(head[i]);
        let hi = tail[i].max(head[i]);
        assert!(
            v >= lo - 1e-6 && v <= hi + 1e-6,
            "sample {i}: {v} not in [{lo}, {hi}]"
        );
    }
}

#[test]
fn test_mel_filterbank_triangular_shape() {
    // Build a minimal mel filterbank: 3 bands between 0 Hz and 8000 Hz (HTK)
    let n_mels = 3;
    let n_fft = 16;
    let sr = 16000.0;
    let fmin = 0.0;
    let fmax = sr / 2.0;

    let mel_min = hz_to_mel_htk(fmin);
    let mel_max = hz_to_mel_htk(fmax);

    // n_mels + 2 evenly spaced mel points (including edges)
    let mel_points: Vec<f64> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_htk(m)).collect();

    // Verify hz_points are monotonically increasing
    for i in 1..hz_points.len() {
        assert!(
            hz_points[i] > hz_points[i - 1],
            "hz_points not increasing at {i}"
        );
    }
    // Verify first is fmin, last is fmax
    assert!((hz_points[0] - fmin).abs() < 1e-6);
    assert!((hz_points[hz_points.len() - 1] - fmax).abs() < 1e-6);

    // Build triangular filters
    let fft_freqs: Vec<f64> = (0..=n_fft / 2)
        .map(|i| sr * f64::from(i) / f64::from(n_fft))
        .collect();

    for band in 0..n_mels {
        let f_left = hz_points[band];
        let f_center = hz_points[band + 1];
        let f_right = hz_points[band + 2];

        // Filter should be 0 outside [f_left, f_right] and peak at f_center
        for &freq in &fft_freqs {
            let weight = if freq < f_left || freq > f_right {
                0.0
            } else if freq <= f_center {
                (freq - f_left) / (f_center - f_left)
            } else {
                (f_right - freq) / (f_right - f_center)
            };
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&weight),
                "band {band} weight {weight} at freq {freq} out of [0, 1]"
            );
        }
    }
}

// ==========================================================================
// 4. Device and DType tests (8+ tests)
// ==========================================================================

#[test]
fn test_dtype_size_bytes_power_of_two() {
    // All dtypes should have power-of-two byte sizes (except Bool/U8 which are 1)
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
fn test_dtype_f64_is_largest_float() {
    assert!(DType::F64.size_bytes() > DType::F32.size_bytes());
    assert!(DType::F32.size_bytes() > DType::F16.size_bytes());
    assert_eq!(DType::F16.size_bytes(), DType::BF16.size_bytes());
}

#[test]
fn test_dtype_bool_not_int_not_float() {
    assert!(!DType::Bool.is_float());
    assert!(!DType::Bool.is_int());
}

#[test]
fn test_dtype_every_variant_is_float_xor_int_or_neither() {
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
        // No dtype should be both float and int
        assert!(
            !(dt.is_float() && dt.is_int()),
            "{dt:?} is both float and int"
        );
    }
}

#[test]
fn test_dtype_display_roundtrip_recognizable() {
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
    for (dt, expected) in all {
        assert_eq!(format!("{dt}"), expected);
    }
}

#[test]
fn test_device_hash_consistency() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Device::Cpu);
    set.insert(Device::metal());
    set.insert(Device::cuda());
    set.insert(Device::vulkan());
    set.insert(Device::Ane);
    assert_eq!(set.len(), 5);
    // Inserting the same device again should not increase the set size.
    set.insert(Device::Cpu);
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

#[test]
fn test_device_cross_variant_inequality() {
    assert_ne!(Device::Cpu, Device::metal());
    assert_ne!(Device::Cpu, Device::cuda());
    assert_ne!(Device::Cpu, Device::vulkan());
    assert_ne!(Device::Cpu, Device::Ane);
    assert_ne!(Device::metal(), Device::cuda());
    assert_ne!(Device::metal(), Device::vulkan());
    assert_ne!(Device::metal(), Device::Ane);
    assert_ne!(Device::cuda(), Device::vulkan());
    assert_ne!(Device::cuda(), Device::Ane);
    assert_ne!(Device::vulkan(), Device::Ane);
}

#[test]
fn test_device_display_contains_variant_name() {
    assert!(format!("{}", Device::Cpu).contains("CPU"));
    assert!(format!("{}", Device::metal()).contains("Metal"));
    assert!(format!("{}", Device::cuda()).contains("CUDA"));
    assert!(format!("{}", Device::vulkan()).contains("Vulkan"));
    assert!(format!("{}", Device::Ane).contains("ANE"));
}

#[test]
fn test_device_copy_semantics() {
    let d = Device::Metal { device_id: 42 };
    let d2 = d; // Copy
    assert_eq!(d, d2);
}

// ==========================================================================
// 5. Cross-module: policy + manifest interaction
// ==========================================================================

#[test]
fn test_manifest_weight_names_contain_expected_patterns() {
    // Verify that weight names follow naming conventions
    for name in AudioEncoder::WEIGHT_NAMES {
        assert!(
            name.contains("weight") || name.contains("bias"),
            "unexpected weight name pattern: '{name}'"
        );
    }
}

#[test]
fn test_policy_resolve_full_pipeline() {
    // Simulate resolving dtypes for a typical model pipeline:
    // input -> conv1d (Compute) -> layer_norm (Accumulate) -> relu (Inherit) -> linear (Compute)
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ops = ["conv1d", "layer_norm", "relu", "linear"];
    let expected = [DType::F16, DType::F32, DType::F16, DType::F16];

    for (op, &expected_dt) in ops.iter().zip(expected.iter()) {
        let cat = default_op_category(op);
        let dt = policy.dtype_for_op(cat);
        assert_eq!(
            dt, expected_dt,
            "op '{op}' expected {expected_dt:?}, got {dt:?}"
        );
    }
}
