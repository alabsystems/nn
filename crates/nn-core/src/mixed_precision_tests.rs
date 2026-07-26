#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`MixedPrecisionPolicy`] and [`OpDTypeCategory`].

use super::*;
use crate::DType;

// -- Preset constructors ------------------------------------------------------

#[test]
fn test_f32_only_all_f32() {
    let p = MixedPrecisionPolicy::f32_only();
    assert_eq!(p.weight_dtype, DType::F32);
    assert_eq!(p.compute_dtype, DType::F32);
    assert_eq!(p.accumulate_dtype, DType::F32);
}

#[test]
fn test_apple_silicon_default_bf16_f16_f32() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(p.weight_dtype, DType::BF16);
    assert_eq!(p.compute_dtype, DType::F16);
    assert_eq!(p.accumulate_dtype, DType::F32);
}

#[test]
fn test_cuda_bf16_bf16_compute() {
    let p = MixedPrecisionPolicy::cuda_bf16();
    assert_eq!(p.weight_dtype, DType::BF16);
    assert_eq!(p.compute_dtype, DType::BF16);
    assert_eq!(p.accumulate_dtype, DType::F32);
}

#[test]
fn test_default_is_f32_only() {
    let p = MixedPrecisionPolicy::default();
    assert_eq!(p, MixedPrecisionPolicy::f32_only());
}

// -- dtype_for_op resolution --------------------------------------------------

#[test]
fn test_dtype_for_op_compute_uses_compute_dtype() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::F16);
}

#[test]
fn test_dtype_for_op_accumulate_uses_accumulate_dtype() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
}

#[test]
fn test_dtype_for_op_inherit_uses_compute_dtype() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    // Inherit defaults to compute_dtype (the tensor already has this dtype)
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::F16);
}

#[test]
fn test_dtype_for_op_f32_only_all_same() {
    let p = MixedPrecisionPolicy::f32_only();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::F32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::F32);
}

// -- Op classification --------------------------------------------------------

#[test]
fn test_default_op_category_compute_ops() {
    assert_eq!(default_op_category("matmul"), OpDTypeCategory::Compute);
    assert_eq!(default_op_category("conv1d"), OpDTypeCategory::Compute);
    assert_eq!(
        default_op_category("conv_transpose1d"),
        OpDTypeCategory::Compute
    );
    assert_eq!(default_op_category("linear"), OpDTypeCategory::Compute);
    assert_eq!(default_op_category("embedding"), OpDTypeCategory::Compute);
    assert_eq!(default_op_category("lstm_gates"), OpDTypeCategory::Compute);
    assert_eq!(default_op_category("conv2d"), OpDTypeCategory::Compute);
    assert_eq!(
        default_op_category("conv_transpose2d"),
        OpDTypeCategory::Compute
    );
    assert_eq!(default_op_category("attention"), OpDTypeCategory::Compute);
    assert_eq!(
        default_op_category("flash_attention"),
        OpDTypeCategory::Compute
    );
    assert_eq!(
        default_op_category("norm_activ_conv1d"),
        OpDTypeCategory::Compute
    );
    assert_eq!(
        default_op_category("fused_res_block"),
        OpDTypeCategory::Compute
    );
    assert_eq!(default_op_category("norm_linear"), OpDTypeCategory::Compute);
    assert_eq!(
        default_op_category("batched_linear_projection"),
        OpDTypeCategory::Compute
    );
    assert_eq!(default_op_category("adain_snake"), OpDTypeCategory::Compute);
    assert_eq!(
        default_op_category("adain_leaky_relu"),
        OpDTypeCategory::Compute
    );
}

#[test]
fn test_default_op_category_accumulate_ops() {
    assert_eq!(default_op_category("softmax"), OpDTypeCategory::Accumulate);
    assert_eq!(
        default_op_category("log_softmax"),
        OpDTypeCategory::Accumulate
    );
    assert_eq!(
        default_op_category("layer_norm"),
        OpDTypeCategory::Accumulate
    );
    assert_eq!(
        default_op_category("group_norm"),
        OpDTypeCategory::Accumulate
    );
    assert_eq!(
        default_op_category("instance_norm"),
        OpDTypeCategory::Accumulate
    );
    assert_eq!(default_op_category("rms_norm"), OpDTypeCategory::Accumulate);
    assert_eq!(
        default_op_category("batch_norm"),
        OpDTypeCategory::Accumulate
    );
    assert_eq!(default_op_category("sum"), OpDTypeCategory::Accumulate);
    assert_eq!(default_op_category("mean"), OpDTypeCategory::Accumulate);
    assert_eq!(default_op_category("log"), OpDTypeCategory::Accumulate);
    assert_eq!(default_op_category("pow"), OpDTypeCategory::Accumulate);
    assert_eq!(default_op_category("cumsum"), OpDTypeCategory::Accumulate);
}

#[test]
fn test_default_op_category_inherit_for_elementwise() {
    // Element-wise ops default to Inherit
    assert_eq!(default_op_category("relu"), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("gelu"), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("silu"), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("tanh"), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("sigmoid"), OpDTypeCategory::Inherit);
    assert_eq!(default_op_category("snake"), OpDTypeCategory::Inherit);
}

#[test]
fn test_default_op_category_unknown_is_inherit() {
    // Unknown ops safely default to Inherit
    assert_eq!(
        default_op_category("unknown_custom_op"),
        OpDTypeCategory::Inherit
    );
    assert_eq!(default_op_category(""), OpDTypeCategory::Inherit);
}

// -- Clone, Debug, PartialEq -------------------------------------------------

#[test]
fn test_policy_clone_eq() {
    let p1 = MixedPrecisionPolicy::apple_silicon_default();
    let p2 = p1.clone();
    assert_eq!(p1, p2);
}

#[test]
fn test_policy_debug_format() {
    let p = MixedPrecisionPolicy::f32_only();
    let dbg = format!("{p:?}");
    assert!(dbg.contains("MixedPrecisionPolicy"));
    assert!(dbg.contains("F32"));
}

#[test]
fn test_category_copy() {
    let c = OpDTypeCategory::Compute;
    let c2 = c; // Copy
    assert_eq!(c, c2);
}

// -- DynTensor _with_policy integration tests ---------------------------------

#[test]
fn test_matmul_with_policy_f32_only() {
    use crate::dyn_tensor::DynTensor;
    use crate::Device;

    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap();
    let policy = MixedPrecisionPolicy::f32_only();

    let result = a.matmul_with_policy(&b, &policy).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_with_policy_matches_plain_matmul() {
    use crate::dyn_tensor::DynTensor;
    use crate::Device;

    let a = DynTensor::from_vec(vec![1.0, 0.5, 0.5, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2], &Device::Cpu).unwrap();
    let policy = MixedPrecisionPolicy::f32_only();

    let with_policy = a.matmul_with_policy(&b, &policy).unwrap();
    let without_policy = a.matmul(&b).unwrap();

    let data_p = with_policy.to_flat_vec::<f32>().unwrap();
    let data_np = without_policy.to_flat_vec::<f32>().unwrap();
    for (a, b) in data_p.iter().zip(data_np.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
    }
}

#[test]
fn test_softmax_with_policy_f32_accumulate() {
    use crate::dyn_tensor::DynTensor;
    use crate::Device;

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let policy = MixedPrecisionPolicy::f32_only();

    let result = x.softmax_with_policy(0, &policy).unwrap();
    let data = result.to_flat_vec::<f32>().unwrap();
    // Should sum to 1
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Monotonically increasing
    assert!(data[0] < data[1]);
    assert!(data[1] < data[2]);
}

#[test]
fn test_layer_norm_with_policy() {
    use crate::dyn_tensor::DynTensor;
    use crate::Device;

    // [1, 2, 3, 4, 5] — mean=3, std=sqrt(2)
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], &Device::Cpu).unwrap();
    let policy = MixedPrecisionPolicy::f32_only();

    let result = x.layer_norm_with_policy(0, 1e-5, &policy).unwrap();
    let data = result.to_flat_vec::<f32>().unwrap();
    // After layer norm, mean should be ~0 and std ~1
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
    assert!((var - 1.0).abs() < 0.1, "var should be ~1, got {var}");
}

// -- Custom policy construction -----------------------------------------------

#[test]
fn test_custom_policy_manual_construction() {
    // Build a policy with F64 weights, BF16 compute, F32 accumulate
    let p = MixedPrecisionPolicy {
        weight_dtype: DType::F64,
        compute_dtype: DType::BF16,
        accumulate_dtype: DType::F32,
    };
    assert_eq!(p.weight_dtype, DType::F64);
    assert_eq!(p.compute_dtype, DType::BF16);
    assert_eq!(p.accumulate_dtype, DType::F32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::BF16);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
}

// -- Preset safety invariants -------------------------------------------------

#[test]
fn test_all_presets_accumulate_is_f32() {
    // Safety invariant: accumulate dtype should always be F32 for correctness.
    // Downgrading requires NY proof (see doc on accumulate_dtype field).
    let presets = [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
        MixedPrecisionPolicy::default(),
    ];
    for p in &presets {
        assert_eq!(
            p.accumulate_dtype,
            DType::F32,
            "preset {p:?} must have F32 accumulate dtype"
        );
    }
}

#[test]
fn test_all_presets_use_float_dtypes() {
    // All dtypes in every preset must be floating-point.
    let presets = [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
    ];
    for p in &presets {
        assert!(
            p.weight_dtype.is_float(),
            "preset {p:?} weight_dtype must be float"
        );
        assert!(
            p.compute_dtype.is_float(),
            "preset {p:?} compute_dtype must be float"
        );
        assert!(
            p.accumulate_dtype.is_float(),
            "preset {p:?} accumulate_dtype must be float"
        );
    }
}

// -- Op classification determinism --------------------------------------------

#[test]
fn test_category_determinism() {
    // Same input must always produce the same category.
    let ops = [
        "matmul",
        "softmax",
        "relu",
        "conv1d",
        "layer_norm",
        "gelu",
        "",
    ];
    for op in &ops {
        let c1 = default_op_category(op);
        let c2 = default_op_category(op);
        assert_eq!(c1, c2, "non-deterministic category for op '{op}'");
    }
}

// -- dtype_for_op with cuda preset --------------------------------------------

#[test]
fn test_cuda_bf16_dtype_for_matmul() {
    let p = MixedPrecisionPolicy::cuda_bf16();
    let cat = default_op_category("matmul");
    assert_eq!(cat, OpDTypeCategory::Compute);
    assert_eq!(p.dtype_for_op(cat), DType::BF16);
}

#[test]
fn test_cuda_bf16_dtype_for_softmax() {
    let p = MixedPrecisionPolicy::cuda_bf16();
    let cat = default_op_category("softmax");
    assert_eq!(cat, OpDTypeCategory::Accumulate);
    assert_eq!(p.dtype_for_op(cat), DType::F32);
}

// -- Inequality ---------------------------------------------------------------

#[test]
fn test_policy_inequality() {
    let f32_policy = MixedPrecisionPolicy::f32_only();
    let apple_policy = MixedPrecisionPolicy::apple_silicon_default();
    let cuda_policy = MixedPrecisionPolicy::cuda_bf16();
    assert_ne!(f32_policy, apple_policy);
    assert_ne!(f32_policy, cuda_policy);
    assert_ne!(apple_policy, cuda_policy);
}

// -- OpDTypeCategory traits ---------------------------------------------------

#[test]
fn test_category_clone_eq() {
    let c1 = OpDTypeCategory::Accumulate;
    let c2 = c1;
    assert_eq!(c1, c2);
}

#[test]
fn test_category_debug_format() {
    assert_eq!(format!("{:?}", OpDTypeCategory::Compute), "Compute");
    assert_eq!(format!("{:?}", OpDTypeCategory::Accumulate), "Accumulate");
    assert_eq!(format!("{:?}", OpDTypeCategory::Inherit), "Inherit");
}

#[test]
fn test_category_variants_are_distinct() {
    assert_ne!(OpDTypeCategory::Compute, OpDTypeCategory::Accumulate);
    assert_ne!(OpDTypeCategory::Compute, OpDTypeCategory::Inherit);
    assert_ne!(OpDTypeCategory::Accumulate, OpDTypeCategory::Inherit);
}
