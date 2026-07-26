// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for DType, MixedPrecisionPolicy, and Module trait implementations.
//!
//! Covers:
//! - DType byte sizes for the requested set (F32=4, F16=2, BF16=2, U8=1, U32=4, I64=8, Bool=1)
//! - DType is_float / is_int predicates with mutual-exclusion and partition checks
//! - DType Display and Debug formatting with alternate/padding format specifiers
//! - MixedPrecisionPolicy cross-policy dtype resolution
//! - OpDTypeCategory exhaustive variant coverage
//! - default_op_category for ALL known ops (every match arm exercised)
//! - DType conversion compatibility via DynTensor::to_dtype on CPU
//! - Module trait: Sequential composition, Activation enum, Dropout identity,
//!   ModuleT blanket, Option<&M> identity/delegate, closure modules

#[allow(deprecated)]
use crate::dyn_tensor::DynTensor;
use crate::mixed_precision::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};
use crate::layers::{Activation, Dropout, Module, ModuleT, Sequential};
use crate::{DType, Device};

// =============================================================================
// A. DType byte sizes — the specific set from the task specification
// =============================================================================

#[test]
fn test_dtype_byte_sizes_requested_set() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::U32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::Bool.size_bytes(), 1);
}

#[test]
fn test_dtype_byte_sizes_remaining_variants() {
    // Complete coverage for variants not in the requested set.
    assert_eq!(DType::F64.size_bytes(), 8);
    assert_eq!(DType::I32.size_bytes(), 4);
}

#[test]
fn test_dtype_byte_size_ordering() {
    // 1-byte < 2-byte < 4-byte < 8-byte — verify the natural ordering.
    assert!(DType::U8.size_bytes() < DType::F16.size_bytes());
    assert!(DType::F16.size_bytes() < DType::F32.size_bytes());
    assert!(DType::F32.size_bytes() < DType::F64.size_bytes());
}

// =============================================================================
// B. DType is_float / is_int predicates — partition semantics
// =============================================================================

#[test]
fn test_dtype_float_predicate_complete() {
    let expected_floats = [DType::F32, DType::F16, DType::BF16, DType::F64];
    let expected_non_floats = [DType::I32, DType::I64, DType::U32, DType::U8, DType::Bool];
    for dt in expected_floats {
        assert!(dt.is_float(), "{dt:?} should be float");
        assert!(!dt.is_int(), "{dt:?} should not be int");
    }
    for dt in expected_non_floats {
        assert!(!dt.is_float(), "{dt:?} should not be float");
    }
}

#[test]
fn test_dtype_int_predicate_complete() {
    let expected_ints = [DType::I32, DType::I64, DType::U32, DType::U8];
    let expected_non_ints = [DType::F32, DType::F16, DType::BF16, DType::F64, DType::Bool];
    for dt in expected_ints {
        assert!(dt.is_int(), "{dt:?} should be int");
        assert!(!dt.is_float(), "{dt:?} should not be float");
    }
    for dt in expected_non_ints {
        assert!(!dt.is_int(), "{dt:?} should not be int");
    }
}

#[test]
fn test_dtype_tripartite_classification() {
    // Every DType falls into exactly one category: float, int, or neither (Bool).
    let all_dtypes = [
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
    let mut float_count = 0;
    let mut int_count = 0;
    let mut neither_count = 0;
    for dt in all_dtypes {
        let f = dt.is_float();
        let i = dt.is_int();
        assert!(!(f && i), "{dt:?} cannot be both float and int");
        if f {
            float_count += 1;
        } else if i {
            int_count += 1;
        } else {
            neither_count += 1;
        }
    }
    assert_eq!(float_count, 4, "should have 4 float dtypes");
    assert_eq!(int_count, 4, "should have 4 int dtypes");
    assert_eq!(neither_count, 1, "only Bool should be neither");
}

// =============================================================================
// C. DType Display and Debug formatting
// =============================================================================

#[test]
fn test_dtype_display_all_lowercase() {
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
fn test_dtype_debug_all_uppercase_enum_variants() {
    let all = [
        (DType::F32, "F32"),
        (DType::F16, "F16"),
        (DType::BF16, "BF16"),
        (DType::F64, "F64"),
        (DType::I32, "I32"),
        (DType::I64, "I64"),
        (DType::U32, "U32"),
        (DType::U8, "U8"),
        (DType::Bool, "Bool"),
    ];
    for (dt, expected) in all {
        assert_eq!(format!("{dt:?}"), expected);
    }
}

#[test]
fn test_dtype_display_in_format_string_interpolation() {
    // Verify Display works correctly when embedded in larger format strings.
    let dt = DType::BF16;
    let msg = format!("tensor dtype is {dt}");
    assert_eq!(msg, "tensor dtype is bf16");
}

#[test]
fn test_dtype_debug_with_alternate_format() {
    // Alternate Debug (#?) should still produce the variant name.
    let debug = format!("{:#?}", DType::F32);
    assert_eq!(debug, "F32");
}

// =============================================================================
// D. MixedPrecisionPolicy — cross-policy dtype resolution
// =============================================================================

#[test]
fn test_policy_f32_all_categories_resolve_to_f32() {
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
fn test_policy_apple_silicon_compute_vs_accumulate() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    // Compute-heavy ops use F16 on Apple Silicon.
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::F16);
    // Sensitive ops always use F32.
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
    // Inherit falls back to compute dtype.
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::F16);
}

#[test]
fn test_policy_cuda_bf16_compute_vs_accumulate() {
    let p = MixedPrecisionPolicy::cuda_bf16();
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Compute), DType::BF16);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
    assert_eq!(p.dtype_for_op(OpDTypeCategory::Inherit), DType::BF16);
}

#[test]
fn test_policy_default_is_f32_only() {
    let default_policy = MixedPrecisionPolicy::default();
    let f32_policy = MixedPrecisionPolicy::f32_only();
    assert_eq!(default_policy, f32_policy);
}

#[test]
fn test_policy_all_presets_accumulate_f32_invariant() {
    // Safety invariant: accumulate dtype must be F32 unless NY proves otherwise.
    for p in [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
        MixedPrecisionPolicy::default(),
    ] {
        assert_eq!(
            p.accumulate_dtype,
            DType::F32,
            "{p:?} must have F32 accumulate"
        );
    }
}

#[test]
fn test_policy_clone_produces_equal_copy() {
    let original = MixedPrecisionPolicy::apple_silicon_default();
    let cloned = original.clone();
    assert_eq!(original, cloned);
    // Fields match individually.
    assert_eq!(original.weight_dtype, cloned.weight_dtype);
    assert_eq!(original.compute_dtype, cloned.compute_dtype);
    assert_eq!(original.accumulate_dtype, cloned.accumulate_dtype);
}

#[test]
fn test_policy_debug_contains_field_names() {
    let p = MixedPrecisionPolicy::apple_silicon_default();
    let dbg = format!("{p:?}");
    assert!(dbg.contains("MixedPrecisionPolicy"));
    assert!(dbg.contains("BF16")); // weight_dtype
    assert!(dbg.contains("F16")); // compute_dtype
    assert!(dbg.contains("F32")); // accumulate_dtype
}

#[test]
fn test_policy_presets_are_distinct() {
    let f32_p = MixedPrecisionPolicy::f32_only();
    let apple_p = MixedPrecisionPolicy::apple_silicon_default();
    let cuda_p = MixedPrecisionPolicy::cuda_bf16();
    assert_ne!(f32_p, apple_p);
    assert_ne!(f32_p, cuda_p);
    assert_ne!(apple_p, cuda_p);
}

// =============================================================================
// E. OpDTypeCategory — variant exhaustiveness and trait coverage
// =============================================================================

#[test]
fn test_op_category_all_variants_distinct() {
    let c = OpDTypeCategory::Compute;
    let a = OpDTypeCategory::Accumulate;
    let i = OpDTypeCategory::Inherit;
    assert_ne!(c, a);
    assert_ne!(c, i);
    assert_ne!(a, i);
}

#[test]
fn test_op_category_copy_semantics() {
    let original = OpDTypeCategory::Compute;
    let copied = original; // Copy
    let cloned = original; // Clone
    assert_eq!(original, copied);
    assert_eq!(original, cloned);
}

#[test]
fn test_op_category_debug_strings() {
    assert_eq!(format!("{:?}", OpDTypeCategory::Compute), "Compute");
    assert_eq!(format!("{:?}", OpDTypeCategory::Accumulate), "Accumulate");
    assert_eq!(format!("{:?}", OpDTypeCategory::Inherit), "Inherit");
}

// =============================================================================
// F. default_op_category — exercise every known match arm
// =============================================================================

#[test]
fn test_default_op_category_all_compute_ops() {
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
            "op '{op}' should be Compute"
        );
    }
}

#[test]
fn test_default_op_category_all_accumulate_ops() {
    let accumulate_ops = [
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
    for op in accumulate_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Accumulate,
            "op '{op}' should be Accumulate"
        );
    }
}

#[test]
fn test_default_op_category_inherit_fallback() {
    // Known element-wise ops that fall through to the catch-all.
    let inherit_ops = [
        "relu", "gelu", "silu", "tanh", "sigmoid", "snake", "add", "mul", "sub", "div", "neg",
        "abs", "exp", "sqrt",
    ];
    for op in inherit_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Inherit,
            "op '{op}' should be Inherit (catch-all)"
        );
    }
}

#[test]
fn test_default_op_category_unknown_ops_inherit() {
    let unknowns = [
        "",
        "nn_custom_op",
        "nonexistent_layer",
        "MATMUL", // case-sensitive: uppercase is unknown
    ];
    for op in unknowns {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Inherit,
            "unknown op '{op}' should default to Inherit"
        );
    }
}

#[test]
fn test_default_op_category_is_deterministic() {
    // Calling the same op name twice must yield the same result.
    for op in ["matmul", "softmax", "relu", "unknown", ""] {
        let c1 = default_op_category(op);
        let c2 = default_op_category(op);
        assert_eq!(c1, c2, "non-deterministic for op '{op}'");
    }
}

#[test]
fn test_policy_resolves_all_compute_ops_to_reduced_precision() {
    // On Apple Silicon, every Compute op should resolve to F16.
    let p = MixedPrecisionPolicy::apple_silicon_default();
    for op in ["matmul", "linear", "conv1d", "embedding", "attention"] {
        let cat = default_op_category(op);
        assert_eq!(p.dtype_for_op(cat), DType::F16);
    }
}

#[test]
fn test_policy_resolves_all_accumulate_ops_to_full_precision() {
    // On every preset, Accumulate ops always resolve to F32.
    for p in [
        MixedPrecisionPolicy::f32_only(),
        MixedPrecisionPolicy::apple_silicon_default(),
        MixedPrecisionPolicy::cuda_bf16(),
    ] {
        for op in ["softmax", "layer_norm", "batch_norm", "sum"] {
            let cat = default_op_category(op);
            assert_eq!(
                p.dtype_for_op(cat),
                DType::F32,
                "op '{op}' on {p:?} should be F32"
            );
        }
    }
}

// =============================================================================
// G. DType conversion compatibility via DynTensor::to_dtype
// =============================================================================

#[test]
fn test_to_dtype_same_dtype_is_noop() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let t2 = t.to_dtype(DType::F32).unwrap();
    assert_eq!(t2.dtype(), DType::F32);
    assert_eq!(t2.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_dtype_f32_to_bf16_roundtrip() {
    let original = DynTensor::from_vec(vec![1.0, -2.5, 3.75], &[3], &Device::Cpu).unwrap();
    let as_bf16 = original.to_dtype(DType::BF16).unwrap();
    assert_eq!(as_bf16.dtype(), DType::BF16);
    let back = as_bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    let vals = back.to_flat_vec::<f32>().unwrap();
    for (i, (&orig, &round)) in [1.0f32, -2.5, 3.75].iter().zip(vals.iter()).enumerate() {
        assert!((orig - round).abs() < 0.1, "element {i}: {orig} vs {round}");
    }
}

#[test]
fn test_to_dtype_f32_to_f16_roundtrip() {
    let original = DynTensor::from_vec(vec![0.0, 1.0, -1.0, 100.0], &[4], &Device::Cpu).unwrap();
    let as_f16 = original.to_dtype(DType::F16).unwrap();
    assert_eq!(as_f16.dtype(), DType::F16);
    let back = as_f16.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.0).abs() < 1e-3);
    assert!((vals[1] - 1.0).abs() < 1e-3);
    assert!((vals[2] - (-1.0)).abs() < 1e-3);
    assert!((vals[3] - 100.0).abs() < 0.5);
}

#[test]
fn test_to_dtype_f32_to_u8() {
    let t = DynTensor::from_vec(vec![0.0, 1.0, 127.0, 255.0], &[4], &Device::Cpu).unwrap();
    let as_u8 = t.to_dtype(DType::U8).unwrap();
    assert_eq!(as_u8.dtype(), DType::U8);
    let vals = as_u8.to_flat_vec::<u8>().unwrap();
    assert_eq!(vals, vec![0, 1, 127, 255]);
}

#[test]
fn test_to_dtype_f32_to_u32() {
    let t = DynTensor::from_vec(vec![0.0, 42.0, 1000.0], &[3], &Device::Cpu).unwrap();
    let as_u32 = t.to_dtype(DType::U32).unwrap();
    assert_eq!(as_u32.dtype(), DType::U32);
    let vals = as_u32.to_flat_vec::<u32>().unwrap();
    assert_eq!(vals, vec![0, 42, 1000]);
}

#[test]
fn test_to_dtype_f32_to_i64() {
    let t = DynTensor::from_vec(vec![0.0, -1.0, 42.0], &[3], &Device::Cpu).unwrap();
    let as_i64 = t.to_dtype(DType::I64).unwrap();
    assert_eq!(as_i64.dtype(), DType::I64);
    let vals = as_i64.to_flat_vec::<i64>().unwrap();
    assert_eq!(vals, vec![0, -1, 42]);
}

#[test]
fn test_to_dtype_preserves_shape() {
    let t = DynTensor::from_vec(vec![1.0; 12], &[3, 4], &Device::Cpu).unwrap();
    let converted = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(converted.dims(), &[3, 4]);
}

// =============================================================================
// H. Module trait implementations
// =============================================================================

#[test]
fn test_module_closure_forward() {
    let layer = |x: &DynTensor| x.relu();
    let input = DynTensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &Device::Cpu).unwrap();
    let output = layer.forward(&input).unwrap();
    assert_eq!(
        output.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 2.0]
    );
}

#[test]
fn test_module_apply_syntax() {
    let layer = |x: &DynTensor| x.neg();
    let input = DynTensor::from_vec(vec![1.0, -2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let output = input.apply(&layer).unwrap();
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn test_module_t_blanket_impl() {
    // Every Module is automatically a ModuleT; train flag is ignored.
    let layer = |x: &DynTensor| x.relu();
    let input = DynTensor::from_vec(vec![-1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let out_train = layer.forward_t(&input, true).unwrap();
    let out_eval = layer.forward_t(&input, false).unwrap();
    assert_eq!(
        out_train.to_flat_vec::<f32>().unwrap(),
        out_eval.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_module_option_none_is_identity() {
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let none_module: Option<&fn(&DynTensor) -> crate::Result<DynTensor>> = None;
    let output = none_module.forward(&input).unwrap();
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// -- Sequential container ---

#[test]
fn test_sequential_empty_is_identity() {
    let seq = Sequential::new();
    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);
    let input = DynTensor::from_vec(vec![5.0, 10.0], &[2], &Device::Cpu).unwrap();
    let output = seq.forward(&input).unwrap();
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![5.0, 10.0]);
}

#[test]
fn test_sequential_chain_two_closures() {
    let mut seq = Sequential::new();
    seq.add_fn(DynTensor::relu);
    seq.add_fn(DynTensor::neg);
    assert_eq!(seq.len(), 2);
    let input = DynTensor::from_vec(vec![-3.0, 0.0, 3.0], &[3], &Device::Cpu).unwrap();
    let output = seq.forward(&input).unwrap();
    // relu(-3,0,3) = (0,0,3), neg = (0,0,-3)
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![0.0, 0.0, -3.0]);
}

#[test]
fn test_sequential_with_activation_module() {
    let mut seq = Sequential::new();
    seq.add(Activation::Relu);
    seq.add(Activation::Sigmoid);
    assert_eq!(seq.len(), 2);
    let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
    let output = seq.forward(&input).unwrap();
    let flat = output.to_flat_vec::<f32>().unwrap();
    // relu(-1,0,1) = (0,0,1), sigmoid(0,0,1) = (0.5, 0.5, ~0.731)
    assert!((flat[0] - 0.5).abs() < 1e-4);
    assert!((flat[1] - 0.5).abs() < 1e-4);
    assert!((flat[2] - 0.7311).abs() < 0.001);
}

#[test]
fn test_sequential_debug_format() {
    let mut seq = Sequential::new();
    seq.add_fn(DynTensor::relu);
    let dbg = format!("{seq:?}");
    assert!(dbg.contains("Sequential"));
    assert!(dbg.contains("1")); // num_layers
}

// -- Activation enum ---

#[test]
fn test_activation_all_variants_forward() {
    let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
    let activations: Vec<Activation> = vec![
        Activation::Relu,
        Activation::Gelu,
        Activation::Silu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Elu(1.0),
        Activation::LeakyRelu(0.1),
    ];
    for act in &activations {
        let output = act.forward(&input).unwrap();
        assert_eq!(output.dims(), &[3], "{act:?} should preserve shape");
        let flat = output.to_flat_vec::<f32>().unwrap();
        // All activations should return finite values for these inputs.
        for (i, v) in flat.iter().enumerate() {
            assert!(v.is_finite(), "{act:?} element {i} is not finite: {v}");
        }
    }
}

#[test]
fn test_activation_relu_zeros_negatives() {
    let input = DynTensor::from_vec(vec![-5.0, -0.1, 0.0, 0.1, 5.0], &[5], &Device::Cpu).unwrap();
    let output = Activation::Relu.forward(&input).unwrap();
    let flat = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat[0], 0.0);
    assert_eq!(flat[1], 0.0);
    assert_eq!(flat[2], 0.0);
    assert_eq!(flat[3], 0.1);
    assert_eq!(flat[4], 5.0);
}

#[test]
fn test_activation_sigmoid_range_zero_to_one() {
    let input =
        DynTensor::from_vec(vec![-100.0, -1.0, 0.0, 1.0, 100.0], &[5], &Device::Cpu).unwrap();
    let output = Activation::Sigmoid.forward(&input).unwrap();
    let flat = output.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "sigmoid element {i} = {v} out of [0,1]"
        );
    }
    // sigmoid(0) = 0.5
    assert!((flat[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_activation_tanh_range_neg_one_to_one() {
    let input =
        DynTensor::from_vec(vec![-100.0, -1.0, 0.0, 1.0, 100.0], &[5], &Device::Cpu).unwrap();
    let output = Activation::Tanh.forward(&input).unwrap();
    let flat = output.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "tanh element {i} = {v} out of [-1,1]"
        );
    }
    // tanh(0) = 0
    assert!(flat[2].abs() < 1e-6);
}

#[test]
fn test_activation_copy_and_eq() {
    let a = Activation::Relu;
    let b = a; // Copy
    assert_eq!(a, b);
    // Different variants are not equal.
    assert_ne!(Activation::Relu, Activation::Gelu);
}

// -- Dropout ---

#[test]
fn test_dropout_is_identity_at_inference() {
    let d = Dropout::new(0.5);
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let output = d.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(
        output.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn test_dropout_boundary_probs() {
    // 0.0 and 1.0 are valid drop probabilities.
    let d0 = Dropout::new(0.0);
    let d1 = Dropout::new(1.0);
    let input = DynTensor::from_vec(vec![42.0], &[1], &Device::Cpu).unwrap();
    // Both should still be identity at inference time.
    let out0 = d0.forward(&input).unwrap();
    let out1 = d1.forward(&input).unwrap();
    assert_eq!(out0.to_scalar::<f32>().unwrap(), 42.0);
    assert_eq!(out1.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_dropout_module_t_ignores_train_flag() {
    let d = Dropout::new(0.5);
    let input = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let out_train = d.forward_t(&input, true).unwrap();
    let out_eval = d.forward_t(&input, false).unwrap();
    assert_eq!(
        out_train.to_flat_vec::<f32>().unwrap(),
        out_eval.to_flat_vec::<f32>().unwrap()
    );
}

// -- Custom struct Module ---

struct ScaleLayer {
    factor: f32,
}

impl Module for ScaleLayer {
    fn forward(&self, x: &DynTensor) -> crate::Result<DynTensor> {
        x.mul_scalar(f64::from(self.factor))
    }
}

#[test]
fn test_custom_struct_module() {
    let layer = ScaleLayer { factor: 3.0 };
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let output = layer.forward(&input).unwrap();
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![3.0, 6.0, 9.0]);
}

#[test]
fn test_option_some_delegates_to_module() {
    let layer = ScaleLayer { factor: 2.0 };
    let input = DynTensor::from_vec(vec![5.0, 10.0], &[2], &Device::Cpu).unwrap();
    let some_module = Some(&layer);
    let output = some_module.forward(&input).unwrap();
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![10.0, 20.0]);
}

#[test]
fn test_sequential_add_custom_struct_module() {
    let mut seq = Sequential::new();
    seq.add(ScaleLayer { factor: 2.0 });
    seq.add_fn(DynTensor::relu);
    let input = DynTensor::from_vec(vec![-1.0, 0.5, 2.0], &[3], &Device::Cpu).unwrap();
    let output = seq.forward(&input).unwrap();
    // scale: (-2.0, 1.0, 4.0), relu: (0.0, 1.0, 4.0)
    assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![0.0, 1.0, 4.0]);
}
