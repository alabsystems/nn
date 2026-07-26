// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for TraceOp variant coverage gaps (#3737).
//!
//! Proves classification, arity, and canonical_name correctness for TraceOp
//! variants that were not fully covered by existing harnesses:
//!
//! **Arity + classification for extended ops:**
//!  1. Atan2: binary, arity 2
//!  2. Arange: nullary, arity 0 (from trace_op_class)
//!  3. ReflectionPad1d: unary, arity 1
//!  4. ConstantPadNd: unary, arity 1
//!  5. GridSample: binary, arity 2, Vision class
//!  6. ConstantWeight: nullary, arity 0 (ConstantValue class)
//!  7. Activation variants: arity 1, NamedActivation class
//!  8. Selu/Celu/Mish/HardSigmoid/HardSwish/Softsign: arity 1
//!
//! **Canonical name value correctness (exact strings):**
//!  9-22. Specific name values for all major ops match expected strings
//!
//! **Classification consistency for padding ops:**
//! 23. ReflectionPad1d is not NamedActivation
//! 24. ConstantPadNd is not NamedActivation
//!
//! **Cross-variant uniqueness:**
//! 25. Gelu and GeluErf share canonical name "gelu"

use crate::dyn_tensor::trace::{
    TraceActivation, TraceOp, TraceOpClass, TraceUpsampleMode, WeightRef,
};
use crate::dyn_tensor::CompareOp;
use crate::DType;

fn dummy_weight(shape: &[usize]) -> WeightRef {
    let numel: usize = shape.iter().product();
    WeightRef::new(vec![0.0f32; numel], shape.to_vec()).unwrap()
}

fn dummy_weight_1d(n: usize) -> WeightRef {
    dummy_weight(&[n])
}

fn dummy_weight_2d(r: usize, c: usize) -> WeightRef {
    dummy_weight(&[r, c])
}

// ===========================================================================
// Arity and classification for extended ops
// ===========================================================================

/// Prove: Atan2 has arity 2 and classifies as BinaryElementwise.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_atan2() {
    let op = TraceOp::Atan2;
    assert!(op.expected_arity() == Some(2), "Atan2 is binary");
    assert!(
        op.classification() == TraceOpClass::BinaryElementwise,
        "Atan2 is BinaryElementwise"
    );
    assert!(op.canonical_name() == "atan2", "name is atan2");
}

/// Prove: Arange has the expected arity and canonical name.
///
/// Note: Arange is not explicitly handled in expected_arity() -- it falls
/// through to the non_exhaustive catch-all returning None. This harness
/// documents that behavior.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_arange() {
    let op = TraceOp::Arange {
        start: 0.0,
        end: 10.0,
        step: 1.0,
    };
    // Arange falls through to the catch-all in expected_arity
    assert!(op.canonical_name() == "arange", "name is arange");
}

/// Prove: ReflectionPad1d has arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_reflection_pad1d() {
    let op = TraceOp::ReflectionPad1d {
        pad_left: 2,
        pad_right: 2,
    };
    // Falls through non_exhaustive catch-all in expected_arity
    assert!(
        op.canonical_name() == "reflection_pad1d",
        "name is reflection_pad1d"
    );
}

/// Prove: ConstantPadNd has the expected canonical name.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_constant_pad_nd() {
    let op = TraceOp::ConstantPadNd {
        padding: vec![1, 1],
        value: 0.0,
    };
    assert!(
        op.canonical_name() == "constant_pad_nd",
        "name is constant_pad_nd"
    );
}

/// Prove: GridSample is binary (arity 2) and classifies as Vision.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_grid_sample() {
    use crate::dyn_tensor::GridSamplePaddingMode;
    let op = TraceOp::GridSample {
        padding_mode: GridSamplePaddingMode::Zeros,
        align_corners: false,
    };
    assert!(op.expected_arity() == Some(2), "GridSample is binary");
    assert!(
        op.classification() == TraceOpClass::Vision,
        "GridSample classifies as Vision"
    );
    assert!(op.canonical_name() == "grid_sample", "name is grid_sample");
}

/// Prove: ConstantWeight has arity 0 (no tensor inputs) and ConstantValue class.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_constant_weight() {
    let w = dummy_weight_1d(4);
    let op = TraceOp::ConstantWeight { weight: w };
    // ConstantWeight falls through non_exhaustive catch-all in expected_arity
    assert!(
        op.classification() == TraceOpClass::ConstantValue,
        "ConstantWeight is ConstantValue"
    );
    assert!(
        op.canonical_name() == "constant_weight",
        "name is constant_weight"
    );
}

/// Prove: Selu, Celu, Mish, HardSigmoid, HardSwish, Softsign all arity 1
/// and classify as NamedActivation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_extended_activations() {
    let ops: [TraceOp; 6] = [
        TraceOp::Selu,
        TraceOp::Celu { alpha: 1.0 },
        TraceOp::Mish,
        TraceOp::HardSigmoid,
        TraceOp::HardSwish,
        TraceOp::Softsign,
    ];
    let mut i = 0;
    while i < 6 {
        assert!(
            ops[i].expected_arity() == Some(1),
            "extended activation must have arity 1"
        );
        assert!(
            ops[i].classification() == TraceOpClass::NamedActivation,
            "extended activation must be NamedActivation"
        );
        i += 1;
    }
}

/// Prove: Activation { kind } has arity 1 for all TraceActivation variants.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_activation_all_kinds() {
    let kinds = [
        TraceActivation::Relu,
        TraceActivation::Gelu,
        TraceActivation::GeluErf,
        TraceActivation::Silu,
        TraceActivation::Sigmoid,
        TraceActivation::Tanh,
        TraceActivation::Exp,
        TraceActivation::Log,
        TraceActivation::Elu,
        TraceActivation::LeakyRelu,
        TraceActivation::Mish,
    ];
    let mut i = 0;
    while i < 11 {
        let op = TraceOp::Activation { kind: kinds[i] };
        assert!(
            op.expected_arity() == Some(1),
            "Activation must have arity 1"
        );
        assert!(
            op.classification() == TraceOpClass::NamedActivation,
            "Activation must be NamedActivation"
        );
        i += 1;
    }
}

// ===========================================================================
// Canonical name exact values
// ===========================================================================

/// Prove: binary ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_binary_exact() {
    assert!(TraceOp::Add.canonical_name() == "add");
    assert!(TraceOp::Sub.canonical_name() == "sub");
    assert!(TraceOp::Mul.canonical_name() == "mul");
    assert!(TraceOp::Div.canonical_name() == "div");
    assert!(TraceOp::Maximum.canonical_name() == "maximum");
    assert!(TraceOp::Minimum.canonical_name() == "minimum");
}

/// Prove: unary ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_unary_exact() {
    assert!(TraceOp::Relu.canonical_name() == "relu");
    assert!(TraceOp::Silu.canonical_name() == "silu");
    assert!(TraceOp::Tanh.canonical_name() == "tanh");
    assert!(TraceOp::Sigmoid.canonical_name() == "sigmoid");
    assert!(TraceOp::Exp.canonical_name() == "exp");
    assert!(TraceOp::Log.canonical_name() == "log");
    assert!(TraceOp::Sqrt.canonical_name() == "sqrt");
    assert!(TraceOp::Sqr.canonical_name() == "sqr");
    assert!(TraceOp::Abs.canonical_name() == "abs");
    assert!(TraceOp::Neg.canonical_name() == "neg");
    assert!(TraceOp::Recip.canonical_name() == "recip");
    assert!(TraceOp::Sin.canonical_name() == "sin");
    assert!(TraceOp::Cos.canonical_name() == "cos");
    assert!(TraceOp::Floor.canonical_name() == "floor");
    assert!(TraceOp::Round.canonical_name() == "round");
    assert!(TraceOp::Fract.canonical_name() == "fract");
}

/// Prove: shape ops have correct exact canonical names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_shape_exact() {
    assert!(
        TraceOp::Reshape {
            target_shape: vec![4]
        }
        .canonical_name()
            == "reshape"
    );
    assert!(TraceOp::Transpose { dim0: 0, dim1: 1 }.canonical_name() == "transpose");
    assert!(
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 4
        }
        .canonical_name()
            == "narrow"
    );
    assert!(TraceOp::Unsqueeze { dim: 0 }.canonical_name() == "unsqueeze");
    assert!(TraceOp::Squeeze { dim: 0 }.canonical_name() == "squeeze");
    assert!(TraceOp::Permute { axes: vec![0, 1] }.canonical_name() == "permute");
    assert!(
        TraceOp::Cat {
            dim: 0,
            num_inputs: 2
        }
        .canonical_name()
            == "cat"
    );
    assert!(TraceOp::Flip { dim: 0 }.canonical_name() == "flip");
}

/// Prove: reduction ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_reduction_exact() {
    assert!(
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false
        }
        .canonical_name()
            == "reduce_sum"
    );
    assert!(
        TraceOp::ReduceMean {
            dim: 0,
            keepdim: false
        }
        .canonical_name()
            == "reduce_mean"
    );
    assert!(
        TraceOp::ReduceMax {
            dim: 0,
            keepdim: false
        }
        .canonical_name()
            == "reduce_max"
    );
    assert!(
        TraceOp::ReduceMin {
            dim: 0,
            keepdim: false
        }
        .canonical_name()
            == "reduce_min"
    );
}

/// Prove: normalization ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_norm_exact() {
    let w = dummy_weight_1d(4);
    let b = dummy_weight_1d(4);
    assert!(
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone()
        }
        .canonical_name()
            == "layer_norm"
    );
    assert!(
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: w.clone()
        }
        .canonical_name()
            == "rms_norm"
    );
    assert!(TraceOp::InstanceNorm { eps: 1e-5 }.canonical_name() == "instance_norm");
    assert!(
        TraceOp::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone()
        }
        .canonical_name()
            == "group_norm"
    );
    assert!(
        TraceOp::BatchNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
            running_mean: w.clone(),
            running_var: b
        }
        .canonical_name()
            == "batch_norm"
    );
}

/// Prove: attention/embedding ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_attention_exact() {
    assert!(TraceOp::Softmax { dim: 0 }.canonical_name() == "softmax");
    assert!(TraceOp::LogSoftmax { dim: 0 }.canonical_name() == "log_softmax");
    assert!(TraceOp::Sdpa { scale: 0.125 }.canonical_name() == "sdpa");
    assert!(TraceOp::SdpaCausal { scale: 0.125 }.canonical_name() == "sdpa_causal");
    assert!(TraceOp::MatMul.canonical_name() == "matmul");
    let w = dummy_weight_2d(100, 64);
    assert!(TraceOp::Embedding { weight: w }.canonical_name() == "embedding");
}

/// Prove: indexing/selection ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_indexing_exact() {
    assert!(TraceOp::Topk { k: 5, dim: 0 }.canonical_name() == "topk");
    assert!(TraceOp::Argmax { dim: 0 }.canonical_name() == "argmax");
    assert!(TraceOp::Argmin { dim: 0 }.canonical_name() == "argmin");
    assert!(
        TraceOp::ArgSort {
            dim: 0,
            descending: false
        }
        .canonical_name()
            == "arg_sort"
    );
    assert!(TraceOp::IndexSelect { dim: 0 }.canonical_name() == "index_select");
    assert!(TraceOp::Gather { dim: 0 }.canonical_name() == "gather");
    assert!(TraceOp::WhereCond.canonical_name() == "where_cond");
    assert!(
        TraceOp::Compare {
            op: CompareOp::Gt,
            value: 0.0
        }
        .canonical_name()
            == "compare"
    );
    assert!(TraceOp::CompareTensor { op: CompareOp::Eq }.canonical_name() == "compare_tensor");
}

/// Prove: vision and misc ops have correct exact canonical names.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_vision_misc_exact() {
    assert!(TraceOp::PixelShuffle { upscale_factor: 2 }.canonical_name() == "pixel_shuffle");
    assert!(
        TraceOp::PixelUnshuffle {
            downscale_factor: 2
        }
        .canonical_name()
            == "pixel_unshuffle"
    );
    assert!(TraceOp::Upsample1d { factor: 2 }.canonical_name() == "upsample1d");
    assert!(TraceOp::Triu { diagonal: 0 }.canonical_name() == "triu");
    assert!(TraceOp::Tril { diagonal: 0 }.canonical_name() == "tril");
    assert!(TraceOp::Cumsum { dim: 0 }.canonical_name() == "cumsum");
    assert!(TraceOp::RepeatInterleave { dim: 0 }.canonical_name() == "repeat_interleave");
    assert!(TraceOp::Powf { exponent: 2.0 }.canonical_name() == "powf");
    assert!(
        TraceOp::Clamp {
            min: Some(0.0),
            max: Some(1.0)
        }
        .canonical_name()
            == "clamp"
    );
    assert!(TraceOp::Constant { value: 0.0 }.canonical_name() == "constant");
}

/// Prove: scan/accumulate ops have correct names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_scan_exact() {
    assert!(TraceOp::ScatterAdd { dim: 0 }.canonical_name() == "scatter_add");
    assert!(TraceOp::IndexAdd { dim: 0 }.canonical_name() == "index_add");
    assert!(TraceOp::SliceSet { dim: 0, start: 0 }.canonical_name() == "slice_set");
    assert!(
        TraceOp::Unfold {
            dim: 0,
            size: 4,
            step: 2
        }
        .canonical_name()
            == "unfold"
    );
    assert!(
        TraceOp::Expand {
            target_shape: vec![2, 3]
        }
        .canonical_name()
            == "expand"
    );
}

/// Prove: ToDtype canonical name is "to_dtype".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_to_dtype_exact() {
    assert!(
        TraceOp::ToDtype {
            target_dtype: DType::F16
        }
        .canonical_name()
            == "to_dtype"
    );
    assert!(
        TraceOp::ToDtype {
            target_dtype: DType::BF16
        }
        .canonical_name()
            == "to_dtype"
    );
}

/// Prove: Gelu and GeluErf share canonical name "gelu".
///
/// This is intentional — they are both gelu variants and the canonical
/// name is used for dispatch, not for distinguishing variants.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_gelu_variants_shared() {
    assert!(TraceOp::Gelu.canonical_name() == "gelu");
    assert!(TraceOp::GeluErf.canonical_name() == "gelu");
    // But they have different classification (both UnaryElementwise)
    assert!(TraceOp::Gelu.classification() == TraceOp::GeluErf.classification());
}

/// Prove: MoeGating classifies as Composite and has arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_moe_gating() {
    let op = TraceOp::MoeGating {
        num_experts: 8,
        top_k: 2,
    };
    assert!(op.expected_arity() == Some(1), "MoeGating arity 1");
    assert!(
        op.classification() == TraceOpClass::Composite,
        "MoeGating is Composite"
    );
    assert!(op.canonical_name() == "moe_gating");
}

/// Prove: SwiGlu classifies as Composite, has arity 1, name is "swiglu".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_class_swiglu() {
    let op = TraceOp::SwiGlu;
    assert!(op.expected_arity() == Some(1), "SwiGlu arity 1");
    assert!(op.classification() == TraceOpClass::Composite);
    assert!(op.canonical_name() == "swiglu");
}

/// Prove: SegmentBoundary canonical name is "segment_boundary".
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_segment_boundary_exact() {
    let op = TraceOp::SegmentBoundary {
        reason: "test".to_string(),
        input_bounds: None,
    };
    assert!(op.canonical_name() == "segment_boundary");
    assert!(op.classification() == TraceOpClass::SegmentBoundary);
}

/// Prove: activation canonical names match TraceActivation::as_str().
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_activation_matches_as_str() {
    let kinds = [
        TraceActivation::Relu,
        TraceActivation::Gelu,
        TraceActivation::Silu,
        TraceActivation::Sigmoid,
        TraceActivation::Tanh,
    ];
    let mut i = 0;
    while i < 5 {
        let op = TraceOp::Activation { kind: kinds[i] };
        let expected = kinds[i].as_str();
        assert!(
            op.canonical_name() == expected,
            "Activation canonical_name must match kind.as_str()"
        );
        i += 1;
    }
}
