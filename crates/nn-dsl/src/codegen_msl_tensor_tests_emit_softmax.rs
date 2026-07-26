// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax MSL emission tests (#737 Direction 4).
//!
//! Extracted from `codegen_msl_tensor_tests_emit.rs` to keep both under 500 lines.

use crate::codegen_msl_tensor::{build_dispatch_plan, TensorMSLCodegenError};
use crate::codegen_msl_tensor_emit::emit_tensor_msl;
use crate::ir::ScalarType;

#[test]
fn test_emit_tensor_msl_softmax_2d() {
    use crate::softmax::build_softmax;
    let def = build_softmax("sm_emit", &[4, 8], -1).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Softmax MSL emission");
    assert!(msl.starts_with("#include <metal_stdlib>"), "prelude");
    assert!(msl.contains("[[kernel]]"), "kernel attribute");
    assert!(msl.contains("sm_emit_softmax_n"), "kernel name");
    // Verify three-phase structure
    assert!(msl.contains("shared_max"), "phase 1: shared max array");
    assert!(msl.contains("shared_sum"), "phase 2: shared sum array");
    assert!(msl.contains("metal::precise::exp"), "uses precise exp");
    assert!(
        msl.contains("/ sum_val"),
        "phase 3: normalize divides by sum_val"
    );
}

#[test]
fn test_emit_tensor_msl_softmax_3d() {
    use crate::softmax::build_softmax;
    // Softmax over axis=2 of [2, 4, 8] — 8 outer slices (2*4), axis_size=8
    let def = build_softmax("sm_3d", &[2, 4, 8], 2).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Softmax MSL emission");
    assert!(msl.contains("[[kernel]]"), "kernel attribute");
    assert!(msl.contains("axis_size"), "axis_size uniform");
    assert!(msl.contains("outer_size"), "outer_size uniform");
}

#[test]
fn test_emit_tensor_msl_softmax_negative_axis() {
    use crate::softmax::build_softmax;
    // Negative axis -1 on [4, 8] should resolve to axis=1, same as positive
    let def_neg = build_softmax("sm_neg", &[4, 8], -1).expect("build");
    let def_pos = build_softmax("sm_pos", &[4, 8], 1).expect("build");
    let msl_neg = emit_tensor_msl(&def_neg, ScalarType::F32).expect("neg emission");
    let msl_pos = emit_tensor_msl(&def_pos, ScalarType::F32).expect("pos emission");
    // Both should produce structurally identical kernels (different names)
    assert!(msl_neg.contains("shared_max"), "neg: has shared_max");
    assert!(msl_pos.contains("shared_max"), "pos: has shared_max");
}

#[test]
fn test_emit_tensor_msl_softmax_dispatch_plan() {
    use crate::softmax::build_softmax;
    let def = build_softmax("sm_plan", &[4, 8], -1).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1, "softmax should be a single dispatch step");
    assert!(
        matches!(
            &plan[0],
            crate::DispatchStep::Softmax {
                axis_size: 8,
                outer_size: 4,
                ..
            }
        ),
        "expected Softmax step with axis_size=8, outer_size=4, got {:?}",
        &plan[0]
    );
}

#[test]
fn test_emit_tensor_msl_softmax_rejects_nonlast_axis() {
    use crate::softmax::build_softmax;
    // axis=0 on [4, 8] is not the last axis — should be rejected
    let def = build_softmax("sm_nonlast", &[4, 8], 0).expect("build");
    let err = emit_tensor_msl(&def, ScalarType::F32)
        .expect_err("non-last-axis softmax should be rejected");
    assert!(
        matches!(
            &err,
            TensorMSLCodegenError::NonLastAxisSoftmax {
                axis: 0,
                shape,
                ..
            } if shape.as_slice() == [4, 8]
        ),
        "unexpected error: {err:?}"
    );
}
