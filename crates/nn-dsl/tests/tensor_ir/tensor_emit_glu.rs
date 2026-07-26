// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for GLU MSL emission: Narrow × 2 + Sigmoid + BinaryMul.
//!
//! Part of #660 AC4.

use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::{build_dispatch_plan, emit_tensor_msl, DispatchStep};

/// AC4: GLU pattern produces 4 dispatch steps (2 Narrow + 1 Sigmoid + 1 BinaryMul).
#[test]
fn test_glu_dispatch_plan() {
    let mut b = TensorBlockBuilder::new("glu_plan");
    let x = b.add_input("x", &[8, 16]);
    let glu = b.add_glu(x, 0, &[8, 16]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("GLU dispatch plan");

    // 2 Narrow + 1 Sigmoid + 1 BinaryMul = 4 steps
    assert_eq!(
        plan.len(),
        4,
        "GLU should produce 4 dispatch steps, got {}",
        plan.len()
    );

    // Verify step types in order
    assert!(
        matches!(&plan[0], DispatchStep::Narrow { .. }),
        "step 0 should be Narrow (data), got {:?}",
        &plan[0]
    );
    assert!(
        matches!(&plan[1], DispatchStep::Narrow { .. }),
        "step 1 should be Narrow (gate), got {:?}",
        &plan[1]
    );
    assert!(
        matches!(&plan[2], DispatchStep::Sigmoid { .. }),
        "step 2 should be Sigmoid, got {:?}",
        &plan[2]
    );
    assert!(
        matches!(&plan[3], DispatchStep::BinaryMul { .. }),
        "step 3 should be BinaryMul, got {:?}",
        &plan[3]
    );
}

/// AC4: GLU MSL emission produces valid Metal source with all 4 kernels.
#[test]
fn test_glu_msl_emission() {
    let mut b = TensorBlockBuilder::new("glu_emit");
    let x = b.add_input("x", &[8, 16]);
    let glu = b.add_glu(x, 0, &[8, 16]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("GLU MSL emission");

    // Must start with prelude
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "MSL prelude must be present"
    );

    // Must contain two narrow kernels
    assert!(
        msl.contains("glu_emit_narrow_n1"),
        "first narrow kernel must appear in MSL:\n{msl}"
    );
    assert!(
        msl.contains("glu_emit_narrow_n2"),
        "second narrow kernel must appear in MSL:\n{msl}"
    );

    // Must contain sigmoid kernel
    assert!(
        msl.contains("glu_emit_sigmoid_n3"),
        "sigmoid kernel must appear in MSL:\n{msl}"
    );
    assert!(
        msl.contains("metal::precise::exp(-x)"),
        "sigmoid formula must use metal::precise::exp(-x) for numerical accuracy:\n{msl}"
    );

    // Must contain binary mul kernel
    assert!(
        msl.contains("glu_emit_binary_mul_n4"),
        "binary mul kernel must appear in MSL:\n{msl}"
    );
    assert!(
        msl.contains("left[tid] * right[tid]"),
        "binary mul formula must be present:\n{msl}"
    );
}

/// GLU along axis 1 (time dimension) also emits valid MSL.
#[test]
fn test_glu_axis1_msl_emission() {
    let mut b = TensorBlockBuilder::new("glu_ax1");
    // Input [C=4, T=32], GLU along axis 1 → [C=4, T=16]
    let x = b.add_input("x", &[4, 32]);
    let glu = b.add_glu(x, 1, &[4, 32]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("GLU axis=1 MSL");
    assert!(
        msl.contains("glu_ax1_narrow_n1"),
        "narrow kernel must appear:\n{msl}"
    );
    assert!(
        msl.contains("glu_ax1_binary_mul_n4"),
        "binary mul kernel must appear:\n{msl}"
    );
}
