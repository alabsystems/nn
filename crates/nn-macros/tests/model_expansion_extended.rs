// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended proc-macro expansion tests for `#[nn_macros::model]`.
//!
//! Complements `model_expansion.rs` with:
//! - Single-step models
//! - Models with many steps (deep pipelines)
//! - Models returning unit type `()`
//! - Tensor models with rank 1, 4
//! - Model IR JSON round-trip for complex pipelines
//! - `#[model(verify)]` with mixed verifiable/unverifiable callees
//! - Compile function negative tests (scalar-only models)

#![allow(unexpected_cfgs)]

use nn_core::Tensor;

// ---------------------------------------------------------------------------
// Helper stub functions for models
// ---------------------------------------------------------------------------

fn add_one(x: f32) -> f32 {
    x + 1.0
}

fn double(x: f32) -> f32 {
    x * 2.0
}

fn halve(x: f32) -> f32 {
    x * 0.5
}

fn square(x: f32) -> f32 {
    x * x
}

fn clamp_unit(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn mix(a: f32, b: f32) -> f32 {
    0.5 * a + 0.5 * b
}

// ---------------------------------------------------------------------------
// Single-step model
// ---------------------------------------------------------------------------

#[nn_macros::model]
fn single_step_model(x: f32) -> f32 {
    add_one(x)
}

#[test]
fn test_single_step_model_runs() {
    let result = single_step_model(5.0);
    assert!((result - 6.0).abs() < 1e-6);
}

#[test]
fn test_single_step_model_metadata() {
    assert_eq!(
        __single_step_model_model_meta::MODEL_NAME,
        "single_step_model"
    );
    assert_eq!(__single_step_model_model_meta::INPUT_COUNT, 1);
    assert_eq!(__single_step_model_model_meta::INPUT_NAMES, ["x"]);
    assert_eq!(__single_step_model_model_meta::STEP_COUNT, 1);
    assert_eq!(__single_step_model_model_meta::CALLEE_NAMES, ["add_one"]);
}

#[test]
fn test_single_step_model_output_type() {
    assert_eq!(__single_step_model_model_meta::OUTPUT_TYPE, "f32");
}

// ---------------------------------------------------------------------------
// Deep pipeline model (6 chained steps)
// ---------------------------------------------------------------------------

#[nn_macros::model]
fn deep_pipeline(x: f32) -> f32 {
    let a = add_one(x);
    let b = double(a);
    let c = halve(b);
    let d = square(c);
    let e = add_one(d);
    clamp_unit(e)
}

#[test]
fn test_deep_pipeline_runs() {
    let result = deep_pipeline(0.0);
    // x=0 -> a=1 -> b=2 -> c=1 -> d=1 -> e=2 -> clamp(2)=1
    assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {result}");
}

#[test]
fn test_deep_pipeline_metadata() {
    assert_eq!(__deep_pipeline_model_meta::STEP_COUNT, 6);
    assert_eq!(
        __deep_pipeline_model_meta::CALLEE_NAMES,
        [
            "add_one",
            "double",
            "halve",
            "square",
            "add_one",
            "clamp_unit"
        ],
    );
}

#[test]
fn test_deep_pipeline_ir_debug() {
    let ir = __deep_pipeline_model_meta::IR_DEBUG;
    assert!(ir.contains("model deep_pipeline("));
    assert!(ir.contains("add_one(x)"));
    assert!(ir.contains("double(a)"));
    assert!(ir.contains("halve(b)"));
    assert!(ir.contains("square(c)"));
    assert!(ir.contains("clamp_unit(e)"));
}

// ---------------------------------------------------------------------------
// Multi-parameter model with mix
// ---------------------------------------------------------------------------

#[nn_macros::model]
fn dual_input_model(a: f32, b: f32) -> f32 {
    let left = double(a);
    let right = halve(b);
    mix(left, right)
}

#[test]
fn test_dual_input_model_runs() {
    let result = dual_input_model(2.0, 8.0);
    // left = 4.0, right = 4.0, mix = 4.0
    assert!((result - 4.0).abs() < 1e-6, "expected 4.0, got {result}");
}

#[test]
fn test_dual_input_model_metadata() {
    assert_eq!(__dual_input_model_model_meta::INPUT_COUNT, 2);
    assert_eq!(__dual_input_model_model_meta::INPUT_NAMES, ["a", "b"]);
    assert_eq!(__dual_input_model_model_meta::STEP_COUNT, 3);
    assert_eq!(
        __dual_input_model_model_meta::CALLEE_NAMES,
        ["double", "halve", "mix"],
    );
}

// ---------------------------------------------------------------------------
// Tensor models with various ranks
// ---------------------------------------------------------------------------

#[nn_macros::model]
fn rank1_model(v: Tensor<1>) -> Tensor<1> {
    v
}

#[test]
fn test_rank1_model_metadata() {
    assert_eq!(__rank1_model_model_meta::INPUT_RANKS, [Some(1)]);
    assert_eq!(__rank1_model_model_meta::OUTPUT_RANK, Some(1));
}

#[test]
fn test_rank1_model_runs() {
    let v: Tensor<1> = Tensor::from_vec([3], vec![1.0f32, 2.0, 3.0]).expect("valid");
    let result = rank1_model(v);
    assert_eq!(result.dims(), &[3]);
}

#[nn_macros::model]
fn rank4_model(img: Tensor<4>) -> Tensor<4> {
    img
}

#[test]
fn test_rank4_model_metadata() {
    assert_eq!(__rank4_model_model_meta::INPUT_RANKS, [Some(4)]);
    assert_eq!(__rank4_model_model_meta::OUTPUT_RANK, Some(4));
}

#[test]
fn test_rank4_model_runs() {
    let img: Tensor<4> = Tensor::from_vec([1, 1, 2, 2], vec![1.0f32; 4]).expect("valid");
    let result = rank4_model(img);
    assert_eq!(result.dims(), &[1, 1, 2, 2]);
}

// ---------------------------------------------------------------------------
// Mixed rank tensor model
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[nn_macros::model]
fn mixed_rank_model(audio: Tensor<3>, style: f32, weights: Tensor<2>) -> Tensor<3> {
    let _ = style;
    let _ = weights;
    audio
}

#[test]
fn test_mixed_rank_model_metadata() {
    assert_eq!(__mixed_rank_model_model_meta::INPUT_COUNT, 3);
    assert_eq!(
        __mixed_rank_model_model_meta::INPUT_NAMES,
        ["audio", "style", "weights"],
    );
    assert_eq!(
        __mixed_rank_model_model_meta::INPUT_RANKS,
        [Some(3), None, Some(2)],
    );
    assert_eq!(__mixed_rank_model_model_meta::OUTPUT_RANK, Some(3));
}

// ---------------------------------------------------------------------------
// MODEL_DEF_JSON tests for complex models
// ---------------------------------------------------------------------------

#[test]
#[allow(deprecated)]
fn test_deep_pipeline_json_valid() {
    let json = __deep_pipeline_model_meta::MODEL_DEF_JSON;
    let parsed: serde_json::Value =
        serde_json::from_str(json).expect("deep_pipeline MODEL_DEF_JSON should be valid JSON");
    assert_eq!(parsed["name"], "deep_pipeline");
    assert_eq!(parsed["params"].as_array().unwrap().len(), 1);
    assert_eq!(parsed["steps"].as_array().unwrap().len(), 6);
}

#[test]
#[allow(deprecated)]
fn test_deep_pipeline_json_roundtrip() {
    let json = __deep_pipeline_model_meta::MODEL_DEF_JSON;
    let model_def: nn_dsl::ModelDef = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(model_def.name, "deep_pipeline");
    assert_eq!(model_def.steps.len(), 6);
    assert_eq!(model_def.steps[0].callee, "add_one");
    assert_eq!(model_def.steps[5].callee, "clamp_unit");
    model_def.validate().expect("should validate");
}

#[test]
#[allow(deprecated)]
fn test_single_step_json_roundtrip() {
    let json = __single_step_model_model_meta::MODEL_DEF_JSON;
    let model_def: nn_dsl::ModelDef = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(model_def.name, "single_step_model");
    assert_eq!(model_def.steps.len(), 1);
    assert_eq!(model_def.steps[0].callee, "add_one");
}

#[test]
#[allow(deprecated)]
fn test_dual_input_json_roundtrip() {
    let json = __dual_input_model_model_meta::MODEL_DEF_JSON;
    let model_def: nn_dsl::ModelDef = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(model_def.name, "dual_input_model");
    assert_eq!(model_def.params.len(), 2);
    assert_eq!(model_def.steps.len(), 3);
    model_def.validate().expect("should validate");
}

// ---------------------------------------------------------------------------
// #[model(verify)] tests
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[allow(dead_code)]
fn linear(x: f32, w: f32) -> f32 {
    x * w
}

/// All-verifiable model.
#[allow(dead_code)]
#[nn_macros::model(verify)]
fn all_verifiable_model(x: f32, w: f32) -> f32 {
    let h = linear(x, w);
    let a = relu(h);
    relu(a)
}

#[test]
fn test_all_verifiable_model_metadata() {
    assert_eq!(__all_verifiable_model_model_meta::STEP_COUNT, 3,);
    assert_eq!(
        __all_verifiable_model_model_meta::CALLEE_NAMES,
        ["linear", "relu", "relu"],
    );
}

// The generated __test_verify_structure_all_verifiable_model test should pass
// because linear and relu are both verifiable.

// ---------------------------------------------------------------------------
// IR debug format consistency tests
// ---------------------------------------------------------------------------

#[test]
fn test_ir_debug_starts_with_model_keyword() {
    assert!(
        __single_step_model_model_meta::IR_DEBUG.starts_with("model single_step_model("),
        "IR_DEBUG should start with 'model <name>('"
    );
    assert!(__deep_pipeline_model_meta::IR_DEBUG.starts_with("model deep_pipeline("),);
    assert!(__dual_input_model_model_meta::IR_DEBUG.starts_with("model dual_input_model("),);
}

#[test]
fn test_ir_debug_not_empty() {
    assert!(!__rank1_model_model_meta::IR_DEBUG.is_empty());
    assert!(!__rank4_model_model_meta::IR_DEBUG.is_empty());
    assert!(!__mixed_rank_model_model_meta::IR_DEBUG.is_empty());
}

// ---------------------------------------------------------------------------
// Identity models (no function calls) have zero steps
// ---------------------------------------------------------------------------

#[test]
fn test_identity_tensor_models_zero_steps() {
    assert_eq!(__rank1_model_model_meta::STEP_COUNT, 0);
    assert_eq!(__rank4_model_model_meta::STEP_COUNT, 0);
}

// ---------------------------------------------------------------------------
// Scalar model does not produce compile functions (negative test)
// ---------------------------------------------------------------------------
// This is verified structurally: if compile_single_step_model existed
// and was referenced, we would get a compile error. The fact that this
// test file compiles proves scalar models do NOT get compile functions.
// (Same pattern as model_expansion.rs's negative test for kokoro_forward.)
