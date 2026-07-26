// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for model-level IR validation and pretty-printing.
//!
//! Extracted from `model_ir.rs` via `#[path]` pattern (#571 AC2).

use super::*;

fn sample_model() -> ModelDef {
    ModelDef::new(
        "test_model",
        vec![ModelParam::new("x", "f32"), ModelParam::new("y", "f32")],
        vec![
            ModelStep::new(
                ModelStepId(0),
                "a",
                "encoder",
                vec![ModelValueRef::Param("x".into())],
            ),
            ModelStep::new(
                ModelStepId(1),
                "b",
                "decoder",
                vec![
                    ModelValueRef::StepOutput(ModelStepId(0)),
                    ModelValueRef::Param("y".into()),
                ],
            ),
        ],
        ModelOutput::StepOutput(ModelStepId(1)),
        "f32",
    )
}

#[test]
fn test_model_def_validates() {
    let model = sample_model();
    assert!(model.validate().is_ok());
}

#[test]
fn test_model_def_rejects_forward_ref() {
    let model = ModelDef::new(
        "bad_model",
        vec![ModelParam::new("x", "f32")],
        vec![ModelStep::new(
            ModelStepId(0),
            "a",
            "encoder",
            vec![ModelValueRef::StepOutput(ModelStepId(0))],
        )],
        ModelOutput::StepOutput(ModelStepId(0)),
        "f32",
    );
    let err = model.validate().unwrap_err();
    assert!(matches!(err, ModelIRError::ForwardRef { .. }));
}

#[test]
fn test_model_def_rejects_unknown_param() {
    let model = ModelDef::new(
        "bad_model",
        vec![ModelParam::new("x", "f32")],
        vec![ModelStep::new(
            ModelStepId(0),
            "a",
            "encoder",
            vec![ModelValueRef::Param("nonexistent".into())],
        )],
        ModelOutput::StepOutput(ModelStepId(0)),
        "f32",
    );
    let err = model.validate().unwrap_err();
    assert!(matches!(err, ModelIRError::UnknownParam { .. }));
}

#[test]
fn test_model_def_rejects_invalid_output() {
    let model = ModelDef::new(
        "bad_model",
        vec![ModelParam::new("x", "f32")],
        vec![],
        ModelOutput::StepOutput(ModelStepId(0)),
        "f32",
    );
    let err = model.validate().unwrap_err();
    assert!(matches!(err, ModelIRError::InvalidOutputRef(_)));
}

#[test]
fn test_model_def_step_id_mismatch() {
    let model = ModelDef::new(
        "bad_model",
        vec![ModelParam::new("x", "f32")],
        vec![ModelStep::new(
            ModelStepId(5),
            "a",
            "encoder",
            vec![ModelValueRef::Param("x".into())],
        )],
        ModelOutput::StepOutput(ModelStepId(5)),
        "f32",
    );
    let err = model.validate().unwrap_err();
    assert!(matches!(err, ModelIRError::StepIdMismatch { .. }));
}

#[test]
fn test_model_param_output() {
    let model = ModelDef::new(
        "identity",
        vec![ModelParam::new("x", "f32")],
        vec![],
        ModelOutput::Param("x".into()),
        "f32",
    );
    assert!(model.validate().is_ok());
}

#[test]
fn test_model_pretty_print() {
    let model = sample_model();
    let output = model_ir_pretty_print(&model);
    assert!(output.contains("model test_model(x: f32, y: f32) -> f32"));
    assert!(output.contains("let a = encoder(x)"));
    assert!(output.contains("let b = decoder(a, y)"));
}

#[test]
fn test_model_pretty_print_oob_step_id_no_panic() {
    // Out-of-bounds step references should produce error markers, not panics.
    let model = ModelDef::new(
        "bad_model",
        vec![ModelParam::new("x", "f32")],
        vec![ModelStep::new(
            ModelStepId(0),
            "a",
            "encoder",
            vec![ModelValueRef::StepOutput(ModelStepId(99))],
        )],
        ModelOutput::StepOutput(ModelStepId(77)),
        "f32",
    );
    let output = model_ir_pretty_print(&model);
    assert!(output.contains("<invalid step 99>"), "arg ref: {output}");
    assert!(output.contains("<invalid step 77>"), "output ref: {output}");
}
