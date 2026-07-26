// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn parse_fn(src: &str) -> syn::ItemFn {
    syn::parse_str(src).expect("failed to parse test function")
}

#[test]
fn test_lower_simple_chain() {
    let func = parse_fn(
        r#"
        fn nn_model(x: f32, y: f32) -> f32 {
            let a = encoder(x);
            let b = decoder(a, y);
            b
        }
        "#,
    );
    let model = lower_model_fn(&func).expect("lowering failed");
    assert_eq!(model.name, "nn_model");
    assert_eq!(model.params.len(), 2);
    assert_eq!(model.steps.len(), 2);
    assert_eq!(model.steps[0].callee, "encoder");
    assert_eq!(model.steps[1].callee, "decoder");

    // Validate the DAG
    model.validate().expect("validation failed");
}

#[test]
fn test_lower_trailing_call() {
    let func = parse_fn(
        r#"
        fn nn_model(x: f32) -> f32 {
            let a = encode(x);
            decode(a)
        }
        "#,
    );
    let model = lower_model_fn(&func).expect("lowering failed");
    assert_eq!(model.steps.len(), 2);
    assert_eq!(model.steps[1].callee, "decode");
    assert!(matches!(model.output, ModelOutput::StepOutput(_)));
    model.validate().expect("validation failed");
}

#[test]
fn test_lower_param_return() {
    let func = parse_fn(
        r#"
        fn identity(x: f32) -> f32 {
            x
        }
        "#,
    );
    let model = lower_model_fn(&func).expect("lowering failed");
    assert_eq!(model.steps.len(), 0);
    assert!(matches!(model.output, ModelOutput::Param(ref p) if p == "x"));
    model.validate().expect("validation failed");
}

#[test]
fn test_lower_kokoro_shape() {
    let func = parse_fn(
        r#"
        fn kokoro_forward(phoneme_energy: f32, reference_mel: f32) -> f32 {
            let text_hidden = text_encoder_stub(phoneme_energy);
            let style = style_encoder_stub(reference_mel);
            let durations = duration_predictor_stub(text_hidden, style);
            let pitch = pitch_predictor_stub(text_hidden, style);
            let mel = decoder_stub(text_hidden, durations, pitch, style);
            istft_vocoder_stub(mel)
        }
        "#,
    );
    let model = lower_model_fn(&func).expect("lowering failed");
    assert_eq!(model.name, "kokoro_forward");
    assert_eq!(model.params.len(), 2);
    assert_eq!(model.steps.len(), 6);
    assert_eq!(model.steps[0].callee, "text_encoder_stub");
    assert_eq!(model.steps[1].callee, "style_encoder_stub");
    assert_eq!(model.steps[2].callee, "duration_predictor_stub");
    assert_eq!(model.steps[3].callee, "pitch_predictor_stub");
    assert_eq!(model.steps[4].callee, "decoder_stub");
    assert_eq!(model.steps[5].callee, "istft_vocoder_stub");
    model.validate().expect("validation failed");
}

#[test]
fn test_lower_rejects_arithmetic() {
    let func = parse_fn(
        r#"
        fn bad_model(x: f32) -> f32 {
            let a = x + 1.0;
            a
        }
        "#,
    );
    let err = lower_model_fn(&func).expect_err("arithmetic should be rejected");
    assert!(
        matches!(err, ModelLowerError::UnsupportedLetBinding(_)),
        "expected UnsupportedLetBinding, got: {err}"
    );
}

#[test]
fn test_lower_rejects_unknown_variable() {
    let func = parse_fn(
        r#"
        fn bad_model(x: f32) -> f32 {
            let a = encode(z);
            a
        }
        "#,
    );
    let err = lower_model_fn(&func).unwrap_err();
    assert!(matches!(err, ModelLowerError::UnknownVariable(ref v) if v == "z"));
}

#[test]
fn test_lower_rejects_missing_trailing_return_expression() {
    let func = parse_fn(
        r#"
        fn bad_model(x: f32) -> f32 {
            let a = encode(x);
            a;
        }
        "#,
    );
    let err = lower_model_fn(&func).unwrap_err();
    assert!(matches!(err, ModelLowerError::UnsupportedBody));
}

#[test]
fn test_lower_rejects_param_shadowing() {
    let func = parse_fn(
        r#"
        fn bad_model(x: f32) -> f32 {
            let x = encode(x);
            x
        }
        "#,
    );
    let err = lower_model_fn(&func).unwrap_err();
    assert!(matches!(err, ModelLowerError::ShadowedVariable(ref v) if v == "x"));
}

#[test]
fn test_lower_rejects_let_shadowing() {
    let func = parse_fn(
        r#"
        fn bad_model(x: f32) -> f32 {
            let a = encode(x);
            let a = decode(a);
            a
        }
        "#,
    );
    let err = lower_model_fn(&func).unwrap_err();
    assert!(matches!(err, ModelLowerError::ShadowedVariable(ref v) if v == "a"));
}
