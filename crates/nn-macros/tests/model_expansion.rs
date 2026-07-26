// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(unexpected_cfgs)]

#[nn_macros::kernel(bounds(alpha = "0.1..1e6"))]
fn snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

fn text_encoder_stub(phoneme_energy: f32) -> f32 {
    snake(phoneme_energy, 1.5)
}

fn style_encoder_stub(reference_mel: f32) -> f32 {
    reference_mel * 0.25 + 0.1
}

fn duration_predictor_stub(text_hidden: f32, style: f32) -> f32 {
    (text_hidden + style).max(0.0)
}

fn pitch_predictor_stub(text_hidden: f32, style: f32) -> f32 {
    text_hidden - style
}

fn decoder_stub(text_hidden: f32, durations: f32, pitch: f32, style: f32) -> f32 {
    text_hidden + durations * 0.5 + pitch * 0.25 + style
}

fn istft_vocoder_stub(mel: f32) -> f32 {
    mel.tanh()
}

#[nn_macros::model]
fn kokoro_forward(phoneme_energy: f32, reference_mel: f32) -> f32 {
    let text_hidden = text_encoder_stub(phoneme_energy);
    let style = style_encoder_stub(reference_mel);
    let durations = duration_predictor_stub(text_hidden, style);
    let pitch = pitch_predictor_stub(text_hidden, style);
    let mel = decoder_stub(text_hidden, durations, pitch, style);
    istft_vocoder_stub(mel)
}

#[test]
fn test_model_reference_fn_runs() {
    let phoneme_energy = 0.3;
    let reference_mel = 0.7;

    let text_hidden = text_encoder_stub(phoneme_energy);
    let style = style_encoder_stub(reference_mel);
    let durations = duration_predictor_stub(text_hidden, style);
    let pitch = pitch_predictor_stub(text_hidden, style);
    let mel = decoder_stub(text_hidden, durations, pitch, style);
    let expected = istft_vocoder_stub(mel);

    let got = kokoro_forward(phoneme_energy, reference_mel);
    assert!(
        (got - expected).abs() < 1e-6,
        "model output mismatch: got={got}, expected={expected}",
    );
}

#[test]
fn test_model_metadata_generated() {
    assert_eq!(__kokoro_forward_model_meta::MODEL_NAME, "kokoro_forward");
    assert_eq!(__kokoro_forward_model_meta::INPUT_COUNT, 2);
    assert_eq!(
        __kokoro_forward_model_meta::INPUT_NAMES,
        ["phoneme_energy", "reference_mel"],
    );
}

#[test]
fn test_model_metadata_output_type() {
    assert_eq!(__kokoro_forward_model_meta::OUTPUT_TYPE, "f32");
}

#[test]
fn test_model_metadata_input_types_scalar() {
    assert_eq!(__kokoro_forward_model_meta::INPUT_TYPES, ["f32", "f32"],);
}

#[test]
fn test_model_metadata_input_ranks_scalar() {
    assert_eq!(__kokoro_forward_model_meta::INPUT_RANKS, [None, None],);
}

#[test]
fn test_model_metadata_output_rank_scalar() {
    assert_eq!(__kokoro_forward_model_meta::OUTPUT_RANK, None);
}

// --- Tensor-typed model test ---

use nn_core::Tensor;

#[nn_macros::model]
fn simple_encoder(audio: Tensor<2>, weights: Tensor<2>) -> Tensor<2> {
    // Stub: just return the audio tensor unchanged.
    // Real models will call kernel ops; this tests macro type extraction.
    let _ = weights;
    audio
}

#[test]
fn test_tensor_model_reference_fn_runs() {
    let audio: Tensor<2> =
        Tensor::from_vec([1, 4], vec![1.0f32, 2.0, 3.0, 4.0]).expect("valid data");
    let weights: Tensor<2> = Tensor::from_vec([4, 1], vec![0.5f32; 4]).expect("valid data");
    let result = simple_encoder(audio, weights);
    assert_eq!(result.dims(), &[1, 4]);
}

#[test]
fn test_tensor_model_metadata() {
    assert_eq!(__simple_encoder_model_meta::MODEL_NAME, "simple_encoder");
    assert_eq!(__simple_encoder_model_meta::INPUT_COUNT, 2);
    assert_eq!(
        __simple_encoder_model_meta::INPUT_NAMES,
        ["audio", "weights"],
    );
}

#[test]
fn test_tensor_model_input_ranks() {
    assert_eq!(__simple_encoder_model_meta::INPUT_RANKS, [Some(2), Some(2)],);
}

#[test]
fn test_tensor_model_output_rank() {
    assert_eq!(__simple_encoder_model_meta::OUTPUT_RANK, Some(2));
}

#[test]
fn test_tensor_model_output_type() {
    assert!(__simple_encoder_model_meta::OUTPUT_TYPE.contains("Tensor"));
}

// --- Mixed scalar + tensor model test ---

#[nn_macros::model]
fn scaled_encoder(audio: Tensor<3>, scale: f32) -> Tensor<3> {
    let _ = scale;
    audio
}

#[test]
fn test_mixed_model_reference_fn_runs() {
    let audio: Tensor<3> = Tensor::from_vec([1, 2, 3], vec![0.0f32; 6]).expect("valid data");
    let result = scaled_encoder(audio, 1.0);
    assert_eq!(result.dims(), &[1, 2, 3]);
}

#[test]
fn test_mixed_model_input_ranks() {
    assert_eq!(__scaled_encoder_model_meta::INPUT_RANKS, [Some(3), None],);
}

#[test]
fn test_mixed_model_output_rank() {
    assert_eq!(__scaled_encoder_model_meta::OUTPUT_RANK, Some(3));
}

// --- Model IR lowering tests (issue #73) ---

#[test]
fn test_model_ir_step_count() {
    assert_eq!(__kokoro_forward_model_meta::STEP_COUNT, 6);
}

#[test]
fn test_model_ir_callee_names() {
    assert_eq!(
        __kokoro_forward_model_meta::CALLEE_NAMES,
        [
            "text_encoder_stub",
            "style_encoder_stub",
            "duration_predictor_stub",
            "pitch_predictor_stub",
            "decoder_stub",
            "istft_vocoder_stub",
        ],
    );
}

#[test]
fn test_model_ir_debug_contains_steps() {
    let ir = __kokoro_forward_model_meta::IR_DEBUG;
    assert!(ir.contains("model kokoro_forward"));
    assert!(ir.contains("text_encoder_stub(phoneme_energy)"));
    assert!(ir.contains("style_encoder_stub(reference_mel)"));
    assert!(ir.contains("duration_predictor_stub(text_hidden, style)"));
    assert!(ir.contains("decoder_stub(text_hidden, durations, pitch, style)"));
    assert!(ir.contains("istft_vocoder_stub(mel)"));
}

#[test]
fn test_model_ir_identity_model_has_zero_steps() {
    // simple_encoder is an identity-like model (returns param directly)
    // but it has a `let _ = weights;` which is not a function call
    // The model body returns `audio` directly → 0 steps
    assert_eq!(__simple_encoder_model_meta::STEP_COUNT, 0);
}

#[test]
fn test_model_ir_debug_roundtrip() {
    // Verify the pretty-printed IR is not empty and has the model name
    let ir = __kokoro_forward_model_meta::IR_DEBUG;
    assert!(!ir.is_empty());
    assert!(ir.starts_with("model kokoro_forward("));
}

// --- #[model(verify)] structural verification tests (Part of #3020 D3) ---

#[allow(dead_code)] // called by proc-macro-generated model body, not directly by tests
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[allow(dead_code)] // called by proc-macro-generated model body, not directly by tests
fn linear(x: f32, w: f32) -> f32 {
    x * w
}

/// Model with all-verifiable ops: `linear` and `relu` are both in the
/// `classify_callee_name` verifiable set. The generated `__test_verify_structure_*`
/// test should pass.
#[allow(dead_code)]
#[nn_macros::model(verify)]
fn verifiable_linear_relu(x: f32, w: f32) -> f32 {
    let h = linear(x, w);
    relu(h)
}

#[test]
fn test_verify_model_metadata_exists() {
    assert_eq!(
        __verifiable_linear_relu_model_meta::MODEL_NAME,
        "verifiable_linear_relu"
    );
    assert_eq!(__verifiable_linear_relu_model_meta::STEP_COUNT, 2);
    assert_eq!(
        __verifiable_linear_relu_model_meta::CALLEE_NAMES,
        ["linear", "relu"],
    );
}

// The `__test_verify_structure_verifiable_linear_relu` test is auto-generated by
// the `#[model(verify)]` attribute. It passes because both `linear` and `relu`
// are classified as verifiable by `nn_dsl::classify_callee_name`.

// --- MODEL_DEF_JSON tests (Part of #3051 D1) ---

#[test]
#[allow(deprecated)]
fn test_model_def_json_is_valid() {
    let json = __kokoro_forward_model_meta::MODEL_DEF_JSON;
    let parsed: serde_json::Value =
        serde_json::from_str(json).expect("MODEL_DEF_JSON should be valid JSON");
    assert_eq!(parsed["name"], "kokoro_forward");
    assert_eq!(parsed["params"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["steps"].as_array().unwrap().len(), 6);
}

#[test]
#[allow(deprecated)]
fn test_model_def_json_roundtrip() {
    let json = __kokoro_forward_model_meta::MODEL_DEF_JSON;
    let model_def: nn_dsl::ModelDef =
        serde_json::from_str(json).expect("MODEL_DEF_JSON should deserialize to ModelDef");
    assert_eq!(model_def.name, "kokoro_forward");
    assert_eq!(model_def.params.len(), 2);
    assert_eq!(model_def.steps.len(), 6);
    assert_eq!(model_def.steps[0].callee, "text_encoder_stub");
    assert_eq!(model_def.steps[5].callee, "istft_vocoder_stub");
    model_def
        .validate()
        .expect("deserialized ModelDef should be valid");
}

#[test]
fn test_identity_model_def_json() {
    let json = __simple_encoder_model_meta::MODEL_DEF_JSON;
    let parsed: serde_json::Value =
        serde_json::from_str(json).expect("identity model MODEL_DEF_JSON should be valid JSON");
    assert_eq!(parsed["name"], "simple_encoder");
    assert_eq!(parsed["steps"].as_array().unwrap().len(), 0);
}

// --- DynTensor compile function generation tests (Part of #3051 D5) ---

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result as NnResult;

/// Stub functions for DynTensor model test — the model lowerer requires
/// function-call syntax (not method calls) for IR extraction.
fn relu_dt(x: DynTensor) -> NnResult<DynTensor> {
    x.relu()
}

fn add_dt(a: DynTensor, b: DynTensor) -> NnResult<DynTensor> {
    a.add(&b)
}

/// DynTensor model with owned tensor input.
/// The proc-macro should generate `compile_dyntensor_relu` and
/// `compile_verified_dyntensor_relu`.
#[nn_macros::model]
fn dyntensor_relu(x: DynTensor) -> NnResult<DynTensor> {
    relu_dt(x)
}

/// DynTensor model with multiple tensor inputs + non-tensor param.
#[nn_macros::model]
fn dyntensor_add(a: DynTensor, b: DynTensor, _scale: f32) -> NnResult<DynTensor> {
    add_dt(a, b)
}

// Verify compile functions exist on macOS (Metal backend available).
#[cfg(target_os = "macos")]
#[test]
fn test_compile_fn_generated_for_single_tensor() {
    // Type-check only — verifies the function was generated with correct signature.
    // Actually calling it would require Metal GPU initialization.
    let _ = compile_dyntensor_relu;
}

#[cfg(target_os = "macos")]
#[test]
fn test_compile_fn_generated_for_multi_tensor() {
    let _ = compile_dyntensor_add;
}

// Verify scalar-only models do NOT get compile functions.
// `kokoro_forward` has only f32 params — `compile_kokoro_forward` should not exist.
// This is a negative test: if someone accidentally generates compile fns for
// scalar models, the test below would fail to compile (uncomment to verify).
// let _ = compile_kokoro_forward; // <-- would be a compile error

#[test]
fn test_dyntensor_model_metadata() {
    assert_eq!(__dyntensor_relu_model_meta::MODEL_NAME, "dyntensor_relu");
    assert_eq!(__dyntensor_relu_model_meta::INPUT_COUNT, 1);
    assert_eq!(__dyntensor_relu_model_meta::INPUT_NAMES, ["x"]);
}

#[test]
fn test_multi_tensor_model_metadata() {
    assert_eq!(__dyntensor_add_model_meta::MODEL_NAME, "dyntensor_add");
    assert_eq!(__dyntensor_add_model_meta::INPUT_COUNT, 3);
    assert_eq!(
        __dyntensor_add_model_meta::INPUT_NAMES,
        ["a", "b", "_scale"],
    );
}
