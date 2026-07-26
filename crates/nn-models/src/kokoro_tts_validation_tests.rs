#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NaN/Inf validation tests for Kokoro TTS forward paths (Part of #1078).

use super::{cpu, make_kokoro_model_weights, test_kokoro_config, T_HIDDEN, T_STYLE};
use nn_core::dyn_tensor::DynTensor;
use nn_core::DType;

#[test]
fn test_kokoro_model_forward_text_finiteness() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let r = model
        .forward_text(&input_ids, &bert_out, &style, 1.0)
        .unwrap();

    let aligned_vals = r.aligned_dur.to_flat_vec::<f32>().unwrap();
    assert!(
        aligned_vals.iter().all(|v| v.is_finite()),
        "aligned_dur features must be finite"
    );
    let regulated_vals = r.regulated.to_flat_vec::<f32>().unwrap();
    assert!(
        regulated_vals.iter().all(|v| v.is_finite()),
        "regulated features must be finite"
    );
    let dur_vals = r.dur_logits.to_flat_vec::<f32>().unwrap();
    assert!(
        dur_vals.iter().all(|v| v.is_finite()),
        "dur_logits must be finite"
    );
}

// -- NaN/Inf validation tests (Part of #1078) --------------------------------

/// AC7: speed=0.0 returns descriptive InvalidSpeed error.
#[test]
fn test_forward_text_speed_zero_returns_error() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward_text(&input_ids, &bert_out, &style, 0.0);
    let err = result.unwrap_err();
    assert!(
        matches!(err, crate::kokoro_error::KokoroError::InvalidSpeed { value } if value == 0.0),
        "expected InvalidSpeed for speed=0.0, got: {err:?}"
    );
}

/// AC1: negative speed returns descriptive InvalidSpeed error.
#[test]
fn test_forward_text_negative_speed_returns_error() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward_text(&input_ids, &bert_out, &style, -1.0);
    let err = result.unwrap_err();
    assert!(
        matches!(err, crate::kokoro_error::KokoroError::InvalidSpeed { value } if value == -1.0),
        "expected InvalidSpeed for speed=-1.0, got: {err:?}"
    );
}

/// AC1: Inf speed returns InvalidSpeed error.
#[test]
fn test_forward_text_inf_speed_returns_error() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward_text(&input_ids, &bert_out, &style, f32::INFINITY);
    assert!(
        matches!(
            result.unwrap_err(),
            crate::kokoro_error::KokoroError::InvalidSpeed { .. }
        ),
        "expected InvalidSpeed for Inf speed"
    );
}

/// AC5: exp() overflow prevented by log_mag clamping in Generator.
/// With zero weights, exp(clamp(0, -88, 88)) = exp(0) = 1.0 (safe).
/// The clamp guard ensures non-zero weights producing log_mag > 88 are capped.
#[test]
fn test_full_forward_exp_overflow_guarded() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward(&input_ids, &style, 1.0);
    assert!(
        result.is_ok(),
        "zero-weight forward should succeed: {:?}",
        result.err()
    );
    let (mag, phase) = result.unwrap();
    let mag_vals = mag.to_flat_vec::<f32>().unwrap();
    let phase_vals = phase.to_flat_vec::<f32>().unwrap();
    assert!(
        mag_vals.iter().all(|v| v.is_finite()),
        "magnitude must be finite"
    );
    assert!(
        phase_vals.iter().all(|v| v.is_finite()),
        "phase must be finite"
    );
}

/// AC8: NaN input caught at first validation boundary (bert_encoder).
#[test]
fn test_forward_text_nan_input_caught_at_boundary() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let nan_data: Vec<f32> = vec![f32::NAN; 3 * T_HIDDEN];
    let bert_out = DynTensor::new(&nan_data, &[1, 3, T_HIDDEN], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward_text(&input_ids, &bert_out, &style, 1.0);
    let err = result.unwrap_err();
    assert!(
        matches!(err, crate::kokoro_error::KokoroError::NonFiniteIntermediate { stage, .. } if stage == "bert_encoder"),
        "NaN should be caught at bert_encoder boundary, got: {err:?}"
    );
}

/// AC1 + full forward: speed=0 rejected in full forward path too.
#[test]
fn test_full_forward_speed_zero_returns_error() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward(&input_ids, &style, 0.0);
    assert!(
        matches!(
            result.unwrap_err(),
            crate::kokoro_error::KokoroError::InvalidSpeed { .. }
        ),
        "full forward should reject speed=0"
    );
}

#[test]
fn test_full_forward_oversized_style_rejected() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE + 1], DType::F32, &cpu()).unwrap();
    let err = model.forward(&input_ids, &style, 1.0).unwrap_err();
    match err {
        crate::kokoro_error::KokoroError::Tensor(nn_core::TensorError::ShapeMismatch {
            expected,
            actual,
            ..
        }) => {
            assert_eq!(expected, vec![0, 2 * T_STYLE]);
            assert_eq!(actual, vec![1, 2 * T_STYLE + 1]);
        }
        other => panic!("expected Tensor(ShapeMismatch), got: {other:?}"),
    }
}

#[test]
fn test_full_forward_rank1_style_rejected() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[2 * T_STYLE], DType::F32, &cpu()).unwrap();
    let err = model.forward(&input_ids, &style, 1.0).unwrap_err();
    match err {
        crate::kokoro_error::KokoroError::Tensor(nn_core::TensorError::ShapeMismatch {
            expected,
            actual,
            ..
        }) => {
            assert_eq!(expected, vec![0, 2 * T_STYLE]);
            assert_eq!(actual, vec![2 * T_STYLE]);
        }
        other => panic!("expected Tensor(ShapeMismatch), got: {other:?}"),
    }
}

#[test]
fn test_full_forward_undersized_style_rejected() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE - 1], DType::F32, &cpu()).unwrap();
    let err = model.forward(&input_ids, &style, 1.0).unwrap_err();
    match err {
        crate::kokoro_error::KokoroError::Tensor(nn_core::TensorError::ShapeMismatch {
            expected,
            actual,
            ..
        }) => {
            assert_eq!(expected, vec![0, 2 * T_STYLE]);
            assert_eq!(actual, vec![1, 2 * T_STYLE - 1]);
        }
        other => panic!("expected Tensor(ShapeMismatch), got: {other:?}"),
    }
}
