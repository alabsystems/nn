// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::{DType, Device, Result};

/// Helper: create a Linear layer with given constant weight value (no bias).
fn make_linear(out_f: usize, in_f: usize, val: f32) -> Linear {
    let w = DynTensor::from_vec(vec![val; out_f * in_f], &[out_f, in_f], &Device::Cpu)
        .expect("linear weight");
    Linear::new(w, None).unwrap()
}

/// Helper: create an MtpHead with uniform weights for shape testing.
fn make_mtp_head(hidden: usize, vocab: usize, num_predict: usize, val: f32) -> MtpHead {
    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    let heads: Vec<Linear> = (0..num_predict)
        .map(|_| make_linear(vocab, hidden, val))
        .collect();
    MtpHead::new(heads, None, vec![], cfg).unwrap()
}

// -- Shape tests --------------------------------------------------------------

#[test]
fn test_mtp_head_output_shape() -> Result<()> {
    let hidden = 256;
    let vocab = 1000;
    let num_predict = 4;
    let mtp = make_mtp_head(hidden, vocab, num_predict, 0.01);

    let x = DynTensor::ones(&[2, 8, hidden], DType::F32, &Device::Cpu)?;
    let out = mtp.forward(&x)?;

    assert_eq!(out.dims(), &[2, 8, num_predict, vocab]);
    Ok(())
}

#[test]
fn test_mtp_head_single_token() -> Result<()> {
    let hidden = 64;
    let vocab = 100;
    let num_predict = 2;
    let mtp = make_mtp_head(hidden, vocab, num_predict, 0.01);

    // Single token input [1, 1, D].
    let x = DynTensor::ones(&[1, 1, hidden], DType::F32, &Device::Cpu)?;
    let out = mtp.forward(&x)?;

    assert_eq!(out.dims(), &[1, 1, num_predict, vocab]);
    Ok(())
}

#[test]
fn test_mtp_head_per_head_output() -> Result<()> {
    let hidden = 64;
    let vocab = 100;
    let num_predict = 3;
    let mtp = make_mtp_head(hidden, vocab, num_predict, 0.01);

    let x = DynTensor::ones(&[1, 5, hidden], DType::F32, &Device::Cpu)?;
    let per_head = mtp.forward_per_head(&x)?;

    assert_eq!(per_head.len(), num_predict);
    for logits in &per_head {
        assert_eq!(logits.dims(), &[1, 5, vocab]);
    }
    Ok(())
}

// -- Value correctness --------------------------------------------------------

#[test]
fn test_mtp_head_different_heads_produce_different_logits() -> Result<()> {
    // Each head has different weights -> different outputs.
    let hidden = 4;
    let vocab = 3;
    let num_predict = 2;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };

    let head0 = make_linear(vocab, hidden, 0.1);
    let head1 = make_linear(vocab, hidden, 0.5);
    let mtp = MtpHead::new(vec![head0, head1], None, vec![], cfg)?;

    let x = DynTensor::ones(&[1, 1, hidden], DType::F32, &Device::Cpu)?;
    let per_head = mtp.forward_per_head(&x)?;

    let vals0 = per_head[0].to_flat_vec::<f32>()?;
    let vals1 = per_head[1].to_flat_vec::<f32>()?;

    // With weight 0.1 and input ones: each output = 0.1 * hidden = 0.4
    // With weight 0.5: each output = 0.5 * hidden = 2.0
    let expected0 = 0.1 * hidden as f32;
    let expected1 = 0.5 * hidden as f32;

    assert!(
        (vals0[0] - expected0).abs() < 1e-4,
        "head0: expected {expected0}, got {}",
        vals0[0]
    );
    assert!(
        (vals1[0] - expected1).abs() < 1e-4,
        "head1: expected {expected1}, got {}",
        vals1[0]
    );
    Ok(())
}

#[test]
fn test_mtp_head_stacked_matches_per_head() -> Result<()> {
    // forward() stacked output should match forward_per_head() sliced.
    let hidden = 8;
    let vocab = 5;
    let num_predict = 3;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };

    let heads: Vec<Linear> = (0..num_predict)
        .map(|i| make_linear(vocab, hidden, 0.1 * (i as f32 + 1.0)))
        .collect();
    let mtp = MtpHead::new(heads, None, vec![], cfg)?;

    // Use a deterministic non-uniform input so heads produce distinguishable outputs.
    let elem_count = 2 * hidden;
    let data: Vec<f32> = (0..elem_count).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let x = DynTensor::from_vec(data, &[1, 2, hidden], &Device::Cpu)?;

    let stacked = mtp.forward(&x)?; // [1, 2, 3, 5]
    let per_head = mtp.forward_per_head(&x)?; // Vec of [1, 2, 5]

    // Extract each head from the stacked tensor and compare.
    for i in 0..num_predict {
        // Narrow along dim 2 to get [1, 2, 1, 5], then squeeze to [1, 2, 5].
        let slice = stacked.narrow(2, i, 1)?;
        let squeezed = slice.squeeze(2)?;
        let stacked_vals = squeezed.to_flat_vec::<f32>()?;
        let per_head_vals = per_head[i].to_flat_vec::<f32>()?;

        assert_eq!(stacked_vals.len(), per_head_vals.len());
        for (j, (&sv, &pv)) in stacked_vals.iter().zip(per_head_vals.iter()).enumerate() {
            assert!(
                (sv - pv).abs() < 1e-5,
                "head {i}, element {j}: stacked={sv}, per_head={pv}"
            );
        }
    }
    Ok(())
}

// -- Shared trunk tests -------------------------------------------------------

#[test]
fn test_mtp_head_shared_trunk() -> Result<()> {
    let hidden = 4;
    let vocab = 3;
    let num_predict = 2;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: true,
        per_head_norm: false,
        norm_eps: 1e-5,
    };

    let trunk = make_linear(hidden, hidden, 0.5);
    let heads: Vec<Linear> = (0..num_predict)
        .map(|_| make_linear(vocab, hidden, 0.1))
        .collect();
    let mtp = MtpHead::new(heads, Some(trunk), vec![], cfg)?;

    let x = DynTensor::ones(&[1, 2, hidden], DType::F32, &Device::Cpu)?;
    let out = mtp.forward(&x)?;

    assert_eq!(out.dims(), &[1, 2, num_predict, vocab]);

    // All values should be finite.
    let vals = out.to_flat_vec::<f32>()?;
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    Ok(())
}

#[test]
fn test_mtp_head_shared_trunk_affects_output() -> Result<()> {
    // Trunk with identity weights should pass through; non-identity should transform.
    let hidden = 2;
    let vocab = 2;

    let cfg_no_trunk = MtpHeadConfig {
        num_predict_tokens: 1,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    let cfg_with_trunk = MtpHeadConfig {
        num_predict_tokens: 1,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: true,
        per_head_norm: false,
        norm_eps: 1e-5,
    };

    let head_weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
                                                // Override with identity weights.
    let head_w = DynTensor::from_vec(head_weight, &[vocab, hidden], &Device::Cpu)?;
    let head_no_trunk = Linear::new(head_w.clone(), None)?;
    let head_with_trunk = Linear::new(head_w, None)?;

    // Trunk that scales by 2.
    let trunk_w = DynTensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[hidden, hidden], &Device::Cpu)?;
    let trunk = Linear::new(trunk_w, None)?;

    let mtp_no = MtpHead::new(vec![head_no_trunk], None, vec![], cfg_no_trunk)?;
    let mtp_with = MtpHead::new(vec![head_with_trunk], Some(trunk), vec![], cfg_with_trunk)?;

    let x = DynTensor::from_vec(vec![1.0, 3.0], &[1, 1, hidden], &Device::Cpu)?;

    let out_no = mtp_no.forward(&x)?;
    let out_with = mtp_with.forward(&x)?;

    let vals_no = out_no.to_flat_vec::<f32>()?;
    let vals_with = out_with.to_flat_vec::<f32>()?;

    // Without trunk: identity head -> [1.0, 3.0]
    // With trunk (2x scale) + identity head -> [2.0, 6.0]
    assert!((vals_no[0] - 1.0).abs() < 1e-5, "no trunk: {}", vals_no[0]);
    assert!((vals_no[1] - 3.0).abs() < 1e-5, "no trunk: {}", vals_no[1]);
    assert!(
        (vals_with[0] - 2.0).abs() < 1e-5,
        "with trunk: {}",
        vals_with[0]
    );
    assert!(
        (vals_with[1] - 6.0).abs() < 1e-5,
        "with trunk: {}",
        vals_with[1]
    );
    Ok(())
}

// -- Per-head norm tests ------------------------------------------------------

#[test]
fn test_mtp_head_per_head_norm() -> Result<()> {
    let hidden = 4;
    let vocab = 3;
    let num_predict = 2;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: true,
        norm_eps: 1e-5,
    };

    let heads: Vec<Linear> = (0..num_predict)
        .map(|_| make_linear(vocab, hidden, 0.1))
        .collect();

    let norms: Vec<RmsNorm> = (0..num_predict)
        .map(|_| {
            let w = DynTensor::ones(&[hidden], DType::F32, &Device::Cpu).unwrap();
            RmsNorm::new(w, 1e-5).unwrap()
        })
        .collect();

    let mtp = MtpHead::new(heads, None, norms, cfg)?;

    let elem_count = 3 * hidden;
    let data: Vec<f32> = (0..elem_count).map(|i| (i as f32) * 0.2 - 1.0).collect();
    let x = DynTensor::from_vec(data, &[1, 3, hidden], &Device::Cpu)?;
    let out = mtp.forward(&x)?;

    assert_eq!(out.dims(), &[1, 3, num_predict, vocab]);
    let vals = out.to_flat_vec::<f32>()?;
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    Ok(())
}

// -- VarBuilder loading -------------------------------------------------------

#[test]
fn test_mtp_head_load_from_var_builder() -> Result<()> {
    use std::collections::HashMap;

    let hidden = 8;
    let vocab = 5;
    let num_predict = 3;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };

    // Build tensor map with expected weight names.
    let mut tensors = HashMap::new();
    for i in 0..num_predict {
        let w = DynTensor::from_vec(vec![0.1; vocab * hidden], &[vocab, hidden], &Device::Cpu)?;
        tensors.insert(format!("mtp.heads.{i}.weight"), w);
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let mtp = MtpHead::load(vb.pp("mtp"), cfg)?;

    assert_eq!(mtp.num_predict_tokens(), num_predict);

    let x = DynTensor::ones(&[1, 2, hidden], DType::F32, &Device::Cpu)?;
    let out = mtp.forward(&x)?;
    assert_eq!(out.dims(), &[1, 2, num_predict, vocab]);
    Ok(())
}

#[test]
fn test_mtp_head_load_with_shared_trunk() -> Result<()> {
    use std::collections::HashMap;

    let hidden = 4;
    let vocab = 3;
    let num_predict = 2;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: true,
        per_head_norm: false,
        norm_eps: 1e-5,
    };

    let mut tensors = HashMap::new();
    for i in 0..num_predict {
        tensors.insert(
            format!("mtp.heads.{i}.weight"),
            DynTensor::from_vec(vec![0.1; vocab * hidden], &[vocab, hidden], &Device::Cpu)?,
        );
    }
    tensors.insert(
        "mtp.shared_trunk.weight".into(),
        DynTensor::from_vec(vec![0.1; hidden * hidden], &[hidden, hidden], &Device::Cpu)?,
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let mtp = MtpHead::load(vb.pp("mtp"), cfg)?;

    assert!(mtp.shared_trunk().is_some());
    Ok(())
}

#[test]
fn test_mtp_head_load_with_per_head_norm() -> Result<()> {
    use std::collections::HashMap;

    let hidden = 4;
    let vocab = 3;
    let num_predict = 2;

    let cfg = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: true,
        norm_eps: 1e-6,
    };

    let mut tensors = HashMap::new();
    for i in 0..num_predict {
        tensors.insert(
            format!("mtp.heads.{i}.weight"),
            DynTensor::from_vec(vec![0.1; vocab * hidden], &[vocab, hidden], &Device::Cpu)?,
        );
        tensors.insert(
            format!("mtp.head_norms.{i}.weight"),
            DynTensor::ones(&[hidden], DType::F32, &Device::Cpu)?,
        );
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let mtp = MtpHead::load(vb.pp("mtp"), cfg)?;

    let data: Vec<f32> = (0..(2 * hidden))
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();
    let x = DynTensor::from_vec(data, &[1, 2, hidden], &Device::Cpu)?;
    let out = mtp.forward(&x)?;
    assert_eq!(out.dims(), &[1, 2, num_predict, vocab]);
    Ok(())
}

// -- Validation / error tests ------------------------------------------------

#[test]
fn test_mtp_head_zero_predict_tokens_error() {
    let cfg = MtpHeadConfig {
        num_predict_tokens: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_mtp_head_wrong_input_dim_error() -> Result<()> {
    let mtp = make_mtp_head(8, 5, 2, 0.1);
    // Input hidden dim = 4, but head expects 8.
    let x = DynTensor::ones(&[1, 2, 4], DType::F32, &Device::Cpu)?;
    let result = mtp.forward(&x);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_mtp_head_wrong_rank_error() -> Result<()> {
    let mtp = make_mtp_head(4, 3, 2, 0.1);
    // 2D input instead of 3D.
    let x = DynTensor::ones(&[2, 4], DType::F32, &Device::Cpu)?;
    let result = mtp.forward(&x);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_mtp_head_head_count_mismatch_error() {
    let cfg = MtpHeadConfig {
        num_predict_tokens: 3,
        hidden_size: 4,
        vocab_size: 5,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    // Only provide 2 heads when config says 3.
    let heads: Vec<Linear> = (0..2).map(|_| make_linear(5, 4, 0.1)).collect();
    let result = MtpHead::new(heads, None, vec![], cfg);
    assert!(result.is_err());
}

#[test]
fn test_mtp_head_accessors() {
    let mtp = make_mtp_head(8, 5, 3, 0.1);
    assert_eq!(mtp.num_predict_tokens(), 3);
    assert_eq!(mtp.config().hidden_size, 8);
    assert_eq!(mtp.config().vocab_size, 5);
    assert!(mtp.head(0).is_some());
    assert!(mtp.head(2).is_some());
    assert!(mtp.head(3).is_none());
    assert!(mtp.shared_trunk().is_none());
}

// -- Finiteness ---------------------------------------------------------------

#[test]
fn test_mtp_head_output_finite() -> Result<()> {
    let hidden = 16;
    let vocab = 10;
    let mtp = make_mtp_head(hidden, vocab, 4, 0.01);

    // Various magnitudes (deterministic, no training feature needed).
    let elem_count = 2 * 5 * hidden;
    let data: Vec<f32> = (0..elem_count)
        .map(|i| ((i as f32) * 1.7 - 50.0).sin() * 10.0)
        .collect();
    let x = DynTensor::from_vec(data, &[2, 5, hidden], &Device::Cpu)?;
    let out = mtp.forward(&x)?;
    let vals = out.to_flat_vec::<f32>()?;

    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    Ok(())
}
