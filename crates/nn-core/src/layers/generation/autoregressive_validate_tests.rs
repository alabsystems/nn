#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation tests for [`GenerationConfig::validate`] and generate() error
//! rejection. Extracted from `autoregressive_tests.rs` for 500-line compliance.

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCache;
use crate::Device;

use super::super::{generate, GenerationConfig};

/// Mock model: returns logits where token `(step % vocab)` has highest logit.
fn deterministic_model(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let input_f32 = input.to_dtype(crate::DType::F32)?;
    let flat = input_f32.to_flat_vec::<f32>()?;
    let last_val = flat[flat.len() - 1];
    let next_token = (last_val as usize + 1) % 5;
    let mut logits = vec![0.0f32; 5];
    logits[next_token] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

// -- GenerationConfig::validate() tests ----------------------------------------

#[test]
fn test_validate_default_config_ok() {
    let config = GenerationConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_positive_temperature_ok() {
    let config = GenerationConfig {
        temperature: 1.5,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_nan_temperature_rejected() {
    let config = GenerationConfig {
        temperature: f64::NAN,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("temperature"),
        "error should mention temperature: {msg}"
    );
}

#[test]
fn test_validate_inf_temperature_rejected() {
    let config = GenerationConfig {
        temperature: f64::INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_neg_inf_temperature_rejected() {
    let config = GenerationConfig {
        temperature: f64::NEG_INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_negative_temperature_rejected() {
    let config = GenerationConfig {
        temperature: -0.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_zero_temperature_ok() {
    // Zero means greedy decoding — must be allowed.
    let config = GenerationConfig {
        temperature: 0.0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_valid_top_p_ok() {
    let config = GenerationConfig {
        top_p: Some(0.9),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_top_p_1_0_ok() {
    let config = GenerationConfig {
        top_p: Some(1.0),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_top_p_zero_rejected() {
    let config = GenerationConfig {
        top_p: Some(0.0),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_top_p_negative_rejected() {
    let config = GenerationConfig {
        top_p: Some(-0.5),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_top_p_above_1_rejected() {
    let config = GenerationConfig {
        top_p: Some(1.5),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_top_p_nan_rejected() {
    let config = GenerationConfig {
        top_p: Some(f64::NAN),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_top_p_inf_rejected() {
    let config = GenerationConfig {
        top_p: Some(f64::INFINITY),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_none_top_p_ok() {
    // None means no top-p filtering — must be allowed.
    let config = GenerationConfig {
        top_p: None,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_generate_nan_temperature_returns_error() {
    // Integration test: generate() rejects invalid config at entry.
    let config = GenerationConfig {
        max_new_tokens: 5,
        temperature: f64::NAN,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_generate_negative_temperature_returns_error() {
    let config = GenerationConfig {
        max_new_tokens: 5,
        temperature: -1.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_generate_invalid_top_p_returns_error() {
    let config = GenerationConfig {
        max_new_tokens: 5,
        top_p: Some(0.0),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}
