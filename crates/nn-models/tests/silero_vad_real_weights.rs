// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real-weight tests for Silero VAD weight loading and shape validation.
//!
//! Gate: `SILERO_VAD_WEIGHTS` env var pointing to the converted safetensors file
//! (e.g., `models/silero_vad/silero_vad_16k.safetensors`).
//!
//! These tests validate weight shapes, value ranges, and structural properties
//! using the backend-agnostic `load_safetensors` path (no Metal required).
//!
//! Forward-pass tests with PyTorch reference comparison are in
//! `crates/nn-metal/tests/model_forward/silero_vad_e2e.rs`.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::DynTensor;
use nn_core::load_safetensors;

/// Expected tensor names and shapes for the 16kHz Silero VAD model.
/// These match the converter output from
const EXPECTED_TENSORS: &[(&str, &[usize])] = &[
    ("stft_forward_basis_buffer", &[258, 1, 256]),
    ("encoder_0_weight", &[128, 129, 3]),
    ("encoder_0_bias", &[128]),
    ("encoder_1_weight", &[64, 128, 3]),
    ("encoder_1_bias", &[64]),
    ("encoder_2_weight", &[64, 64, 3]),
    ("encoder_2_bias", &[64]),
    ("encoder_3_weight", &[128, 64, 3]),
    ("encoder_3_bias", &[128]),
    ("decoder_rnn_weight_ih", &[512, 128]),
    ("decoder_rnn_weight_hh", &[512, 128]),
    ("decoder_rnn_bias_ih", &[512]),
    ("decoder_rnn_bias_hh", &[512]),
    ("decoder_output_weight", &[1, 128, 1]),
    ("decoder_output_bias", &[1]),
];

/// Load weights from the env-var-specified path, returning None if unavailable.
fn load_weights() -> Option<HashMap<String, DynTensor>> {
    let path = std::env::var("SILERO_VAD_WEIGHTS").ok()?;
    let p = Path::new(&path);
    if !p.exists() {
        eprintln!("SKIP: SILERO_VAD_WEIGHTS path does not exist: {path}");
        return None;
    }
    Some(load_safetensors(p).expect("load_safetensors should succeed"))
}

// ============================================================================
// Weight loading and tensor count
// ============================================================================

#[test]
fn real_weights_load_successfully() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };
    assert_eq!(
        tensors.len(),
        15,
        "Silero VAD 16kHz model should have exactly 15 tensors, got {}",
        tensors.len()
    );
    eprintln!("Loaded {} tensors from Silero VAD weights", tensors.len());
}

// ============================================================================
// Shape validation for all 15 tensors
// ============================================================================

#[test]
fn real_weights_shapes_match_expected() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    for (name, expected_shape) in EXPECTED_TENSORS {
        let tensor = tensors
            .get(*name)
            .unwrap_or_else(|| panic!("missing tensor: {name}"));
        assert_eq!(
            tensor.dims(),
            *expected_shape,
            "tensor '{name}': shape {:?} != expected {:?}",
            tensor.dims(),
            expected_shape,
        );
    }
    eprintln!("All 15 tensor shapes validated successfully");
}

// ============================================================================
// All tensors contain finite values (no NaN/Inf)
// ============================================================================

#[test]
fn real_weights_all_finite() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    for (name, tensor) in &tensors {
        let has_non_finite = tensor
            .any_non_finite()
            .unwrap_or_else(|e| panic!("any_non_finite failed for '{name}': {e}"));
        assert!(
            !has_non_finite,
            "tensor '{name}' contains NaN or Inf values"
        );
    }
    eprintln!("All tensors are finite (no NaN/Inf)");
}

// ============================================================================
// STFT basis buffer properties
// ============================================================================

#[test]
fn real_weights_stft_basis_in_expected_range() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let stft = tensors
        .get("stft_forward_basis_buffer")
        .expect("stft tensor");
    let data = stft.to_flat_vec::<f32>().expect("f32 conversion");

    // STFT basis should be bounded by [-1, 1] (DFT coefficients)
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min >= -1.001 && max <= 1.001,
        "STFT basis should be in [-1, 1], got [{min}, {max}]"
    );
    eprintln!("STFT basis range: [{min:.6}, {max:.6}]");
}

// ============================================================================
// Encoder weight statistics
// ============================================================================

#[test]
fn real_weights_encoder_statistics() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    for i in 0..4 {
        let w_name = format!("encoder_{i}_weight");
        let b_name = format!("encoder_{i}_bias");

        let weight = tensors
            .get(&w_name)
            .unwrap_or_else(|| panic!("missing {w_name}"));
        let bias = tensors
            .get(&b_name)
            .unwrap_or_else(|| panic!("missing {b_name}"));

        let w_data = weight.to_flat_vec::<f32>().expect("f32 conversion");
        let b_data = bias.to_flat_vec::<f32>().expect("f32 conversion");

        // Weights should not be all zeros (indicates failed loading)
        let w_nonzero = w_data.iter().any(|&v| v != 0.0);
        assert!(w_nonzero, "encoder_{i}_weight should not be all zeros");

        // Bias values should be reasonable (not extreme)
        let b_max = b_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let b_min = b_data.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            b_max.abs() < 100.0 && b_min.abs() < 100.0,
            "encoder_{i}_bias has extreme values: [{b_min}, {b_max}]"
        );

        let w_abs_max = w_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        eprintln!("encoder_{i}: weight |max|={w_abs_max:.4}, bias range=[{b_min:.4}, {b_max:.4}]");
    }
}

// ============================================================================
// Decoder RNN weight dimensions (LSTM: 4 * hidden_size)
// ============================================================================

#[test]
fn real_weights_decoder_rnn_dimensions() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    // RNN weight_ih: [4*hidden, input_size] = [512, 128]
    // RNN weight_hh: [4*hidden, hidden_size] = [512, 128]
    // The factor 4 comes from LSTM gates: input, forget, cell, output
    let hidden_size = 128_usize;
    let gates = 4_usize;

    let wih = tensors.get("decoder_rnn_weight_ih").expect("weight_ih");
    assert_eq!(wih.dims(), &[gates * hidden_size, hidden_size]);

    let whh = tensors.get("decoder_rnn_weight_hh").expect("weight_hh");
    assert_eq!(whh.dims(), &[gates * hidden_size, hidden_size]);

    let bih = tensors.get("decoder_rnn_bias_ih").expect("bias_ih");
    assert_eq!(bih.dims(), &[gates * hidden_size]);

    let bhh = tensors.get("decoder_rnn_bias_hh").expect("bias_hh");
    assert_eq!(bhh.dims(), &[gates * hidden_size]);

    // RNN weights should not be all zeros
    let wih_data = wih.to_flat_vec::<f32>().expect("f32");
    let whh_data = whh.to_flat_vec::<f32>().expect("f32");
    assert!(
        wih_data.iter().any(|&v| v != 0.0),
        "decoder_rnn_weight_ih should not be all zeros"
    );
    assert!(
        whh_data.iter().any(|&v| v != 0.0),
        "decoder_rnn_weight_hh should not be all zeros"
    );

    eprintln!(
        "RNN dimensions valid: weight_ih={:?}, weight_hh={:?}",
        wih.dims(),
        whh.dims()
    );
}

// ============================================================================
// Decoder output layer (Conv1d acting as linear: [1, 128, 1])
// ============================================================================

#[test]
fn real_weights_decoder_output_properties() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let weight = tensors.get("decoder_output_weight").expect("output weight");
    let bias = tensors.get("decoder_output_bias").expect("output bias");

    // Shape: Conv1d(128, 1, 1) => weight [1, 128, 1], bias [1]
    assert_eq!(weight.dims(), &[1, 128, 1]);
    assert_eq!(bias.dims(), &[1]);

    let w_data = weight.to_flat_vec::<f32>().expect("f32");
    let b_data = bias.to_flat_vec::<f32>().expect("f32");

    // Output weight should not be all zeros
    assert!(
        w_data.iter().any(|&v| v != 0.0),
        "decoder_output_weight should not be all zeros"
    );

    // Bias is a single value — report it
    eprintln!(
        "Output layer: weight |max|={:.4}, bias={:.6}",
        w_data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max),
        b_data[0]
    );
}

// ============================================================================
// Total parameter count
// ============================================================================

#[test]
fn real_weights_total_parameter_count() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let total: usize = tensors.values().map(DynTensor::elem_count).sum();

    // Expected total from converter output: 309,633 parameters
    assert_eq!(
        total, 309_633,
        "total parameter count should be 309,633, got {total}"
    );
    eprintln!(
        "Total parameters: {total} ({:.2} MB f32)",
        total as f64 * 4.0 / 1024.0 / 1024.0
    );
}

// ============================================================================
// VarBuilder integration: construct from loaded tensors
// ============================================================================

#[test]
fn real_weights_var_builder_construction() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let vb = nn_core::var_builder::VarBuilder::from_tensors(
        tensors,
        nn_core::DType::F32,
        &nn_core::Device::Cpu,
    );

    // Verify we can retrieve tensors by name with shape validation
    let stft = vb.get(&[258, 1, 256], "stft_forward_basis_buffer");
    assert!(
        stft.is_ok(),
        "VarBuilder should load STFT basis: {:?}",
        stft.err()
    );

    let enc0_w = vb.get(&[128, 129, 3], "encoder_0_weight");
    assert!(
        enc0_w.is_ok(),
        "VarBuilder should load encoder_0_weight: {:?}",
        enc0_w.err()
    );

    // Shape mismatch should fail
    let bad = vb.get(&[999, 999], "encoder_0_weight");
    assert!(bad.is_err(), "wrong shape should fail");

    // Missing tensor should fail
    let missing = vb.get(&[1], "nonexistent_tensor");
    assert!(missing.is_err(), "missing tensor should fail");

    eprintln!("VarBuilder construction and retrieval validated");
}

// ============================================================================
// Encoder channel chain consistency with builder constants
// ============================================================================

#[test]
fn real_weights_encoder_chain_matches_builder_constants() {
    use nn_models::silero_vad_builders::ENCODER_BLOCKS;

    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let w_name = format!("encoder_{i}_weight");
        let b_name = format!("encoder_{i}_bias");

        let weight = tensors.get(&w_name).unwrap();
        let bias = tensors.get(&b_name).unwrap();

        // Conv1d weight shape: [out_channels, in_channels, kernel_size]
        assert_eq!(
            weight.dims(),
            &[block.out_channels, block.in_channels, block.kernel_size],
            "encoder_{i}_weight shape should match ENCODER_BLOCKS"
        );
        assert_eq!(
            bias.dims(),
            &[block.out_channels],
            "encoder_{i}_bias shape should match ENCODER_BLOCKS"
        );
    }
    eprintln!("All encoder weights match ENCODER_BLOCKS constants");
}
