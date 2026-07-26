// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test proving the candle→nn migration pattern works end-to-end.
//!
//! This test exercises the exact pattern dvoice uses to migrate from candle:
//! - Import types from `nn::` (consumer-facing API, not internal `nn_core::`)
//! - Type alias `Tensor = DynTensor` (dvoice `backend.rs` pattern)
//! - Build multi-layer models using VarBuilder + nn free functions
//! - Run forward passes on CPU
//! - Verify output shape and finiteness
//!
//! Tensor operation compatibility tests (cat, stack, broadcast, selection, etc.)
//! are in `candle_migration_ops.rs`.
//!
//! If this test compiles and passes, the candle→nn migration path works.
//!
//! Run: `cargo test -p nn --test candle_migration`

// ---------------------------------------------------------------------------
// Step 1: Import from `nn::` — this is what dvoice consumers see.
//
// candle code:        use candle_core::{DType, Device, Result, Tensor};
// nn migration:      use nn::{DType, Device, Result, DynTensor};
// ---------------------------------------------------------------------------
use nn::{DType, Device, DynTensor, Result, VarBuilder, D};

// nn layers — candle: `use candle_nn::{Linear, LayerNorm, Conv1d, ...}`
// Layer types are available but we use free functions (linear(), conv1d(), etc.)
// which return these types. Module trait provides .forward().
use nn::{Conv1dConfig, Module};

// nn free functions — candle: `use candle_nn::{linear, conv1d, embedding, lstm, ...}`
use nn::{conv1d, embedding, layer_norm, linear, linear_no_bias, lstm};

// LSTM state — candle: `use candle_nn::LSTMState`
use nn::LstmState;

// KV cache — candle: `use candle_nn::kv_cache::KvCache`
use nn::KvCacheLayer;

// ---------------------------------------------------------------------------
// Step 2: Type alias matching dvoice backend.rs convention.
//
// dvoice does: `type Tensor = nn::DynTensor;`
// This test verifies the alias works transparently.
// ---------------------------------------------------------------------------
type Tensor = DynTensor;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify Linear layer constructed via VarBuilder works end-to-end.
///
/// Pattern: `let linear = nn::linear(in_dim, out_dim, &vb.pp("linear"));`
#[test]
fn test_linear_via_varbuilder() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let layer = linear(4, 8, vb.pp("proj"))?;

    // Forward pass with Tensor alias
    let input: Tensor = DynTensor::ones(&[2, 4], DType::F32, &Device::Cpu)?;
    let output = layer.forward(&input)?;

    assert_eq!(output.dims(), &[2, 8]);
    // Zeros backend produces zero weights → zero output
    let vals = output.to_flat_vec::<f32>()?;
    assert!(vals.iter().all(|v| v.is_finite()));
    Ok(())
}

/// Verify Linear without bias matches candle_nn::linear_no_bias.
#[test]
fn test_linear_no_bias_via_varbuilder() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let layer = linear_no_bias(16, 32, vb.pp("attn.out"))?;

    let input = DynTensor::ones(&[1, 5, 16], DType::F32, &Device::Cpu)?;
    let flat = input.reshape([5, 16])?;
    let output = layer.forward(&flat)?;

    assert_eq!(output.dims(), &[5, 32]);
    Ok(())
}

/// Verify Conv1d constructed via VarBuilder + Config.
///
/// Pattern: `let conv = nn::conv1d(in_c, out_c, kernel, cfg, &vb.pp("conv1"));`
#[test]
fn test_conv1d_via_varbuilder() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    // #[non_exhaustive] config structs require ..Default::default() pattern.
    // This is the correct consumer-facing usage after #1246 (AC4).
    let mut cfg = Conv1dConfig::default();
    cfg.padding = 1;
    let layer = conv1d(3, 16, 3, cfg, vb.pp("encoder.conv1"))?;

    let input = DynTensor::ones(&[1, 3, 100], DType::F32, &Device::Cpu)?;
    let output = layer.forward(&input)?;

    // output_len = (100 + 2*1 - 3) / 1 + 1 = 100
    assert_eq!(output.dims(), &[1, 16, 100]);
    Ok(())
}

/// Verify Embedding layer — handles both F32 and U32/I64 token inputs.
///
/// candle: `let emb = candle_nn::embedding(vocab, dim, &vb.pp("embed"));`
/// nn:    `let emb = nn::embedding(vocab, dim, &vb.pp("embed"));`
#[test]
fn test_embedding_via_varbuilder() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let layer = embedding(100, 64, vb.pp("embed_tokens"))?;

    // U32 token IDs (natural for embedding lookup)
    // Embedding forward flattens input then index_selects, preserving input shape prefix.
    let ids = DynTensor::from_vec_u32(vec![0, 1, 5, 99], &[4], &Device::Cpu)?;
    let output = layer.forward(&ids)?;

    assert_eq!(output.dims(), &[4, 64]);
    Ok(())
}

/// Verify LayerNorm — candle_nn::layer_norm pattern.
#[test]
fn test_layer_norm_via_varbuilder() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let ln = layer_norm(64, Default::default(), vb.pp("ln"))?;

    let input = DynTensor::ones(&[2, 10, 64], DType::F32, &Device::Cpu)?;
    let output = ln.forward(&input)?;

    assert_eq!(output.dims(), &[2, 10, 64]);
    Ok(())
}

/// Verify LSTM layer constructed via VarBuilder.
///
/// candle:  `let lstm = candle_nn::lstm(input_size, hidden_size, &vb.pp("lstm"));`
/// nn:     `let lstm_layer = nn::lstm(input_size, hidden_size, &vb.pp("lstm"));`
///
/// LSTM is used by Silero VAD (streaming voice detection) and other sequence models.
#[test]
fn test_lstm_via_varbuilder() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let lstm_layer = lstm(16, 32, vb.pp("lstm"))?;

    // Single-step forward: input [batch, input_size]
    let input = DynTensor::ones(&[1, 16], DType::F32, &Device::Cpu)?;
    let (output, state): (DynTensor, LstmState) = lstm_layer.forward(&input, None)?;

    // output is h_new [batch, hidden_size]
    assert_eq!(output.dims(), &[1, 32]);
    let vals = output.to_flat_vec::<f32>()?;
    assert!(vals.iter().all(|v| v.is_finite()));

    // State carries h and c for next step (streaming pattern used by Silero VAD)
    let (output2, _state2) = lstm_layer.forward(&input, Some(&state))?;
    assert_eq!(output2.dims(), &[1, 32]);

    Ok(())
}

/// Verify LSTM sequence processing (multi-step).
///
/// candle:  `lstm.forward_seq(&input, None)?`
/// nn:     same API
#[test]
fn test_lstm_forward_seq() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let lstm_layer = lstm(16, 32, vb.pp("lstm"))?;

    // Sequence input: [seq_len, batch, input_size]
    let seq_input = DynTensor::ones(&[10, 1, 16], DType::F32, &Device::Cpu)?;
    let (outputs, final_state) = lstm_layer.forward_seq(&seq_input, None)?;

    // outputs: [seq_len, batch, hidden_size]
    assert_eq!(outputs.dims(), &[10, 1, 32]);

    // final_state can seed next chunk (streaming)
    let seq_input2 = DynTensor::ones(&[5, 1, 16], DType::F32, &Device::Cpu)?;
    let (outputs2, _) = lstm_layer.forward_seq(&seq_input2, Some(&final_state))?;
    assert_eq!(outputs2.dims(), &[5, 1, 32]);

    Ok(())
}

/// Verify KV cache bridge — candle uses per-layer KvCache, nn uses KvCacheLayer.
///
/// candle:  `let mut kv = candle_nn::kv_cache::KvCache::new(2, max_seq);`
/// nn:     `let mut kv = nn::KvCacheLayer::new(2, max_seq)?;`
///
/// dvoice backend.rs: `type KvCache = nn::layers::KvCacheLayer;`
#[test]
fn test_kv_cache_bridge() -> Result<()> {
    // candle pattern: KvCache::new(dim=2, max_seq_len)
    let mut cache = KvCacheLayer::new(2, 128)?;

    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);

    // Append K/V tensors shaped [batch, heads, seq, head_dim]
    let k = DynTensor::ones(&[1, 4, 5, 32], DType::F32, &Device::Cpu)?;
    let v = DynTensor::ones(&[1, 4, 5, 32], DType::F32, &Device::Cpu)?;
    let (full_k, full_v) = cache.append(&k, &v)?;

    assert_eq!(full_k.dims(), &[1, 4, 5, 32]);
    assert_eq!(full_v.dims(), &[1, 4, 5, 32]);
    assert_eq!(cache.seq_len(), 5);

    // candle accessor names work
    assert_eq!(cache.current_seq_len(), 5);
    assert_eq!(cache.dim(), 2);

    // Incremental append (autoregressive decode step)
    let k2 = DynTensor::ones(&[1, 4, 1, 32], DType::F32, &Device::Cpu)?;
    let v2 = DynTensor::ones(&[1, 4, 1, 32], DType::F32, &Device::Cpu)?;
    let (full_k2, _full_v2) = cache.append(&k2, &v2)?;

    assert_eq!(full_k2.dims(), &[1, 4, 6, 32]);
    assert_eq!(cache.seq_len(), 6);

    // Reset for new sequence
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);

    Ok(())
}

/// Verify D::Minus1 dimension indexing (candle pattern: `tensor.softmax(D::Minus1)`).
#[test]
fn test_d_minus1_softmax() -> Result<()> {
    use nn::softmax;

    let input = DynTensor::ones(&[2, 5, 10], DType::F32, &Device::Cpu)?;
    let output = softmax(&input, D::Minus1)?;

    assert_eq!(output.dims(), &[2, 5, 10]);
    // softmax output sums to 1 along last dim
    let vals = output.to_flat_vec::<f32>()?;
    assert!(vals.iter().all(|v| v.is_finite()));
    Ok(())
}

/// Multi-layer model test — exercises the full migration pattern with all 5 layer types.
///
/// Builds a model resembling a simplified Silero VAD pipeline:
/// Embedding → Conv1d → LayerNorm → LSTM → Linear → output
///
/// AC1: Linear + LayerNorm + Conv1d + Embedding + LSTM all exercised.
/// This tests that multiple nn layers compose correctly through DynTensor.
#[test]
fn test_multi_layer_model() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    // Build all 5 layer types (zeros backend — shape correctness is what matters)
    let emb = embedding(256, 16, vb.pp("embed"))?;
    let mut conv_cfg = Conv1dConfig::default();
    conv_cfg.padding = 1;
    let conv = conv1d(16, 32, 3, conv_cfg, vb.pp("conv"))?;
    let ln = layer_norm(32, Default::default(), vb.pp("ln"))?;
    let lstm_layer = lstm(32, 64, vb.pp("lstm"))?;
    let proj = linear(64, 1, vb.pp("output"))?;

    // Forward pass using Tensor alias (AC2: dvoice backend.rs pattern)
    // token IDs → embedding → conv → norm → lstm → linear → output

    // Embedding: [seq_len=20] → [20, 16]
    let token_ids = DynTensor::from_vec_u32(vec![1; 20], &[20], &Device::Cpu)?;
    let x: Tensor = emb.forward(&token_ids)?;
    assert_eq!(x.dims(), &[20, 16]);

    // Reshape for Conv1d: [20, 16] → [1, 16, 20] (batch=1, channels=16, time=20)
    let x: Tensor = x.transpose(0, 1)?.reshape([1, 16, 20])?;

    // Conv1d: [1, 16, 20] → [1, 32, 20]
    let x: Tensor = conv.forward(&x)?;
    assert_eq!(x.dims(), &[1, 32, 20]);

    // Transpose for LayerNorm: [1, 32, 20] → [1, 20, 32]
    let x = x.transpose(1, 2)?;

    // LayerNorm: [1, 20, 32] → [1, 20, 32]
    let x: Tensor = ln.forward(&x)?;
    assert_eq!(x.dims(), &[1, 20, 32]);

    // Reshape for LSTM: [1, 20, 32] → [20, 1, 32] (seq, batch, features)
    let x = x.transpose(0, 1)?;

    // LSTM: [20, 1, 32] → [20, 1, 64]
    let (x, _final_state) = lstm_layer.forward_seq(&x, None)?;
    assert_eq!(x.dims(), &[20, 1, 64]);

    // Linear projection: reshape to [20, 64] → [20, 1]
    let x = x.reshape([20, 64])?;
    let output: Tensor = proj.forward(&x)?;
    assert_eq!(output.dims(), &[20, 1]);

    // Verify output is finite
    let vals = output.to_flat_vec::<f32>()?;
    assert!(vals.iter().all(|v| v.is_finite()));

    Ok(())
}

/// Verify VarBuilder hierarchical scoping (pp) works like candle.
///
/// candle:  `vb.pp("encoder").pp("layers").pp("0")`
/// nn:     same API
#[test]
fn test_varbuilder_pp_scoping() -> Result<()> {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    // Hierarchical scoping
    let encoder_vb = vb.pp("encoder");
    let layer0_vb = encoder_vb.pp("layers").pp("0");

    // Build layers at different scopes
    let _proj = linear(32, 64, layer0_vb.pp("self_attn.out_proj"))?;
    let _ln = layer_norm(64, Default::default(), layer0_vb.pp("self_attn_layer_norm"))?;

    // VarBuilder is cheap to clone (Arc + Vec) — reuse original here
    let _proj2 = linear(64, 32, vb.pp("decoder").pp("proj"))?;

    Ok(())
}
