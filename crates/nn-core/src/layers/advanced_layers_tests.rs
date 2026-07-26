// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for advanced nn layers: BiLSTM, MoE, DiT, GatedDeltaNet,
//! PixelShuffle, PixelUnshuffle, SqueezeExcitation.
//!
//! Part of #4186: Increase coverage for less-tested nn layers.

use crate::dyn_tensor::DynTensor;
use crate::layers::{
    AdaLnZero, AdaLnZeroDual, BiLstm, DiTBlock, DiTBlockDual, GatedDeltaNet, Linear, LstmState,
    Module, MoeDispatch, MoeDispatchConfig, MoeRouter, PixelShuffle, PixelUnshuffle, RmsNorm,
    SqueezeExcitation, SwiGluExpert,
};
use crate::{DType, Device, Result};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a Linear layer with uniform weights (no bias).
fn make_linear(out_f: usize, in_f: usize, val: f32) -> Linear {
    let w = DynTensor::from_vec(vec![val; out_f * in_f], &[out_f, in_f], &Device::Cpu).unwrap();
    Linear::new(w, None).unwrap()
}

/// Create a Linear layer with uniform weights and bias.
fn make_linear_with_bias(out_f: usize, in_f: usize, w_val: f32, b_val: f32) -> Linear {
    let w = DynTensor::from_vec(vec![w_val; out_f * in_f], &[out_f, in_f], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![b_val; out_f], &[out_f], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

/// Identity module (passthrough).
fn identity_module() -> Box<dyn Module + Send + Sync> {
    Box::new(|x: &DynTensor| -> Result<DynTensor> { Ok(x.clone()) })
}

/// Identity RmsNorm (weight = ones, eps = 1e-6).
fn identity_rms_norm(dim: usize) -> RmsNorm {
    let w = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    RmsNorm::new(w, 1e-6).unwrap()
}

/// Build a SwiGluExpert with uniform weights.
fn make_expert(dim: usize, ff_dim: usize, scale: f32) -> SwiGluExpert {
    SwiGluExpert::new(
        make_linear(ff_dim, dim, scale),
        make_linear(ff_dim, dim, scale),
        make_linear(dim, ff_dim, scale),
    )
    .unwrap()
}

/// Build a BiLstm with uniform weights.
fn make_bilstm(hidden: usize, input_size: usize, val: f64) -> BiLstm {
    let mk = |r: usize, c: usize| DynTensor::full(&[r, c], val, DType::F32, &Device::Cpu).unwrap();
    BiLstm::from_weights(
        mk(4 * hidden, input_size),
        mk(4 * hidden, hidden),
        None,
        None,
        mk(4 * hidden, input_size),
        mk(4 * hidden, hidden),
        None,
        None,
        hidden,
    )
    .unwrap()
}

// ===========================================================================
// BiLSTM tests
// ===========================================================================

#[test]
fn test_bilstm_forward_output_shape() {
    let hidden = 4;
    let input_size = 6;
    let seq_len = 7;
    let batch = 2;
    let bilstm = make_bilstm(hidden, input_size, 0.05);

    let input =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let (out, fwd_s, bwd_s) = bilstm.forward_seq(&input, None, None).unwrap();

    assert_eq!(out.dims(), &[seq_len, batch, 2 * hidden]);
    assert_eq!(fwd_s.h.dims(), &[batch, hidden]);
    assert_eq!(bwd_s.h.dims(), &[batch, hidden]);
}

#[test]
fn test_bilstm_bidirectional_concat() {
    // Verify the output is forward-concat-backward along the last dim.
    let h = 3;
    let input_size = 4;
    let seq_len = 5;
    let batch = 1;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input =
        DynTensor::full(&[seq_len, batch, input_size], 0.5, DType::F32, &Device::Cpu).unwrap();
    let (outputs, _, _) = bilstm.forward_seq(&input, None, None).unwrap();

    // Run forward LSTM independently.
    let (fwd_out, _) = bilstm.forward_lstm().forward_seq(&input, None).unwrap();
    // Run backward LSTM independently (reverse input, run, reverse output).
    let reversed = input.flip(0).unwrap();
    let (bwd_rev, _) = bilstm.backward_lstm().forward_seq(&reversed, None).unwrap();
    let bwd_out = bwd_rev.flip(0).unwrap();

    let out_v = outputs.to_flat_vec::<f32>().unwrap();
    let fwd_v = fwd_out.to_flat_vec::<f32>().unwrap();
    let bwd_v = bwd_out.to_flat_vec::<f32>().unwrap();

    for t in 0..seq_len {
        for i in 0..h {
            let idx = t * (2 * h) + i;
            assert!(
                (out_v[idx] - fwd_v[t * h + i]).abs() < 1e-5,
                "fwd mismatch at t={t} i={i}"
            );
            assert!(
                (out_v[t * (2 * h) + h + i] - bwd_v[t * h + i]).abs() < 1e-5,
                "bwd mismatch at t={t} i={i}"
            );
        }
    }
}

#[test]
fn test_bilstm_with_initial_hidden_state() {
    let h = 2;
    let input_size = 3;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input = DynTensor::full(&[3, 1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();

    // Run with default (zero) state.
    let (out_zero, _, _) = bilstm.forward_seq(&input, None, None).unwrap();

    // Run with non-zero forward state.
    let state = LstmState::new(
        DynTensor::full(&[1, h], 0.5, DType::F32, &Device::Cpu).unwrap(),
        DynTensor::full(&[1, h], 0.3, DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();
    let (out_nz, _, _) = bilstm.forward_seq(&input, Some(&state), None).unwrap();

    // Outputs must differ.
    let z = out_zero.to_flat_vec::<f32>().unwrap();
    let nz = out_nz.to_flat_vec::<f32>().unwrap();
    assert!(
        z.iter().zip(nz.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "non-zero initial state should change outputs"
    );
}

// ===========================================================================
// MoeRouter tests
// ===========================================================================

#[test]
fn test_moe_router_output_shape() {
    let num_experts = 4;
    let top_k = 2;
    let dim = 8;
    let gate = make_linear(num_experts, dim, 0.1);
    let router = MoeRouter::new(gate, num_experts, top_k).unwrap();

    let x = DynTensor::ones(&[2, 5, dim], DType::F32, &Device::Cpu).unwrap();
    let routing = router.forward(&x).unwrap();

    // weights and indices: [2, 5, 2]
    assert_eq!(routing.weights.dims(), &[2, 5, top_k]);
    assert_eq!(routing.indices.dims(), &[2, 5, top_k]);
}

#[test]
fn test_moe_router_topk_selection() {
    // Gate with identity-ish weights: expert i fires when feature i is high.
    let num_experts = 3;
    let top_k = 1;
    let dim = 3;

    let mut gate_data = vec![0.0f32; num_experts * dim];
    for e in 0..num_experts {
        gate_data[e * dim + e] = 10.0; // strong diagonal
    }
    let gate_w = DynTensor::from_vec(gate_data, &[num_experts, dim], &Device::Cpu).unwrap();
    let gate = Linear::new(gate_w, None).unwrap();
    let router = MoeRouter::new(gate, num_experts, top_k).unwrap();

    // Token with feature 0 dominant -> expert 0
    let x = DynTensor::from_vec(vec![10.0, 0.0, 0.0], &[1, 1, dim], &Device::Cpu).unwrap();
    let routing = router.forward(&x).unwrap();
    let idx = routing.indices.to_flat_vec::<u32>().unwrap();
    assert_eq!(idx[0], 0, "dominant feature 0 should route to expert 0");

    // Token with feature 2 dominant -> expert 2
    let x = DynTensor::from_vec(vec![0.0, 0.0, 10.0], &[1, 1, dim], &Device::Cpu).unwrap();
    let routing = router.forward(&x).unwrap();
    let idx = routing.indices.to_flat_vec::<u32>().unwrap();
    assert_eq!(idx[0], 2, "dominant feature 2 should route to expert 2");
}

#[test]
fn test_moe_router_weights_sum_to_one() {
    let num_experts = 4;
    let top_k = 2;
    let dim = 6;
    let gate = make_linear(num_experts, dim, 0.2);
    let router = MoeRouter::new(gate, num_experts, top_k).unwrap();

    let x = DynTensor::from_vec(
        (0..18).map(|i| i as f32 * 0.1).collect(),
        &[1, 3, dim],
        &Device::Cpu,
    )
    .unwrap();
    let routing = router.forward(&x).unwrap();
    let w = routing.weights.to_flat_vec::<f32>().unwrap();

    // Each token's top-k weights should sum to ~1.0 (renormalized).
    for t in 0..3 {
        let sum: f32 = w[t * top_k..(t + 1) * top_k].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "token {t} weights sum {sum} != 1.0"
        );
    }
}

// ===========================================================================
// MoeDispatch tests
// ===========================================================================

#[test]
fn test_moe_dispatch_forward_shape() {
    let dim = 8;
    let ff_dim = 16;
    let num_experts = 4;
    let top_k = 2;
    let cfg = MoeDispatchConfig::new(num_experts, top_k, dim, ff_dim, true).unwrap();

    let router = make_linear(num_experts, dim, 0.1);
    let experts: Vec<SwiGluExpert> = (0..num_experts)
        .map(|_| make_expert(dim, ff_dim, 0.1))
        .collect();
    let dispatch = MoeDispatch::new(router, experts, cfg).unwrap();

    let x = DynTensor::ones(&[2, 3, dim], DType::F32, &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, dim]);
}

#[test]
fn test_moe_dispatch_different_expert_counts() {
    // Verify dispatch works with 2 experts top-1.
    for n_exp in [2, 6, 8] {
        let dim = 4;
        let ff_dim = 8;
        let top_k = 1;
        let cfg = MoeDispatchConfig::new(n_exp, top_k, dim, ff_dim, true).unwrap();
        let router = make_linear(n_exp, dim, 0.05);
        let experts: Vec<SwiGluExpert> =
            (0..n_exp).map(|_| make_expert(dim, ff_dim, 0.05)).collect();
        let dispatch = MoeDispatch::new(router, experts, cfg).unwrap();

        let x = DynTensor::ones(&[1, 2, dim], DType::F32, &Device::Cpu).unwrap();
        let out = dispatch.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 2, dim], "failed for num_experts={n_exp}");
    }
}

#[test]
fn test_moe_dispatch_aux_loss_is_positive_scalar() {
    let dim = 8;
    let ff_dim = 16;
    let num_experts = 4;
    let top_k = 2;
    let cfg = MoeDispatchConfig::new(num_experts, top_k, dim, ff_dim, true).unwrap();
    let router = make_linear(num_experts, dim, 0.1);
    let experts: Vec<SwiGluExpert> = (0..num_experts)
        .map(|_| make_expert(dim, ff_dim, 0.1))
        .collect();
    let dispatch = MoeDispatch::new(router, experts, cfg).unwrap();

    let x = DynTensor::ones(&[2, 4, dim], DType::F32, &Device::Cpu).unwrap();
    let output = dispatch.forward_with_aux_loss(&x).unwrap();

    // aux_loss should be a scalar.
    assert_eq!(output.aux_loss.rank(), 0, "aux_loss should be scalar");
    let loss_val = output.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        loss_val >= 0.0,
        "aux_loss should be non-negative: {loss_val}"
    );
    assert!(loss_val.is_finite(), "aux_loss should be finite");

    // hidden_states should match input shape.
    assert_eq!(output.hidden_states.dims(), &[2, 4, dim]);
}

// ===========================================================================
// DiTBlock tests
// ===========================================================================

#[test]
fn test_dit_block_forward_output_shape() -> Result<()> {
    let dim = 8;
    let cond_dim = 8;

    // AdaLnZero projects cond_dim -> 3*dim
    let proj1 = make_linear(3 * dim, cond_dim, 0.01);
    let proj2 = make_linear(3 * dim, cond_dim, 0.01);
    let adaln_attn = AdaLnZero::new(proj1, Box::new(identity_rms_norm(dim)), dim)?;
    let adaln_ffn = AdaLnZero::new(proj2, Box::new(identity_rms_norm(dim)), dim)?;

    let block = DiTBlock::new(adaln_attn, identity_module(), adaln_ffn, identity_module())?;

    let x = DynTensor::ones(&[2, 5, dim], DType::F32, &Device::Cpu)?;
    let cond = DynTensor::ones(&[2, 5, cond_dim], DType::F32, &Device::Cpu)?;

    let out = block.forward(&x, &cond)?;
    assert_eq!(out.dims(), &[2, 5, dim]);

    // All values should be finite.
    let vals = out.to_flat_vec::<f32>()?;
    assert!(vals.iter().all(|v| v.is_finite()));
    Ok(())
}

#[test]
fn test_dit_block_adaln_modulation_affects_output() -> Result<()> {
    let dim = 4;
    let cond_dim = 4;

    // Non-zero projections so modulation has effect.
    let proj1 = make_linear(3 * dim, cond_dim, 0.5);
    let proj2 = make_linear(3 * dim, cond_dim, 0.5);
    let adaln_attn = AdaLnZero::new(proj1, Box::new(identity_rms_norm(dim)), dim)?;
    let adaln_ffn = AdaLnZero::new(proj2, Box::new(identity_rms_norm(dim)), dim)?;
    let block = DiTBlock::new(adaln_attn, identity_module(), adaln_ffn, identity_module())?;

    let x = DynTensor::ones(&[1, 2, dim], DType::F32, &Device::Cpu)?;

    // Two different conditioning signals.
    let cond_a = DynTensor::full(&[1, 2, cond_dim], 1.0, DType::F32, &Device::Cpu)?;
    let cond_b = DynTensor::full(&[1, 2, cond_dim], -1.0, DType::F32, &Device::Cpu)?;

    let out_a = block.forward(&x, &cond_a)?;
    let out_b = block.forward(&x, &cond_b)?;

    let a_vals = out_a.to_flat_vec::<f32>()?;
    let b_vals = out_b.to_flat_vec::<f32>()?;
    assert!(
        a_vals
            .iter()
            .zip(b_vals.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "different conditioning should produce different outputs"
    );
    Ok(())
}

#[test]
fn test_dit_block_dual_forward_shape() -> Result<()> {
    let dim = 8;

    // AdaLnZeroDual projects dim -> 6*dim
    let modulation = make_linear(6 * dim, dim, 0.01);
    let adaln = AdaLnZeroDual::new(modulation, dim)?;

    let block = DiTBlockDual::new(
        adaln,
        identity_module(), // norm_attn
        identity_module(), // attn
        identity_module(), // norm_ffn
        identity_module(), // ffn
    )?;

    let x = DynTensor::ones(&[2, 3, dim], DType::F32, &Device::Cpu)?;
    let t_emb = DynTensor::ones(&[2, dim], DType::F32, &Device::Cpu)?;

    let out = block.forward(&x, &t_emb)?;
    assert_eq!(out.dims(), &[2, 3, dim]);
    Ok(())
}

// ===========================================================================
// GatedDeltaNet tests
// ===========================================================================

fn make_gated_delta_net(dim: usize, num_heads: usize) -> GatedDeltaNet {
    let key_dim = dim / num_heads;
    let value_dim = dim / num_heads;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;

    GatedDeltaNet::new(
        make_linear(qk_total, dim, 0.01),  // q_proj
        make_linear(qk_total, dim, 0.01),  // k_proj
        make_linear(v_total, dim, 0.01),   // v_proj
        make_linear(num_heads, dim, 0.01), // gate_proj
        make_linear(num_heads, dim, 0.01), // beta_proj
        make_linear(dim, v_total, 0.01),   // out_proj
        num_heads,
        key_dim,
        value_dim,
    )
    .unwrap()
}

#[test]
fn test_gated_delta_net_forward_shape() {
    let dim = 8;
    let num_heads = 2;
    let gdn = make_gated_delta_net(dim, num_heads);

    let x = DynTensor::ones(&[1, 4, dim], DType::F32, &Device::Cpu).unwrap();
    let (out, state) = gdn.forward(&x, None).unwrap();

    assert_eq!(out.dims(), &[1, 4, dim]);
    // State: [B, H, K, V]
    let key_dim = dim / num_heads;
    let value_dim = dim / num_heads;
    assert_eq!(state.state.dims(), &[1, num_heads, key_dim, value_dim]);
}

#[test]
fn test_gated_delta_net_state_update() {
    // Run two forward passes: first creates state, second uses it.
    // Output with state should differ from output without state.
    let dim = 8;
    let num_heads = 2;
    let gdn = make_gated_delta_net(dim, num_heads);

    let x = DynTensor::full(&[1, 3, dim], 0.5, DType::F32, &Device::Cpu).unwrap();

    // First pass: creates initial state.
    let (out1, state1) = gdn.forward(&x, None).unwrap();

    // Second pass: uses returned state.
    let (out2, _state2) = gdn.forward(&x, Some(&state1)).unwrap();

    let v1 = out1.to_flat_vec::<f32>().unwrap();
    let v2 = out2.to_flat_vec::<f32>().unwrap();
    assert!(
        v1.iter().zip(v2.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "output with accumulated state should differ from initial pass"
    );
}

#[test]
fn test_gated_delta_net_batch_dimension() {
    let dim = 8;
    let num_heads = 2;
    let gdn = make_gated_delta_net(dim, num_heads);

    let batch = 3;
    let x = DynTensor::ones(&[batch, 2, dim], DType::F32, &Device::Cpu).unwrap();
    let (out, state) = gdn.forward(&x, None).unwrap();

    assert_eq!(out.dim(0).unwrap(), batch);
    assert_eq!(state.state.dim(0).unwrap(), batch);
}

#[test]
fn test_gated_delta_net_single_timestep() {
    let dim = 8;
    let num_heads = 2;
    let gdn = make_gated_delta_net(dim, num_heads);

    let x = DynTensor::ones(&[1, 1, dim], DType::F32, &Device::Cpu).unwrap();
    let (out, _state) = gdn.forward(&x, None).unwrap();

    assert_eq!(out.dims(), &[1, 1, dim]);
    let vals = out.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all values should be finite"
    );
}

// ===========================================================================
// PixelShuffle / PixelUnshuffle tests
// ===========================================================================

#[test]
fn test_pixel_shuffle_shape_factor_2() {
    let ps = PixelShuffle::new(2).unwrap();

    // [B=1, C=4, H=3, W=3] -> [B=1, C=4/(2*2)=1, H=3*2=6, W=3*2=6]
    let x = DynTensor::ones(&[1, 4, 3, 3], DType::F32, &Device::Cpu).unwrap();
    let out = ps.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 1, 6, 6]);
}

#[test]
fn test_pixel_shuffle_shape_factor_3() {
    let ps = PixelShuffle::new(3).unwrap();

    // [B=1, C=9, H=2, W=2] -> [B=1, C=9/9=1, H=2*3=6, W=2*3=6]
    let x = DynTensor::ones(&[1, 9, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let out = ps.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 1, 6, 6]);
}

#[test]
fn test_pixel_unshuffle_shape_factor_2() {
    let pus = PixelUnshuffle::new(2).unwrap();

    // [B=1, C=1, H=6, W=6] -> [B=1, C=1*4=4, H=3, W=3]
    let x = DynTensor::ones(&[1, 1, 6, 6], DType::F32, &Device::Cpu).unwrap();
    let out = pus.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, 3, 3]);
}

#[test]
fn test_pixel_shuffle_unshuffle_roundtrip() {
    let factor = 2;
    let ps = PixelShuffle::new(factor).unwrap();
    let pus = PixelUnshuffle::new(factor).unwrap();

    // Start with [1, 4, 3, 3], shuffle to [1, 1, 6, 6], unshuffle back.
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 4, 3, 3], &Device::Cpu).unwrap();

    let shuffled = ps.forward(&x).unwrap();
    assert_eq!(shuffled.dims(), &[1, 1, 6, 6]);

    let roundtrip = pus.forward(&shuffled).unwrap();
    assert_eq!(roundtrip.dims(), &[1, 4, 3, 3]);

    let rt_vals = roundtrip.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in data.iter().zip(rt_vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "roundtrip mismatch at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_pixel_shuffle_preserves_batch_dim() {
    let ps = PixelShuffle::new(2).unwrap();
    let batch = 3;
    let x = DynTensor::ones(&[batch, 4, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let out = ps.forward(&x).unwrap();
    assert_eq!(out.dim(0).unwrap(), batch);
    assert_eq!(out.dims(), &[batch, 1, 4, 4]);
}

// ===========================================================================
// SqueezeExcitation tests
// ===========================================================================

#[test]
fn test_squeeze_excitation_output_shape() {
    let channels = 8;
    let reduced = 2;
    let se = SqueezeExcitation::new(
        make_linear_with_bias(reduced, channels, 0.1, 0.0),
        make_linear_with_bias(channels, reduced, 0.1, 0.0),
        channels,
    )
    .unwrap();

    let x = DynTensor::ones(&[2, channels, 4, 4], DType::F32, &Device::Cpu).unwrap();
    let out = se.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, channels, 4, 4]);
}

#[test]
fn test_squeeze_excitation_attention_weights_positive() {
    // SE applies sigmoid, so attention weights are in (0, 1).
    // Output = input * sigmoid(excitation), so output should be positive
    // when input is positive.
    let channels = 4;
    let reduced = 2;
    let se = SqueezeExcitation::new(
        make_linear_with_bias(reduced, channels, 0.1, 0.0),
        make_linear_with_bias(channels, reduced, 0.1, 0.0),
        channels,
    )
    .unwrap();

    let x = DynTensor::full(&[1, channels, 3, 3], 1.0, DType::F32, &Device::Cpu).unwrap();
    let out = se.forward(&x).unwrap();
    let vals = out.to_flat_vec::<f32>().unwrap();

    // All outputs should be > 0 (positive input * sigmoid > 0).
    assert!(
        vals.iter().all(|&v| v > 0.0),
        "SE output should be positive for positive input"
    );
    // Sigmoid output is in (0, 1), so output should be <= input.
    assert!(
        vals.iter().all(|&v| v <= 1.0 + 1e-5),
        "SE output should be <= input when input=1.0"
    );
}

#[test]
fn test_squeeze_excitation_channel_specific() {
    // With enough weights, SE should produce different attention per channel.
    let channels = 4;
    let reduced = 2;

    // Use distinct weights for each row to create channel-specific attention.
    let mut w1_data = vec![0.0f32; reduced * channels];
    for r in 0..reduced {
        for c in 0..channels {
            w1_data[r * channels + c] = (r as f32 + 1.0) * (c as f32 + 1.0) * 0.1;
        }
    }
    let w1 = DynTensor::from_vec(w1_data, &[reduced, channels], &Device::Cpu).unwrap();
    let fc1 = Linear::new(w1, None).unwrap();

    let mut w2_data = vec![0.0f32; channels * reduced];
    for r in 0..channels {
        for c in 0..reduced {
            w2_data[r * reduced + c] = (r as f32 + 1.0) * (c as f32 + 1.0) * 0.1;
        }
    }
    let w2 = DynTensor::from_vec(w2_data, &[channels, reduced], &Device::Cpu).unwrap();
    let fc2 = Linear::new(w2, None).unwrap();

    let se = SqueezeExcitation::new(fc1, fc2, channels).unwrap();

    // Different input per channel.
    let mut input_data = vec![1.0f32; channels * 2 * 2];
    for c in 0..channels {
        for hw in 0..4 {
            input_data[c * 4 + hw] = (c as f32 + 1.0) * 0.5;
        }
    }
    let x = DynTensor::from_vec(input_data, &[1, channels, 2, 2], &Device::Cpu).unwrap();
    let out = se.forward(&x).unwrap();
    let vals = out.to_flat_vec::<f32>().unwrap();

    // Check that different channels get different scaling.
    let ch0_mean: f32 = vals[0..4].iter().sum::<f32>() / 4.0;
    let ch3_mean: f32 = vals[12..16].iter().sum::<f32>() / 4.0;
    assert!(
        (ch0_mean - ch3_mean).abs() > 1e-4,
        "SE should produce channel-specific attention: ch0={ch0_mean} ch3={ch3_mean}"
    );
}

// ===========================================================================
// Edge case / validation tests
// ===========================================================================

#[test]
fn test_pixel_shuffle_factor_zero_rejected() {
    let result = PixelShuffle::new(0);
    assert!(result.is_err(), "factor 0 should be rejected");
}

#[test]
fn test_pixel_unshuffle_factor_zero_rejected() {
    let result = PixelUnshuffle::new(0);
    assert!(result.is_err(), "factor 0 should be rejected");
}

#[test]
fn test_moe_router_topk_exceeds_experts_rejected() {
    let gate = make_linear(3, 4, 0.1);
    let result = MoeRouter::new(gate, 3, 4); // top_k > num_experts
    assert!(result.is_err(), "top_k > num_experts should be rejected");
}
