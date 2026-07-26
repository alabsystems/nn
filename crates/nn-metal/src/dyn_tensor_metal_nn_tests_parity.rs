#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU parity tests for nn layers missing Metal GPU coverage.
//! Part of #1304: 13 nn layers missing GPU parity tests.
//!
//! AC1: BatchNorm (1e-4 tolerance)
//! AC2: BiLstm (1e-4 tolerance)
//! AC3: WeightNormConv1d (1e-4 tolerance)
//! AC4: GatedDeltaNet (1e-3 tolerance, complex computation)

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BatchNorm, BiLstm, Conv1dConfig, GatedDeltaNet, Linear, Module, WeightNormConv1d};
use nn_core::{DType, Device};

use crate::test_common::{assert_close, assert_gpu_matches_cpu, init};

// -- AC1: BatchNorm GPU parity tests ------------------------------------------

#[test]
fn test_gpu_batch_norm_basic() {
    // BatchNorm with zero mean, unit variance, no affine — should be identity-like.
    assert_gpu_matches_cpu(
        |dev| {
            let channels = 3;
            let mean = DynTensor::zeros(&[channels], DType::F32, dev).unwrap();
            let var = DynTensor::ones(&[channels], DType::F32, dev).unwrap();
            let layer = BatchNorm::new(mean, var, None, None, 1e-5).unwrap();
            // Input: [B=2, C=3, T=4]
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                ],
                &[2, 3, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "batch_norm_basic",
    );
}

#[test]
fn test_gpu_batch_norm_with_affine() {
    // BatchNorm with non-trivial running stats and affine parameters.
    assert_gpu_matches_cpu(
        |dev| {
            let channels = 2;
            // Running mean=[1.0, 2.0], var=[4.0, 9.0]
            let mean = DynTensor::new(&[1.0, 2.0], &[channels], dev).unwrap();
            let var = DynTensor::new(&[4.0, 9.0], &[channels], dev).unwrap();
            // Affine: weight=[2.0, 0.5], bias=[10.0, -5.0]
            let weight = DynTensor::new(&[2.0, 0.5], &[channels], dev).unwrap();
            let bias = DynTensor::new(&[10.0, -5.0], &[channels], dev).unwrap();
            let layer = BatchNorm::new(mean, var, Some(weight), Some(bias), 1e-5).unwrap();
            // Input: [B=1, C=2, T=3]
            let x = DynTensor::new(&[3.0, 5.0, 7.0, 11.0, 14.0, 17.0], &[1, 2, 3], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "batch_norm_affine",
    );
}

#[test]
fn test_gpu_batch_norm_2d_fused_4d_input() {
    // BatchNorm2d with 4D input [N, C, H, W] -- the ResNet use case (#4324).
    // Verifies the fused Metal kernel correctly indexes channels in NCHW layout.
    use nn_core::layers::BatchNorm2d;
    init();
    let channels = 3;
    // Running stats: mean=[0.5, -0.5, 1.0], var=[2.0, 0.5, 4.0]
    let cpu = Device::Cpu;
    let mean_cpu = DynTensor::new(&[0.5, -0.5, 1.0], &[channels], &cpu).unwrap();
    let var_cpu = DynTensor::new(&[2.0, 0.5, 4.0], &[channels], &cpu).unwrap();
    let weight_cpu = DynTensor::new(&[1.0, 2.0, 0.5], &[channels], &cpu).unwrap();
    let bias_cpu = DynTensor::new(&[0.0, 1.0, -1.0], &[channels], &cpu).unwrap();
    // Input: [B=1, C=3, H=2, W=2]
    let x_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let x_cpu = DynTensor::from_vec(x_data, &[1, 3, 2, 2], &cpu).unwrap();

    let bn_cpu = BatchNorm2d::new(
        channels,
        mean_cpu.clone(),
        var_cpu.clone(),
        Some(weight_cpu.clone()),
        Some(bias_cpu.clone()),
        1e-5,
    )
    .unwrap();
    let y_cpu = bn_cpu.forward(&x_cpu).unwrap();

    // GPU path.
    let gpu = Device::metal();
    let mean_gpu = mean_cpu.to_device(&gpu).unwrap();
    let var_gpu = var_cpu.to_device(&gpu).unwrap();
    let weight_gpu = weight_cpu.to_device(&gpu).unwrap();
    let bias_gpu = bias_cpu.to_device(&gpu).unwrap();
    let x_gpu = x_cpu.to_device(&gpu).unwrap();

    let bn_gpu = BatchNorm2d::new(
        channels,
        mean_gpu,
        var_gpu,
        Some(weight_gpu),
        Some(bias_gpu),
        1e-5,
    )
    .unwrap();
    let y_gpu = bn_gpu.forward(&x_gpu).unwrap();
    let y_gpu_cpu = y_gpu.to_device(&cpu).unwrap();

    let cpu_vals = y_cpu.to_flat_vec::<f32>().unwrap();
    let gpu_vals = y_gpu_cpu.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "batch_norm_2d_fused_4d");
}

#[test]
fn test_gpu_batch_norm_2d_no_affine() {
    // BatchNorm2d without weight/bias -- pure normalization (#4324).
    use nn_core::layers::BatchNorm2d;
    init();
    let channels = 2;
    let cpu = Device::Cpu;
    let mean_cpu = DynTensor::new(&[1.0, 2.0], &[channels], &cpu).unwrap();
    let var_cpu = DynTensor::new(&[4.0, 9.0], &[channels], &cpu).unwrap();
    // Input: [B=2, C=2, H=1, W=3]
    let x_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x_cpu = DynTensor::from_vec(x_data, &[2, 2, 1, 3], &cpu).unwrap();

    let bn_cpu = BatchNorm2d::new(channels, mean_cpu.clone(), var_cpu.clone(), None, None, 1e-5).unwrap();
    let y_cpu = bn_cpu.forward(&x_cpu).unwrap();

    let gpu = Device::metal();
    let mean_gpu = mean_cpu.to_device(&gpu).unwrap();
    let var_gpu = var_cpu.to_device(&gpu).unwrap();
    let x_gpu = x_cpu.to_device(&gpu).unwrap();

    let bn_gpu = BatchNorm2d::new(channels, mean_gpu, var_gpu, None, None, 1e-5).unwrap();
    let y_gpu = bn_gpu.forward(&x_gpu).unwrap();
    let y_gpu_cpu = y_gpu.to_device(&cpu).unwrap();

    let cpu_vals = y_cpu.to_flat_vec::<f32>().unwrap();
    let gpu_vals = y_gpu_cpu.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "batch_norm_2d_no_affine");
}

// -- AC2: BiLstm GPU parity tests --------------------------------------------

/// Build a BiLstm on the given device with small dimensions for testing.
fn build_bilstm(dev: &Device) -> (BiLstm, DynTensor) {
    let input_size = 4;
    let hidden_size = 3;

    // Forward LSTM weights
    let w_ih_fwd = DynTensor::new(
        &vec![0.1f32; 4 * hidden_size * input_size],
        &[4 * hidden_size, input_size],
        dev,
    )
    .unwrap();
    let w_hh_fwd = DynTensor::new(
        &vec![0.02f32; 4 * hidden_size * hidden_size],
        &[4 * hidden_size, hidden_size],
        dev,
    )
    .unwrap();

    // Backward LSTM weights (different values to distinguish directions)
    let w_ih_rev = DynTensor::new(
        &vec![0.05f32; 4 * hidden_size * input_size],
        &[4 * hidden_size, input_size],
        dev,
    )
    .unwrap();
    let w_hh_rev = DynTensor::new(
        &vec![0.01f32; 4 * hidden_size * hidden_size],
        &[4 * hidden_size, hidden_size],
        dev,
    )
    .unwrap();

    let bilstm = BiLstm::from_weights(
        w_ih_fwd,
        w_hh_fwd,
        None,
        None,
        w_ih_rev,
        w_hh_rev,
        None,
        None,
        hidden_size,
    )
    .unwrap();

    // Input: [seq_len=3, batch=2, input_size=4]
    let x = DynTensor::new(
        &[
            1.0, 0.5, -0.5, 2.0, -1.0, 1.5, 0.0, 0.5, 0.3, -0.3, 1.0, -1.0, 2.0, 0.0, -0.5, 1.0,
            -0.5, 0.5, 1.5, -1.5, 0.0, 1.0, -1.0, 0.5,
        ],
        &[3, 2, input_size],
        dev,
    )
    .unwrap();

    (bilstm, x)
}

#[test]
fn test_gpu_bilstm_forward_seq() {
    // BiLstm forward_seq: runs forward and backward LSTMs, concatenates outputs.
    init();

    // CPU reference
    let (cpu_bilstm, cpu_x) = build_bilstm(&Device::Cpu);
    let (cpu_out, _fwd_state, _bwd_state) = cpu_bilstm.forward_seq(&cpu_x, None, None).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU forward
    let (gpu_bilstm, gpu_x) = build_bilstm(&Device::metal());
    assert_eq!(gpu_x.device(), Device::metal());
    let (gpu_out, gpu_fwd_state, gpu_bwd_state) =
        gpu_bilstm.forward_seq(&gpu_x, None, None).unwrap();

    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "BiLstm output should stay on GPU"
    );
    // Output: [seq_len=3, batch=2, 2*hidden_size=6]
    assert_eq!(gpu_out.dims(), cpu_out.dims());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "bilstm_forward_seq");

    // Verify state devices
    assert_eq!(
        gpu_fwd_state.h.device(),
        Device::metal(),
        "BiLstm fwd state should be on GPU"
    );
    assert_eq!(
        gpu_bwd_state.h.device(),
        Device::metal(),
        "BiLstm bwd state should be on GPU"
    );
}

// -- AC3: WeightNormConv1d GPU parity tests -----------------------------------

#[test]
fn test_gpu_weight_norm_conv1d_basic() {
    // WeightNormConv1d: weight normalization applied at construction, then delegates
    // to Conv1d. The normalization (g * v / ||v||) uses broadcast_div/mul on GPU.
    assert_gpu_matches_cpu(
        |dev| {
            let out_ch = 2;
            let in_ch = 3;
            let kernel_size = 3;
            // weight_v: [out_ch, in_ch, kernel_size]
            let weight_v = DynTensor::new(
                &[
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.7, 0.8,
                    0.9, 1.0, 1.1, 1.2,
                ],
                &[out_ch, in_ch, kernel_size],
                dev,
            )
            .unwrap();
            // weight_g: [out_ch, 1, 1]
            let weight_g = DynTensor::new(&[1.5, 2.0], &[out_ch, 1, 1], dev).unwrap();
            let bias = DynTensor::new(&[0.1, -0.1], &[out_ch], dev).unwrap();

            let layer =
                WeightNormConv1d::new(weight_v, weight_g, Some(bias), Conv1dConfig::default())
                    .unwrap();
            // Input: [B=1, C=3, T=5]
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 4.5, -0.5, 0.5, 1.5, 2.5, 3.5,
                ],
                &[1, in_ch, 5],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "weight_norm_conv1d_basic",
    );
}

#[test]
fn test_gpu_weight_norm_conv1d_no_bias() {
    // WeightNormConv1d without bias, stride=2.
    assert_gpu_matches_cpu(
        |dev| {
            let out_ch = 1;
            let in_ch = 2;
            let kernel_size = 3;
            let weight_v = DynTensor::new(
                &[0.3, 0.4, 0.5, -0.3, -0.4, -0.5],
                &[out_ch, in_ch, kernel_size],
                dev,
            )
            .unwrap();
            let weight_g = DynTensor::new(&[1.0], &[out_ch, 1, 1], dev).unwrap();
            let mut config = Conv1dConfig::default();
            config.stride = 2;
            let layer = WeightNormConv1d::new(weight_v, weight_g, None, config).unwrap();
            // Input: [B=1, C=2, T=8]
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                ],
                &[1, in_ch, 8],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "weight_norm_conv1d_no_bias",
    );
}

// -- AC4: GatedDeltaNet GPU parity tests --------------------------------------

/// Build a GatedDeltaNet on the given device with small dimensions.
fn build_gated_delta_net(dev: &Device) -> (GatedDeltaNet, DynTensor) {
    let dim = 8; // model dimension
    let num_heads = 2;
    let key_dim = 4; // per-head key dim
    let value_dim = 4; // per-head value dim

    let make_linear = |out_features: usize| -> Linear {
        let w = DynTensor::new(
            &vec![0.05f32; out_features * dim],
            &[out_features, dim],
            dev,
        )
        .unwrap();
        Linear::new(w, None).unwrap()
    };

    let q_proj = make_linear(num_heads * key_dim); // [H*K, D] = [8, 8]
    let k_proj = make_linear(num_heads * key_dim); // [8, 8]
    let v_proj = make_linear(num_heads * value_dim); // [8, 8]
    let gate_proj = make_linear(num_heads); // [2, 8]
    let beta_proj = make_linear(num_heads); // [2, 8]
    let out_proj = make_linear(dim); // [8, 8]

    let gdn = GatedDeltaNet::new(
        q_proj, k_proj, v_proj, gate_proj, beta_proj, out_proj, num_heads, key_dim, value_dim,
    )
    .unwrap();

    // Input: [B=1, S=2, D=8]
    let x = DynTensor::new(
        &[
            0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, -0.3, -0.1, 0.5, 0.2, 0.1, 0.3, -0.4, 0.2, 0.1,
        ],
        &[1, 2, dim],
        dev,
    )
    .unwrap();

    (gdn, x)
}

#[test]
fn test_gpu_gated_delta_net_forward() {
    // GatedDeltaNet: complex multi-step recurrence (projections + gate + beta + state update).
    // Uses matmul, sigmoid, reshape, narrow, squeeze, unsqueeze, broadcast_mul, sub, add.
    init();

    // CPU reference
    let (cpu_gdn, cpu_x) = build_gated_delta_net(&Device::Cpu);
    let (cpu_out, cpu_state) = cpu_gdn.forward(&cpu_x, None).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    let cpu_state_vals = cpu_state.state.to_flat_vec::<f32>().unwrap();

    // GPU forward
    let (gpu_gdn, gpu_x) = build_gated_delta_net(&Device::metal());
    assert_eq!(gpu_x.device(), Device::metal());
    let (gpu_out, gpu_state) = gpu_gdn.forward(&gpu_x, None).unwrap();

    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "GatedDeltaNet output should stay on GPU"
    );
    assert_eq!(gpu_out.dims(), cpu_out.dims(), "output shape mismatch");

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // Higher tolerance for complex multi-step computation
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "gated_delta_net_output");

    // Verify state parity
    let gpu_state_vals = gpu_state
        .state
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &gpu_state_vals,
        &cpu_state_vals,
        1e-3,
        "gated_delta_net_state",
    );
}

#[test]
fn test_gpu_gated_delta_net_with_initial_state() {
    // GatedDeltaNet forward with pre-existing recurrent state (streaming scenario).
    init();

    let dim = 8;
    let num_heads = 2;
    let key_dim = 4;
    let value_dim = 4;

    // CPU: build, forward once to get state, forward again with that state
    let (cpu_gdn, cpu_x) = build_gated_delta_net(&Device::Cpu);
    let (_cpu_out1, cpu_state1) = cpu_gdn.forward(&cpu_x, None).unwrap();
    // Second forward with state from first (S=2 to avoid single-element cat on GPU)
    let cpu_x2 = DynTensor::new(
        &[
            -0.2, 0.1, 0.3, -0.1, 0.2, 0.4, -0.3, 0.1, 0.05, -0.15, 0.25, -0.05, 0.1, 0.3, -0.2,
            0.15,
        ],
        &[1, 2, dim],
        &Device::Cpu,
    )
    .unwrap();
    let (cpu_out2, _cpu_state2) = cpu_gdn.forward(&cpu_x2, Some(&cpu_state1)).unwrap();
    let cpu_vals2 = cpu_out2.to_flat_vec::<f32>().unwrap();

    // GPU: same sequence
    let (gpu_gdn, gpu_x) = build_gated_delta_net(&Device::metal());
    let (_gpu_out1, gpu_state1) = gpu_gdn.forward(&gpu_x, None).unwrap();
    let gpu_x2 = DynTensor::new(
        &[
            -0.2, 0.1, 0.3, -0.1, 0.2, 0.4, -0.3, 0.1, 0.05, -0.15, 0.25, -0.05, 0.1, 0.3, -0.2,
            0.15,
        ],
        &[1, 2, dim],
        &Device::metal(),
    )
    .unwrap();
    let (gpu_out2, gpu_state2) = gpu_gdn.forward(&gpu_x2, Some(&gpu_state1)).unwrap();

    assert_eq!(
        gpu_out2.device(),
        Device::metal(),
        "GatedDeltaNet output should stay on GPU"
    );

    let gpu_vals2 = gpu_out2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &gpu_vals2,
        &cpu_vals2,
        1e-3,
        "gated_delta_net_with_state_output",
    );

    // Verify state shape
    assert_eq!(
        gpu_state2.state.dims(),
        &[1, num_heads, key_dim, value_dim],
        "state shape"
    );
}
