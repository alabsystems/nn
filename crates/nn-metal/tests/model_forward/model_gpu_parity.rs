// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU forward pass parity tests: CPU vs GPU comparison.
//!
//! Three categories:
//! 1. **Whisper encoder** (gated on `WHISPER_WEIGHTS` env var for real weights,
//!    always-run zero-weight CPU vs GPU comparison).
//! 2. **Silero VAD** (gated on model file existence for real weights,
//!    always-run zero-weight CPU vs GPU comparison via DynTensor path).
//! 3. **DynTensor op sequence parity** (always run, no weights needed):
//!    - Linear -> ReLU -> Linear -> Softmax
//!    - Conv1d -> BatchNorm -> ReLU
//!    - Embedding -> LayerNorm -> MatMul
//!
//! These tests exercise composed model-building-block sequences on both CPU
//! and Metal, comparing outputs element-wise within floating-point tolerance.

use super::test_utils::{assert_gpu_cpu_close, gpu_init, rand_f32_vec};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BatchNormConfig, Conv1dConfig, Embedding, LayerNorm, Linear, Module};
use nn_core::{DType, Device, VarBuilder};

/// Tolerance for f32 CPU vs GPU comparison on composed multi-layer sequences.
const TOL: f32 = 1e-3;

fn init() {
    gpu_init();
}

// ===========================================================================
// Section A: Whisper encoder CPU vs GPU parity
// ===========================================================================

/// Whisper encoder: zero-weight CPU vs GPU parity (always runs).
///
/// Loads identical zero-weight tiny Whisper models on CPU and Metal, runs the
/// encoder forward pass on both, and compares outputs element-wise.
/// Zero weights produce deterministic (though degenerate) outputs through the
/// full encoder path: Conv1d stem -> positional embedding -> transformer layers.
#[test]
fn test_whisper_encoder_cpu_gpu_parity_zeros() {
    init();
    let config = nn_whisper::test_utils::tiny_config();

    // CPU reference.
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model_cpu =
        nn_whisper::WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let mel_cpu = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::Cpu)
        .expect("CPU mel input");
    let enc_cpu = model_cpu.encode(&mel_cpu).expect("CPU encode");

    // GPU.
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_gpu =
        nn_whisper::WhisperModel::load(&vb_gpu, config.clone()).expect("GPU model load");
    let mel_gpu = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::metal())
        .expect("GPU mel input");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("GPU encode");

    // Shape validation.
    assert_eq!(enc_gpu.rank(), 3, "encoder output should be rank 3");
    assert_eq!(enc_gpu.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(enc_gpu.dim(2).unwrap(), config.d_model, "d_model dim");
    assert_eq!(enc_gpu.dims(), enc_cpu.dims(), "shapes must match");
    assert_eq!(enc_gpu.device(), Device::metal(), "output stays on GPU");

    assert_gpu_cpu_close(&enc_gpu, &enc_cpu, TOL, "whisper_encoder_zeros");
}

/// Whisper encode->decode round-trip: zero-weight CPU vs GPU parity.
///
/// Exercises the full Whisper pipeline: encoder + decoder with KV cache init.
#[test]
fn test_whisper_roundtrip_cpu_gpu_parity_zeros() {
    init();
    let config = nn_whisper::test_utils::tiny_config();

    let run = |device: &Device| -> (DynTensor, DynTensor) {
        let vb = VarBuilder::zeros(DType::F32, device);
        let mut model = nn_whisper::WhisperModel::load(&vb, config.clone()).expect("model load");

        let mel =
            DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, device).expect("mel input");
        let enc_out = model.encode(&mel).expect("encode");

        let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], device).expect("token tensor");
        let logits = model.decode(&tokens, &enc_out, true, 0).expect("decode");
        (enc_out, logits)
    };

    let (enc_cpu, logits_cpu) = run(&Device::Cpu);
    let (enc_gpu, logits_gpu) = run(&Device::metal());

    assert_gpu_cpu_close(&enc_gpu, &enc_cpu, TOL, "whisper_rt_encoder");
    assert_gpu_cpu_close(&logits_gpu, &logits_cpu, TOL, "whisper_rt_logits");
}

/// Whisper encoder with real weights: CPU vs GPU parity.
///
/// Gated on `WHISPER_WEIGHTS` env var. When set, loads real safetensors weights
/// and compares encoder output on CPU vs Metal within tolerance.
#[test]
fn test_whisper_encoder_cpu_gpu_parity_real_weights() {
    init();

    let weights_path = match std::env::var("WHISPER_WEIGHTS") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "SKIP: WHISPER_WEIGHTS not set. Set to path of whisper .safetensors to enable."
            );
            return;
        }
    };

    if !weights_path.exists() {
        eprintln!(
            "SKIP: WHISPER_WEIGHTS={} does not exist.",
            weights_path.display()
        );
        return;
    }

    let tensors =
        nn_core::dyn_tensor::load_safetensors(&weights_path).expect("load whisper weights");

    // Use the tiny config for speed; real weights will be subset-loaded.
    // For full-size models, use the appropriate config.
    let config = nn_whisper::test_utils::tiny_config();

    let vb_cpu = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
    let mut model_cpu =
        nn_whisper::WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");

    let vb_gpu = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model_gpu =
        nn_whisper::WhisperModel::load(&vb_gpu, config.clone()).expect("GPU model load");

    let mel_cpu =
        DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::Cpu).expect("CPU mel");
    let mel_gpu = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::metal())
        .expect("GPU mel");

    let enc_cpu = model_cpu.encode(&mel_cpu).expect("CPU encode");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("GPU encode");

    assert_eq!(enc_gpu.dims(), enc_cpu.dims());
    assert_gpu_cpu_close(&enc_gpu, &enc_cpu, 1e-3, "whisper_real_weights");
}

// ===========================================================================
// Section B: Silero VAD CPU vs GPU parity (via DynTensor nn layers)
// ===========================================================================

/// Silero VAD-like pipeline: Conv1d encoder -> LSTM-like path on CPU vs GPU.
///
/// Uses the DynTensor nn layer API to construct a VAD-like pipeline and
/// compare CPU vs Metal output. No real Silero weights needed — tests the
/// composition of Conv1d -> ReLU -> mean-pool -> Linear -> sigmoid.
#[test]
fn test_silero_vad_like_pipeline_cpu_gpu_parity() {
    init();

    let run = |device: &Device| -> DynTensor {
        let vb = VarBuilder::zeros(DType::F32, device);

        // Encoder: Conv1d(1, 16, kernel=3) -> ReLU -> Conv1d(16, 32, kernel=3)
        let conv1 =
            nn_core::layers::conv1d(1, 16, 3, Conv1dConfig::default(), vb.pp("enc.0")).expect("conv1");
        let conv2 = nn_core::layers::conv1d(16, 32, 3, Conv1dConfig::default(), vb.pp("enc.1"))
            .expect("conv2");

        // Output head: Linear(32, 1) -> sigmoid
        let fc = nn_core::layers::linear(32, 1, vb.pp("head")).expect("fc");

        // Input: [1, 1, 64] audio chunk
        let x = DynTensor::zeros(&[1, 1, 64], DType::F32, device).expect("input");
        let h = conv1.forward(&x).expect("conv1 fwd");
        let h = h.relu().expect("relu1");
        let h = conv2.forward(&h).expect("conv2 fwd");
        let h = h.relu().expect("relu2");
        // Global average pool over time -> [1, 32]
        let h = h.mean(2).expect("mean pool");
        let h = fc.forward(&h).expect("fc fwd");
        // Sigmoid for probability output
        h.sigmoid().expect("sigmoid")
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), cpu_out.dims(), "shape mismatch");
    assert_eq!(gpu_out.dims(), &[1, 1], "output should be [B, 1]");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "silero_vad_like");
}

/// Silero VAD with real weights gated on model file existence.
///
/// Loads the real Silero VAD model via the Metal-native SileroVad API
/// and verifies forward pass produces valid probabilities.
#[test]
fn test_silero_vad_real_weights_gpu() {
    init();

    let env_path = std::env::var("SILERO_VAD_WEIGHTS").ok();
    let weights_path = env_path.map(std::path::PathBuf::from).unwrap_or_else(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("project root")
            .to_path_buf();
        root.join("models/silero_vad/silero_vad_16k.safetensors")
    });

    if !weights_path.exists() {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}. \
             Set SILERO_VAD_WEIGHTS or run converter script.",
            weights_path.display()
        );
        return;
    }

    let model = nn_metal::SileroVad::load_safetensors(&weights_path).expect("load silero weights");
    let _backend = nn_metal::MetalBackend::init().expect("Metal backend");
    let cache = nn_metal::PipelineCache::new_global().expect("pipeline cache");
    let mut state = nn_metal::SileroVadState::zero();

    // Run 3 chunks of silence and verify output range.
    let silence = vec![0.0f32; 512];
    for i in 0..3 {
        let prob = model
            .process(&cache, &silence, &mut state)
            .unwrap_or_else(|e| panic!("process chunk {i}: {e}"));
        assert!(
            (0.0..=1.0).contains(&prob),
            "chunk {i}: probability {prob} outside [0, 1]",
        );
    }
}

// ===========================================================================
// Section C: DynTensor op sequence parity (always run, no weights needed)
// ===========================================================================

/// Linear -> ReLU -> Linear -> Softmax: random-weight CPU vs GPU parity.
///
/// Exercises the most common MLP pattern with non-zero weights to catch
/// numerical divergence between CPU and Metal dispatch paths.
#[test]
fn test_op_sequence_linear_relu_linear_softmax() {
    init();
    let batch = 4;
    let in_feat = 32;
    let hidden = 64;
    let out_feat = 16;

    let x_data = rand_f32_vec(300, batch * in_feat, -1.0, 1.0);
    let w1_data = rand_f32_vec(301, hidden * in_feat, -0.3, 0.3);
    let b1_data = rand_f32_vec(302, hidden, -0.1, 0.1);
    let w2_data = rand_f32_vec(303, out_feat * hidden, -0.3, 0.3);
    let b2_data = rand_f32_vec(304, out_feat, -0.1, 0.1);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, in_feat], device).unwrap();
        let w1 = DynTensor::new(&w1_data, &[hidden, in_feat], device).unwrap();
        let b1 = DynTensor::new(&b1_data, &[hidden], device).unwrap();
        let w2 = DynTensor::new(&w2_data, &[out_feat, hidden], device).unwrap();
        let b2 = DynTensor::new(&b2_data, &[out_feat], device).unwrap();

        let linear1 = Linear::new(w1, Some(b1)).unwrap();
        let linear2 = Linear::new(w2, Some(b2)).unwrap();

        let h = linear1.forward(&x).unwrap();
        let h = h.relu().unwrap();
        let y = linear2.forward(&h).unwrap();
        y.softmax(1).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, out_feat]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "linear_relu_linear_softmax");

    // Softmax output should sum to ~1.0 per row.
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for row in 0..batch {
        let row_sum: f32 = gpu_vals[row * out_feat..(row + 1) * out_feat].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "softmax row {row} sum={row_sum}, expected ~1.0"
        );
    }
}

/// Conv1d -> BatchNorm -> ReLU: random-weight CPU vs GPU parity.
///
/// Exercises the convolutional block pattern common in audio models
/// (Silero VAD, HTDemucs encoder stages).
#[test]
fn test_op_sequence_conv1d_batchnorm_relu() {
    init();
    let batch = 2;
    let in_ch = 4;
    let out_ch = 8;
    let kernel = 3;
    let seq_len = 32;

    let run = |device: &Device| -> DynTensor {
        let vb = VarBuilder::zeros(DType::F32, device);

        let conv = nn_core::layers::conv1d(
            in_ch,
            out_ch,
            kernel,
            Conv1dConfig::default(),
            vb.pp("conv"),
        )
        .expect("conv1d");
        let bn = nn_core::layers::batch_norm(out_ch, BatchNormConfig::default(), vb.pp("bn"))
            .expect("batch_norm");

        // Input: [B, C_in, T]
        let x = DynTensor::zeros(&[batch, in_ch, seq_len], DType::F32, device).expect("input");

        let h = conv.forward(&x).expect("conv fwd");
        let h = bn.forward(&h).expect("bn fwd");
        h.relu().expect("relu")
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    // Conv1d with kernel=3, no padding: output time = seq_len - 2
    let expected_time = seq_len - kernel + 1;
    assert_eq!(gpu_out.dims(), &[batch, out_ch, expected_time]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "conv1d_batchnorm_relu");
}

/// Conv1d -> BatchNorm -> ReLU with random (non-zero) weights.
///
/// Same architecture as above but with deterministic random weights to exercise
/// non-trivial numerical paths through the GPU dispatch.
#[test]
fn test_op_sequence_conv1d_batchnorm_relu_random() {
    init();
    let batch = 2;
    let in_ch = 4;
    let out_ch = 8;
    let kernel = 3;
    let seq_len = 32;

    let x_data = rand_f32_vec(400, batch * in_ch * seq_len, -1.0, 1.0);
    let conv_w = rand_f32_vec(401, out_ch * in_ch * kernel, -0.3, 0.3);
    let conv_b = rand_f32_vec(402, out_ch, -0.1, 0.1);
    let bn_mean = rand_f32_vec(403, out_ch, -0.5, 0.5);
    // running_var must be positive
    let bn_var = rand_f32_vec(404, out_ch, 0.5, 2.0);
    let bn_w = rand_f32_vec(405, out_ch, 0.8, 1.2);
    let bn_b = rand_f32_vec(406, out_ch, -0.1, 0.1);

    let run = |device: &Device| -> DynTensor {
        use nn_core::layers::{BatchNorm, Conv1d};

        let x = DynTensor::new(&x_data, &[batch, in_ch, seq_len], device).unwrap();
        let w = DynTensor::new(&conv_w, &[out_ch, in_ch, kernel], device).unwrap();
        let b = DynTensor::new(&conv_b, &[out_ch], device).unwrap();
        let conv = Conv1d::new(w, Some(b), Conv1dConfig::default()).unwrap();

        let rm = DynTensor::new(&bn_mean, &[out_ch], device).unwrap();
        let rv = DynTensor::new(&bn_var, &[out_ch], device).unwrap();
        let bw = DynTensor::new(&bn_w, &[out_ch], device).unwrap();
        let bb = DynTensor::new(&bn_b, &[out_ch], device).unwrap();
        let bn = BatchNorm::new(rm, rv, Some(bw), Some(bb), 1e-5).unwrap();

        let h = conv.forward(&x).unwrap();
        let h = bn.forward(&h).unwrap();
        h.relu().unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    let expected_time = seq_len - kernel + 1;
    assert_eq!(gpu_out.dims(), &[batch, out_ch, expected_time]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    // BatchNorm with non-zero running stats + affine: slightly wider tolerance.
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "conv1d_batchnorm_relu_random");
}

/// Embedding -> LayerNorm -> MatMul: CPU vs GPU parity.
///
/// Exercises the transformer front-end pattern: token embedding lookup,
/// layer normalization, and projection via matrix multiply. Uses random
/// weights and deterministic token IDs.
#[test]
fn test_op_sequence_embedding_layernorm_matmul() {
    init();
    let vocab = 64;
    let embed_dim = 32;
    let seq_len = 8;
    let proj_dim = 16;

    let embed_w = rand_f32_vec(500, vocab * embed_dim, -0.5, 0.5);
    let ln_w = rand_f32_vec(501, embed_dim, 0.8, 1.2);
    let ln_b = rand_f32_vec(502, embed_dim, -0.05, 0.05);
    let proj_w = rand_f32_vec(503, embed_dim * proj_dim, -0.3, 0.3);

    // Token IDs: deterministic sequence [5, 10, 15, 20, 25, 30, 35, 40]
    let token_ids: Vec<f32> = vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];

    let run = |device: &Device| -> DynTensor {
        let we = DynTensor::new(&embed_w, &[vocab, embed_dim], device).unwrap();
        let emb = Embedding::new(we).unwrap();

        let lw = DynTensor::new(&ln_w, &[embed_dim], device).unwrap();
        let lb = DynTensor::new(&ln_b, &[embed_dim], device).unwrap();
        let ln = LayerNorm::new(lw, lb, 1e-5).unwrap();

        let pw = DynTensor::new(&proj_w, &[embed_dim, proj_dim], device).unwrap();

        // Token IDs as DynTensor for Module::forward.
        let ids = DynTensor::new(&token_ids, &[1, seq_len], device).unwrap();

        // Embedding lookup -> [1, seq_len, embed_dim]
        let h = emb.forward(&ids).unwrap();
        // LayerNorm -> [1, seq_len, embed_dim]
        let h = ln.forward(&h).unwrap();
        // MatMul with projection -> [1, seq_len, proj_dim]
        h.matmul(&pw).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[1, seq_len, proj_dim]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "embedding_layernorm_matmul");
}

/// Embedding -> LayerNorm -> MatMul -> Softmax: full transformer-like front end.
///
/// Extends the embedding test with a final softmax to exercise the complete
/// token-to-logits path that appears in every language model.
#[test]
fn test_op_sequence_embedding_to_softmax() {
    init();
    let vocab = 64;
    let embed_dim = 32;
    let seq_len = 4;

    let embed_w = rand_f32_vec(510, vocab * embed_dim, -0.5, 0.5);
    let ln_w = rand_f32_vec(511, embed_dim, 0.8, 1.2);
    let ln_b = rand_f32_vec(512, embed_dim, -0.05, 0.05);
    // Project back to vocab for logits.
    let proj_w = rand_f32_vec(513, embed_dim * vocab, -0.3, 0.3);

    let token_ids: Vec<f32> = vec![1.0, 7.0, 13.0, 42.0];

    let run = |device: &Device| -> DynTensor {
        let we = DynTensor::new(&embed_w, &[vocab, embed_dim], device).unwrap();
        let emb = Embedding::new(we).unwrap();

        let lw = DynTensor::new(&ln_w, &[embed_dim], device).unwrap();
        let lb = DynTensor::new(&ln_b, &[embed_dim], device).unwrap();
        let ln = LayerNorm::new(lw, lb, 1e-5).unwrap();

        let pw = DynTensor::new(&proj_w, &[embed_dim, vocab], device).unwrap();

        let ids = DynTensor::new(&token_ids, &[1, seq_len], device).unwrap();

        let h = emb.forward(&ids).unwrap();
        let h = ln.forward(&h).unwrap();
        let logits = h.matmul(&pw).unwrap();
        logits.softmax(2).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[1, seq_len, vocab]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "embedding_to_softmax");

    // Verify softmax sums to ~1 per position.
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for pos in 0..seq_len {
        let row_sum: f32 = gpu_vals[pos * vocab..(pos + 1) * vocab].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "softmax position {pos} sum={row_sum}, expected ~1.0"
        );
    }
}

/// Multi-layer sequence: Linear -> SiLU -> LayerNorm -> Linear -> Tanh.
///
/// Tests the MLP block variant used in transformer feed-forward networks
/// (SwiGLU uses SiLU, many models use LayerNorm between MLP layers).
#[test]
fn test_op_sequence_linear_silu_layernorm_linear_tanh() {
    init();
    let batch = 4;
    let in_feat = 16;
    let hidden = 32;
    let out_feat = 8;

    let x_data = rand_f32_vec(600, batch * in_feat, -1.0, 1.0);
    let w1_data = rand_f32_vec(601, hidden * in_feat, -0.3, 0.3);
    let b1_data = rand_f32_vec(602, hidden, -0.1, 0.1);
    let ln_w = rand_f32_vec(603, hidden, 0.8, 1.2);
    let ln_b = rand_f32_vec(604, hidden, -0.05, 0.05);
    let w2_data = rand_f32_vec(605, out_feat * hidden, -0.3, 0.3);
    let b2_data = rand_f32_vec(606, out_feat, -0.1, 0.1);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, in_feat], device).unwrap();
        let w1 = DynTensor::new(&w1_data, &[hidden, in_feat], device).unwrap();
        let b1 = DynTensor::new(&b1_data, &[hidden], device).unwrap();
        let lw = DynTensor::new(&ln_w, &[hidden], device).unwrap();
        let lb = DynTensor::new(&ln_b, &[hidden], device).unwrap();
        let w2 = DynTensor::new(&w2_data, &[out_feat, hidden], device).unwrap();
        let b2 = DynTensor::new(&b2_data, &[out_feat], device).unwrap();

        let linear1 = Linear::new(w1, Some(b1)).unwrap();
        let ln = LayerNorm::new(lw, lb, 1e-5).unwrap();
        let linear2 = Linear::new(w2, Some(b2)).unwrap();

        let h = linear1.forward(&x).unwrap();
        let h = h.silu().unwrap();
        let h = ln.forward(&h).unwrap();
        let y = linear2.forward(&h).unwrap();
        y.tanh().unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, out_feat]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "linear_silu_ln_linear_tanh");

    // Tanh output should be in [-1, 1].
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in gpu_vals.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "tanh output[{i}]={v} outside [-1, 1]"
        );
    }
}
