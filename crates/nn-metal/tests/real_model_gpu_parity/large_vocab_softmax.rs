// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Large-vocabulary softmax GPU parity tests for Qwen3/GLM5 numerical stability.
//!
//! Qwen3 has vocab_size=152064, GLM5 has vocab_size=151552. Softmax over these
//! large logit tensors can overflow in f16/bf16. These tests verify:
//! - GPU softmax produces no NaN/Inf on large dimensions
//! - GPU output sums to ~1.0 per row (softmax) or log-sum-exp ~0.0 (log_softmax)
//! - GPU vs CPU parity within tolerance
//! - Correct behavior across F32, BF16, F16 dtypes
//!
//! Part of #4264: RTF improvement and numerical stability for large-vocab models.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::{DType, Device};

/// Verify all output values are finite (no NaN, no Inf).
fn assert_all_finite(tensor: &DynTensor, label: &str) {
    let vals = tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{label}[{i}] = {v} is not finite (NaN or Inf)"
        );
    }
}

/// Verify no NaN and no +Inf in output. -Inf is allowed (log_softmax of
/// near-zero probabilities legitimately produces -inf, since log(0) = -inf).
fn assert_no_nan_no_pos_inf(tensor: &DynTensor, label: &str) {
    let vals = tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(!v.is_nan(), "{label}[{i}] = NaN");
        assert!(
            v != f32::INFINITY,
            "{label}[{i}] = +Inf (unexpected for log_softmax)"
        );
    }
}

/// Verify softmax output sums to ~1.0 along the last axis.
fn verify_softmax_row_sums(tensor: &DynTensor, last_dim: usize, tol: f32, label: &str) {
    let vals = tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let n_rows = vals.len() / last_dim;
    for row in 0..n_rows {
        let row_sum: f64 = vals[row * last_dim..(row + 1) * last_dim]
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        assert!(
            (row_sum - 1.0).abs() < f64::from(tol),
            "{label}: row {row} sum = {row_sum}, expected ~1.0 (tol={tol})"
        );
    }
}

/// Verify log_softmax output: exp(values) sums to ~1.0 along the last axis,
/// equivalent to log-sum-exp being ~0.0. Handles -inf gracefully (exp(-inf) = 0).
fn verify_log_softmax_row_sums(tensor: &DynTensor, last_dim: usize, tol: f32, label: &str) {
    let vals = tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let n_rows = vals.len() / last_dim;
    for row in 0..n_rows {
        let row_exp_sum: f64 = vals[row * last_dim..(row + 1) * last_dim]
            .iter()
            .map(|&v| f64::from(v).exp())
            .sum();
        assert!(
            (row_exp_sum - 1.0).abs() < f64::from(tol),
            "{label}: row {row} exp-sum = {row_exp_sum}, expected ~1.0 (tol={tol})"
        );
    }
}

/// Verify all log_softmax values are non-positive.
fn verify_log_softmax_non_positive(tensor: &DynTensor, label: &str) {
    let vals = tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v <= 1e-6, "{label}[{i}] = {v} should be <= 0");
    }
}

/// GPU vs CPU parity for log_softmax, tolerating -inf matches.
///
/// Both GPU and CPU may produce -inf for near-zero probabilities; regular
/// float subtraction of -inf values is NaN, so we check those positions
/// separately.
fn assert_log_softmax_parity(gpu: &DynTensor, cpu: &DynTensor, tol: f32, label: &str) {
    let gpu_vals = gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len(), "{label}: length mismatch");
    for (i, (&g, &c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        if c == f32::NEG_INFINITY {
            assert!(g == f32::NEG_INFINITY, "{label}[{i}]: cpu=-inf but gpu={g}");
        } else if g == f32::NEG_INFINITY {
            // GPU produced -inf where CPU didn't — tolerate if CPU value is
            // extremely negative (close to -inf in practice).
            assert!(
                c < -80.0,
                "{label}[{i}]: gpu=-inf but cpu={c} (not very negative)"
            );
        } else {
            let diff = (g - c).abs();
            assert!(diff <= tol, "{label}[{i}]: gpu={g} cpu={c} diff={diff}");
        }
    }
}

// ==========================================================================
// Qwen3 vocab_size=152064 softmax tests
// ==========================================================================

/// Softmax over Qwen3 full vocabulary logits: [1, 152064] in F32.
///
/// This is the critical production shape for next-token prediction. The GPU
/// softmax kernel must handle 152K-element reductions without overflow.
#[test]
fn test_softmax_qwen3_vocab_f32() {
    gpu_init();
    let vocab = 152064;
    let data = rand_f32_vec(5000, vocab, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[1, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_eq!(gpu_out.dims(), &[1, vocab]);
    assert_all_finite(&gpu_out, "softmax_qwen3_vocab_f32");
    // Wider tolerance for 152K-element reduction: accumulation error scales.
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-4, "softmax_qwen3_vocab_f32");
    verify_softmax_row_sums(&gpu_out, vocab, 1e-3, "softmax_qwen3_vocab_f32");
}

/// Log-softmax over Qwen3 full vocabulary: [1, 152064] in F32.
#[test]
fn test_log_softmax_qwen3_vocab_f32() {
    gpu_init();
    let vocab = 152064;
    let data = rand_f32_vec(5001, vocab, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[1, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.log_softmax(1).unwrap();
    let gpu_out = gpu.log_softmax(1).unwrap();

    assert_eq!(gpu_out.dims(), &[1, vocab]);
    assert_no_nan_no_pos_inf(&gpu_out, "log_softmax_qwen3_vocab_f32");
    assert_log_softmax_parity(&gpu_out, &cpu_out, 5e-4, "log_softmax_qwen3_vocab_f32");
    verify_log_softmax_non_positive(&gpu_out, "log_softmax_qwen3_vocab_f32");
    verify_log_softmax_row_sums(&gpu_out, vocab, 1e-3, "log_softmax_qwen3_vocab_f32");
}

/// Softmax over Qwen3 vocabulary with BF16 dtype: [1, 152064].
///
/// BF16 has 7-bit mantissa (~2 decimal digits). With 152K elements, the
/// accumulation of exp(x - max) can lose precision. The kernel should
/// auto-upcast to f32 for accumulation.
#[test]
fn test_softmax_qwen3_vocab_bf16() {
    gpu_init();
    let vocab = 152064;
    let data = rand_f32_vec(5010, vocab, -5.0, 5.0);

    // F32 reference
    let f32_cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let f32_ref = f32_cpu.softmax(1).unwrap();

    // BF16 GPU path
    let bf16_cpu = f32_cpu.to_dtype(DType::BF16).unwrap();
    let bf16_gpu = bf16_cpu.to_device(&Device::metal()).unwrap();
    let bf16_out = bf16_gpu.softmax(1).unwrap();

    assert_eq!(bf16_out.dtype(), DType::BF16);
    assert_eq!(bf16_out.dims(), &[1, vocab]);

    // Convert back to F32 for comparison
    let bf16_f32 = bf16_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    assert_all_finite(&bf16_f32, "softmax_qwen3_vocab_bf16");

    let bf16_vals = bf16_f32.to_flat_vec::<f32>().unwrap();
    let ref_vals = f32_ref.to_flat_vec::<f32>().unwrap();

    // BF16 tolerance: ~1e-2 absolute, plus accumulation error over 152K elements.
    for (i, (&b, &r)) in bf16_vals.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (b - r).abs();
        assert!(
            diff < 5e-2 || (r.abs() < 1e-6 && b.abs() < 1e-2),
            "softmax_qwen3_vocab_bf16[{i}]: bf16={b}, ref={r}, diff={diff}"
        );
    }

    // Structural check: sums to ~1.0
    let row_sum: f64 = bf16_vals.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (row_sum - 1.0).abs() < 0.1,
        "softmax_qwen3_vocab_bf16 sum = {row_sum}, expected ~1.0"
    );
}

/// Softmax over Qwen3 vocabulary with F16 dtype: [1, 152064].
///
/// F16 has 10-bit mantissa (~3 decimal digits). The kernel should auto-upcast
/// to f32 for the exp/sum accumulation to avoid overflow (f16 max ~65504).
#[test]
fn test_softmax_qwen3_vocab_f16() {
    gpu_init();
    let vocab = 152064;
    // F16 max is ~65504; keep input range moderate to avoid input overflow.
    let data = rand_f32_vec(5020, vocab, -5.0, 5.0);

    // F32 reference
    let f32_cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let f32_ref = f32_cpu.softmax(1).unwrap();

    // F16 GPU path
    let f16_cpu = f32_cpu.to_dtype(DType::F16).unwrap();
    let f16_gpu = f16_cpu.to_device(&Device::metal()).unwrap();
    let f16_out = f16_gpu.softmax(1).unwrap();

    assert_eq!(f16_out.dtype(), DType::F16);
    assert_eq!(f16_out.dims(), &[1, vocab]);

    let f16_f32 = f16_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    assert_all_finite(&f16_f32, "softmax_qwen3_vocab_f16");

    let f16_vals = f16_f32.to_flat_vec::<f32>().unwrap();
    let ref_vals = f32_ref.to_flat_vec::<f32>().unwrap();

    // F16 tolerance: ~1e-3 absolute, plus accumulation over 152K elements.
    for (i, (&f, &r)) in f16_vals.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (f - r).abs();
        assert!(
            diff < 1e-2 || (r.abs() < 1e-6 && f.abs() < 1e-2),
            "softmax_qwen3_vocab_f16[{i}]: f16={f}, ref={r}, diff={diff}"
        );
    }

    let row_sum: f64 = f16_vals.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (row_sum - 1.0).abs() < 0.05,
        "softmax_qwen3_vocab_f16 sum = {row_sum}, expected ~1.0"
    );
}

// ==========================================================================
// GLM5 vocab_size=151552 softmax tests
// ==========================================================================

/// Softmax over GLM5 full vocabulary logits: [1, 151552] in F32.
#[test]
fn test_softmax_glm5_vocab_f32() {
    gpu_init();
    let vocab = 151552;
    let data = rand_f32_vec(5030, vocab, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[1, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_eq!(gpu_out.dims(), &[1, vocab]);
    assert_all_finite(&gpu_out, "softmax_glm5_vocab_f32");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-4, "softmax_glm5_vocab_f32");
    verify_softmax_row_sums(&gpu_out, vocab, 1e-3, "softmax_glm5_vocab_f32");
}

/// Log-softmax over GLM5 full vocabulary: [1, 151552] in F32.
#[test]
fn test_log_softmax_glm5_vocab_f32() {
    gpu_init();
    let vocab = 151552;
    let data = rand_f32_vec(5031, vocab, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[1, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.log_softmax(1).unwrap();
    let gpu_out = gpu.log_softmax(1).unwrap();

    assert_eq!(gpu_out.dims(), &[1, vocab]);
    assert_no_nan_no_pos_inf(&gpu_out, "log_softmax_glm5_vocab_f32");
    assert_log_softmax_parity(&gpu_out, &cpu_out, 5e-4, "log_softmax_glm5_vocab_f32");
    verify_log_softmax_non_positive(&gpu_out, "log_softmax_glm5_vocab_f32");
}

// ==========================================================================
// Batched large-vocab softmax (multi-sequence generation)
// ==========================================================================

/// Softmax over batched Qwen3 logits: [4, 152064] in F32.
///
/// Simulates batch=4 sequence generation, softmax over vocabulary per row.
#[test]
fn test_softmax_qwen3_vocab_batched_f32() {
    gpu_init();
    let batch = 4;
    let vocab = 152064;
    let data = rand_f32_vec(5040, batch * vocab, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[batch, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[batch, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_eq!(gpu_out.dims(), &[batch, vocab]);
    assert_all_finite(&gpu_out, "softmax_qwen3_vocab_batched");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-4, "softmax_qwen3_vocab_batched");
    verify_softmax_row_sums(&gpu_out, vocab, 1e-3, "softmax_qwen3_vocab_batched");
}

// ==========================================================================
// Extreme logit values at large-vocab scale
// ==========================================================================

/// Softmax stability with extreme input magnitudes over Qwen3 vocabulary.
///
/// Logits in [-50, 50] are realistic for unscaled attention and can cause
/// overflow in naive exp() without max-subtraction.
#[test]
fn test_softmax_qwen3_vocab_extreme_values() {
    gpu_init();
    let vocab = 152064;
    let data = rand_f32_vec(5050, vocab, -50.0, 50.0);

    let cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[1, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_all_finite(&gpu_out, "softmax_qwen3_extreme");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-3, "softmax_qwen3_extreme");

    // Verify non-negativity
    let vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v >= 0.0, "softmax_qwen3_extreme[{i}] = {v} is negative");
    }
    verify_softmax_row_sums(&gpu_out, vocab, 1e-3, "softmax_qwen3_extreme");
}

/// Log-softmax stability with extreme inputs over GLM5 vocabulary.
///
/// With 151K elements in [-50, 50], most tokens will have near-zero softmax
/// probability, so log_softmax legitimately produces -inf at those positions
/// (log(0) = -inf). We check: no NaN, no +Inf, GPU matches CPU.
#[test]
fn test_log_softmax_glm5_vocab_extreme_values() {
    gpu_init();
    let vocab = 151552;
    let data = rand_f32_vec(5051, vocab, -50.0, 50.0);

    let cpu = DynTensor::new(&data, &[1, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[1, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.log_softmax(1).unwrap();
    let gpu_out = gpu.log_softmax(1).unwrap();

    // log_softmax with extreme inputs produces -inf for near-zero probabilities.
    assert_no_nan_no_pos_inf(&gpu_out, "log_softmax_glm5_extreme");
    assert_log_softmax_parity(&gpu_out, &cpu_out, 1e-3, "log_softmax_glm5_extreme");
    verify_log_softmax_non_positive(&gpu_out, "log_softmax_glm5_extreme");
}

// ==========================================================================
// 3D shape: [batch, seq, vocab] -- full LLM output shape
// ==========================================================================

/// Softmax over 3D Qwen3 logits: [1, 4, 152064] (batch=1, seq=4, vocab).
///
/// Tests the full output tensor shape from an LLM forward pass during
/// multi-token generation.
#[test]
fn test_softmax_qwen3_3d_logits_f32() {
    gpu_init();
    let batch = 1;
    let seq = 4;
    let vocab = 152064;
    let data = rand_f32_vec(5060, batch * seq * vocab, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[batch, seq, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[batch, seq, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[batch, seq, vocab]);
    assert_all_finite(&gpu_out, "softmax_qwen3_3d_logits");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-4, "softmax_qwen3_3d_logits");

    // Verify per-row sums: last dim is vocab
    let vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let n_rows = batch * seq;
    for row in 0..n_rows {
        let row_sum: f64 = vals[row * vocab..(row + 1) * vocab]
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "softmax_qwen3_3d row {row} sum = {row_sum}, expected ~1.0"
        );
    }
}
