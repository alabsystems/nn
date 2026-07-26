// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounds verification for 3D LSTM NativeOp execution.
//!
//! LSTM hidden output: `h = sigmoid(o) * tanh(c_new)`.
//! Analytical bound: every element of h is in [-1, 1] because
//! `|sigmoid(o)| <= 1` and `|tanh(c_new)| <= 1`.
//!
//! The 3D LSTM path (`forward_seq` with `[seq_len, batch, input_size]`)
//! compiles to `NativeOpKind::LstmSequence` which delegates to the fused
//! `gpu_lstm_sequence` Metal kernel, bypassing the IR decomposition path.
//! This test verifies the analytical bound holds on actual GPU output,
//! covering the NativeOp execution path that has no NY coverage.
//!
//! Part of #2218 (Kokoro epic).
//! Re: #2427 (LSTM NativeOp bounds gap).

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Lstm;
use nn_core::{DType, Device};
use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
use nn_dsl::NativeOpKind;
use nn_metal::compiled_model::CompiledModel;

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

/// Core test helper: trace LSTM forward_seq, compile, execute on GPU,
/// verify NativeOp is used AND output satisfies analytical bounds.
///
/// Returns (gpu_output_values, cpu_reference_values) for additional checks.
fn trace_compile_execute_lstm_bounds(
    lstm: &Lstm,
    input: &DynTensor,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    // Eager CPU reference.
    let (ref_output, _state) = lstm.forward_seq(input, None).unwrap();
    let ref_vals = ref_output.to_flat_vec::<f32>().unwrap();

    // Trace.
    let (traced_out, mut graph) = trace_graph(|| {
        let mut inp = input.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let (output, _state) = lstm.forward_seq(&inp, None)?;
        Ok::<_, nn_core::TensorError>(output)
    })
    .expect("trace_graph");

    if let Some(id) = traced_out.trace_id() {
        assert!(graph.set_primary_output(id), "{label}: output not in graph");
    }

    // Compile and verify NativeOp is present.
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let has_native_lstm = plan.steps.iter().any(|s| {
        matches!(
            s,
            nn_dsl::trace_compile::CompiledStep::NativeOp {
                op: NativeOpKind::LstmSequence { .. },
                ..
            }
        )
    });
    assert!(
        has_native_lstm,
        "{label}: 3D LSTM should compile to NativeOp::LstmSequence"
    );

    // Execute on GPU.
    let cache = super::test_utils::metal_setup();
    let compiled = CompiledModel::from_plan(&plan, &graph, &cache).expect("from_plan");
    let input_gpu = input.to_device(&gpu()).unwrap();
    let result = compiled
        .execute_dyn(&cache, &[&input_gpu])
        .expect("execute_dyn");

    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Verify analytical bound: every element in [-1, 1].
    for (i, &val) in gpu_vals.iter().enumerate() {
        assert!(
            val.is_finite(),
            "{label}[{i}]: GPU output is non-finite: {val}"
        );
        assert!(
            (-1.0..=1.0).contains(&val),
            "{label}[{i}]: GPU output {val} violates LSTM analytical bound [-1, 1]"
        );
    }

    // Verify GPU/CPU parity.
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "{label}: length mismatch gpu={} ref={}",
        gpu_vals.len(),
        ref_vals.len()
    );
    let mut max_diff: f32 = 0.0;
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (g - r).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "{label}[{i}]: gpu={g}, ref={r}, diff={diff} exceeds tolerance"
        );
    }
    eprintln!("{label}: max GPU/CPU diff = {max_diff:.2e}, all elements in [-1, 1]");

    (gpu_vals, ref_vals)
}

/// Small LSTM with moderate weights — verifies NativeOp compilation and
/// analytical bound h in [-1, 1].
#[test]
fn test_lstm_nativeop_bounds_small_weights() {
    super::test_utils::gpu_init();

    let hidden = 4;
    let input_size = 3;
    let seq_len = 5;
    let batch = 2;

    // Small weights (0.1) — output should be well within [-1, 1].
    let w_ih = DynTensor::new(
        &vec![0.1f32; 4 * hidden * input_size],
        &[4 * hidden, input_size],
        &cpu(),
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &vec![0.1f32; 4 * hidden * hidden],
        &[4 * hidden, hidden],
        &cpu(),
    )
    .unwrap();
    let b_ih = DynTensor::new(&vec![0.0f32; 4 * hidden], &[4 * hidden], &cpu()).unwrap();
    let b_hh = DynTensor::new(&vec![0.0f32; 4 * hidden], &[4 * hidden], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    let input = DynTensor::new(
        &(0..(seq_len * batch * input_size))
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect::<Vec<_>>(),
        &[seq_len, batch, input_size],
        &cpu(),
    )
    .unwrap();

    let (gpu_vals, _) = trace_compile_execute_lstm_bounds(&lstm, &input, "small_weights");

    // With small weights, outputs should be well within bounds.
    let max_abs = gpu_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!("small_weights: max |output| = {max_abs:.4}");
    assert!(max_abs < 1.0, "small weights should produce |h| < 1.0");
}

/// Large weights that push LSTM outputs toward saturation.
/// Verifies the analytical bound still holds even near the extremes.
#[test]
fn test_lstm_nativeop_bounds_large_weights() {
    super::test_utils::gpu_init();

    let hidden = 4;
    let input_size = 3;
    let seq_len = 8;
    let batch = 1;

    // Larger weights (2.0) — outputs should approach [-1, 1] boundaries.
    let w_ih_data: Vec<f32> = (0..(4 * hidden * input_size))
        .map(|i| if i % 3 == 0 { 2.0 } else { -1.5 })
        .collect();
    let w_hh_data: Vec<f32> = (0..(4 * hidden * hidden))
        .map(|i| if i % 2 == 0 { 1.5 } else { -1.0 })
        .collect();
    let b_ih_data: Vec<f32> = (0..(4 * hidden))
        .map(|i| if i < 2 * hidden { 1.0 } else { -1.0 })
        .collect();
    let b_hh_data = vec![0.0f32; 4 * hidden];

    let w_ih = DynTensor::new(&w_ih_data, &[4 * hidden, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[4 * hidden, hidden], &cpu()).unwrap();
    let b_ih = DynTensor::new(&b_ih_data, &[4 * hidden], &cpu()).unwrap();
    let b_hh = DynTensor::new(&b_hh_data, &[4 * hidden], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    // Input with larger values to push gates toward saturation.
    let input = DynTensor::new(
        &(0..(seq_len * batch * input_size))
            .map(|i| (i as f32) * 0.5 - 3.0)
            .collect::<Vec<_>>(),
        &[seq_len, batch, input_size],
        &cpu(),
    )
    .unwrap();

    let (gpu_vals, _) = trace_compile_execute_lstm_bounds(&lstm, &input, "large_weights");

    // With large weights + many timesteps, some outputs should be close to
    // [-1, 1] but never exceed them.
    let max_abs = gpu_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!("large_weights: max |output| = {max_abs:.4}");
    // The bound should hold strictly — that's the whole point.
    assert!(
        max_abs <= 1.0,
        "LSTM analytical bound violated: max |h| = {max_abs}"
    );
}

/// LSTM with no bias — verifies the NativeOp path handles the bias-free case.
#[test]
fn test_lstm_nativeop_bounds_no_bias() {
    super::test_utils::gpu_init();

    let hidden = 3;
    let input_size = 2;
    let seq_len = 4;
    let batch = 2;

    let w_ih = DynTensor::new(
        &vec![0.3f32; 4 * hidden * input_size],
        &[4 * hidden, input_size],
        &cpu(),
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &vec![0.2f32; 4 * hidden * hidden],
        &[4 * hidden, hidden],
        &cpu(),
    )
    .unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden).unwrap();

    let input = DynTensor::new(
        &(0..(seq_len * batch * input_size))
            .map(|i| (i as f32) * 0.2 - 1.0)
            .collect::<Vec<_>>(),
        &[seq_len, batch, input_size],
        &cpu(),
    )
    .unwrap();

    trace_compile_execute_lstm_bounds(&lstm, &input, "no_bias");
}

/// Kokoro-scale LSTM dimensions (hidden=512, input=512, seq=20).
///
/// This matches the dimensions used in Kokoro's text pipeline
/// (`BiLstm` with hidden=256 per direction → 512 concat).
/// Verifies the NativeOp bound at production scale.
#[test]
fn test_lstm_nativeop_bounds_kokoro_scale() {
    super::test_utils::gpu_init();

    let hidden = 64; // Smaller than Kokoro (512) for test speed, still exercises path.
    let input_size = 64;
    let seq_len = 20;
    let batch = 1;

    // Varied weights matching Kokoro-like initialization.
    let w_ih_data: Vec<f32> = (0..(4 * hidden * input_size))
        .map(|i| {
            let x = (i as f32) * 0.001;
            (x * 7.3).sin() * 0.15
        })
        .collect();
    let w_hh_data: Vec<f32> = (0..(4 * hidden * hidden))
        .map(|i| {
            let x = (i as f32) * 0.001;
            (x * 13.7).cos() * 0.1
        })
        .collect();
    let b_ih = DynTensor::new(&vec![0.0f32; 4 * hidden], &[4 * hidden], &cpu()).unwrap();
    let b_hh = DynTensor::new(&vec![0.0f32; 4 * hidden], &[4 * hidden], &cpu()).unwrap();

    let w_ih = DynTensor::new(&w_ih_data, &[4 * hidden, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[4 * hidden, hidden], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    // Varied input data.
    let input_data: Vec<f32> = (0..(seq_len * batch * input_size))
        .map(|i| {
            let x = (i as f32) * 0.01;
            (x * 3.1).sin() * 0.5
        })
        .collect();
    let input = DynTensor::new(&input_data, &[seq_len, batch, input_size], &cpu()).unwrap();

    let (gpu_vals, _) = trace_compile_execute_lstm_bounds(&lstm, &input, "kokoro_scale");
    let max_abs = gpu_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "kokoro_scale: {} output elements, max |h| = {max_abs:.6}",
        gpu_vals.len()
    );
}

// -- NarrowView after LSTM in mixed-precision (#3116) -------------------------

/// Regression test: NarrowView after LSTM in F16 mixed-precision mode must use
/// the runtime buffer dtype (F16 after LSTM output cast) for byte offset
/// computation, not the static step_scalar_types (F32).
///
/// Before fix: NarrowView used `step_scalar_types[step_idx]` which for the
/// NarrowView step was F16 (uniform override). But if the source step was
/// LSTM (stays F32 statically), its output gets cast F32→F16 at runtime by
/// `execute_native_op_mixed`. The NarrowView needs to use the runtime dtype
/// of the *source* step to compute the byte offset.
///
/// Graph: input [4, 2, 3] → LSTM(hidden=4) → [4, 2, 4] → narrow(dim=0, start=1, len=2) → [2, 2, 4].
#[test]
fn test_f16_narrowview_after_lstm() {
    super::test_utils::gpu_init();

    let hidden = 4;
    let input_size = 3;
    let seq_len = 4;
    let batch = 2;

    let w_ih = DynTensor::new(
        &vec![0.1f32; 4 * hidden * input_size],
        &[4 * hidden, input_size],
        &cpu(),
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &vec![0.05f32; 4 * hidden * hidden],
        &[4 * hidden, hidden],
        &cpu(),
    )
    .unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden).unwrap();

    let input_data: Vec<f32> = (0..(seq_len * batch * input_size))
        .map(|i| ((i as f32) * 0.1).sin() * 0.5)
        .collect();
    let input = DynTensor::new(&input_data, &[seq_len, batch, input_size], &cpu()).unwrap();

    // Eager CPU reference: LSTM → narrow.
    let (ref_output, _) = lstm.forward_seq(&input, None).unwrap();
    let ref_narrow = ref_output.narrow(0, 1, 2).expect("narrow");
    let ref_vals = ref_narrow.to_flat_vec::<f32>().unwrap();

    // Trace: LSTM → narrow.
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    let (traced_out, mut graph) = trace_graph(|| {
        let mut inp = input.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let (output, _) = lstm.forward_seq(&inp, None)?;
        let narrow = output.narrow(0, 1, 2)?;
        Ok::<_, nn_core::TensorError>(narrow)
    })
    .expect("trace_graph");

    if let Some(id) = traced_out.trace_id() {
        assert!(graph.set_primary_output(id), "output not in graph");
    }

    // F32 compiled reference.
    let cache = super::test_utils::metal_setup();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let f32_model = CompiledModel::from_plan(&plan, &graph, &cache).expect("from_plan f32");

    let input_gpu = input.to_device(&gpu()).unwrap();
    let f32_result = f32_model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("execute f32")
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // F16 mixed-precision: LSTM stays F32 internally, output cast to F16,
    // NarrowView must use F16 byte offset (2 bytes/elem, not 4).
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    let f16_result = f16_model
        .execute_dyn(&cache, &[&input_gpu])
        .expect("execute f16")
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // All three should agree within tolerance.
    assert_eq!(ref_vals.len(), f32_result.len(), "length mismatch f32");
    assert_eq!(ref_vals.len(), f16_result.len(), "length mismatch f16");
    let expected_len = 2 * batch * hidden; // narrow [2, 2, 4]
    assert_eq!(ref_vals.len(), expected_len, "unexpected output length");

    for (i, ((&r, &g32), &g16)) in ref_vals
        .iter()
        .zip(f32_result.iter())
        .zip(f16_result.iter())
        .enumerate()
    {
        let diff_f32 = (r - g32).abs();
        assert!(
            diff_f32 < 1e-3,
            "f32[{i}]: ref={r}, gpu={g32}, diff={diff_f32}"
        );
        let diff_f16 = (r - g16).abs();
        assert!(
            diff_f16 < 0.05,
            "f16[{i}]: ref={r}, gpu={g16}, diff={diff_f16} (exceeds F16 tolerance)"
        );
    }
    eprintln!(
        "f16_narrowview_after_lstm: {} elements, all within tolerance",
        ref_vals.len()
    );
}

// -- Precomputed GEMM path validation (#3491) ---------------------------------

/// Exercises the precomputed LSTM GEMM path restored in #3491.
///
/// The precomputed path fires when ALL alignment conditions are met:
/// - `weight_ih_t` present (always true from trace compiler)
/// - `seq_len * batch % 8 == 0`
/// - `input_size % 8 == 0`
/// - `4 * hidden_size % 8 == 0`
///
/// This test uses dimensions that satisfy all conditions:
/// input_size=64, hidden_size=32, seq_len=8, batch=1 → m=8 (aligned).
///
/// The test verifies GPU/CPU parity, proving the precomputed matmul +
/// recurrence kernel produces the same result as the CPU LSTM reference.
#[test]
fn test_lstm_precomputed_gemm_aligned_dimensions() {
    super::test_utils::gpu_init();

    let hidden = 32;
    let input_size = 64;
    let seq_len = 8;
    let batch = 1;
    let four_h = 4 * hidden;

    // Verify alignment conditions (m % 8, input_size % 8, n % 8).
    let m = seq_len * batch;
    let n = four_h;
    assert_eq!(m % 8, 0, "m must be 8-aligned for precomputed path");
    assert_eq!(input_size % 8, 0, "input_size must be 8-aligned");
    assert_eq!(n % 8, 0, "n (4*hidden) must be 8-aligned");

    // Deterministic weights with varied values (not uniform).
    let w_ih_data: Vec<f32> = (0..(four_h * input_size))
        .map(|i| ((i as f32) * 0.0031).sin() * 0.12)
        .collect();
    let w_hh_data: Vec<f32> = (0..(four_h * hidden))
        .map(|i| ((i as f32) * 0.0047).sin() * 0.08)
        .collect();
    let b_ih_data: Vec<f32> = (0..four_h)
        .map(|i| ((i as f32) * 0.13).sin() * 0.05)
        .collect();
    let b_hh_data: Vec<f32> = (0..four_h)
        .map(|i| ((i as f32) * 0.17).cos() * 0.03)
        .collect();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden], &cpu()).unwrap();
    let b_ih = DynTensor::new(&b_ih_data, &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&b_hh_data, &[four_h], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    let input_data: Vec<f32> = (0..(seq_len * batch * input_size))
        .map(|i| ((i as f32) * 0.007).sin() * 0.3)
        .collect();
    let input = DynTensor::new(&input_data, &[seq_len, batch, input_size], &cpu()).unwrap();

    let (gpu_vals, ref_vals) = trace_compile_execute_lstm_bounds(&lstm, &input, "precomputed_gemm");

    // The precomputed path should produce results within tight tolerance of CPU.
    let max_diff = gpu_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "precomputed_gemm: {} output elements, max GPU/CPU diff = {max_diff:.2e}",
        gpu_vals.len()
    );
    assert!(
        max_diff < 1e-4,
        "precomputed GEMM path exceeds tolerance: max_diff = {max_diff:.2e}"
    );
}

/// Non-aligned M: seq_len=13, batch=1 → m=13 (not multiple of 8).
///
/// Verifies that the precomputed GEMM path works for arbitrary sequence
/// lengths (not just multiples of 8). The simdgroup matmul kernel handles
/// edge tiles with bounds-checked loads and writes. Production Kokoro has
/// variable seq_len (e.g., 100 phonemes).
#[test]
fn test_lstm_precomputed_gemm_unaligned_m() {
    super::test_utils::gpu_init();

    let hidden = 32;
    let input_size = 64;
    let seq_len = 13; // NOT a multiple of 8
    let batch = 1;
    let four_h = 4 * hidden;

    let m = seq_len * batch;
    assert_ne!(m % 8, 0, "test requires non-aligned m");
    assert_eq!(input_size % 8, 0);
    assert_eq!(four_h % 8, 0);

    let w_ih_data: Vec<f32> = (0..(four_h * input_size))
        .map(|i| ((i as f32) * 0.0031).sin() * 0.12)
        .collect();
    let w_hh_data: Vec<f32> = (0..(four_h * hidden))
        .map(|i| ((i as f32) * 0.0047).sin() * 0.08)
        .collect();
    let b_ih_data: Vec<f32> = (0..four_h)
        .map(|i| ((i as f32) * 0.13).sin() * 0.05)
        .collect();
    let b_hh_data: Vec<f32> = (0..four_h)
        .map(|i| ((i as f32) * 0.17).cos() * 0.03)
        .collect();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden], &cpu()).unwrap();
    let b_ih = DynTensor::new(&b_ih_data, &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&b_hh_data, &[four_h], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    let input_data: Vec<f32> = (0..(seq_len * batch * input_size))
        .map(|i| ((i as f32) * 0.007).sin() * 0.3)
        .collect();
    let input = DynTensor::new(&input_data, &[seq_len, batch, input_size], &cpu()).unwrap();

    let (gpu_vals, ref_vals) =
        trace_compile_execute_lstm_bounds(&lstm, &input, "precomputed_gemm_unaligned");

    let max_diff = gpu_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "precomputed_gemm_unaligned: {} output elements, max GPU/CPU diff = {max_diff:.2e}",
        gpu_vals.len()
    );
    assert!(
        max_diff < 1e-4,
        "precomputed GEMM unaligned M path exceeds tolerance: max_diff = {max_diff:.2e}"
    );
}

/// Production-scale non-aligned: seq_len=100, batch=1, D=512 dimensions.
///
/// Matches Kokoro production TextEncoder: input_size=512, hidden=256.
/// seq_len=100 → m=100 (100%8=4, NOT aligned). Before the M alignment
/// relaxation, this would fall back to the slower fused path.
#[test]
fn test_lstm_precomputed_gemm_production_unaligned() {
    super::test_utils::gpu_init();

    let hidden = 256;
    let input_size = 512;
    let seq_len = 100;
    let batch = 1;
    let four_h = 4 * hidden;

    let m = seq_len * batch;
    assert_ne!(
        m % 8,
        0,
        "test requires non-aligned m for production scenario"
    );

    let w_ih_data: Vec<f32> = (0..(four_h * input_size))
        .map(|i| ((i as f32) * 0.00011).sin() * 0.04)
        .collect();
    let w_hh_data: Vec<f32> = (0..(four_h * hidden))
        .map(|i| ((i as f32) * 0.00017).sin() * 0.03)
        .collect();
    let b_ih = DynTensor::new(&vec![0.01f32; four_h], &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&vec![-0.005f32; four_h], &[four_h], &cpu()).unwrap();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    let input_data: Vec<f32> = (0..(seq_len * batch * input_size))
        .map(|i| ((i as f32) * 0.0003).sin() * 0.15)
        .collect();
    let input = DynTensor::new(&input_data, &[seq_len, batch, input_size], &cpu()).unwrap();

    let (gpu_vals, ref_vals) =
        trace_compile_execute_lstm_bounds(&lstm, &input, "production_unaligned");

    let max_diff = gpu_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "production_unaligned: {} output elements, max GPU/CPU diff = {max_diff:.2e}",
        gpu_vals.len()
    );
    assert!(
        max_diff < 1e-3,
        "production unaligned LSTM exceeds tolerance: max_diff = {max_diff:.2e}"
    );
}

/// Same as above but with batch=2 (m=16) and larger hidden to exercise
/// the precomputed path under multi-batch conditions.
#[test]
fn test_lstm_precomputed_gemm_batch2() {
    super::test_utils::gpu_init();

    let hidden = 64;
    let input_size = 32;
    let seq_len = 8;
    let batch = 2;
    let four_h = 4 * hidden;

    let m = seq_len * batch;
    assert_eq!(m % 8, 0);
    assert_eq!(input_size % 8, 0);
    assert_eq!((four_h) % 8, 0);

    let w_ih_data: Vec<f32> = (0..(four_h * input_size))
        .map(|i| ((i as f32) * 0.0023).sin() * 0.1)
        .collect();
    let w_hh_data: Vec<f32> = (0..(four_h * hidden))
        .map(|i| ((i as f32) * 0.0037).cos() * 0.06)
        .collect();
    let b_ih = DynTensor::new(&vec![0.01f32; four_h], &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&vec![-0.01f32; four_h], &[four_h], &cpu()).unwrap();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    let input_data: Vec<f32> = (0..(seq_len * batch * input_size))
        .map(|i| ((i as f32) * 0.005).cos() * 0.25)
        .collect();
    let input = DynTensor::new(&input_data, &[seq_len, batch, input_size], &cpu()).unwrap();

    let (gpu_vals, ref_vals) =
        trace_compile_execute_lstm_bounds(&lstm, &input, "precomputed_gemm_batch2");

    let max_diff = gpu_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "precomputed_gemm_batch2: {} output elements, max GPU/CPU diff = {max_diff:.2e}",
        gpu_vals.len()
    );
    assert!(
        max_diff < 1e-3,
        "precomputed GEMM batch2 path exceeds tolerance: max_diff = {max_diff:.2e}"
    );
}
