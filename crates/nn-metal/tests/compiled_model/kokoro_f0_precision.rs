// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! F0EnergyPredictor GPU precision investigation (#2449).
//!
//! Tests whether `PrecisionTier::Strict` (Kahan-compensated reductions)
//! reduces the ~5e-3 max diff observed between GPU compiled execution
//! and CPU eager execution in the F0EnergyPredictor.
//!
//! The F0EnergyPredictor has 12 InstanceNorm calls (6 AdainResBlk1d × 2 AdaIN
//! each), plus a BiLSTM. InstanceNorm decomposes to ReduceMean → center →
//! ReduceMean(sq) → rsqrt → normalize. Parallel GPU reduction accumulates
//! differently than sequential CPU, and normalization amplifies the difference.
//!
//! Part of #2449.
//! Part of #2218.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_dsl::ir::ScalarType;
use nn_dsl::{PrecisionContract, PrecisionTier};
use nn_metal::compiled_model::CompiledModel;
use nn_models::kokoro_error::KokoroError;

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const BILSTM_HIDDEN: usize = D_EN / 2;

fn z(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 0.01, DType::F32, &cpu()).unwrap(),
    );
}

fn f0_energy_predictor_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let d = D_EN;
    let bh = BILSTM_HIDDEN;
    let bo = 2 * bh;
    let s = STYLE_DIM;

    // Shared BiLSTM — input_size is d_model + style_dim (style included from DurationEncoder)
    let bilstm_in = d + s;
    z(&mut m, "shared.forward.weight_ih_l0", &[4 * bh, bilstm_in]);
    z(&mut m, "shared.forward.weight_hh_l0", &[4 * bh, bh]);
    z(&mut m, "shared.forward.bias_ih_l0", &[4 * bh]);
    z(&mut m, "shared.forward.bias_hh_l0", &[4 * bh]);
    z(&mut m, "shared.backward.weight_ih_l0", &[4 * bh, bilstm_in]);
    z(&mut m, "shared.backward.weight_hh_l0", &[4 * bh, bh]);
    z(&mut m, "shared.backward.bias_ih_l0", &[4 * bh]);
    z(&mut m, "shared.backward.bias_hh_l0", &[4 * bh]);

    let adain_blk = |m: &mut HashMap<String, DynTensor>,
                     pfx: &str,
                     dim_in: usize,
                     dim_out: usize,
                     upsample: bool| {
        z(m, &format!("{pfx}.n1.fc.weight"), &[2 * dim_in, s]);
        z(m, &format!("{pfx}.n1.fc.bias"), &[2 * dim_in]);
        z(m, &format!("{pfx}.n2.fc.weight"), &[2 * dim_out, s]);
        z(m, &format!("{pfx}.n2.fc.bias"), &[2 * dim_out]);
        z(m, &format!("{pfx}.c1.weight"), &[dim_out, dim_in, 3]);
        z(m, &format!("{pfx}.c1.bias"), &[dim_out]);
        z(m, &format!("{pfx}.c2.weight"), &[dim_out, dim_out, 3]);
        z(m, &format!("{pfx}.c2.bias"), &[dim_out]);
        if dim_in != dim_out {
            z(m, &format!("{pfx}.skip.weight"), &[dim_out, dim_in, 1]);
            z(m, &format!("{pfx}.skip.bias"), &[dim_out]);
        }
        if upsample {
            z(m, &format!("{pfx}.pool.weight"), &[dim_in, 1, 3]);
            z(m, &format!("{pfx}.pool.bias"), &[dim_in]);
        }
    };

    adain_blk(&mut m, "F0.0", bo, bo, false);
    adain_blk(&mut m, "F0.1", bo, bh, true);
    adain_blk(&mut m, "F0.2", bh, bh, false);
    z(&mut m, "F0_proj.weight", &[1, bh]);
    z(&mut m, "F0_proj.bias", &[1]);

    adain_blk(&mut m, "N.0", bo, bo, false);
    adain_blk(&mut m, "N.1", bo, bh, true);
    adain_blk(&mut m, "N.2", bh, bh, false);
    z(&mut m, "N_proj.weight", &[1, bh]);
    z(&mut m, "N_proj.bias", &[1]);

    m
}

/// Execute compiled F0EnergyPredictor with the given precision tier and return
/// (max_diff, mean_diff, gpu_values).
fn run_f0_with_precision(tier: PrecisionTier, t_mel: usize) -> (f32, f32, Vec<f32>) {
    use nn_models::kokoro_f0::F0EnergyPredictor;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weights = f0_energy_predictor_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let f0_pred =
        F0EnergyPredictor::load(&vb, D_EN, STYLE_DIM, BILSTM_HIDDEN).expect("F0EnergyPredictor");

    let batch = 1;
    let bilstm_in = D_EN + STYLE_DIM;
    let aligned_data = super::test_utils::rand_f32_vec(70, batch * bilstm_in * t_mel, -0.5, 0.5);
    let aligned = DynTensor::new(&aligned_data, &[batch, bilstm_in, t_mel], &cpu()).unwrap();
    let style_data = super::test_utils::rand_f32_vec(71, batch * STYLE_DIM, -0.5, 0.5);
    let style = DynTensor::new(&style_data, &[batch, STYLE_DIM], &cpu()).unwrap();

    // CPU reference
    let (ref_f0, _) = f0_pred.forward(&aligned, &style).expect("eager forward");
    let ref_vals = ref_f0.to_flat_vec::<f32>().unwrap();

    // Trace
    let (out, mut graph) = trace_graph(|| {
        let mut inp = aligned.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (f0_out, _) = f0_pred
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(f0_out)
    })
    .expect("trace_graph");
    if let Some(id) = out.trace_id() {
        assert!(graph.set_primary_output(id), "trace_id not in graph");
    }

    // Compile with specified precision
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let contract = PrecisionContract::bootstrap(tier, ScalarType::F32);
    let compiled = CompiledModel::from_plan(&plan, &graph, &cache)
        .expect("from_plan")
        .with_precision(contract);

    eprintln!(
        "F0 [{tier:?}] compiled: {} steps, {} dispatches",
        compiled.num_steps(),
        compiled.num_dispatches()
    );

    // Execute
    let aligned_gpu = aligned.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    let result = compiled
        .execute_dyn(&cache, &[&aligned_gpu, &style_gpu])
        .expect("execute_dyn");
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let (max_diff, mean_diff) = compute_diff_stats(&gpu_vals, &ref_vals);
    eprintln!(
        "F0 [{tier:?}] t_mel={t_mel}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, \
         n_elements={}",
        gpu_vals.len()
    );
    (max_diff, mean_diff, gpu_vals)
}

/// Compute max and mean absolute difference between two float slices.
fn compute_diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "output length mismatch");
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    (max_diff, sum_diff / a.len() as f32)
}

/// #2449 AC2: Test whether PrecisionTier::Strict reduces F0EnergyPredictor drift.
///
/// **Finding:** Strict (Kahan-compensated reductions) has ZERO effect — the
/// max diff is identical to Normal. This conclusively rules out ReduceMean
/// accumulation order as the drift source. The drift originates from:
///
/// 1. **BiLSTM NativeOp**: The fused GPU LSTM kernel uses GPU matmul + GPU
///    sigmoid/tanh, while CPU reference uses sequential f32 math. Gate
///    computations involve 4*H multiplied-accumulated values per timestep.
/// 2. **InstanceNorm amplification**: 12 InstanceNorm calls (6 blocks × 2
///    AdaIN) amplify upstream drift via normalization (division by std dev).
/// 3. **rsqrt vs sqrt+recip**: Compiled path uses GPU `rsqrt()` (hardware),
///    CPU reference uses `sqrt()` then `recip()`.
///
/// Run with: `cargo test -p nn-metal --test compiled_model_kokoro_f0_precision
///            -- --nocapture`
#[test]
fn test_f0_strict_vs_normal_precision() {
    let t_mel = 16;

    let (normal_max, normal_mean, _) = run_f0_with_precision(PrecisionTier::Normal, t_mel);
    let (strict_max, strict_mean, _) = run_f0_with_precision(PrecisionTier::Strict, t_mel);

    eprintln!("\n--- #2449 Precision Comparison (t_mel={t_mel}) ---");
    eprintln!("Normal: max={normal_max:.2e}, mean={normal_mean:.2e}");
    eprintln!("Strict: max={strict_max:.2e}, mean={strict_mean:.2e}");

    if strict_max < normal_max {
        let improvement = 1.0 - (strict_max / normal_max);
        eprintln!("Strict improves max diff by {:.0}%", improvement * 100.0);
    } else {
        eprintln!(
            "Strict did NOT improve (ratio: {:.2}x)",
            strict_max / normal_max
        );
    }

    // Kahan summation does NOT reduce drift (confirmed empirically).
    // The drift comes from BiLSTM GPU matmul + InstanceNorm amplification,
    // not from reduction accumulation order.
    if strict_max < 1e-3 {
        eprintln!("Strict achieves <1e-3 — reduction order dominates drift");
    } else {
        eprintln!(
            "Strict does NOT achieve <1e-3 — non-reduction ops contribute: \
             BiLSTM fused GPU kernel, GPU rsqrt, InstanceNorm amplification"
        );
    }

    // Tolerance: 1e-2 for this model with 12 chained InstanceNorm + BiLSTM.
    // The ~7e-3 max diff is expected given:
    //   - BiLSTM GPU vs CPU arithmetic (NativeOp fused kernel)
    //   - 12 normalization passes amplifying upstream drift
    //   - GPU rsqrt vs CPU sqrt+recip
    assert!(
        normal_max < 1e-2,
        "Normal max diff {normal_max:.2e} exceeds 1e-2 at t_mel={t_mel}"
    );
    assert!(
        strict_max < 1e-2,
        "Strict max diff {strict_max:.2e} exceeds 1e-2"
    );
}

/// #2449 AC1: Spatial dimension effect on drift.
///
/// Tests F0EnergyPredictor at t_mel=4 (original, high drift), t_mel=16 (current),
/// and t_mel=64 (large spatial) to quantify how spatial dimension size affects
/// InstanceNorm reduction precision.
///
/// With more elements, GPU parallel reduction has more accurate partial sums,
/// so drift should decrease with larger t_mel.
#[test]
fn test_f0_spatial_dimension_effect() {
    let tiers = [4, 16, 64];
    let mut results = Vec::new();

    for &t_mel in &tiers {
        let (max_diff, mean_diff, _) = run_f0_with_precision(PrecisionTier::Normal, t_mel);
        results.push((t_mel, max_diff, mean_diff));
    }

    eprintln!("\n--- #2449 Spatial Dimension Effect ---");
    for &(t, max_d, mean_d) in &results {
        eprintln!("t_mel={t:3}: max={max_d:.2e}, mean={mean_d:.2e}");
    }

    // t_mel=16 should be tighter than t_mel=4
    // (more elements = more accurate parallel reduction)
    let diff_4 = results[0].1;
    let diff_16 = results[1].1;
    eprintln!(
        "t_mel=4 vs 16 improvement: {:.0}%",
        (1.0 - diff_16 / diff_4) * 100.0
    );

    // All tiers must stay within 1e-2 (same tolerance as precision test).
    for &(t, max_d, _) in &results {
        assert!(max_d < 1e-2, "t_mel={t}: max_diff {max_d:.2e} exceeds 1e-2");
    }
}
