// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU vs CPU equivalence tests for AdainResBlk1d (LeakyRelu) and ResBlock (Snake).
//!
//! Verifies that the compiled GPU path (FusedResBlock NativeOp with fused
//! NormActivConv1d kernel) produces identical output to the CPU path.
//!
//! Originally from #2459 (FusedAdainResBlock). Updated by #2590 to use
//! NativeOps. Peephole fusion creates a single FusedResBlock NativeOp that
//! uses the fused stats+norm+conv kernel for LeakyRelu (#2780).
//! Snake path coverage added by #2980 (Generator ResBlock gap).
//! Part of #2218 (Kokoro epic).

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_f0::AdainResBlk1d;
use nn_models::kokoro_resblock::ResBlock;

/// Miniaturized dimensions for fast execution.
const DIM: usize = 4;
const STYLE_DIM: usize = 4;
const SEQ_LEN: usize = 16;

fn cpu() -> Device {
    Device::Cpu
}

/// Build an AdainResBlk1d with deterministic weights (no upsample, same dim).
fn build_block() -> (AdainResBlk1d, HashMap<String, DynTensor>) {
    let mut m = HashMap::new();

    // Small positive weights for numerical stability.
    let fill = |shape: &[usize]| -> DynTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        DynTensor::from_vec(data, shape, &cpu()).unwrap()
    };

    // AdaIN1 style projection: [2*DIM, STYLE_DIM] -> gamma, beta
    m.insert("n1.fc.weight".into(), fill(&[2 * DIM, STYLE_DIM]));
    m.insert("n1.fc.bias".into(), fill(&[2 * DIM]));
    // AdaIN2 style projection
    m.insert("n2.fc.weight".into(), fill(&[2 * DIM, STYLE_DIM]));
    m.insert("n2.fc.bias".into(), fill(&[2 * DIM]));
    // Conv1d layers (k=3, pad=1)
    m.insert("c1.weight".into(), fill(&[DIM, DIM, 3]));
    m.insert("c1.bias".into(), fill(&[DIM]));
    m.insert("c2.weight".into(), fill(&[DIM, DIM, 3]));
    m.insert("c2.bias".into(), fill(&[DIM]));

    let vb = VarBuilder::from_tensors(m.clone(), DType::F32, &cpu());
    let block = AdainResBlk1d::load(&vb, DIM, DIM, STYLE_DIM, false).expect("load AdainResBlk1d");
    (block, m)
}

/// Verify dispatch count for 3 chained AdainResBlk1d blocks (F0 head pattern).
///
/// Without FusedResBlock peephole: 3 blocks × ~7 = ~21 logical dispatches.
/// With FusedResBlock peephole:    3 FusedResBlock NativeOps = 3 logical, ~24 Metal.
/// This verifies #2780 AC4: F0-style dispatch reduction from peephole fusion.
#[test]
#[allow(deprecated)]
fn test_chained_resblock_dispatch_count() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Build 3 blocks with the same dim (non-upsample, no skip_conv).
    let blocks: Vec<AdainResBlk1d> = (0..3)
        .map(|i| {
            let mut m = HashMap::new();
            let fill = |shape: &[usize]| -> DynTensor {
                let n: usize = shape.iter().product();
                let data: Vec<f32> = (0..n).map(|j| 0.01 * ((j + i) as f32 + 1.0)).collect();
                DynTensor::from_vec(data, shape, &cpu()).unwrap()
            };
            let prefix = format!("b{i}.");
            m.insert(format!("{prefix}n1.fc.weight"), fill(&[2 * DIM, STYLE_DIM]));
            m.insert(format!("{prefix}n1.fc.bias"), fill(&[2 * DIM]));
            m.insert(format!("{prefix}n2.fc.weight"), fill(&[2 * DIM, STYLE_DIM]));
            m.insert(format!("{prefix}n2.fc.bias"), fill(&[2 * DIM]));
            m.insert(format!("{prefix}c1.weight"), fill(&[DIM, DIM, 3]));
            m.insert(format!("{prefix}c1.bias"), fill(&[DIM]));
            m.insert(format!("{prefix}c2.weight"), fill(&[DIM, DIM, 3]));
            m.insert(format!("{prefix}c2.bias"), fill(&[DIM]));
            let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
            AdainResBlk1d::load(vb.pp(format!("b{i}")), DIM, DIM, STYLE_DIM, false)
                .expect("load block")
        })
        .collect();

    let x_data = super::test_utils::rand_f32_vec(42, DIM * SEQ_LEN, -0.5, 0.5);
    let x = DynTensor::from_vec(x_data, &[1, DIM, SEQ_LEN], &cpu()).unwrap();
    let style_data = super::test_utils::rand_f32_vec(99, STYLE_DIM, -0.3, 0.3);
    let style = DynTensor::from_vec(style_data, &[1, STYLE_DIM], &cpu()).unwrap();

    let compiled = CompiledModel::compile_forward(
        &[&x, &style],
        |inputs| {
            let mut h = inputs[0].clone();
            for block in &blocks {
                h = block
                    .forward(&h, &inputs[1])
                    .map_err(KokoroError::into_tensor_error)?;
            }
            Ok(h)
        },
        &cache,
    )
    .expect("compile 3-block chain");

    let native_ops = compiled.num_native_ops();
    let ir_dispatches = compiled.num_ir_dispatches();
    let metal = compiled.num_metal_dispatches();
    eprintln!(
        "3-block chain: {native_ops} NativeOps, {ir_dispatches} IR, {metal} Metal dispatches"
    );

    // Each non-upsample block should fuse into 1 FusedResBlock NativeOp.
    // 3 blocks = 3 FusedResBlock NativeOps + style projection steps.
    assert!(
        native_ops >= 3,
        "Expected >= 3 FusedResBlock NativeOps for 3 chained blocks, got {native_ops}"
    );
    // Total should be much less than unfused (3 * 7 = 21).
    let total = native_ops + ir_dispatches;
    assert!(
        total <= 15,
        "3-block chain should have <=15 total dispatches (3 FusedResBlocks + overhead), got {total}"
    );
}

/// GPU vs CPU equivalence: compiled output matches CPU reference.
///
/// Both paths run decomposed AdaIN → LeakyRelu → Conv1d → ... → residual.
/// The GPU path compiles AdainLeakyRelu as NativeOp (single Metal dispatch
/// each), giving fewer total dispatches than the old FusedAdainResBlock. #2590.
#[test]
#[allow(deprecated)]
fn test_fused_resblock_matches_decomposed() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (block, _weights) = build_block();

    // Deterministic inputs.
    let x_data = super::test_utils::rand_f32_vec(42, DIM * SEQ_LEN, -0.5, 0.5);
    let x = DynTensor::from_vec(x_data, &[1, DIM, SEQ_LEN], &cpu()).unwrap();

    let style_data = super::test_utils::rand_f32_vec(99, STYLE_DIM, -0.3, 0.3);
    let style = DynTensor::from_vec(style_data, &[1, STYLE_DIM], &cpu()).unwrap();

    // CPU reference: decomposed path (tracing inactive, CPU device).
    let cpu_out = block.forward(&x, &style).expect("CPU forward");
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU compiled path: trace -> compile -> execute.
    let compiled = CompiledModel::compile_forward(
        &[&x, &style],
        |inputs| {
            block
                .forward(&inputs[0], &inputs[1])
                .map_err(KokoroError::into_tensor_error)
        },
        &cache,
    )
    .expect("compile_forward");

    // Peephole fusion should produce FusedResBlock NativeOp (LeakyRelu):
    // 2× fused stats+norm+conv dispatches, with residual in phase 2 (#2780).
    let ir_dispatches = compiled.num_ir_dispatches();
    let native_ops = compiled.num_native_ops();
    let metal_dispatches = compiled.num_metal_dispatches();
    eprintln!(
        "AdainResBlk1d: {ir_dispatches} IR, {native_ops} NativeOps, {metal_dispatches} Metal dispatches"
    );

    // Execute on GPU.
    let x_gpu = x.to_device(&Device::metal()).unwrap();
    let style_gpu = style.to_device(&Device::metal()).unwrap();
    let gpu_out = compiled
        .execute_dyn(&cache, &[&x_gpu, &style_gpu])
        .expect("GPU execute");

    let gpu_vals = gpu_out
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // AC3 from #2459: max_diff < 1e-4.
    assert_eq!(cpu_vals.len(), gpu_vals.len(), "output length mismatch");
    let max_diff = cpu_vals
        .iter()
        .zip(gpu_vals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "GPU vs CPU: {} elements, max_diff={max_diff:.6e}",
        cpu_vals.len()
    );
    assert!(
        max_diff < 1e-4,
        "GPU path should match CPU within 1e-4, got max_diff={max_diff:.6e}"
    );
}

// -- Snake ResBlock (Generator) -----------------------------------------------

/// Kernel size for Snake ResBlock (matches Kokoro Generator default).
const SNAKE_KERNEL_SIZE: usize = 3;

/// Build a Snake ResBlock with deterministic weights (single dilation layer).
///
/// Matches `kokoro_resblock::ResBlock::load` weight naming:
/// `convs1.{i}`, `convs2.{i}`, `adain1.{i}`, `adain2.{i}`, `alpha1.{i}`, `alpha2.{i}`.
fn build_snake_resblock() -> ResBlock {
    let mut m = HashMap::new();

    let fill = |shape: &[usize]| -> DynTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        DynTensor::from_vec(data, shape, &cpu()).unwrap()
    };

    // Single dilation layer (i=0, dilation=1).
    // Conv1d pairs: dilated conv1 + non-dilated conv2
    m.insert(
        "convs1.0.weight".into(),
        fill(&[DIM, DIM, SNAKE_KERNEL_SIZE]),
    );
    m.insert("convs1.0.bias".into(), fill(&[DIM]));
    m.insert(
        "convs2.0.weight".into(),
        fill(&[DIM, DIM, SNAKE_KERNEL_SIZE]),
    );
    m.insert("convs2.0.bias".into(), fill(&[DIM]));

    // AdaIN style projections: [2*DIM, STYLE_DIM] -> gamma, beta
    m.insert("adain1.0.fc.weight".into(), fill(&[2 * DIM, STYLE_DIM]));
    m.insert("adain1.0.fc.bias".into(), fill(&[2 * DIM]));
    m.insert("adain2.0.fc.weight".into(), fill(&[2 * DIM, STYLE_DIM]));
    m.insert("adain2.0.fc.bias".into(), fill(&[2 * DIM]));

    // Snake alpha: per-channel learnable parameter [1, C, 1].
    // Small positive values (Snake alpha controls activation frequency).
    let alpha_data: Vec<f32> = (0..DIM).map(|i| 0.5 + 0.1 * i as f32).collect();
    let alpha1 = DynTensor::from_vec(alpha_data.clone(), &[1, DIM, 1], &cpu()).unwrap();
    let alpha2 = DynTensor::from_vec(alpha_data, &[1, DIM, 1], &cpu()).unwrap();
    m.insert("alpha1.0".into(), alpha1);
    m.insert("alpha2.0".into(), alpha2);

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    ResBlock::load(&vb, DIM, SNAKE_KERNEL_SIZE, &[1], STYLE_DIM).expect("load Snake ResBlock")
}

/// GPU vs CPU equivalence for Snake-activated FusedResBlock (Generator path).
///
/// The Kokoro Generator uses Snake activation in its ResBlocks. The peephole
/// pass fuses AdainSnake + Conv1d pairs into a FusedResBlock NativeOp with
/// `NormActivation::Snake`, executing via `native_norm_activ_conv1d_snake`
/// with residual folding in phase 2.
///
/// This is the Snake counterpart to `test_fused_resblock_matches_decomposed`
/// (LeakyRelu). Filed as #2980.
#[test]
#[allow(deprecated)]
fn test_fused_resblock_snake_matches_decomposed() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let block = build_snake_resblock();

    // Deterministic inputs (different seeds from LeakyRelu test).
    let x_data = super::test_utils::rand_f32_vec(77, DIM * SEQ_LEN, -0.5, 0.5);
    let x = DynTensor::from_vec(x_data, &[1, DIM, SEQ_LEN], &cpu()).unwrap();

    let style_data = super::test_utils::rand_f32_vec(88, STYLE_DIM, -0.3, 0.3);
    let style = DynTensor::from_vec(style_data, &[1, STYLE_DIM], &cpu()).unwrap();

    // CPU reference: decomposed path (tracing inactive, CPU device).
    let cpu_out = block.forward(&x, &style).expect("CPU forward");
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU compiled path: trace -> compile -> execute.
    let compiled = CompiledModel::compile_forward(
        &[&x, &style],
        |inputs| {
            block
                .forward(&inputs[0], &inputs[1])
                .map_err(KokoroError::into_tensor_error)
        },
        &cache,
    )
    .expect("compile_forward (Snake ResBlock)");

    // Peephole fusion should produce FusedResBlock NativeOp (Snake):
    // 2× fused stats+norm+snake+conv dispatches per dilation layer,
    // with residual add in phase 2.
    let ir_dispatches = compiled.num_ir_dispatches();
    let native_ops = compiled.num_native_ops();
    let metal_dispatches = compiled.num_metal_dispatches();
    eprintln!(
        "Snake ResBlock: {ir_dispatches} IR, {native_ops} NativeOps, {metal_dispatches} Metal dispatches"
    );

    // Execute on GPU.
    let x_gpu = x.to_device(&Device::metal()).unwrap();
    let style_gpu = style.to_device(&Device::metal()).unwrap();
    let gpu_out = compiled
        .execute_dyn(&cache, &[&x_gpu, &style_gpu])
        .expect("GPU execute (Snake ResBlock)");

    let gpu_vals = gpu_out
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Snake activation uses sin²(αx)/α which is more numerically sensitive
    // than LeakyRelu. Use same 1e-4 tolerance as LeakyRelu test.
    assert_eq!(cpu_vals.len(), gpu_vals.len(), "output length mismatch");
    let max_diff = cpu_vals
        .iter()
        .zip(gpu_vals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "Snake GPU vs CPU: {} elements, max_diff={max_diff:.6e}",
        cpu_vals.len()
    );
    assert!(
        max_diff < 1e-4,
        "Snake GPU path should match CPU within 1e-4, got max_diff={max_diff:.6e}"
    );
}
