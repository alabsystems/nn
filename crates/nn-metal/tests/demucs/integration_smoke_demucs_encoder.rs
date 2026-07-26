// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration smoke test: Demucs encoder block consumption pattern.
//!
//! Exercises the full dvoice integration path:
//! 1. Create synthetic weights matching a Demucs encoder block
//! 2. Load via `WeightMap` (safetensors → zero-copy Metal buffers)
//! 3. Extract weight data from `WeightMap` for GPU dispatch
//! 4. Build `TensorKernelDef` using `TensorBlockBuilder`
//! 5. Dispatch on Metal GPU via `execute_tensor_dispatch()`
//! 6. Verify output within NY IBP bounds
//!
//! # Safety
//!
//! `WeightMap::load` is `unsafe` because it creates an mmap-backed Metal buffer
//! from a file path. Callers must ensure the file is a valid safetensors file,
//! the Metal context is initialized, and the file outlives the WeightMap.
//! All tests create temp files and clean up after assertions.
//!
//! This is the first test that exercises the WeightMap → TensorBlockBuilder →
//! GPU dispatch → bounds verification pipeline end-to-end, which is the
//! actual pattern dvoice will use to consume nn.
//!
//! Part of #710.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::adain::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_dsl::TensorKernelDef;
use nn_metal::{execute_tensor_dispatch, MetalContext, WeightMap};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

// ---------------------------------------------------------------------------
// Safetensors helpers
// ---------------------------------------------------------------------------

/// Create a safetensors file with named tensors of specified shapes.
/// Returns the generated tensor data for round-trip verification.
fn create_encoder_safetensors(
    path: &Path,
    tensors: &[(&str, &[usize])],
    seed_base: u64,
) -> Vec<(String, Vec<f32>)> {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let mut tensor_data: Vec<(String, Vec<f32>)> = Vec::new();
    for (i, &(name, shape)) in tensors.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let seed = seed_base.wrapping_add(i as u64 * 0x1234_5678);
        tensor_data.push((name.to_string(), rand_f32_vec(seed, numel, -0.3, 0.3)));
    }

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .zip(tensor_data.iter())
        .map(|(&(name, shape), (_n, data))| {
            let bytes = bytemuck::cast_slice::<f32, u8>(data);
            let view = TensorView::new(StDtype::F32, shape.to_vec(), bytes).expect("tensor view");
            (name.to_string(), view)
        })
        .collect();

    let serialized = serialize(views, None).expect("serialize safetensors");
    let mut file = std::fs::File::create(path).expect("create file");
    file.write_all(&serialized).expect("write file");
    tensor_data
}

/// Extract f32 data from a WeightMap tensor by name.
fn extract_f32(wm: &WeightMap, name: &str) -> Vec<f32> {
    let bytes = wm.tensor_data(name).expect("tensor data");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_smoke_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

// ---------------------------------------------------------------------------
// Verification + dispatch helpers
// ---------------------------------------------------------------------------

/// Prove IBP bounds and return (proved_lo, proved_hi).
fn prove_ibp_bounds(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    in_shape: &[usize],
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph build");
    let lower_in = ArrayD::from_elem(IxDyn(in_shape), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(in_shape), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (lo, hi) = output_bounds.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    (lo.clone(), hi.clone())
}

/// Dispatch on Metal GPU and verify output within proved bounds.
fn dispatch_and_verify(
    label: &str,
    def: &TensorKernelDef,
    inputs: HashMap<&str, Vec<f32>>,
    expected_len: usize,
    proved_lo: &ArrayD<f32>,
    proved_hi: &ArrayD<f32>,
) {
    let cache = metal_setup();
    let gpu_out =
        execute_tensor_dispatch(&cache, def, ScalarType::F32, &inputs).expect("GPU dispatch");
    assert_eq!(gpu_out.len(), expected_len, "{label}: GPU output length");
    assert_gpu_within_bounds(label, &gpu_out, proved_lo, proved_hi);
}

// ---------------------------------------------------------------------------
// Integration smoke tests
// ---------------------------------------------------------------------------

/// Smoke test: single Demucs encoder block (Conv1d → Snake).
///
/// Models the full dvoice consumption pattern: safetensors file →
/// WeightMap load → extract weights → TensorBlockBuilder → GPU dispatch →
/// NY IBP bounds verification.
///
/// Part of #710.
#[test]
fn smoke_demucs_encoder_block_conv1d_snake() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 48, 8, 64, 4, 2);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    // Step 1: Create safetensors, load via WeightMap, extract weights
    let dir = temp_dir("enc_block");
    let st_path = dir.join("encoder_block.safetensors");
    let conv_shape = [out_ch, in_ch, kernel_size];
    let alpha_shape = [1];
    let specs: Vec<(&str, &[usize])> =
        vec![("conv.weight", &conv_shape), ("snake.alpha", &alpha_shape)];
    let written = create_encoder_safetensors(&st_path, &specs, 0x5E0C_E001);

    let ctx = MetalContext::new().expect("Metal device required");
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&st_path, &ctx).expect("load safetensors") };
    assert_eq!(wm.tensor_count(), 2);

    let conv_weight = extract_f32(&wm, "conv.weight");
    let snake_alpha = extract_f32(&wm, "snake.alpha");
    assert_eq!(conv_weight, written[0].1, "conv weight round-trip");
    assert_eq!(snake_alpha, written[1].1, "alpha round-trip");

    // Step 2: Build TensorKernelDef via TensorBlockBuilder
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let out_shape = [out_ch, out_len];
    let mut b = TensorBlockBuilder::new("smoke_demucs_enc");
    let data_node = b.add_input("data", &[in_ch, in_len]);
    let weight_node = b.add_input("conv.weight", &[out_ch, in_ch, kernel_size]);
    let alpha_node = b.add_input("snake.alpha", &[1]);
    let conv_out = b.add_conv1d(data_node, weight_node, None, stride, padding, &out_shape);
    let alpha_bc = b.add_broadcast(alpha_node, &out_shape);
    let snake_out = b.add_elementwise(snake_kernel, &[conv_out, alpha_bc], &out_shape);
    let def = b.build(snake_out).expect("valid graph");

    // Step 3: Prove bounds, dispatch on GPU, verify
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&conv_shape), conv_weight.clone()).expect("w"),
        ),
        TensorParamBinding::ConstantScalar(snake_alpha[0]),
    ];
    let (proved_lo, proved_hi) = prove_ibp_bounds(&def, &bindings, &[in_ch, in_len]);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let mut inputs = HashMap::new();
    inputs.insert("data", rand_f32_vec(0x5E0C_E099, in_ch * in_len, -1.0, 1.0));
    inputs.insert("conv.weight", conv_weight);
    inputs.insert("snake.alpha", snake_alpha);
    dispatch_and_verify(
        "smoke_enc",
        &def,
        inputs,
        out_ch * out_len,
        &proved_lo,
        &proved_hi,
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// Two-block encoder config for the smoke test.
struct TwoBlockConfig {
    in_ch_1: usize,
    out_ch_1: usize,
    out_ch_2: usize,
    kernel_size: usize,
    in_len: usize,
    stride: usize,
    padding: usize,
}

/// Build a two-block Conv1d→Snake encoder from extracted WeightMap data.
fn build_two_block_encoder(
    cfg: &TwoBlockConfig,
    w1: &[f32],
    a1: f32,
    w2: &[f32],
    a2: f32,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let out_len_1 = (cfg.in_len + 2 * cfg.padding - cfg.kernel_size) / cfg.stride + 1;
    let out_len_2 = (out_len_1 + 2 * cfg.padding - cfg.kernel_size) / cfg.stride + 1;
    let (out_shape_1, out_shape_2) = ([cfg.out_ch_1, out_len_1], [cfg.out_ch_2, out_len_2]);
    let w1_shape = [cfg.out_ch_1, cfg.in_ch_1, cfg.kernel_size];
    let w2_shape = [cfg.out_ch_2, cfg.out_ch_1, cfg.kernel_size];

    let mut b = TensorBlockBuilder::new("smoke_enc_2block");
    let data = b.add_input("data", &[cfg.in_ch_1, cfg.in_len]);
    let w1_n = b.add_input("block1.conv.weight", &w1_shape);
    let a1_n = b.add_input("block1.snake.alpha", &[1]);
    let w2_n = b.add_input("block2.conv.weight", &w2_shape);
    let a2_n = b.add_input("block2.snake.alpha", &[1]);

    let c1 = b.add_conv1d(data, w1_n, None, cfg.stride, cfg.padding, &out_shape_1);
    let a1_bc = b.add_broadcast(a1_n, &out_shape_1);
    let sk = build_snake_scalar_kernel().expect("snake 1");
    let s1 = b.add_elementwise(sk, &[c1, a1_bc], &out_shape_1);
    let c2 = b.add_conv1d(s1, w2_n, None, cfg.stride, cfg.padding, &out_shape_2);
    let a2_bc = b.add_broadcast(a2_n, &out_shape_2);
    let sk2 = build_snake_scalar_kernel().expect("snake 2");
    let s2 = b.add_elementwise(sk2, &[c2, a2_bc], &out_shape_2);
    let def = b.build(s2).expect("valid graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&w1_shape), w1.to_vec()).expect("w1"),
        ),
        TensorParamBinding::ConstantScalar(a1),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&w2_shape), w2.to_vec()).expect("w2"),
        ),
        TensorParamBinding::ConstantScalar(a2),
    ];
    (def, bindings)
}

/// Smoke test: two-block Demucs encoder chain.
///
///   Block 1: Conv1d(1→48, k=8, s=4, p=2) → Snake(α)
///   Block 2: Conv1d(48→96, k=8, s=4, p=2) → Snake(α)
///
/// All weight data loaded from a single safetensors file via WeightMap.
///
/// Part of #710.
#[test]
fn smoke_demucs_encoder_two_block_chain() {
    let cfg = TwoBlockConfig {
        in_ch_1: 1,
        out_ch_1: 48,
        out_ch_2: 96,
        kernel_size: 8,
        in_len: 64,
        stride: 4,
        padding: 2,
    };
    let out_len_1 = (cfg.in_len + 2 * cfg.padding - cfg.kernel_size) / cfg.stride + 1;
    let out_len_2 = (out_len_1 + 2 * cfg.padding - cfg.kernel_size) / cfg.stride + 1;

    // Step 1: Create safetensors, load via WeightMap
    let dir = temp_dir("enc_2block");
    let st_path = dir.join("encoder_2block.safetensors");
    let w1_shape = [cfg.out_ch_1, cfg.in_ch_1, cfg.kernel_size];
    let a_shape = [1];
    let w2_shape = [cfg.out_ch_2, cfg.out_ch_1, cfg.kernel_size];
    let specs: Vec<(&str, &[usize])> = vec![
        ("block1.conv.weight", &w1_shape),
        ("block1.snake.alpha", &a_shape),
        ("block2.conv.weight", &w2_shape),
        ("block2.snake.alpha", &a_shape),
    ];
    let _written = create_encoder_safetensors(&st_path, &specs, 0x2B1C_0001);

    let ctx = MetalContext::new().expect("Metal device");
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&st_path, &ctx).expect("load") };
    assert_eq!(wm.tensor_count(), 4);
    let w1 = extract_f32(&wm, "block1.conv.weight");
    let a1 = extract_f32(&wm, "block1.snake.alpha");
    let w2 = extract_f32(&wm, "block2.conv.weight");
    let a2 = extract_f32(&wm, "block2.snake.alpha");

    // Step 2+3: Build graph, prove bounds, dispatch, verify
    let (def, bindings) = build_two_block_encoder(&cfg, &w1, a1[0], &w2, a2[0]);
    let (proved_lo, proved_hi) = prove_ibp_bounds(&def, &bindings, &[cfg.in_ch_1, cfg.in_len]);
    assert_eq!(proved_lo.shape(), &[cfg.out_ch_2, out_len_2]);

    let mut inputs = HashMap::new();
    inputs.insert(
        "data",
        rand_f32_vec(0x2B1C_0099, cfg.in_ch_1 * cfg.in_len, -1.0, 1.0),
    );
    inputs.insert("block1.conv.weight", w1);
    inputs.insert("block1.snake.alpha", a1);
    inputs.insert("block2.conv.weight", w2);
    inputs.insert("block2.snake.alpha", a2);
    dispatch_and_verify(
        "smoke_2block",
        &def,
        inputs,
        cfg.out_ch_2 * out_len_2,
        &proved_lo,
        &proved_hi,
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}
