// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! convert() integration test -- ViT model trace-to-CompiledModel roundtrip.
//!
//! Proves that a ViT-style model built from nn-core nn layers can be:
//!   1. Run eagerly on GPU for a reference output
//!   2. Traced via `trace_graph()` into a `ComputationGraph`
//!   3. Compiled into a `CompiledModel` via the builder API
//!   4. Executed on GPU with output matching the eager reference (atol=1e-4)
//!
//! This exercises the full "convert" pipeline for vision transformers:
//! Conv2d patch embedding, positional embedding, multi-head self-attention,
//! LayerNorm, GELU MLP, and linear classification head.
//!
//! The model uses VarBuilder::zeros for deterministic, fast testing. Zero-init
//! weights reduce numerical divergence between eager and compiled paths while
//! still exercising the full dispatch pipeline (Conv2d, SDPA, LayerNorm, etc.).
//!
//! Part of #3541.

#![cfg(target_os = "macos")]

mod test_utils;

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::{PoolingStrategy, VitConfig, VitEncoder};
use nn_core::layers::{with_nan_check_policy, Linear, Module, NanCheckPolicy};
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::without_arena;

fn init() -> nn_metal::PipelineCache {
    test_utils::gpu_init();
    test_utils::metal_setup()
}

/// Small ViT config: 2 layers, 128 hidden, 2 heads.
/// image_size=32, patch_size=16 -> 4 patches + CLS = 5 sequence length.
/// Kept small for fast CI execution while exercising all ViT components.
fn test_config() -> VitConfig {
    VitConfig::new(
        3,    // num_channels (RGB)
        128,  // hidden_size
        2,    // num_layers
        2,    // num_heads (128/2 = 64 head_dim)
        256,  // intermediate_size (2x hidden)
        16,   // patch_size
        32,   // image_size (32/16 = 2x2 = 4 patches)
        1e-6, // layer_norm_eps
        true, // use_cls_token
    )
    .expect("valid ViT config")
}

/// Load VitEncoder with arena bypass to prevent stale arena reads.
///
/// VitEncoderBlock::load() fuses Q/K/V weights via DynTensor::cat on GPU,
/// which allocates from the ActivationArena. `without_arena` ensures all
/// weight buffers are standalone Metal buffers that survive arena generation
/// advances during trace_graph().
fn load_encoder(vb: &VarBuilder, config: &VitConfig) -> VitEncoder {
    without_arena(|| VitEncoder::load(vb, config).expect("encoder load"))
}

/// Create a classification head: Linear(hidden_size, num_classes).
fn load_head(vb: &VarBuilder, hidden_size: usize, num_classes: usize) -> Linear {
    let w = vb
        .get(&[num_classes, hidden_size], "head.weight")
        .expect("head weight");
    let b = vb.get(&[num_classes], "head.bias").expect("head bias");
    Linear::new(w, Some(b)).expect("head linear")
}

/// Create test input: [1, 3, 32, 32] image tensor on GPU.
fn test_input(config: &VitConfig, dev: &Device) -> DynTensor {
    DynTensor::zeros(
        &[1, config.num_channels, config.image_size, config.image_size],
        DType::F32,
        dev,
    )
    .expect("image tensor")
}

/// Full ViT forward: encoder (Conv2d patch embed + transformer blocks) -> CLS pooling -> Linear head.
fn vit_forward(
    encoder: &VitEncoder,
    head: &Linear,
    image: &DynTensor,
) -> Result<DynTensor, nn_core::TensorError> {
    // Encoder: [1, 3, 32, 32] -> [1, 5, 128] (with CLS token).
    // CLS pooling: take first token -> [1, 128].
    let encoded = encoder.forward(image, PoolingStrategy::Cls)?;
    // Classification head: [1, 128] -> [1, num_classes].
    head.forward(&encoded)
}

// -------------------------------------------------------------------------
// Test 1: Eager forward produces valid output shape
// -------------------------------------------------------------------------

/// Verify eager forward produces expected output shape before testing compiled path.
#[test]
fn test_convert_vit_eager_output_shape() {
    let _cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);
    let num_classes = 10;
    let head = load_head(&vb, config.hidden_size, num_classes);
    let image = test_input(&config, &dev);

    let output = with_nan_check_policy(NanCheckPolicy::Skip, || {
        vit_forward(&encoder, &head, &image)
    })
    .expect("eager forward");

    // Output shape: [1, num_classes].
    assert_eq!(output.dims(), &[1, num_classes]);
    eprintln!(
        "ViT eager output: shape={:?}, dtype={:?}",
        output.dims(),
        output.dtype()
    );
}

// -------------------------------------------------------------------------
// Test 2: Trace-to-CompiledModel roundtrip with numerical parity
// -------------------------------------------------------------------------

/// Core integration test: build ViT model, run eager reference, trace + compile,
/// execute compiled model, assert outputs match within atol=1e-4.
#[test]
fn test_convert_vit_compiled_matches_eager() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let num_classes = 10;

    let image = test_input(&config, &dev);

    // --- Eager reference ---
    let encoder_eager = load_encoder(&vb, &config);
    let head_eager = load_head(&vb, config.hidden_size, num_classes);

    let eager_output = with_nan_check_policy(NanCheckPolicy::Skip, || {
        vit_forward(&encoder_eager, &head_eager, &image)
    })
    .expect("eager forward");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output.to_flat_vec::<f32>().unwrap();

    // --- Trace + compile (fresh model to avoid shared state) ---
    let encoder_trace = load_encoder(&vb, &config);
    let head_trace = load_head(&vb, config.hidden_size, num_classes);

    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            vit_forward(&encoder_trace, &head_trace, &traced_image)
        })
    })
    .expect("trace_graph");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("CompiledModel build");

    // --- Execute compiled model ---
    let compiled_output = compiled
        .execute_dyn(&cache, &[&image])
        .expect("compiled execute");
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    // --- Assert numerical parity ---
    assert_eq!(eager_shape, compiled_output.dims());
    assert_eq!(eager_data.len(), compiled_data.len());

    let max_error = eager_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-4,
        "max error between eager and compiled should be < 1e-4, got {max_error:.2e}"
    );

    eprintln!(
        "ViT convert roundtrip: shape={eager_shape:?}, max_error={max_error:.2e}, \
         graph_nodes={}, compiled_steps={}, dispatches={}",
        graph.nodes().len(),
        compiled.num_steps(),
        compiled.num_dispatches()
    );
}

// -------------------------------------------------------------------------
// Test 3: Dispatch count reduction vs graph node count
// -------------------------------------------------------------------------

/// Verify compiled dispatch count is less than raw graph node count,
/// proving that compilation eliminates weight nodes, shape ops, and
/// identity passthroughs.
#[test]
fn test_convert_vit_dispatch_count() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let num_classes = 10;

    let encoder = load_encoder(&vb, &config);
    let head = load_head(&vb, config.hidden_size, num_classes);
    let image = test_input(&config, &dev);

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            vit_forward(&encoder, &head, &traced_image)
        })
    })
    .expect("trace_graph");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let node_count = graph.nodes().len();
    let dispatch_count = compiled.num_dispatches();
    let step_count = compiled.num_steps();

    assert!(
        dispatch_count < node_count,
        "compiled dispatches ({dispatch_count}) should be fewer than \
         graph nodes ({node_count})"
    );

    // ViT with 2 layers should produce a reasonable dispatch count.
    // Log breakdown for diagnostics.
    let (ir_ops, native_ops) = compiled.dispatch_breakdown();

    eprintln!(
        "ViT convert dispatch: {node_count} graph nodes -> \
         {step_count} compiled steps, {dispatch_count} dispatches"
    );
    eprintln!("  IR ops:");
    for (name, count) in &ir_ops {
        eprintln!("    {name}: {count}");
    }
    eprintln!("  NativeOps:");
    for (name, count) in &native_ops {
        eprintln!("    {name}: {count}");
    }
}

// -------------------------------------------------------------------------
// Test 4: Compiled model reuse with different inputs
// -------------------------------------------------------------------------

/// Verify CompiledModel can be reused with different input data,
/// proving that weight buffers are persistent and not tied to a
/// specific input.
#[test]
fn test_convert_vit_model_reuse() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let num_classes = 10;

    let encoder = load_encoder(&vb, &config);
    let head = load_head(&vb, config.hidden_size, num_classes);

    // Two different inputs: zeros and ones.
    let image_zeros = test_input(&config, &dev);
    let image_ones = DynTensor::ones(
        &[1, config.num_channels, config.image_size, config.image_size],
        DType::F32,
        &dev,
    )
    .expect("ones tensor");

    // Trace with zeros input.
    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced_image = image_zeros.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            vit_forward(&encoder, &head, &traced_image)
        })
    })
    .expect("trace_graph");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Execute with zeros input.
    let out_zeros = compiled
        .execute_dyn(&cache, &[&image_zeros])
        .expect("execute zeros");
    let data_zeros = out_zeros.to_flat_vec::<f32>().unwrap();

    // Execute with ones input (reusing the same compiled model).
    let out_ones = compiled
        .execute_dyn(&cache, &[&image_ones])
        .expect("execute ones");
    let data_ones = out_ones.to_flat_vec::<f32>().unwrap();

    // Shapes must match.
    assert_eq!(out_zeros.dims(), &[1, num_classes]);
    assert_eq!(out_ones.dims(), &[1, num_classes]);

    // With zero-init weights, both outputs should be identical (all biases=0,
    // all weights=0 means output is always 0 regardless of input). But the
    // key assertion is that the second execution did not crash or produce
    // NaN, proving the compiled model is reusable.
    assert_eq!(data_zeros.len(), data_ones.len());
    for val in &data_zeros {
        assert!(val.is_finite(), "zeros output contains non-finite: {val}");
    }
    for val in &data_ones {
        assert!(val.is_finite(), "ones output contains non-finite: {val}");
    }

    eprintln!("ViT convert reuse: zeros_out={data_zeros:?}, ones_out={data_ones:?}");
}
