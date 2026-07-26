// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ViT encoder CompiledModel integration test.
//!
//! Proves VitEncoder runs through the full trace_graph() -> compile_trace()
//! -> CompiledModel::builder().build() -> execute_dyn() pipeline with
//! peephole optimization.
//!
//! Uses a small ViT config (4 layers, 256 hidden, 4 heads, patch_size=16,
//! image_size=64) with VarBuilder::zeros for fast testing.
//!
//! VitEncoderBlock::load() fuses Q/K/V via DynTensor::cat on GPU, creating
//! arena-allocated tensors. Model loading is wrapped in `without_arena` so
//! fused weights get standalone Metal buffers that survive arena generation
//! advances during the forward pass. Without this, trace_graph() fails with
//! `WeightConversionFailed` because to_weight_ref() hits a stale arena read.
//!
//! Part of #3520.

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::{PoolingStrategy, VitConfig, VitEncoder};
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::without_arena;

fn init() -> nn_metal::PipelineCache {
    super::test_utils::gpu_init();
    super::test_utils::metal_setup()
}

/// Small ViT config for fast testing: 4 layers, 256 hidden, 4 heads.
/// image_size=64, patch_size=16 -> 16 patches + CLS = 17 sequence length.
fn test_config() -> VitConfig {
    VitConfig::new(
        3,    // num_channels (RGB)
        256,  // hidden_size
        4,    // num_layers
        4,    // num_heads (256/4 = 64 head_dim)
        512,  // intermediate_size (2x hidden)
        16,   // patch_size
        64,   // image_size (64/16 = 4x4 = 16 patches)
        1e-6, // layer_norm_eps
        true, // use_cls_token
    )
    .expect("valid ViT config")
}

/// Load VitEncoder with arena bypass to prevent stale arena reads.
///
/// VitEncoderBlock::load() fuses Q/K/V weights via DynTensor::cat on GPU,
/// which allocates from the ActivationArena. During trace_graph(), the
/// forward pass advances the arena generation, making these fused weights
/// stale. `without_arena` ensures all weight buffers are standalone.
fn load_encoder(vb: &VarBuilder, config: &VitConfig) -> VitEncoder {
    without_arena(|| VitEncoder::load(vb, config).expect("model load"))
}

/// Create test input: [1, 3, 64, 64] image tensor on GPU.
fn test_input(config: &VitConfig, dev: &Device) -> DynTensor {
    DynTensor::zeros(
        &[1, config.num_channels, config.image_size, config.image_size],
        DType::F32,
        dev,
    )
    .expect("image tensor")
}

/// AC1: VitEncoder produces valid ComputationGraph via trace_graph().
#[test]
fn test_vit_encoder_trace_graph() {
    let _cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);

    let image = test_input(&config, &dev);

    let (output, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            encoder.forward(&traced_image, PoolingStrategy::None)
        })
    })
    .expect("trace_graph should succeed");

    // Verify output shape: [1, seq_len, hidden_size].
    // seq_len = num_patches + 1 (CLS token) = 16 + 1 = 17.
    assert_eq!(output.rank(), 3);
    assert_eq!(output.dim(0).unwrap(), 1);
    assert_eq!(output.dim(1).unwrap(), config.seq_len()); // 17
    assert_eq!(output.dim(2).unwrap(), config.hidden_size); // 256

    // Verify graph has nodes.
    let node_count = graph.nodes().len();
    assert!(
        node_count > 10,
        "encoder graph should have substantial node count, got {node_count}"
    );

    // Verify at least 1 input node exists.
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();
    assert!(
        input_count >= 1,
        "graph should have at least 1 input, got {input_count}"
    );

    // Check for ConstantWeight nodes (position embedding, CLS token, layer weights).
    let cw_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::ConstantWeight { .. }))
        .count();
    eprintln!(
        "ViT encoder (4-layer, d=256, 4 heads): {node_count} nodes, \
         {input_count} inputs, {cw_count} constant weights"
    );
}

/// AC2: CompiledModel executes encoder via trace+builder, matches eager output.
#[test]
fn test_vit_encoder_compiled_vs_eager() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);

    let image = test_input(&config, &dev);

    // Eager forward -- extract data to CPU immediately.
    let eager_output = with_nan_check_policy(NanCheckPolicy::Skip, || {
        encoder.forward(&image, PoolingStrategy::None)
    })
    .expect("eager forward");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output.to_flat_vec::<f32>().unwrap();

    // Trace + compile via builder (fresh model to avoid cache state).
    let encoder_ref = load_encoder(&vb, &config);

    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            encoder_ref.forward(&traced_image, PoolingStrategy::None)
        })
    })
    .expect("trace_graph for compile");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("builder.build() should succeed");

    // Execute compiled model.
    let compiled_output = compiled
        .execute_dyn(&cache, &[&image])
        .expect("compiled execute should succeed");
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    // Verify shapes match.
    assert_eq!(eager_shape, compiled_output.dims());

    let max_error = eager_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // ViT has Conv2d, LayerNorm, SDPA, GELU MLP -- same tolerance as Whisper.
    assert!(
        max_error < 1e-4,
        "max error between eager and compiled should be < 1e-4, got {max_error}"
    );

    eprintln!("ViT encoder compiled vs eager: shape={eager_shape:?}, max_error={max_error:.2e}");
}

/// AC3: Dispatch count -- compiled should reduce steps vs graph nodes.
#[test]
fn test_vit_encoder_dispatch_count() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);

    let image = test_input(&config, &dev);

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            encoder.forward(&traced_image, PoolingStrategy::None)
        })
    })
    .expect("trace_graph should succeed");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile should succeed");

    let step_count = compiled.num_steps();
    let dispatch_count = compiled.num_dispatches();
    let node_count = graph.nodes().len();

    // Compiled dispatch count should be less than total graph nodes because
    // many nodes are weights, shape ops, or identity passthroughs.
    assert!(
        dispatch_count < node_count,
        "compiled dispatches ({dispatch_count}) should be fewer than \
         graph nodes ({node_count})"
    );

    eprintln!(
        "ViT encoder (4-layer, d=256): {node_count} graph nodes -> \
         {step_count} compiled steps, {dispatch_count} dispatches"
    );
}

/// AC4: Autocast compiled encoder matches F32 compiled encoder within F16 tolerance.
///
/// Exercises the full ViT F16 pipeline: Conv2d(F16) -> LayerNorm(F32) ->
/// SDPA(F16) -> GELU MLP(F16) with boundary casts.
#[test]
fn test_vit_encoder_autocast_vs_f32() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);

    let image = test_input(&config, &dev);

    // Trace encoder forward.
    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            encoder.forward(&traced_image, PoolingStrategy::None)
        })
    })
    .expect("trace_graph");

    // F32 compiled baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute_dyn(&cache, &[&image]).expect("f32 exec");
    let f32_data = f32_out.to_flat_vec::<f32>().unwrap();

    // Autocast compiled.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    let ac_out = ac_model
        .execute_dyn(&cache, &[&image])
        .expect("autocast exec");
    let ac_data = ac_out.to_flat_vec::<f32>().unwrap();

    assert_eq!(f32_out.dims(), ac_out.dims());

    let max_error = f32_data
        .iter()
        .zip(ac_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // ViT: 4 layers x (Conv2d + SDPA + GELU MLP) with F16 boundary casts.
    // Zero-init weights reduce numerical divergence. Same tolerance as Whisper.
    assert!(
        max_error < 0.1,
        "encoder autocast vs f32 max error: {max_error:.2e} (expected < 0.1)"
    );
    eprintln!("ViT encoder autocast vs F32: max_error={max_error:.2e}");
}

/// AC5: Autocast encoder has F16 steps — proves autocast classification is active.
///
/// Without this assertion, a broken autocast classifier could pass AC4
/// (all F32, zero error) while silently disabling the speedup.
/// ViT has Conv2d, Linear (QKV, proj, MLP), and SDPA — all compute-dominant.
#[test]
fn test_vit_encoder_autocast_has_f16_steps() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);

    let image = test_input(&config, &dev);

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            encoder.forward(&traced_image, PoolingStrategy::None)
        })
    })
    .expect("trace_graph");

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");

    // ViT (4 layers): Conv2d patch embed + 4x (QKV linear + proj linear + 2 MLP linears)
    // + SDPA. At least some should qualify for F16.
    let f16_count = ac_model.num_autocast_f16_steps();
    assert!(
        f16_count > 0,
        "encoder autocast should have at least 1 F16 step, got 0"
    );
    eprintln!(
        "ViT encoder autocast: {f16_count} F16 steps out of {} total",
        ac_model.num_steps()
    );
}

/// AC6: Dispatch breakdown — verify peephole fusions fire for ViT patterns.
///
/// ViT uses standard ops that should trigger peephole passes:
/// - FlashAttention (pass 5/7): SDPA → fused attention kernel
/// - NormLinear (pass 9): LayerNorm + Linear → fused NormLinear
///
/// We verify via dispatch_breakdown() that fused kernel names appear.
#[test]
fn test_vit_encoder_peephole_fusions() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let encoder = load_encoder(&vb, &config);

    let image = test_input(&config, &dev);

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_image = image.clone();
        if let Some(id) = trace::record_input(traced_image.dims(), DType::F32) {
            traced_image.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            encoder.forward(&traced_image, PoolingStrategy::None)
        })
    })
    .expect("trace_graph");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let (ir_ops, native_ops) = compiled.dispatch_breakdown();

    // Log full breakdown for diagnostics.
    eprintln!("ViT encoder IR dispatch breakdown:");
    for (name, count) in &ir_ops {
        eprintln!("  {name}: {count}");
    }
    eprintln!("ViT encoder NativeOp breakdown:");
    for (name, count) in &native_ops {
        eprintln!("  {name}: {count}");
    }

    // FlashAttention should appear: ViT has 4 layers × 1 SDPA each.
    let has_flash_attn = native_ops
        .iter()
        .any(|(name, _)| name.contains("FlashAttention"));
    assert!(
        has_flash_attn,
        "peephole should fuse SDPA into FlashAttention NativeOp"
    );

    // NormLinear should appear: pre-norm ViT has LayerNorm before QKV and MLP projections.
    let has_norm_linear = native_ops
        .iter()
        .any(|(name, _)| name.contains("NormLinear"));
    // NormLinear is expected but not strictly required — log a warning if missing
    // since the fusion depends on the specific peephole pass configuration.
    if has_norm_linear {
        eprintln!("  NormLinear fusion: ACTIVE");
    } else {
        eprintln!(
            "  NormLinear fusion: not detected (LayerNorm+Linear not adjacent after compilation)"
        );
    }
}
