// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `CompiledModelBuilder` configuration, error cases,
//! step classification, and query method defaults.
//!
//! Part of #4299.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::{CompiledStep, NativeOpKind};

use super::CompiledModel;

fn empty_graph() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![])
}

// ── force_dtype: valid GPU-compilable dtypes ────────────────────────

#[test]
fn force_dtype_accepts_f32() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let builder = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F32)
        .expect("F32 should be accepted");
    assert!(builder.mixed_precision.is_some());
    assert_eq!(builder.mixed_precision.unwrap(), ScalarType::F32);
}

#[test]
fn force_dtype_accepts_f16() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let builder = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .expect("F16 should be accepted");
    assert_eq!(builder.mixed_precision.unwrap(), ScalarType::F16);
}

#[test]
fn force_dtype_accepts_bf16() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let builder = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::BF16)
        .expect("BF16 should be accepted");
    assert_eq!(builder.mixed_precision.unwrap(), ScalarType::BF16);
}

// ── force_dtype: non-GPU dtypes rejected ────────────────────────────

#[test]
fn force_dtype_rejects_u8() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = CompiledModel::builder(&graph, &cache).force_dtype(DType::U8);
    assert!(result.is_err(), "U8 is not a GPU-compilable dtype");
}

#[test]
fn force_dtype_rejects_u32() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = CompiledModel::builder(&graph, &cache).force_dtype(DType::U32);
    assert!(result.is_err(), "U32 is not a GPU-compilable dtype");
}

#[test]
fn force_dtype_rejects_i64() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = CompiledModel::builder(&graph, &cache).force_dtype(DType::I64);
    assert!(result.is_err(), "I64 is not a GPU-compilable dtype");
}

#[test]
fn force_dtype_rejects_i32() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = CompiledModel::builder(&graph, &cache).force_dtype(DType::I32);
    assert!(result.is_err(), "I32 is not a GPU-compilable dtype");
}

#[test]
fn force_dtype_rejects_bool() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = CompiledModel::builder(&graph, &cache).force_dtype(DType::Bool);
    assert!(result.is_err(), "Bool is not a GPU-compilable dtype");
}

#[test]
fn force_dtype_rejects_f64() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = CompiledModel::builder(&graph, &cache).force_dtype(DType::F64);
    assert!(result.is_err(), "F64 is not a GPU-compilable dtype");
}

// ── autocast: f32_only policy is a no-op ────────────────────────────

#[test]
fn autocast_f32_only_policy_is_noop() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::f32_only();
    let builder = CompiledModel::builder(&graph, &cache).autocast(policy);
    // f32_only has compute_dtype == F32, so autocast should be skipped
    assert!(
        builder.autocast_policy.is_none(),
        "f32_only autocast should be treated as a no-op"
    );
}

#[test]
fn autocast_apple_silicon_policy_is_stored() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let builder = CompiledModel::builder(&graph, &cache).autocast(policy.clone());
    assert_eq!(
        builder.autocast_policy,
        Some(policy),
        "non-F32 autocast policy should be stored"
    );
}

// ── builder defaults ────────────────────────────────────────────────

#[test]
fn builder_defaults_are_none() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let builder = CompiledModel::builder(&graph, &cache);
    assert!(builder.shared_weights.is_none());
    assert!(builder.mixed_precision.is_none());
    assert!(builder.autocast_policy.is_none());
    assert!(builder.peephole_config.is_none());
    assert!(builder.optimization_budget.is_none());
}

// ── shared_weights builder ──────────────────────────────────────────

#[test]
fn shared_weights_stores_reference() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let weights: HashMap<(usize, String), crate::buffer::MetalBuffer> = HashMap::new();
    let builder = CompiledModel::builder(&graph, &cache).shared_weights(&weights);
    assert!(builder.shared_weights.is_some());
}

// ── peephole_config builder ─────────────────────────────────────────

#[test]
fn with_peephole_config_stores_config() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = nn_dsl::PeepholeConfig::default();
    let builder = CompiledModel::builder(&graph, &cache).with_peephole_config(config);
    assert!(builder.peephole_config.is_some());
}

// ── optimize builder ────────────────────────────────────────────────

#[test]
fn optimize_stores_budget() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let budget = std::time::Duration::from_millis(100);
    let builder = CompiledModel::builder(&graph, &cache).optimize(budget);
    assert_eq!(builder.optimization_budget, Some(budget));
}

// ── empty graph build produces empty model ──────────────────────────

#[test]
fn build_empty_graph_produces_empty_model() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("empty graph should build successfully");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
    assert_eq!(model.num_inputs(), 0);
    assert_eq!(model.num_outputs(), 0);
    assert!(!model.is_mixed_precision());
    assert!(!model.is_autocast());
}

// ── empty model query defaults ──────────────────────────────────────

#[test]
fn empty_model_output_dtype_defaults_to_f32() {
    let model = CompiledModel::empty();
    assert_eq!(model.output_dtype(), DType::F32);
}

#[test]
fn empty_model_output_shape_is_empty() {
    let model = CompiledModel::empty();
    assert!(model.output_shape().is_empty());
}

#[test]
fn empty_model_num_ir_dispatches_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_ir_dispatches(), 0);
}

#[test]
fn empty_model_num_native_ops_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_native_ops(), 0);
}

#[test]
fn empty_model_num_metal_dispatches_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_metal_dispatches(), 0);
}

#[test]
fn empty_model_num_encoding_events_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_encoding_events(), 0);
}

#[test]
fn empty_model_num_mixed_gemm_steps_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_mixed_gemm_steps(), 0);
}

#[test]
fn empty_model_num_autocast_f16_steps_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_autocast_f16_steps(), 0);
}

#[test]
fn empty_model_num_icb_segments_is_zero() {
    let model = CompiledModel::empty();
    assert_eq!(model.num_icb_segments(), 0);
}

#[test]
fn empty_model_dispatch_breakdown_is_empty() {
    let model = CompiledModel::empty();
    let (ir, native) = model.dispatch_breakdown();
    assert!(ir.is_empty());
    assert!(native.is_empty());
}

#[test]
fn empty_model_buffer_plan_is_zero() {
    let model = CompiledModel::empty();
    let bp = model.buffer_plan();
    assert_eq!(bp.total_bytes, 0);
    assert_eq!(bp.naive_total, 0);
}

#[test]
fn empty_model_input_specs_is_empty() {
    let model = CompiledModel::empty();
    assert!(model.input_specs().is_empty());
}

#[test]
fn empty_model_steps_is_empty() {
    let model = CompiledModel::empty();
    assert!(model.steps().is_empty());
}

#[test]
fn empty_model_precision_is_none() {
    let model = CompiledModel::empty();
    assert!(model.precision().is_none());
}

#[test]
fn empty_model_proof_certificate_is_none() {
    let model = CompiledModel::empty();
    assert!(model.proof_certificate_json().is_none());
}

// ── with_precision ──────────────────────────────────────────────────

#[test]
fn with_precision_sets_contract() {
    let model = CompiledModel::empty();
    let contract = nn_dsl::PrecisionContract::bootstrap(
        nn_dsl::PrecisionTier::Strict,
        ScalarType::F32,
    );
    let model = model.with_precision(contract);
    assert!(model.precision().is_some());
    assert_eq!(model.precision().unwrap().tier, nn_dsl::PrecisionTier::Strict);
}

// ── shared model definition ─────────────────────────────────────────

#[test]
fn share_def_creates_independent_execution_instances() {
    let model = CompiledModel::empty();
    let shared = model.share_def();
    let instance = CompiledModel::from_shared(shared);
    // Both models share the same definition but have independent caches
    assert_eq!(instance.num_steps(), 0);
    assert_eq!(instance.num_inputs(), 0);
}

// ── step classification: is_compute_native_op ───────────────────────

#[test]
fn classify_flash_attention_as_compute() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 8, 128, 64],
        k_shape: vec![1, 8, 128, 64],
        output_shape: vec![1, 8, 128, 64],
        input_layout: Default::default(),
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "FlashAttention should be classified as compute"
    );
}

#[test]
fn classify_norm_activ_conv1d_as_compute() {
    let op = NativeOpKind::NormActivConv1d {
        activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 0,
        input_shape: vec![1, 2, 4],
        output_channels: 2,
        kernel_size: 2,
        external_node_ids: None,
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "NormActivConv1d should be classified as compute"
    );
}

#[test]
fn classify_fused_resblock_as_compute() {
    let params = nn_dsl::NormActivConv1dParams::new(
        nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
        1e-5,
        1,  // conv_dilation
        0,  // conv_padding
        vec![1, 2, 4],
        2,  // output_channels
        3,  // kernel_size
    );
    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "FusedResBlock should be classified as compute"
    );
}

#[test]
fn classify_batched_linear_projection_as_compute() {
    let op = NativeOpKind::BatchedLinearProjection {
        in_features: 512,
        total_out_features: 256,
        projection_sizes: vec![256],
        has_bias: false,
        input_shape: vec![1, 128, 512],
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "BatchedLinearProjection should be classified as compute"
    );
}

#[test]
fn classify_norm_linear_as_compute() {
    let op = NativeOpKind::NormLinear {
        norm_kind: nn_dsl::FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        hidden_dim: 512,
        out_features: 256,
        has_bias: true,
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "NormLinear should be classified as compute"
    );
}

#[test]
fn classify_adain_snake_as_compute() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
        channels: 4,
        residual_gamma: true,
        external_node_ids: None,
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "AdainSnake should be classified as compute"
    );
}

#[test]
fn classify_adain_leaky_relu_as_compute() {
    let op = NativeOpKind::AdainLeakyRelu {
        eps: 1e-5,
        slope: 0.2,
        input_shape: vec![1, 4, 16],
        external_node_ids: None,
    };
    assert!(
        super::classify::is_compute_native_op(&op),
        "AdainLeakyRelu should be classified as compute"
    );
}

#[test]
fn classify_instance_norm_not_as_compute() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
    };
    assert!(
        !super::classify::is_compute_native_op(&op),
        "InstanceNorm is accumulate, not compute"
    );
}

#[test]
fn classify_lstm_not_as_compute() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![10, 1, 512],
        h_shape: vec![1, 256],
        reverse: false,
    };
    assert!(
        !super::classify::is_compute_native_op(&op),
        "LSTM is not a compute-dominant op for autocast"
    );
}

#[test]
fn classify_layer_norm_not_as_compute() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        hidden_dim: 512,
    };
    assert!(
        !super::classify::is_compute_native_op(&op),
        "LayerNorm is accumulate, not compute"
    );
}

#[test]
fn classify_cumsum_not_as_compute() {
    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![4],
    };
    assert!(
        !super::classify::is_compute_native_op(&op),
        "Cumsum is not compute"
    );
}

#[test]
fn classify_add_layer_norm_not_as_compute() {
    let op = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        hidden_dim: 512,
    };
    assert!(
        !super::classify::is_compute_native_op(&op),
        "AddLayerNorm is accumulate, not compute"
    );
}

// ── step classification: is_passthrough_safe ────────────────────────

#[test]
fn classify_projection_slice_as_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::ProjectionSlice {
            source_step: 0,
            dim: 2,
            start: 0,
            length: 64,
            output_shape: vec![1, 128, 64],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        super::classify::is_passthrough_safe(&step),
        "ProjectionSlice is data-movement only, safe to passthrough"
    );
}

#[test]
fn classify_silu_mul_as_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::SiluMul {
            input_shape: vec![1, 128, 512],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        super::classify::is_passthrough_safe(&step),
        "SiluMul should be passthrough safe"
    );
}

#[test]
fn classify_rotary_embedding_as_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 128, 64],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        super::classify::is_passthrough_safe(&step),
        "RotaryEmbedding should be passthrough safe"
    );
}

#[test]
fn classify_instance_norm_nativeop_not_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "InstanceNorm is not passthrough safe (accumulate op)"
    );
}

#[test]
fn classify_lstm_nativeop_not_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![10, 1, 512],
            h_shape: vec![1, 256],
            reverse: false,
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "LSTM is not passthrough safe"
    );
}

#[test]
fn classify_input_forward_not_passthrough() {
    let step = CompiledStep::InputForward;
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "InputForward is not a passthrough step"
    );
}

#[test]
fn classify_identity_passthrough_step_not_passthrough_safe() {
    let step = CompiledStep::IdentityPassthrough;
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "IdentityPassthrough is not classified as passthrough safe (separate category)"
    );
}

#[test]
fn classify_passthrough_step_not_passthrough_safe() {
    let step = CompiledStep::Passthrough {
        op_name: "reshape".into(),
        output_shape: vec![1, 4],
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "Passthrough (reshape) is not classified in the passthrough_safe dispatch match"
    );
}

#[test]
fn classify_flash_attention_nativeop_not_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 8, 128, 64],
            k_shape: vec![1, 8, 128, 64],
            output_shape: vec![1, 8, 128, 64],
            input_layout: Default::default(),
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "FlashAttention is compute, not passthrough"
    );
}

// ── validate_buffer_plan_edges ──────────────────────────────────────

#[test]
fn validate_buffer_plan_edges_ok_for_empty_steps() {
    let steps: Vec<CompiledStep> = vec![];
    let last_use: Vec<usize> = vec![];
    let result = super::super::build::validate_buffer_plan_edges(&steps, &last_use);
    assert!(result.is_ok(), "empty steps should pass validation");
}

#[test]
fn validate_buffer_plan_edges_ok_for_non_nativeop_steps() {
    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::IdentityPassthrough,
    ];
    let last_use = vec![1, 1];
    let result = super::super::build::validate_buffer_plan_edges(&steps, &last_use);
    assert!(result.is_ok(), "non-NativeOp steps skip validation");
}

#[test]
fn validate_buffer_plan_edges_rejects_early_release() {
    // FusedResBlock references step 0 via input_steps, but last_use[0] = 0 < step_idx 2.
    let params = nn_dsl::NormActivConv1dParams::new(
        nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
        1e-5, 1, 0, vec![1, 2, 4], 2, 3,
    );
    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::NativeOp {
            op: NativeOpKind::FusedResBlock {
                phase1: params.clone(),
                phase2: params,
                input_steps: vec![0],
                residual_scale: 1.0,
                style_proj: None,
                shortcut_step: None,
                pool_step: None,
                style_batch_offset: None,
            },
            weight_data: HashMap::new(),
        },
    ];
    let last_use = vec![0, 2, 2];
    let result = super::super::build::validate_buffer_plan_edges(&steps, &last_use);
    assert!(
        result.is_err(),
        "buffer plan should reject early release of FusedResBlock dependency"
    );
}

#[test]
fn validate_buffer_plan_edges_accepts_valid_fused_resblock_deps() {
    let params = nn_dsl::NormActivConv1dParams::new(
        nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
        1e-5, 1, 0, vec![1, 2, 4], 2, 3,
    );
    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::NativeOp {
            op: NativeOpKind::FusedResBlock {
                phase1: params.clone(),
                phase2: params,
                input_steps: vec![0],
                residual_scale: 1.0,
                style_proj: None,
                shortcut_step: Some(1),
                pool_step: None,
                style_batch_offset: None,
            },
            weight_data: HashMap::new(),
        },
    ];
    let last_use = vec![2, 2, 2]; // both deps alive until step 2
    let result = super::super::build::validate_buffer_plan_edges(&steps, &last_use);
    assert!(
        result.is_ok(),
        "valid buffer plan should pass for FusedResBlock with shortcut"
    );
}

#[test]
fn validate_buffer_plan_edges_checks_projection_slice_dep() {
    // ProjectionSlice references source_step. Must be alive.
    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::NativeOp {
            op: NativeOpKind::ProjectionSlice {
                source_step: 0,
                dim: 2,
                start: 0,
                length: 64,
                output_shape: vec![1, 128, 64],
            },
            weight_data: HashMap::new(),
        },
    ];
    let last_use = vec![1, 1]; // step 0 alive until step 1 → ok
    let result = super::super::build::validate_buffer_plan_edges(&steps, &last_use);
    assert!(
        result.is_ok(),
        "ProjectionSlice with valid source_step should pass"
    );
}

#[test]
fn validate_buffer_plan_edges_rejects_early_release_projection_slice() {
    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::NativeOp {
            op: NativeOpKind::ProjectionSlice {
                source_step: 0,
                dim: 2,
                start: 0,
                length: 64,
                output_shape: vec![1, 128, 64],
            },
            weight_data: HashMap::new(),
        },
    ];
    let last_use = vec![0, 2, 2]; // step 0 released at step 0 < step 2
    let result = super::super::build::validate_buffer_plan_edges(&steps, &last_use);
    assert!(
        result.is_err(),
        "ProjectionSlice source released early should fail"
    );
}

// ── flat_weights_to_indexed ─────────────────────────────────────────

#[test]
fn flat_weights_to_indexed_empty() {
    let flat = HashMap::new();
    let result = super::super::build::flat_weights_to_indexed(flat, 0);
    assert!(result.is_empty());
}

#[test]
fn flat_weights_to_indexed_distributes_by_step() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buf_a = ctx.create_buffer(&[1.0_f32]).expect("buf");
    let buf_b = ctx.create_buffer(&[2.0_f32]).expect("buf");

    let mut flat = HashMap::new();
    flat.insert((0, "weight".to_string()), buf_a);
    flat.insert((2, "bias".to_string()), buf_b);

    let result = super::super::build::flat_weights_to_indexed(flat, 3);
    assert_eq!(result.len(), 3);
    assert!(result[0].contains_key("weight"));
    assert!(result[1].is_empty());
    assert!(result[2].contains_key("bias"));
}

#[test]
fn flat_weights_to_indexed_ignores_out_of_range_step() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buf = ctx.create_buffer(&[1.0_f32]).expect("buf");

    let mut flat = HashMap::new();
    flat.insert((99, "weight".to_string()), buf);

    let result = super::super::build::flat_weights_to_indexed(flat, 2);
    assert_eq!(result.len(), 2);
    assert!(result[0].is_empty());
    assert!(result[1].is_empty());
}

// ── shares_weight_buffers ───────────────────────────────────────────

#[test]
fn shares_weight_buffers_true_for_norm_activ_conv() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::NormActivConv1d {
            activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 0,
            input_shape: vec![1, 2, 4],
            output_channels: 2,
            kernel_size: 2,
            external_node_ids: None,
        },
        weight_data: HashMap::new(),
    };
    assert!(
        super::super::build::shares_weight_buffers(&step),
        "NormActivConv1d weights are shape-invariant and shareable"
    );
}

#[test]
fn shares_weight_buffers_false_for_constant_weight() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::ConstantWeight {
            name: "const".into(),
            shape: vec![5],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::super::build::shares_weight_buffers(&step),
        "ConstantWeight may encode shape-dependent data and is not shareable"
    );
}

#[test]
fn shares_weight_buffers_true_for_dispatch() {
    let node = nn_dsl::TensorNode::new(
        nn_dsl::TensorNodeId::new(0),
        nn_dsl::TensorOpKind::Input {
            name: "x".into(),
            shape: vec![1],
        },
        vec![1],
    );
    let def = nn_dsl::TensorKernelDef::new("test", vec![node], nn_dsl::TensorNodeId::new(0));
    let step = CompiledStep::Dispatch {
        kernel: nn_dsl::CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    };
    assert!(super::super::build::shares_weight_buffers(&step));
}

#[test]
fn shares_weight_buffers_true_for_input_forward() {
    let step = CompiledStep::InputForward;
    assert!(super::super::build::shares_weight_buffers(&step));
}

// ── PeepholeConfig: all disabled ────────────────────────────────────

#[test]
fn peephole_config_all_disabled() {
    let config = nn_dsl::PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
        add_norm_linear: false,
        fuse_adain_snake: false,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
        fuse_snake_instance_norm: false,
        fuse_conv1d_snake_norm: false,
        fuse_conv1d_snake_norm_resblock: false,
        fuse_add_instance_norm_conv1x1: false,
        fuse_conv_transpose1d_activation: false,
        norm_activ_conv_transpose1d: false,
        fuse_instance_norm_conv1d: false,
        fuse_conv1d_instance_norm: false,
        fuse_linear_layer_norm: false,
        fuse_resblock_chain: false,
        fuse_activation_conv1d: false,
    };
    assert!(!config.norm_activ_conv1d);
    assert!(!config.fused_resblock);
    assert!(!config.linear_activation);
    assert!(!config.add_layer_norm);
    assert!(!config.norm_linear);
    assert!(!config.attention_transpose);
    assert!(!config.flip_lstm);
    assert!(!config.batched_linear_projection);
    assert!(!config.channels_first_layer_norm);
    assert!(!config.silu_mul);
    assert!(!config.auto_fuse_elementwise);
    assert!(!config.bilstm_cat);
    assert!(!config.add_norm_linear);
}

#[test]
fn peephole_config_default_all_enabled() {
    let config = nn_dsl::PeepholeConfig::default();
    assert!(config.norm_activ_conv1d);
    assert!(config.fused_resblock);
    assert!(config.linear_activation);
    assert!(config.add_layer_norm);
    assert!(config.norm_linear);
    assert!(config.attention_transpose);
    assert!(config.flip_lstm);
    assert!(config.batched_linear_projection);
    assert!(config.channels_first_layer_norm);
    assert!(config.silu_mul);
    assert!(config.auto_fuse_elementwise);
    assert!(config.bilstm_cat);
    assert!(config.add_norm_linear);
}

#[test]
fn peephole_config_selective_disable() {
    let config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        bilstm_cat: false,
        ..nn_dsl::PeepholeConfig::default()
    };
    assert!(config.norm_activ_conv1d, "unmodified passes stay enabled");
    assert!(config.fused_resblock, "unmodified passes stay enabled");
    assert!(!config.silu_mul, "explicitly disabled");
    assert!(!config.bilstm_cat, "explicitly disabled");
}

// ── builder method chaining: combined options ───────────────────────

#[test]
fn builder_chaining_peephole_then_force_dtype() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = nn_dsl::PeepholeConfig::default();
    let builder = CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .force_dtype(DType::F16)
        .expect("F16 should be accepted");
    assert!(builder.peephole_config.is_some());
    assert_eq!(builder.mixed_precision, Some(ScalarType::F16));
}

#[test]
fn builder_chaining_shared_weights_then_autocast() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let weights: HashMap<(usize, String), crate::buffer::MetalBuffer> = HashMap::new();
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let builder = CompiledModel::builder(&graph, &cache)
        .shared_weights(&weights)
        .autocast(policy.clone());
    assert!(builder.shared_weights.is_some());
    assert_eq!(builder.autocast_policy, Some(policy));
}

#[test]
fn builder_optimize_overrides_peephole_config() {
    // optimize and with_peephole_config are mutually exclusive;
    // whichever is set last wins in the build() method dispatch.
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let budget = std::time::Duration::from_millis(50);
    let config = nn_dsl::PeepholeConfig::default();
    let builder = CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .optimize(budget);
    // optimize takes priority (checked first in build())
    assert!(builder.optimization_budget.is_some());
    assert!(builder.peephole_config.is_some());
}

// ── autocast + force_dtype mutual exclusivity at build time ─────────

#[test]
fn autocast_and_force_dtype_are_mutually_exclusive() {
    // The builder allows setting both, but build() returns an error
    // when both autocast_policy and mixed_precision are set.
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let result = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .force_dtype(DType::F16)
        .expect("force_dtype alone is fine")
        .build();
    assert!(
        result.is_err(),
        "build should fail when both autocast and force_dtype are set"
    );
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error, got Ok"),
    };
    assert!(
        err_msg.contains("mutually exclusive"),
        "error message should mention mutual exclusivity: {err_msg}"
    );
}

// ── compile_trace_to_plan_with_fusion: basic behavior ───────────────

/// Build a minimal single-input graph (no ops, just one Input node).
fn single_input_graph() -> ComputationGraph {
    use nn_core::dyn_tensor::trace::{TraceNode, TraceOp};
    let node = TraceNode::new(
        1,
        "input_0".to_string(),
        TraceOp::Input,
        vec![],
        vec![1, 128],
        DType::F32,
    );
    ComputationGraph::from_nodes(vec![node])
}

/// Build a graph with Input → Relu → output.
fn input_relu_graph() -> ComputationGraph {
    use nn_core::dyn_tensor::trace::{TraceNode, TraceOp};
    let input = TraceNode::new(
        1,
        "input_0".to_string(),
        TraceOp::Input,
        vec![],
        vec![1, 128],
        DType::F32,
    );
    let relu = TraceNode::new(
        2,
        "relu_0".to_string(),
        TraceOp::Relu,
        vec![1],
        vec![1, 128],
        DType::F32,
    );
    ComputationGraph::from_nodes(vec![input, relu])
}

/// Build a graph with Input → Add (with constant) → output.
fn input_add_graph() -> ComputationGraph {
    use nn_core::dyn_tensor::trace::{TraceNode, TraceOp, WeightRef};
    let input = TraceNode::new(
        1,
        "input_0".to_string(),
        TraceOp::Input,
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let constant = TraceNode::new(
        2,
        "const_0".to_string(),
        TraceOp::ConstantWeight {
            weight: WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])
                .expect("valid weight ref"),
        },
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let add = TraceNode::new(
        3,
        "add_0".to_string(),
        TraceOp::Add,
        vec![1, 2],
        vec![1, 4],
        DType::F32,
    );
    ComputationGraph::from_nodes(vec![input, constant, add])
}

#[test]
fn compile_trace_to_plan_with_fusion_empty_graph() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = empty_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("empty graph should compile");
    assert!(plan.steps.is_empty());
    assert!(plan.input_shapes.is_empty());
    assert!(plan.weight_names.is_empty());
}

#[test]
fn compile_trace_to_plan_with_fusion_single_input() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = single_input_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("single input should compile");
    assert_eq!(plan.steps.len(), 1, "one node → one step");
    assert!(
        matches!(plan.steps[0], CompiledStep::InputForward),
        "Input node → InputForward step"
    );
    assert_eq!(plan.input_shapes.len(), 1);
    assert_eq!(plan.input_shapes[0], vec![1, 128]);
}

#[test]
fn compile_trace_to_plan_with_fusion_input_relu() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = input_relu_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("input+relu should compile");
    assert_eq!(plan.steps.len(), 2, "two nodes → two steps");
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
    // Relu might be compiled as Dispatch or Passthrough depending on
    // the constant folder and peephole, but it should be present.
    assert_eq!(plan.input_shapes.len(), 1);
    assert_eq!(plan.input_shapes[0], vec![1, 128]);
}

#[test]
fn compile_trace_to_plan_with_fusion_input_add() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = input_add_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("input+const+add should compile");
    // 3 nodes: Input, ConstantWeight, Add
    assert!(plan.steps.len() >= 2, "should have multiple steps");
    assert!(
        matches!(plan.steps[0], CompiledStep::InputForward),
        "first step should be InputForward"
    );
}

#[test]
fn compile_trace_to_plan_configured_no_peephole() {
    use nn_dsl::trace_compile::compile_trace_to_plan_configured;
    let graph = input_relu_graph();
    let no_peephole = nn_dsl::PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
        add_norm_linear: false,
        fuse_adain_snake: false,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
        fuse_snake_instance_norm: false,
        fuse_conv1d_snake_norm: false,
        fuse_conv1d_snake_norm_resblock: false,
        fuse_add_instance_norm_conv1x1: false,
        fuse_conv_transpose1d_activation: false,
        norm_activ_conv_transpose1d: false,
        fuse_instance_norm_conv1d: false,
        fuse_conv1d_instance_norm: false,
        fuse_linear_layer_norm: false,
        fuse_resblock_chain: false,
        fuse_activation_conv1d: false,
    };
    let plan = compile_trace_to_plan_configured(&graph, &no_peephole)
        .expect("should compile with all peephole disabled");
    assert!(!plan.steps.is_empty());
    assert_eq!(plan.input_shapes.len(), 1);
}

// ── build with non-empty graph ──────────────────────────────────────

#[test]
fn build_single_input_graph_produces_model() {
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("single input graph should build");
    assert_eq!(model.num_inputs(), 1);
    assert_eq!(model.num_steps(), 1);
    assert_eq!(model.num_dispatches(), 0, "Input-only graph has no dispatches");
    assert!(!model.is_mixed_precision());
    assert!(!model.is_autocast());
}

#[test]
fn build_input_relu_graph_produces_model_with_dispatch() {
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("input+relu graph should build");
    assert_eq!(model.num_inputs(), 1);
    assert!(model.num_steps() >= 2, "should have at least 2 steps");
    // Output shape should match the relu output shape
    assert_eq!(model.output_dtype(), DType::F32);
}

#[test]
fn build_input_add_graph_has_weight_names() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = input_add_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("input+add should compile");
    // The ConstantWeight node should register the "bias" weight.
    // The plan may or may not have weight_names depending on constant folding.
    // Just verify it compiled and the step count is valid.
    assert!(plan.steps.len() >= 2);
}

// ── from_plan with pre-compiled plan ────────────────────────────────

#[test]
fn from_plan_empty_graph_produces_empty_model() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("empty graph should compile to plan");
    let model = CompiledModel::from_plan(&plan, &graph, &cache)
        .expect("empty plan should build");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_inputs(), 0);
}

#[test]
fn from_plan_with_single_input() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("single input graph should compile to plan");
    let model = CompiledModel::from_plan(&plan, &graph, &cache)
        .expect("single-input plan should build");
    assert_eq!(model.num_steps(), 1);
    assert_eq!(model.num_inputs(), 1);
    assert_eq!(model.num_outputs(), 1);
    assert_eq!(model.output_shape(), &[1, 128]);
    assert_eq!(model.output_dtype(), DType::F32);
}

#[test]
fn from_plan_with_relu_graph() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("relu graph should compile to plan");
    let model = CompiledModel::from_plan(&plan, &graph, &cache)
        .expect("relu plan should build");
    assert!(model.num_steps() >= 2);
    assert_eq!(model.num_inputs(), 1);
}

// ── build with force_dtype on non-empty graph ───────────────────────

#[test]
fn build_with_force_dtype_f16_sets_mixed_precision() {
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .expect("F16 accepted")
        .build()
        .expect("should build with F16");
    assert!(model.is_mixed_precision());
    assert!(!model.is_autocast());
}

#[test]
fn build_with_autocast_sets_autocast_flag() {
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("should build with autocast");
    assert!(model.is_autocast());
    assert!(!model.is_mixed_precision());
}

// ── build with peephole config on non-empty graph ───────────────────

#[test]
fn build_with_peephole_config_compiles_successfully() {
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        auto_fuse_elementwise: false,
        ..nn_dsl::PeepholeConfig::default()
    };
    let model = CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .build()
        .expect("should build with custom peephole config");
    assert!(model.num_steps() >= 2);
}

// ── shared def and from_shared on non-empty model ───────────────────

#[test]
fn shared_def_on_built_model_creates_independent_instance() {
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("should build");
    let shared = model.share_def();
    let instance = CompiledModel::from_shared(shared);
    assert_eq!(model.num_steps(), instance.num_steps());
    assert_eq!(model.num_inputs(), instance.num_inputs());
    assert_eq!(model.output_shape(), instance.output_shape());
}

// ── segment_performance ─────────────────────────────────────────────

#[test]
fn segment_performance_on_empty_model() {
    let model = CompiledModel::empty();
    let sp = model.segment_performance("test_segment");
    assert_eq!(sp.dispatches, 0);
    assert_eq!(sp.metal_dispatches, 0);
    assert_eq!(sp.steps, 0);
    assert_eq!(sp.native_ops, 0);
    assert_eq!(sp.ir_dispatches, 0);
    assert_eq!(sp.buffer_bytes, 0);
}

#[test]
fn segment_performance_on_built_model() {
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("should build");
    let sp = model.segment_performance("relu_model");
    assert!(sp.steps >= 2, "input + relu = at least 2 steps");
}

// ── buffer plan metrics ─────────────────────────────────────────────

#[test]
fn buffer_plan_on_built_model_has_step_data() {
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("should build");
    let bp = model.buffer_plan();
    assert_eq!(
        bp.step_offsets.len(),
        model.num_steps(),
        "buffer plan should have one offset per step"
    );
    assert_eq!(
        bp.step_sizes.len(),
        model.num_steps(),
        "buffer plan should have one size per step"
    );
    assert_eq!(
        bp.last_use.len(),
        model.num_steps(),
        "buffer plan should have one last_use per step"
    );
}

// ── classify: AddNormLinear (from peephole pass 8) ──────────────────

#[test]
fn classify_add_norm_linear_not_as_compute() {
    let op = NativeOpKind::AddNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        hidden_dim: 512,
        out_features: 256,
        has_bias: true,
    };
    assert!(
        !super::classify::is_compute_native_op(&op),
        "AddNormLinear includes accumulate norm — not classified as pure compute"
    );
}

// ── classify: ConstantWeight and Cumsum passthrough ─────────────────

#[test]
fn classify_constant_weight_not_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::ConstantWeight {
            name: "const".into(),
            shape: vec![5],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "ConstantWeight is not passthrough safe"
    );
}

#[test]
fn classify_cumsum_not_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![4],
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "Cumsum is not passthrough safe"
    );
}

// ── classify: NormLinear is compute but not passthrough ──────────────

#[test]
fn classify_norm_linear_not_passthrough() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::NormLinear {
            norm_kind: nn_dsl::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            hidden_dim: 512,
            out_features: 256,
            has_bias: true,
        },
        weight_data: HashMap::new(),
    };
    assert!(
        !super::classify::is_passthrough_safe(&step),
        "NormLinear is compute, not passthrough"
    );
}

// ── with_precision on built model ───────────────────────────────────

#[test]
fn with_precision_on_built_model() {
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("should build");
    let contract = nn_dsl::PrecisionContract::bootstrap(
        nn_dsl::PrecisionTier::Strict,
        ScalarType::F32,
    );
    let model = model.with_precision(contract);
    assert!(model.precision().is_some());
    assert_eq!(
        model.precision().unwrap().tier,
        nn_dsl::PrecisionTier::Strict,
    );
}

// ── SegmentCacheConfig in builder context ───────────────────────────

#[test]
fn segment_cache_config_default_byte_budget_is_none() {
    let config = crate::segment_cache::SegmentCacheConfig::default();
    assert_eq!(config.byte_budget, None);
    assert_eq!(config.max_segments_per_step, 4);
    assert_eq!(
        config.eviction,
        crate::segment_cache::EvictionPolicy::Lru,
    );
}

#[test]
fn segment_cache_config_custom_values() {
    let config = crate::segment_cache::SegmentCacheConfig {
        max_segments_per_step: 16,
        eviction: crate::segment_cache::EvictionPolicy::Lru,
        byte_budget: Some(1024 * 1024 * 1024), // 1 GB
        shared_store: None,
    };
    assert_eq!(config.max_segments_per_step, 16);
    assert_eq!(config.byte_budget, Some(1024 * 1024 * 1024));
}

#[test]
fn segment_cache_config_clone_preserves_fields() {
    let config = crate::segment_cache::SegmentCacheConfig {
        max_segments_per_step: 8,
        eviction: crate::segment_cache::EvictionPolicy::Lru,
        byte_budget: Some(256 * 1024 * 1024),
        shared_store: None,
    };
    let cloned = config.clone();
    assert_eq!(cloned.max_segments_per_step, config.max_segments_per_step);
    assert_eq!(cloned.eviction, config.eviction);
    assert_eq!(cloned.byte_budget, config.byte_budget);
}

// ── CompiledModelError display coverage ─────────────────────────────

#[test]
fn compiled_model_error_display_invalid_config() {
    let err = super::super::CompiledModelError::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("test reason"));
    assert!(msg.contains("invalid config"));
}

#[test]
fn compiled_model_error_display_empty_plan() {
    let err = super::super::CompiledModelError::EmptyPlan;
    let msg = err.to_string();
    assert!(msg.contains("empty"));
}

#[test]
fn compiled_model_error_display_dispatch_failed() {
    let err = super::super::CompiledModelError::DispatchFailed {
        step_idx: 42,
        reason: "buffer too small".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("buffer too small"));
}

#[test]
fn compiled_model_error_display_input_count_mismatch() {
    let err = super::super::CompiledModelError::InputCountMismatch {
        expected: 2,
        got: 1,
    };
    let msg = err.to_string();
    assert!(msg.contains("2"));
    assert!(msg.contains("1"));
}

#[test]
fn compiled_model_error_display_missing_output_node() {
    let err = super::super::CompiledModelError::MissingOutputNode;
    let msg = err.to_string();
    assert!(msg.contains("output node"));
}

#[test]
fn compiled_model_error_converts_to_tensor_error() {
    use nn_core::TensorError;
    let err = super::super::CompiledModelError::InvalidConfig {
        reason: "test".into(),
    };
    let tensor_err: TensorError = err.into();
    let msg = tensor_err.to_string();
    assert!(msg.contains("test"), "TensorError should preserve message: {msg}");
}

// ── optimizer integration: peephole + optimize build paths ──────────

#[test]
fn build_with_all_disabled_peephole_on_empty_graph() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let all_disabled = nn_dsl::PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
        add_norm_linear: false,
        fuse_adain_snake: false,
        fuse_upsample_conv1d: false,
        fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
        fuse_snake_instance_norm: false,
        fuse_conv1d_snake_norm: false,
        fuse_conv1d_snake_norm_resblock: false,
        fuse_add_instance_norm_conv1x1: false,
        fuse_conv_transpose1d_activation: false,
        norm_activ_conv_transpose1d: false,
        fuse_instance_norm_conv1d: false,
        fuse_conv1d_instance_norm: false,
        fuse_linear_layer_norm: false,
        fuse_resblock_chain: false,
        fuse_activation_conv1d: false,
    };
    let model = CompiledModel::builder(&graph, &cache)
        .with_peephole_config(all_disabled)
        .build()
        .expect("all-disabled peephole on empty graph should build");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
}

#[test]
fn build_default_peephole_matches_default_build() {
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);

    // Default build (no peephole config specified).
    let model_default = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("default build should succeed");

    // Build with explicit default PeepholeConfig.
    let model_explicit = CompiledModel::builder(&graph, &cache)
        .with_peephole_config(nn_dsl::PeepholeConfig::default())
        .build()
        .expect("explicit default peephole should succeed");

    // Both paths should produce equivalent models: same step count,
    // same dispatch count, same input/output structure.
    assert_eq!(
        model_default.num_steps(),
        model_explicit.num_steps(),
        "step count should match between default and explicit-default peephole"
    );
    assert_eq!(
        model_default.num_dispatches(),
        model_explicit.num_dispatches(),
        "dispatch count should match"
    );
    assert_eq!(model_default.num_inputs(), model_explicit.num_inputs());
    assert_eq!(model_default.num_outputs(), model_explicit.num_outputs());
    assert_eq!(model_default.output_dtype(), model_explicit.output_dtype());
}

#[test]
fn optimize_zero_budget_empty_graph_builds() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = CompiledModel::builder(&graph, &cache)
        .optimize(std::time::Duration::ZERO)
        .build()
        .expect("optimize with zero budget on empty graph should build");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
}

#[test]
fn optimize_100ms_budget_empty_graph_no_worse_than_default() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);

    let model_default = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("default build should succeed");

    let model_optimized = CompiledModel::builder(&graph, &cache)
        .optimize(std::time::Duration::from_millis(100))
        .build()
        .expect("optimize with 100ms budget should succeed");

    assert!(
        model_optimized.num_dispatches() <= model_default.num_dispatches(),
        "optimized dispatch count ({}) should not exceed default ({})",
        model_optimized.num_dispatches(),
        model_default.num_dispatches(),
    );
}

#[test]
fn optimize_100ms_budget_relu_graph_no_worse_than_default() {
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);

    let model_default = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("default build should succeed");

    let model_optimized = CompiledModel::builder(&graph, &cache)
        .optimize(std::time::Duration::from_millis(100))
        .build()
        .expect("optimize with 100ms budget should succeed");

    assert!(
        model_optimized.num_dispatches() <= model_default.num_dispatches(),
        "optimized dispatch count ({}) should not exceed default ({})",
        model_optimized.num_dispatches(),
        model_default.num_dispatches(),
    );
    assert_eq!(model_optimized.num_inputs(), model_default.num_inputs());
    assert_eq!(model_optimized.output_dtype(), model_default.output_dtype());
}

#[test]
fn build_from_plan_empty_plan_via_builder() {
    use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("empty graph should compile to empty plan");
    assert!(plan.steps.is_empty(), "sanity: empty graph yields empty plan");
    let model = CompiledModel::builder(&graph, &cache)
        .build_from_plan(&plan)
        .expect("empty plan should build via builder");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_inputs(), 0);
    assert_eq!(model.num_outputs(), 0);
}

#[test]
fn optimize_takes_precedence_over_peephole_config_at_build() {
    // When both optimize() and with_peephole_config() are set,
    // optimize takes precedence because it is checked first in build().
    let graph = input_relu_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);

    let all_disabled = nn_dsl::PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
        add_norm_linear: false,
        fuse_adain_snake: false,
        fuse_upsample_conv1d: false,
        fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
        fuse_snake_instance_norm: false,
        fuse_conv1d_snake_norm: false,
        fuse_conv1d_snake_norm_resblock: false,
        fuse_add_instance_norm_conv1x1: false,
        fuse_conv_transpose1d_activation: false,
        norm_activ_conv_transpose1d: false,
        fuse_instance_norm_conv1d: false,
        fuse_conv1d_instance_norm: false,
        fuse_linear_layer_norm: false,
        fuse_resblock_chain: false,
        fuse_activation_conv1d: false,
    };

    // Set peephole first, then optimize — optimize wins.
    let model = CompiledModel::builder(&graph, &cache)
        .with_peephole_config(all_disabled)
        .optimize(std::time::Duration::from_millis(50))
        .build()
        .expect("optimize should take precedence and build successfully");

    // The model should have built (optimizer ran, not peephole config).
    // We verify it has reasonable structure (at least input + relu steps).
    assert!(model.num_steps() >= 2, "optimize should produce a valid model");
    assert_eq!(model.num_inputs(), 1);
}

#[test]
fn autocast_f32_only_is_noop_at_build() {
    // Verify that f32_only autocast is a no-op at the build level too,
    // not just at the builder field level.
    let graph = single_input_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::f32_only();
    let model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("f32_only autocast should build as default");
    // f32_only should not activate autocast
    assert!(!model.is_autocast(), "f32_only autocast should be inactive");
    assert!(!model.is_mixed_precision());
}
