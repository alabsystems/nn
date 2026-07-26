// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! nn-dsl — kernel IR, lowering, and code generation.
//!
//! This crate provides the core infrastructure for `#[nn::kernel]` and
//! `#[nn::model]`:
//!
//! - **`ir`**: The `KernelDef` intermediate representation — a DAG of typed
//!   scalar operations.
//! - **`lower`**: Lowers a Rust function AST (`syn::ItemFn`) to `KernelDef`.
//! - **`codegen_msl`**: Emits MSL (Metal Shading Language) from `KernelDef`.
//! - **`codegen_msl_tensor`**: Emits MSL for tensor-level operations (broadcast,
//!   reduce, dispatch plans).
//! - **`codegen_msl_structural`**: MSL codegen for structural tensor ops (reshape,
//!   slice, stack).
//! - **`codegen_msl_structural_conv`**: MSL codegen for convolution structural ops.
//! - **`codegen_msl_tensor_emit`**: Shared MSL emit helpers for tensor dispatch.
//! - **`codegen_kani`**: Emits a Kani verification harness from `KernelDef`.
//! - **`codegen_difftest`**: Emits differential test code (Rust vs MSL).
//! - **`tensor_ir`**: The `TensorKernelDef` — tensor-level IR with broadcast,
//!   reduce, and InstanceNorm operations.
//! - **`tensor_block_builder`**: Builder API for composing tensor op blocks.
//! - **`tensor_builders`**: Internal tensor builder implementations.
//! - **`instance_norm`**: InstanceNorm1d reference implementation and IR builders.
//! - **`adain`**: AdaIN and AdaIN-Snake fused kernel builders.
//! - **`conv1d`**: Conv1d (K1) kernel with stride, padding, and dilation.
//! - **`conv_transpose_1d`**: ConvTranspose1d (K4) transposed convolution kernel.
//! - **`gelu`**: GELU (Gaussian Error Linear Unit) activation kernel.
//! - **`layer_norm`**: LayerNorm (K7) kernel with two-pass variance + affine.
//! - **`lstm_decomposed`**: LSTM cell decomposed into primitive ops for GPU dispatch.
//! - **`rms_norm`**: RMSNorm (K5) kernel with sum-of-squares reduction + reciprocal sqrt.
//! - **`rope`**: RoPE (K6) rotary position embedding with cos/sin rotation pairs.
//! - **`silu_mul`**: SiLU-Mul (K8) elementwise kernel.
//! - **`sigmoid`**: Sigmoid (logistic) elementwise activation for GLU gates.
//! - **`snake`**: Hand-written Snake1d reference kernel (pre-proc-macro POC).
//! - **`kernel_error`**: Error types for kernel validation and scalar output checks.
//! - **`kernel_util`**: Internal utilities for kernel scalar validation.
//! - **`model_ir`** *(deprecated)*: The `ModelDef` — a graph of named function/kernel
//!   calls with tensor data flow (Level 3 IR for `#[nn::model]`). Superseded by
//!   the trace-based pipeline (`trace_compile` -> `CompiledModel`).
//! - **`lower_model`** *(deprecated)*: Lowers a model function AST to `ModelDef`.
//! - **[`precision`]**: Three-tier floating-point precision policy (strict /
//!   normal / relaxed) controlling MSL intrinsic selection, Metal fast-math
//!   flags, and differential test tolerance budgets.
//! - **`kernel_descriptor`**: `KernelDescriptor` bundling MSL source + metadata.
//! - **`kernel_ops`**: `KernelOps` trait for kernel operation dispatch.
//! - **`biquad`**: Biquad IIR filter kernel (Direct Form II Transposed) for
//!   peaking EQ, high-shelf, and bandpass filters.
//! - **`causal_conv1d`**: Causal Conv1d tensor builder — decomposes left-pad-only
//!   1D convolution into ZeroPad1d + Conv1d(padding=0).
//! - **`conv2d`**: Conv2d tensor builder for 2D convolution with stride, padding,
//!   dilation, and groups.
//! - **`ducker`**: Side-chain ducker kernel — envelope tracking with gain reduction
//!   above threshold.
//! - **`gated_delta_net`**: Gated DeltaNet cell decomposition into primitive tensor
//!   ops (MatMul, BinaryMul, BinaryAdd, Reshape) for GPU execution.
//! - **`input_names`**: Standard tensor input name constants for builder and dispatch
//!   APIs.
//! - **`linear`**: Builder for `TensorOpKind::Linear` (y = x @ W^T + b), mapping to
//!   NY's `LinearLayer`.
//! - **`relu`**: ReLU elementwise activation kernel (max(x, 0)).
//! - **`reverb`**: Reverb filter kernels — Freeverb lowpass-feedback comb filters and
//!   Schroeder allpass filters with Kani-proved stability.
//! - **`softmax`**: Softmax builder and reference implementation with log-sum-exp
//!   numerical stability.
//! - **`tanh_kernel`**: Tanh elementwise activation kernel for LSTM gate decomposition.
//! - **`trace_compile`**: Compiles a DynTensor `ComputationGraph` (from `trace_graph()`)
//!   into pre-built `TensorKernelDef` dispatch plans. Bridge between DynTensor runtime
//!   traces and TensorBlockBuilder Metal dispatch. Produces `CompiledPlan` with optional
//!   kernel fusion.
//! - **`buffer_planner`**: Static buffer planner for compiled model execution. Analyzes
//!   intermediate buffer lifetimes in a `CompiledPlan` and assigns byte offsets into a
//!   single contiguous GPU allocation using linear-scan register allocation.
//! - **`waveshaper`**: Audio waveshaper kernel — normalized tanh soft-clipping with
//!   guaranteed output in [-1, 1].
//! - **`weight_norm`**: Weight normalization reparameterization (w = g * v / ||v||)
//!   for Conv1d/Linear layers.

pub mod ada_layer_norm;
pub mod adain;
pub mod auto_fuse_codegen;
pub mod biquad;
pub mod buffer_planner;
pub mod causal_conv1d;
pub(crate) mod codegen_difftest;
pub(crate) mod codegen_kani;
pub(crate) mod codegen_msl;
pub(crate) mod codegen_msl_structural;
pub(crate) mod codegen_msl_structural_conv;
pub(crate) mod codegen_msl_tensor;
pub(crate) mod codegen_msl_tensor_emit;
pub mod codegen_shared;
pub mod codegen_shared_conv;
pub mod codegen_syntax;
pub mod conv1d;
pub mod conv2d;
pub mod conv_transpose_1d;
pub mod cost_model;
pub mod dispatch_plan_optimizer;
pub mod ducker;
pub mod edge_map;
pub mod gap_analysis_schema;
pub mod gated_delta_net;
pub mod gelu;
pub mod input_names;
pub mod instance_norm;
pub mod ir;
#[cfg(kani)]
#[path = "kani_auto_fuse_codegen.rs"]
mod kani_auto_fuse_codegen;
#[cfg(kani)]
#[path = "kani_auto_fuse_codegen_advanced.rs"]
mod kani_auto_fuse_codegen_advanced;
#[cfg(kani)]
mod kani_auto_fuse_codegen_issue3731;
#[cfg(kani)]
#[path = "kani_auto_fuse_codegen_proofs.rs"]
mod kani_auto_fuse_codegen_proofs;
#[cfg(kani)]
mod kani_codegen_msl;
#[cfg(kani)]
#[path = "kani_compiled_plan_serde.rs"]
mod kani_compiled_plan_serde;
#[cfg(kani)]
#[path = "kani_dispatch_step.rs"]
mod kani_dispatch_step;
#[cfg(kani)]
#[path = "kani_dsl_core_proofs.rs"]
mod kani_dsl_core_proofs;
#[cfg(kani)]
#[path = "kani_dsl_extra_proofs.rs"]
mod kani_dsl_extra_proofs;
#[cfg(kani)]
mod kani_fusion_proofs;
#[cfg(kani)]
#[path = "kani_ir_verifiability_enums.rs"]
mod kani_ir_verifiability_enums;
#[cfg(kani)]
mod kani_kernel_expansion;
#[cfg(kani)]
#[path = "kani_msl_auto_fuse.rs"]
mod kani_msl_auto_fuse;
#[cfg(kani)]
#[path = "kani_msl_auto_fuse_3745.rs"]
mod kani_msl_auto_fuse_3745;
#[cfg(kani)]
#[path = "kani_peephole_auto_fuse.rs"]
mod kani_peephole_auto_fuse;
#[cfg(kani)]
#[path = "kani_peephole_resblock_3738.rs"]
mod kani_peephole_resblock_3738;
#[cfg(kani)]
#[path = "kani_peephole_resblock_fusion.rs"]
mod kani_peephole_resblock_fusion;
#[cfg(kani)]
mod kani_peephole_resblock_issue3731;
#[cfg(kani)]
mod kani_precision;
#[cfg(any(test, kani))]
mod kani_reduce;
#[cfg(kani)]
pub(crate) mod kani_stubs;
#[cfg(kani)]
mod kani_trace_compile_native_ops_issue3731;
#[cfg(kani)]
#[path = "kani_trace_compile_ops_dispatch.rs"]
mod kani_trace_compile_ops_dispatch;
#[cfg(kani)]
#[path = "kani_trace_compile_peephole_resblock.rs"]
mod kani_trace_compile_peephole_resblock;
#[cfg(kani)]
#[path = "kani_trace_compile_peephole_resblock_advanced.rs"]
mod kani_trace_compile_peephole_resblock_advanced;
// Kani proofs for polar-to-rect scalar math (#2218 F14).
#[cfg(feature = "plan-serde")]
pub mod compiled_plan_io;
pub(crate) mod kernel_descriptor;
pub mod kernel_error;
pub(crate) mod kernel_ops;
pub(crate) mod kernel_util;
pub mod layer_norm;
pub mod linear;
pub mod lower;
pub(crate) mod lower_model;
pub mod lstm_decomposed;
pub(crate) mod nnc_header;
pub mod model_ir;
pub mod msl_auto_fuse;
pub mod norm_activ_conv_kernels;
pub mod partition_compiler;
#[cfg(feature = "plan-serde")]
pub mod peephole_config_persist;
pub mod performance_report;
#[cfg(all(kani, feature = "kani-stubbing"))]
#[path = "polar_to_rect_kani.rs"]
mod polar_to_rect_kani;
pub mod precision;
pub mod relu;
pub mod reverb;
pub mod rms_norm;
pub mod rope;
pub mod sigmoid;
pub mod silu_mul;
pub mod snake;
pub mod softmax;
pub mod tanh_kernel;
pub mod tensor_block_builder;
pub(crate) mod tensor_builders;
pub mod tensor_ir;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_kernels;
pub mod trace_compile;
pub mod verifiability;
pub mod waveshaper;
pub mod weight_norm;

#[cfg(test)]
#[path = "dsl_expanded_tests.rs"]
mod dsl_expanded_tests;

#[cfg(test)]
#[path = "fusion_chain_detection_tests.rs"]
mod fusion_chain_detection_tests;

#[cfg(test)]
#[path = "trace_compile_coverage_tests.rs"]
mod trace_compile_coverage_tests;

#[cfg(test)]
#[path = "peephole_pass_tests.rs"]
mod peephole_pass_tests;

#[cfg(test)]
#[path = "gap_analysis_tests.rs"]
mod gap_analysis_tests;

#[cfg(test)]
#[path = "peephole_fusion_extended_tests.rs"]
mod peephole_fusion_extended_tests;

#[cfg(test)]
#[path = "trace_compile_extended_tests.rs"]
mod trace_compile_extended_tests;

#[cfg(test)]
#[path = "codegen_msl_extended_tests.rs"]
mod codegen_msl_extended_tests;

#[cfg(test)]
#[path = "fusion_gap_extended_tests.rs"]
mod fusion_gap_extended_tests;

#[cfg(test)]
#[path = "trace_compile_extended_tests2.rs"]
mod trace_compile_extended_tests2;

#[cfg(test)]
#[path = "native_op_extended_tests.rs"]
mod native_op_extended_tests;

#[cfg(test)]
#[path = "native_op_tests.rs"]
mod native_op_tests;

#[cfg(test)]
#[path = "dsl_fusion_extended_tests.rs"]
mod dsl_fusion_extended_tests;

#[cfg(test)]
#[path = "dsl_fusion_detection_extended_tests.rs"]
mod dsl_fusion_detection_extended_tests;

#[cfg(test)]
#[path = "fused_upsample_conv1d_tests.rs"]
mod fused_upsample_conv1d_tests;

#[cfg(test)]
#[path = "dsl_ir_codegen_optim_extended_tests.rs"]
mod dsl_ir_codegen_optim_extended_tests;

/// Reduction intrinsic for the kernel subset: sum explicit scalar terms.
///
/// This is intentionally tiny and deterministic so kernels can call
/// `nn_dsl::sum_reduce([a, b, c])` in plain Rust while lowering emits a
/// dedicated `IRNodeKind::SumReduce`.
#[must_use]
pub fn sum_reduce<T, const N: usize>(values: [T; N]) -> T
where
    T: Copy + Default + std::ops::Add<Output = T>,
{
    let mut acc = T::default();
    for value in values {
        acc = acc + value;
    }
    acc
}

// Explicit re-exports — public API surface of each module (#677).
pub use ada_layer_norm::{
    ada_layer_norm_fused_scalar, adaptive_affine_scalar, build_ada_layer_norm_fused_kernel,
    build_adaptive_affine_kernel,
};
pub use adain::{
    adain_leaky_relu_fused_scalar, adain_scalar, adain_snake_fused_scalar, build_adain1d,
    build_adain_leaky_relu_fused_kernel, build_adain_scalar_kernel, build_adain_snake_fused_kernel,
    build_leaky_relu_scalar_kernel, build_snake_scalar_kernel, leaky_relu_scalar,
};
pub use auto_fuse_codegen::{
    auto_fuse_to_msl, compose_trace_ops_to_kernel_ir, AutoFusedKernel, FuseableOp, OpWiring,
};
pub use biquad::{
    biquad_bandpass, biquad_high_shelf, biquad_peaking, biquad_process_sample_scalar, BiquadCoeffs,
    BiquadSampleOutput, BIQUAD_MIN_Q,
};
pub use buffer_planner::{plan_buffers, plan_buffers_with_dtypes, BufferPlan};
pub use causal_conv1d::build_causal_conv1d;
pub use conv1d::{build_conv1d, build_conv1d_full};
pub use conv2d::{build_conv2d, build_conv2d_full};
pub use conv_transpose_1d::build_conv_transpose_1d;
pub use cost_model::{
    CalibrationData, CalibrationError, CalibrationRecord, CalibrationReport, CostEstimate,
    CostModel,
};
pub use dispatch_plan_optimizer::{
    build_optimizer_edge_map, optimize_dispatch_plan, optimize_dispatch_plan_unconstrained,
    DepEdge, DepGraph, OptimizedPlan,
};
pub use ducker::{
    ducker_process_sample_scalar, validate_ducker_config, DuckerCoeffs, DuckerOutput, DuckerState,
};
pub use edge_map::compute_edge_map;
pub use gap_analysis_schema::{
    fusion_gap_analysis_schema, GapAnalysisReport, GapAnalysisSegment, OptimizationRequest,
    OptimizationResponse, OptimizationSuggestion, PROTOCOL_VERSION,
};
pub use gated_delta_net::{
    build_gated_delta_net_decomposed, decompose_gated_delta_net, GatedDeltaNetOutputs,
};
pub use gelu::{build_gelu_kernel, gelu_scalar, gelu_scalar_bounds};
pub use instance_norm::{
    build_instance_norm, build_instance_norm_affine_scalar_kernel, build_instance_norm_decomposed,
    build_instance_norm_scalar_kernel, instance_norm_ref, instance_norm_scalar,
};
pub use ir::{
    ir_pretty_print, BinOpKind, CompareOpKind, IRError, IRNode, IRNodeKind, KernelDef, MinMaxKind,
    NodeId, Param, ScalarType, UnaryFnKind, ValueType, POWI_MAX_EXPONENT,
};
pub use kernel_error::KernelError;
pub use layer_norm::{
    build_layer_norm_decomposed, build_layer_norm_gelu_fused_kernel,
    build_layer_norm_scalar_kernel, layer_norm_gelu_fused_scalar, layer_norm_ref,
    layer_norm_scalar,
};
pub use linear::{build_linear, build_linear_batched};
pub use lower::{LowerError, Lowerer};
pub use lstm_decomposed::{build_lstm_cell_decomposed_dual, LstmCellOutputs};
#[allow(deprecated)]
pub use model_ir::{
    model_ir_pretty_print, ModelDef, ModelIRError, ModelOutput, ModelParam, ModelStep, ModelStepId,
    ModelValueRef,
};
pub use norm_activ_conv_kernels::{
    build_norm_leaky_relu_kernel, build_norm_leaky_relu_mul_fused_kernel, build_norm_snake_kernel,
    build_norm_snake_mul_fused_kernel, build_weight_mul_kernel, norm_leaky_relu_mul_fused_scalar,
    norm_leaky_relu_scalar, norm_snake_mul_fused_scalar, norm_snake_scalar, weight_mul_scalar,
};
pub use partition_compiler::{
    find_fusion_groups, partition_plan, partition_summary, DagNode, DagNodeKind, FusionGroup,
    PartitionDag, PartitionSummary,
};
pub use performance_report::{MemoryMetrics, PerformanceReport, SegmentPerformance};
pub use precision::{
    bootstrap_budget, differential_tolerance, within_differential_budget, InputBound,
    InputBoundParseError, InputBounds, PrecisionContract, PrecisionParseError, PrecisionTier,
};
pub use relu::build_relu_kernel;
pub use reverb::{
    allpass_process_sample_scalar, comb_process_sample_scalar, validate_allpass_config,
    validate_comb_config, AllpassCoeffs, AllpassOutput, CombCoeffs, CombOutput, CombState,
};
pub use rms_norm::{
    build_rms_norm, build_rms_norm_decomposed, build_rms_norm_scalar_kernel,
    build_rms_norm_silu_mul_fused_kernel, rms_norm_ref, rms_norm_scalar,
    rms_norm_silu_mul_fused_scalar,
};
pub use rope::{
    build_rope_cos_kernel, build_rope_rotate_kernel, build_rope_sin_kernel, rope_cos_scalar,
    rope_cos_scalar_bounds, rope_rotate_ref, rope_sin_scalar, rope_sin_scalar_bounds,
};
pub use sigmoid::{build_sigmoid_kernel, sigmoid_scalar, sigmoid_scalar_bounds};
pub use silu_mul::{build_silu_mul_kernel, silu_mul_scalar, silu_mul_scalar_bounds};
pub use snake::{snake_ref_f16, snake_ref_f32, snake_scalar, snake_scalar_bounds, SNAKE_MIN_ALPHA};
pub use softmax::build_softmax;
pub use tanh_kernel::{build_tanh_kernel, tanh_scalar, tanh_scalar_bounds};
pub use tensor_block_builder::{
    CrossAttentionBlockConfig, CrossAttentionBlockWeights, TensorBlockBuilder,
    TransformerBlockConfig, TransformerBlockWeights,
};
pub use tensor_ir::{
    infer_broadcast_alignment, tensor_ir_pretty_print, AttentionMask, BroadcastAlignment,
    Pool2dParams, ReduceOp, TensorIRConvError, TensorIRError, TensorIRLayerError, TensorKernelDef,
    TensorNode, TensorNodeId, TensorOpKind,
};
pub use trace_compile::{
    analyze_fusion_gaps, analyze_fusion_opportunities, analyze_pass_impact, compile_trace,
    compile_trace_to_plan, compile_trace_to_plan_configured, compile_trace_to_plan_with_fusion,
    compile_trace_with_fusion, count_dispatches, detect_fusion_chains, optimize_plan,
    optimize_plan_with_cost, optimize_segments, scan_fusion_opportunities,
    theoretical_minimum_dispatches, AttentionLayout, BufferPlanMetrics, CompiledKernel,
    CompiledPlan, CompiledStep, ConvActivation, FusedNormKind, FusionBlocker, FusionChainInfo,
    FusionGap, FusionGapAnalysis, FusionOpportunity, FusionPair, FusionScanResult, FusionStats,
    GemmActivation, NativeOpKind, NormActivConv1dParams, NormActivation, OptimizationResult,
    PassImpactEntry, PeepholeConfig, PeepholeStats, PlanDiff, PlanSummary, ResBlockChainEntry,
    RuntimeOpKind, ScannerFusionCategory, ScannerOpportunity, SegmentOptimizationResult,
    StyleBatchOffset, StyleProjectionParams,
};
pub use verifiability::{
    classify_callee_name, classify_op, VerifiabilityClass, VerifiabilitySummary,
};
pub use waveshaper::{tanh_waveshaper_scalar, tanh_waveshaper_scalar_bounds};

// Explicit re-exports for pub(crate) modules — only selected items
// are part of the crate's public API.
pub use codegen_difftest::emit_differential_test_with_bounds;
pub use codegen_kani::emit_kani_harness;
pub use codegen_msl::{
    emit_msl, emit_msl_with_contract, emit_scalar_fn, MAX_DIRECT_BINDING_INPUTS,
};
pub use codegen_msl_tensor::{
    build_dispatch_plan, build_dispatch_plan_full, tiled_transpose_2d_params, Conv1dParams,
    Conv2dParams, ConvTranspose1dParams, DispatchStep, SimdgroupLinearParams,
    SimdgroupMatMulParams, TensorMSLCodegenError, TiledLinearParams, TiledMatMulParams,
    TILED_GEMM_TILE, TILED_TRANSPOSE_TILE_SIZE,
};
pub use codegen_msl_tensor_emit::{
    emit_linear_activation_kernel, emit_simdgroup_linear_activation_kernel,
    emit_simdgroup_linear_standalone_kernel, emit_tensor_msl, emit_tensor_msl_with_contract,
    emit_tensor_msl_with_plan, gemm_activation_msl_var,
};
#[cfg(feature = "plan-serde")]
pub use compiled_plan_io::{load_plan, save_plan};
pub use kernel_descriptor::KernelDescriptor;
pub use kernel_ops::KernelOps;
#[allow(deprecated)]
pub use lower_model::{lower_model_fn, ModelLowerError};
pub use nnc_header::NncError;
pub use msl_auto_fuse::{
    generate_fused_msl, generate_fused_msl_with_contract, FusedKernelMeta, FusedMslError,
    FusedMslResult,
};
#[cfg(feature = "plan-serde")]
pub use peephole_config_persist::{
    load_optimization_result_summary, load_peephole_config, save_optimization_result_summary,
    save_peephole_config, OptimizationResultSummary, PeepholeConfigPersistError,
};

// ---------------------------------------------------------------------------
// Unified error supertype (Part of #690)
// ---------------------------------------------------------------------------

/// Unified error type for the nn-dsl pipeline.
///
/// Consumers building a tensor kernel pipeline (build → lower → codegen → verify)
/// encounter multiple error types with no common supertype. `NnDslError` provides
/// a `?`-friendly unifier: any function returning `Result<T, NnDslError>` can
/// propagate errors from scalar kernels, IR validation, lowering, tensor IR,
/// tensor codegen, and model IR stages.
///
/// Existing function signatures are unchanged — this is an opt-in supertype.
#[allow(deprecated)] // ModelIRError and ModelLowerError use deprecated ModelDef types
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NnDslError {
    /// Scalar kernel function error (e.g., NaN input, overflow).
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// IR validation error (e.g., unresolved parameter, type mismatch).
    #[error(transparent)]
    Ir(#[from] IRError),

    /// Lowering error (syn AST → KernelDef IR).
    #[error(transparent)]
    Lower(#[from] LowerError),

    /// Tensor IR validation error (e.g., shape mismatch, invalid node reference).
    #[error(transparent)]
    TensorIr(#[from] TensorIRError),

    /// Tensor MSL codegen error (e.g., unsupported reduce axis).
    #[error(transparent)]
    TensorCodegen(#[from] TensorMSLCodegenError),

    /// Model IR validation error.
    #[error(transparent)]
    ModelIr(#[from] ModelIRError),

    /// Model lowering error (model AST → ModelDef).
    #[error(transparent)]
    ModelLower(#[from] ModelLowerError),
}
