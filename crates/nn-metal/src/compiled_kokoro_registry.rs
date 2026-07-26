// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Component registry for the compiled Kokoro pipeline.
//!
//! Makes implicit pipeline knowledge explicit and test-checkable. Every
//! NativeOp kernel, CPU bridge, sync point, and compiled segment is
//! registered with its source file, usage status, and expected wiring.
//!
//! Tests assert:
//! - No unwired fused kernels (expected_stages ⊆ kokoro_stages)
//! - No undocumented CPU bridges (all known bridges registered)
//! - Sync point count matches pipeline documentation
//!
//! Part of #2923, #2218.

// Registry data is consumed by tests in compiled_kokoro_registry_tests.rs.
// The statics are intentionally pub(crate) for cross-module test access.
#![allow(dead_code)]

/// A NativeOp kernel registered in the Kokoro pipeline.
pub(crate) struct KernelEntry {
    /// NativeOpKind variant name (e.g., "FusedResBlock").
    pub kind: &'static str,
    /// Source file for the executor dispatch.
    pub dispatch_file: &'static str,
    /// Which compiled Kokoro segments currently use this kernel.
    pub kokoro_stages: &'static [&'static str],
    /// Which segments SHOULD use this kernel (design intent).
    pub expected_stages: &'static [&'static str],
}

/// A CPU bridge in the compiled pipeline (operation that touches CPU).
pub(crate) struct CpuBridgeEntry {
    /// Human-readable name.
    pub name: &'static str,
    /// Source file and approximate line.
    pub file_line: &'static str,
    /// Why this operation runs on CPU.
    pub reason: &'static str,
    /// Issue tracking GPU replacement (if any).
    pub gpu_replacement: Option<&'static str>,
    /// Whether this bridge can be eliminated.
    pub eliminable: bool,
}

/// A GPU-CPU sync point (flush / readback / commit_and_wait).
#[allow(dead_code)] // fields are documentation, read by humans not just tests
pub(crate) struct SyncPointEntry {
    /// Human-readable name.
    pub name: &'static str,
    /// Source file and approximate line.
    pub file_line: &'static str,
    /// What triggers this sync point.
    pub trigger: &'static str,
    /// Whether this sync point can be eliminated.
    pub eliminable: bool,
    /// Issue tracking elimination (if any).
    pub replacement_issue: Option<&'static str>,
}

/// A compiled GPU segment in the pipeline.
#[allow(dead_code)] // fields are documentation, read by humans not just tests
pub(crate) struct SegmentEntry {
    /// Segment name (e.g., "seg_plbert").
    pub name: &'static str,
    /// Pipeline step that uses this segment.
    pub step: &'static str,
    /// LRU cache key dimension.
    pub cache_key: &'static str,
    /// Key NativeOps used within this segment.
    pub native_ops: &'static [&'static str],
}

// ---------------------------------------------------------------------------
// Registry constants — ground truth for the compiled Kokoro pipeline.
// ---------------------------------------------------------------------------

/// All 41 NativeOpKind variants with their Kokoro pipeline usage.
///
/// When adding a new NativeOp, add an entry here. The `test_no_unwired_kernels`
/// test will fail if `expected_stages` has entries not in `kokoro_stages`.
pub(crate) static KERNEL_REGISTRY: &[KernelEntry] = &[
    KernelEntry {
        kind: "LstmSequence",
        dispatch_file: "compiled_model_execute_native.rs",
        kokoro_stages: &["text_encoder", "prosody", "f0_energy"],
        expected_stages: &["text_encoder", "prosody", "f0_energy"],
    },
    KernelEntry {
        kind: "Cumsum",
        dispatch_file: "compiled_model_execute_native.rs",
        // Cumsum is used in harmonic_source (eager bridge, not compiled segment).
        // The NativeOpKind variant exists for other compiled models.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "InstanceNorm",
        dispatch_file: "compiled_model_execute_native.rs",
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "LayerNorm",
        dispatch_file: "compiled_model_execute_native.rs",
        kokoro_stages: &["plbert", "prosody"],
        expected_stages: &["plbert", "prosody"],
    },
    KernelEntry {
        kind: "AddLayerNorm",
        dispatch_file: "compiled_model_execute_native_add_ln.rs",
        // Peephole pass 6 fuses BinaryAdd + LayerNorm in PlBert transformer
        // layers (post-attention and post-FFN residual connections).
        // Saves 1 dispatch per fusion site. Part of #1815 Tier 5 D2.
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "AdainSnake",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Standalone AdainSnake is subsumed by NormActivConv1d and FusedResBlock
        // in the generator. May appear if peephole doesn't fire.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "AdainLeakyRelu",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Standalone AdainLeakyRelu is subsumed by NormActivConv1d and
        // FusedResBlock in f0_energy. May appear if peephole doesn't fire.
        kokoro_stages: &["f0_energy"],
        expected_stages: &["f0_energy"],
    },
    KernelEntry {
        kind: "AdaLayerNorm",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        kokoro_stages: &["prosody"],
        expected_stages: &["prosody"],
    },
    KernelEntry {
        kind: "FlashAttention",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "MaxPool1d",
        dispatch_file: "compiled_model_execute_native.rs",
        // MaxPool1d is for PyanNet (speaker segmentation), not Kokoro.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "ConstantWeight",
        dispatch_file: "compiled_model_execute_native.rs",
        // ConstantWeight is for pre-computed arange constants. Used in
        // harmonic_source (eager, not compiled). No compiled Kokoro segment
        // currently generates ConstantWeight NativeOps. Verify with
        // KOKORO_WEIGHTS if segments grow arange calls.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "FusedResBlock",
        dispatch_file: "compiled_model_execute_native_resblock.rs",
        // Peephole optimizer handles both Generator (Snake) and F0 (LeakyRelu)
        // ResBlock patterns. See trace_compile_peephole_resblock.rs.
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "NormActivConv1d",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "LinearActivation",
        dispatch_file: "compiled_model_execute_native_simple.rs",
        // Peephole pass 5 fuses Linear + activation (e.g., Linear + GELU
        // in PlBert FFN layers). Part of #2256.
        kokoro_stages: &["plbert", "harmonic_source"],
        expected_stages: &["plbert", "harmonic_source"],
    },
    KernelEntry {
        kind: "NormLinear",
        dispatch_file: "compiled_model_execute_native_norm_linear.rs",
        // Peephole pass 7 fuses LayerNorm/RmsNorm + Linear. In PlBert,
        // AddLayerNorm (pass 6) may consume LayerNorms first. Effective
        // usage depends on pass ordering and model topology.
        // Guard: hidden_dim <= 7680 (Metal threadgroup memory).
        // Part of #3089.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "BatchedStyleProjection",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Peephole pass 4 batches per-block style projections into a single
        // matmul across all FusedResBlocks in a segment.
        // Part of #1815 Tier 1.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "BatchedLinearProjection",
        dispatch_file: "compiled_model_execute_native_batched.rs",
        // Peephole pass 12 batches parallel Q/K/V linear projections into a
        // single GEMM. Saves 2 dispatches per attention layer. Part of #3269.
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "ProjectionSlice",
        dispatch_file: "compiled_model_execute_native_batched.rs",
        // GPU narrow from batched projection output. Paired with
        // BatchedLinearProjection. Part of #3269.
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "ChannelsFirstLayerNorm",
        dispatch_file: "compiled_model_execute_native_simple.rs",
        // Peephole pass 13 eliminates transpose pairs around LayerNorm by
        // computing normalization directly on channels-first [B, C, T] layout.
        // Saves 2 dispatches per site. Part of #3457.
        kokoro_stages: &["text_encoder"],
        expected_stages: &["text_encoder"],
    },
    KernelEntry {
        kind: "Int8Gemm",
        dispatch_file: "compiled_model_execute_native_simple.rs",
        // INT8 W8A16 dequantizing matmul. Not used in Kokoro (F32/F16 models).
        // Available for quantized model deployment. Part of #3522.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "SiluMul",
        dispatch_file: "compiled_model_execute_native_simple.rs",
        // Fused Silu + Mul for SwiGLU MLP blocks. Not yet used in Kokoro
        // (Kokoro has no SwiGLU layers). Available for Qwen3/GLM5. Part of #3521.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "Conv1dGemm",
        dispatch_file: "compiled_model_execute_native_simple.rs",
        // Conv1d via im2col + GEMM for large kernels. Used in generator
        // and f0_energy for strided convolutions. Part of #3351.
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "RotaryEmbedding",
        dispatch_file: "compiled_model_execute_native.rs",
        // RoPE for transformer attention layers. Not used in Kokoro
        // (Kokoro has no RoPE). Available for Qwen3/GLM5. Part of #3526.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "AddNormLinear",
        dispatch_file: "compiled_model_execute_native_norm_linear.rs",
        // Fused Add + LayerNorm + Linear. Peephole pass 8 fuses
        // AddLayerNorm + Linear in PlBert transformer layers.
        // Part of #3351 T2.1.
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "MoeGating",
        dispatch_file: "compiled_model_execute_native.rs",
        // MoE top-k gating for mixture-of-experts models. Not used in
        // Kokoro. Available for Qwen3 MoE. Part of #3542.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "FusedAdainSnake",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused InstanceNorm + affine + Snake from trace-level pattern detection.
        // Targets generator AdaIN+Snake blocks not absorbed by FusedResBlock or
        // NormActivConv1d. Part of #4252.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedInstanceNormMulAdd",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused InstanceNorm + Mul + Add from trace-level pattern detection.
        // Targets generator and other segments with AdaIN blocks that lack
        // a following Snake or Conv1d. Part of #4252.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedUpsampleConv1d",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused upsample1d + conv1d. Targets f0_energy segment where 6
        // upsample+conv pairs are used. Part of #4310.
        kokoro_stages: &["f0_energy"],
        expected_stages: &["f0_energy"],
    },
    KernelEntry {
        kind: "FusedConv1dActivation",
        dispatch_file: "compiled_model_execute_native_conv1d_activation.rs",
        // Fused Conv1d + Activation (Snake, ReLU, LeakyReLU, SiLU).
        // Targets generator and f0_energy segments with standalone Conv1d
        // followed by activation that are not part of NormActivConv1d or
        // FusedResBlock patterns. Part of #4264.
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "BiLstmCat",
        dispatch_file: "compiled_model_execute_native_bilstm.rs",
        // Fused bidirectional LSTM + concatenation. Peephole pass 11 fuses
        // forward LSTM + reverse LSTM + Cat. Used in text_encoder, prosody,
        // and f0_energy BiLSTM layers. Part of #4252.
        kokoro_stages: &["text_encoder", "prosody", "f0_energy"],
        expected_stages: &["text_encoder", "prosody", "f0_energy"],
    },
    KernelEntry {
        kind: "FusedMulAdd",
        dispatch_file: "compiled_model_execute_native.rs",
        // Fused multiply-add (FMA). Not yet used in Kokoro.
        // Available for models with Mul+Add patterns. Part of #4431.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "FusedSiGLU",
        dispatch_file: "compiled_model_execute_native.rs",
        // Fused SiGLU (Swish). Not yet used in Kokoro.
        // Available for models with Sigmoid+Mul patterns. Part of #4431.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "FusedGeGLU",
        dispatch_file: "compiled_model_execute_native.rs",
        // Fused GeGLU. Not yet used in Kokoro.
        // Available for Qwen3/GLM5 with GELU+Mul patterns. Part of #4431.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "FusedLayerNormLinear",
        dispatch_file: "compiled_model_execute_native_norm_linear.rs",
        // Fused LayerNorm + Linear. Targets PlBert attention projections
        // where LayerNorm is followed by a Linear. Saves ~12 dispatches
        // in encoder segments. Part of #4252.
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "BatchNorm2d",
        dispatch_file: "compiled_model_execute_native_simple.rs",
        // Fused BatchNorm2d for CNN models (ResNet, Table Transformer).
        // Not used in Kokoro (no BatchNorm layers). Available for
        // vision models. Part of #4324.
        kokoro_stages: &[],
        expected_stages: &[],
    },
    KernelEntry {
        kind: "FusedSnakeInstanceNorm",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused Snake activation + InstanceNorm. Targets generator blocks
        // where the deeper FusedResBlock or NormActivConv1d patterns don't
        // fire but Snake → InstanceNorm pairs remain. Part of #4264.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedConv1dSnakeNorm",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused Conv1d + Snake + InstanceNorm. Targets generator blocks
        // where the 3-step conv1d → snake → instance_norm pattern appears
        // outside deeper FusedResBlock / NormActivConv1d captures. Part of #4264.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedConv1dSnakeNormResBlock",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused 2x (Conv1d + Snake + InstanceNorm) + residual add. Targets
        // generator ResBlock patterns where two FusedConv1dSnakeNorm steps
        // feed into an add with the residual input. Reduces 7 dispatches
        // (2x conv + 2x snake + 2x norm + 1 add) to 1 NativeOp. Part of #4264.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedAddInstanceNormConv1x1",
        dispatch_file: "compiled_model_execute_native_fused.rs",
        // Fused Add + InstanceNorm + Conv1d(K=1). Targets decoder blocks
        // where x + h → instance_norm → 1x1 conv pattern appears.
        // Reduces 3 dispatches (add + norm + conv1d) to 1 NativeOp. Part of #4264.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedConvTranspose1dActivation",
        dispatch_file: "compiled_model_execute_native_conv_transpose1d_activation.rs",
        // Fused ConvTranspose1d + Activation (LeakyReLU, Snake). Targets
        // Kokoro Generator upsample stages (4× ConvTranspose1d stride=2 +
        // LeakyReLU) and F0EnergyPredictor upsampling blocks. These
        // ConvTranspose1d steps are not captured by FusedResBlock's pool_step.
        // Saves 1 dispatch per pair. Part of #4264.
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "NormActivConvTranspose1d",
        dispatch_file: "compiled_model_execute_native_norm_activ_conv_transpose1d.rs",
        // Fused AdainLeakyRelu/AdainSnake + ConvTranspose1d. Transposed-conv
        // dual of NormActivConv1d. Targets Kokoro Generator and F0EnergyPredictor
        // upsample stages where AdaIN normalization is followed by strided
        // ConvTranspose1d. Saves 1 dispatch per pair. Part of #4264.
        kokoro_stages: &["generator", "f0_energy"],
        expected_stages: &["generator", "f0_energy"],
    },
    KernelEntry {
        kind: "FusedInstanceNormConv1d",
        dispatch_file: "compiled_model_execute_native_fused_norm_conv.rs",
        // Fused InstanceNorm + Conv1d. 2-dispatch: stats + fused norm-conv.
        // Targets Kokoro Generator downsample stages. Part of #4264.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
    KernelEntry {
        kind: "FusedConv1dInstanceNorm",
        dispatch_file: "compiled_model_execute_native_fused_norm_conv.rs",
        // Fused Conv1d + InstanceNorm. 2-dispatch: conv + fused norm.
        // Targets Kokoro Generator and PlBert stages. Part of #4264.
        kokoro_stages: &["generator", "plbert"],
        expected_stages: &["generator", "plbert"],
    },
    KernelEntry {
        kind: "FusedLinearLayerNorm",
        dispatch_file: "compiled_model_execute_native_fused_norm_conv.rs",
        // Fused Linear + LayerNorm. Single fused kernel: GEMM + norm in threadgroup.
        // Up to 6 sites in PlBert (2 per layer x 3 layers). Part of #4264.
        kokoro_stages: &["plbert"],
        expected_stages: &["plbert"],
    },
    KernelEntry {
        kind: "FusedResBlockChain",
        dispatch_file: "compiled_model_execute_native_resblock_chain.rs",
        // Chains 2-4 consecutive FusedResBlocks into a single NativeOp.
        // Reduces dispatch count by eliminating inter-block plan overhead.
        // 24 FusedResBlocks in generator → 6-12 FusedResBlockChain ops.
        // Part of #4264.
        kokoro_stages: &["generator"],
        expected_stages: &["generator"],
    },
];

/// CPU bridges — operations in the compiled pipeline that touch CPU.
///
/// **All CPU bridges have been eliminated.** The pipeline is fully GPU from
/// step 1 through step 8. GPU→CPU transfer now occurs at the pipeline
/// boundary (`synthesize` exit via `to_device(&cpu())`), not mid-pipeline.
///
/// Historical bridges kept as documentation:
/// - `sinegen_cumsum_kahan_gpu`: eliminated by Kahan GPU cumsum (#2909).
/// - `istft_terminal_readback`: eliminated — iSTFT now returns GPU-resident
///   `DynTensor` via `gpu_istft_from_polar_gpu`. CPU readback moved to
///   pipeline exit boundary (#3351).
pub(crate) static CPU_BRIDGE_REGISTRY: &[CpuBridgeEntry] = &[];

/// GPU-CPU sync points in the pipeline hot path.
///
/// A sync point is any operation that forces the GPU command buffer to
/// commit and the CPU to wait for results. Fewer sync points = better
/// pipeline throughput.
pub(crate) static SYNC_POINT_REGISTRY: &[SyncPointEntry] = &[
    // sinegen_cumsum_roundtrip ELIMINATED by Kahan GPU cumsum (#2909).
    SyncPointEntry {
        name: "regulate_scalar_readback",
        file_line: "compiled_kokoro_step_regulate.rs:124",
        trigger: "submit()+sync() pipelined 4-byte readback for total_repeats",
        eliminable: false, // need output buffer size before scatter
        replacement_issue: None,
    },
    SyncPointEntry {
        name: "pipeline_exit_transfer",
        file_line: "compiled_kokoro_pipeline.rs:155",
        trigger: "audio.to_device(&cpu()) — single flush of all GPU work at pipeline boundary",
        eliminable: false, // inherent terminal sync
        replacement_issue: None,
    },
];

/// Compiled GPU segments with their cache keys and NativeOp composition.
pub(crate) static SEGMENT_REGISTRY: &[SegmentEntry] = &[
    SegmentEntry {
        name: "seg_plbert",
        step: "step_encode",
        cache_key: "seq_len",
        native_ops: &[
            "FlashAttention",
            "LayerNorm",
            "AddLayerNorm",
            "LinearActivation",
        ],
    },
    SegmentEntry {
        name: "seg_text",
        step: "step_encode",
        cache_key: "seq_len",
        native_ops: &["LstmSequence", "BiLstmCat"],
    },
    SegmentEntry {
        name: "seg_prosody",
        step: "step_predict_prosody",
        cache_key: "seq_len",
        native_ops: &["AdaLayerNorm", "LstmSequence", "LayerNorm", "BiLstmCat"],
    },
    SegmentEntry {
        name: "seg_f0",
        step: "step_predict_f0_energy",
        cache_key: "t_mel",
        native_ops: &[
            "FusedResBlock",
            "NormActivConv1d",
            "AdainLeakyRelu",
            "LstmSequence",
            "InstanceNorm",
            "BiLstmCat",
            "FusedUpsampleConv1d",
        ],
    },
    SegmentEntry {
        name: "seg_generator",
        step: "step_generate",
        cache_key: "total_samples",
        native_ops: &[
            "FusedResBlock",
            "NormActivConv1d",
            "AdainSnake",
            "FusedAdainSnake",
            "InstanceNorm",
            "BatchedStyleProjection",
        ],
    },
    SegmentEntry {
        name: "seg_regulate",
        step: "step_regulate",
        cache_key: "seq_len",
        // Pure elementwise chain — no NativeOps, no model weights.
        // Part of #1815 Tier 6 D2b.
        native_ops: &[],
    },
    SegmentEntry {
        name: "seg_sinegen_pre",
        step: "step_harmonic_source",
        cache_key: "t_frames",
        // Pre-cumsum SineGen: F0 → rad_frames + voiced mask.
        // Pure elementwise/index_select chain — no NativeOps.
        // Part of #1815 Tier 6 D2.
        native_ops: &[],
    },
    SegmentEntry {
        name: "seg_sinegen_post",
        step: "step_harmonic_source",
        cache_key: "t_frames",
        // Post-cumsum SineGen: phase → sin → linear → tanh → transpose.
        // LinearActivation peephole fuses linear+tanh.
        // Part of #1815 Tier 6 D3.
        native_ops: &["LinearActivation"],
    },
];

/// Expected number of pipeline-level sync points (hot path).
///
/// This constant is checked by `test_sync_point_count_matches_pipeline_docs`.
/// Update when sync points are added or removed.
pub(crate) const EXPECTED_SYNC_POINTS: usize = 2;

/// Total number of NativeOpKind variants.
///
/// Checked against the registry to ensure all variants are documented.
/// Bump when adding a new NativeOpKind variant.
pub(crate) const NATIVE_OP_VARIANT_COUNT: usize = 45;

#[cfg(test)]
#[path = "compiled_kokoro_registry_tests.rs"]
mod tests;
