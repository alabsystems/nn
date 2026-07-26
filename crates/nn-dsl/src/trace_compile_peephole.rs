// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Post-compilation peephole optimizer for compiled step sequences.
//!
//! Detects step patterns that can be fused into single `NativeOp`
//! variants and replaces them in-place. Runs after elementwise chain
//! fusion but before `build_plan()`.
//!
//! **18 passes** (order matters — earlier passes create patterns for later ones):
//!
//! 0. **FusedAdainSnake** (pass 0): `InstanceNorm` + `Mul` + `Add` + `Snake`
//!    → single `FusedAdainSnake`. Runs first so standalone AdaIN+Snake blocks
//!    that lack a following Conv1d get fused (#4252).
//! 1. **NormActivConv1d** (pass 1): `AdainLeakyRelu`/`AdainSnake` + `Conv1d`
//!    → single `NormActivConv1d`. Cuts dispatches per block (#2780).
//! 2. **FusedResBlock** (pass 2): 2× NormActivConv1d + residual add → single
//!    `FusedResBlock`. Reduces per-layer overhead (#2218).
//! 3. **Style projection absorption** (pass 3): Absorbs `Linear` projections
//!    for gamma/beta into `FusedResBlock` (#2780).
//! 4. **Batch style projections** (pass 4): Batches per-block style projections
//!    into single matmul across `FusedResBlock`s (#1815 Tier 1, #2964).
//! 5. **LinearActivation** (pass 5): `Linear` + activation → fused GEMM
//!    epilogue (#2256).
//! 6. **AddLayerNorm** (pass 6): `BinaryAdd` + `LayerNorm` → fused
//!    `AddLayerNorm` (#1815 Tier 5 D2). Must run before NormLinear.
//! 7. **NormLinear** (pass 7): `LayerNorm`/`RmsNorm` + `Linear` → fused
//!    norm+GEMM (#3089).
//! 8. **AddNormLinear** (pass 8): `AddLayerNorm` + `Linear` → fused
//!    `AddNormLinear` (#3351 T2.1). Runs after passes 6+7 to catch
//!    AddLayerNorm+Linear pairs that NormLinear couldn't see.
//! 9. **Attention transpose absorption** (pass 9): Absorbs `Transpose(1,2)`
//!    around `FlashAttention` into `SeqFirst` layout (#1815).
//! 10. **Flip+LSTM absorption** (pass 10): Absorbs `Flip(dim=0)` around
//!     `LstmSequence` into reverse mode (#1815). Saves ~192 dispatches.
//! 11. **BiLstmCat** (pass 11): Fwd `LstmSequence` + Rev `LstmSequence` +
//!     `Cat` → single `BiLstmCat`. Must run after Flip+LSTM (#4252).
//! 12. **BatchedLinearProjection** (pass 12): Batches N parallel `Linear`
//!     projections sharing one input into a single matmul (#3269).
//! 13. **ChannelsFirstLayerNorm** (pass 13): Absorbs `Transpose(1,2)` around
//!     `LayerNorm` → normalize over dim 1 in channels-first layout (#3457).
//! 14. **SiluMul** (pass 14): Fuses `Silu` + `Mul` → single `silu_mul`
//!     kernel for SwiGLU MLP blocks (#3521).
//! 15. **AutoFuseElementwiseChains** (pass 15): Fuses remaining consecutive
//!     elementwise Dispatch steps into single composed kernels (#3517).
//!     Runs last so specific named patterns match first.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::tensor_ir::TensorOpKind;

use super::{CompiledStep, NativeOpKind, NormActivation};

#[path = "trace_compile_peephole_resblock.rs"]
mod resblock;

#[path = "trace_compile_peephole_add_ln.rs"]
mod add_ln;

#[path = "trace_compile_peephole_norm_linear.rs"]
mod norm_linear;

#[path = "trace_compile_peephole_attention.rs"]
mod attention;

#[path = "trace_compile_peephole_flip_lstm.rs"]
mod flip_lstm;

#[path = "trace_compile_peephole_linear_activation.rs"]
pub(crate) mod linear_activation;
use linear_activation::{extract_linear_params, LinearInfo};

#[path = "trace_compile_peephole_batched_qkv.rs"]
mod batched_qkv;

#[path = "trace_compile_peephole_conv_ln.rs"]
mod conv_ln;

#[path = "trace_compile_peephole_auto_fuse.rs"]
mod auto_fuse;

#[path = "trace_compile_peephole_silu_mul.rs"]
mod silu_mul_fuse;

#[path = "trace_compile_peephole_bilstm_cat.rs"]
mod bilstm_cat;

#[path = "trace_compile_peephole_add_norm_linear.rs"]
mod add_norm_linear;

#[path = "trace_compile_peephole_adain_snake.rs"]
mod adain_snake;

#[path = "trace_compile_peephole_upsample_conv1d.rs"]
mod upsample_conv1d;

#[path = "trace_compile_peephole_instance_norm_mul_add.rs"]
mod instance_norm_mul_add;

#[path = "trace_compile_peephole_conv1d_activation.rs"]
mod conv1d_activation;

#[path = "trace_compile_peephole_snake_instance_norm.rs"]
mod snake_instance_norm;

#[path = "trace_compile_peephole_conv1d_snake_norm.rs"]
mod conv1d_snake_norm;

#[path = "trace_compile_peephole_conv1d_snake_norm_resblock.rs"]
mod conv1d_snake_norm_resblock;

#[path = "trace_compile_peephole_add_instance_norm_conv1x1.rs"]
mod add_instance_norm_conv1x1;

#[path = "trace_compile_peephole_conv_transpose1d_activation.rs"]
mod conv_transpose1d_activation;

#[path = "trace_compile_peephole_norm_activ_conv_transpose1d.rs"]
mod norm_activ_conv_transpose1d;

#[path = "trace_compile_peephole_instance_norm_conv1d.rs"]
mod instance_norm_conv1d;

#[path = "trace_compile_peephole_conv1d_instance_norm.rs"]
mod conv1d_instance_norm;

#[path = "trace_compile_peephole_linear_layer_norm.rs"]
mod linear_layer_norm;

#[path = "trace_compile_peephole_activation_conv1d.rs"]
mod activation_conv1d;
#[path = "trace_compile_peephole_resblock_chain.rs"]
mod resblock_chain;

/// Per-pass enable flags for the peephole optimizer.
///
/// All passes are enabled by default. Disable individual passes to diagnose
/// performance regressions (e.g., if a specific fusion hurts GPU occupancy
/// for particular tensor shapes). See #3348.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "plan-serde", serde(default))]
pub struct PeepholeConfig {
    /// Pass 1: AdainSnake/LeakyRelu + Conv1d → NormActivConv1d
    pub norm_activ_conv1d: bool,
    /// Pass 2-4: FusedResBlock + style projection absorption + batching
    pub fused_resblock: bool,
    /// Pass 5: Linear + Activation → LinearActivation (GEMM epilogue)
    pub linear_activation: bool,
    /// Pass 6: BinaryAdd + LayerNorm → AddLayerNorm
    pub add_layer_norm: bool,
    /// Pass 7: LayerNorm/RmsNorm + Linear → NormLinear
    pub norm_linear: bool,
    /// Pass 9: Transpose(1,2) absorption into FlashAttention
    pub attention_transpose: bool,
    /// Pass 10: Flip(dim=0) absorption into LstmSequence reverse mode
    pub flip_lstm: bool,
    /// Pass 12: Batch parallel Linear projections into single matmul
    pub batched_linear_projection: bool,
    /// Pass 13: Transpose(1,2) + LayerNorm + Transpose(1,2) → ChannelsFirstLayerNorm
    pub channels_first_layer_norm: bool,
    /// Pass 14: Silu + Mul → SiluMul (SwiGLU MLP fusion)
    pub silu_mul: bool,
    /// Pass 15: Auto-fuse remaining consecutive elementwise Dispatch chains
    pub auto_fuse_elementwise: bool,
    /// Pass 11: Fwd LstmSequence + Rev LstmSequence + Cat → BiLstmCat
    pub bilstm_cat: bool,
    /// Pass 8: AddLayerNorm + Linear → AddNormLinear (fused add+norm+GEMM)
    pub add_norm_linear: bool,
    /// Pass 0: InstanceNorm + Mul + Add + Snake → FusedAdainSnake
    pub fuse_adain_snake: bool,
    /// Upsample1d + Conv1d → FusedUpsampleConv1d (#4310)
    pub fuse_upsample_conv1d: bool,
    /// InstanceNorm + Mul + Add → FusedInstanceNormMulAdd (#4252)
    pub fuse_instance_norm_mul_add: bool,
    /// Conv1d + Activation → FusedConv1dActivation (#4264)
    pub fuse_conv1d_activation: bool,
    /// Snake + InstanceNorm → FusedSnakeInstanceNorm (#4264)
    pub fuse_snake_instance_norm: bool,
    /// Conv1d + Snake + InstanceNorm → FusedConv1dSnakeNorm (#4264)
    pub fuse_conv1d_snake_norm: bool,
    /// 2× FusedConv1dSnakeNorm + add → FusedConv1dSnakeNormResBlock (#4264)
    pub fuse_conv1d_snake_norm_resblock: bool,
    /// Add + InstanceNorm + Conv1d(K=1) → FusedAddInstanceNormConv1x1 (#4264)
    pub fuse_add_instance_norm_conv1x1: bool,
    /// ConvTranspose1d + Activation → FusedConvTranspose1dActivation (#4264)
    pub fuse_conv_transpose1d_activation: bool,
    /// AdainLeakyRelu/AdainSnake + ConvTranspose1d → NormActivConvTranspose1d (#4264)
    pub norm_activ_conv_transpose1d: bool,
    /// InstanceNorm + Conv1d → FusedInstanceNormConv1d (#4264)
    pub fuse_instance_norm_conv1d: bool,
    /// Conv1d + InstanceNorm → FusedConv1dInstanceNorm (#4264)
    pub fuse_conv1d_instance_norm: bool,
    /// Linear + LayerNorm → FusedLinearLayerNorm (#4264)
    pub fuse_linear_layer_norm: bool,
    /// Chain 2-4 consecutive FusedResBlocks into FusedResBlockChain (#4264)
    pub fuse_resblock_chain: bool,
    /// Activation + Conv1d → FusedConv1dActivation (pre_activation=true) (#4264)
    pub fuse_activation_conv1d: bool,
}

impl Default for PeepholeConfig {
    fn default() -> Self {
        Self {
            norm_activ_conv1d: true,
            fused_resblock: true,
            linear_activation: true,
            add_layer_norm: true,
            norm_linear: true,
            attention_transpose: true,
            flip_lstm: true,
            batched_linear_projection: true,
            channels_first_layer_norm: true,
            silu_mul: true,
            auto_fuse_elementwise: true,
            bilstm_cat: true,
            add_norm_linear: true,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: true,
            fuse_instance_norm_mul_add: true,
            fuse_conv1d_activation: true,
            fuse_snake_instance_norm: true,
            fuse_conv1d_snake_norm: true,
            fuse_conv1d_snake_norm_resblock: true,
            fuse_add_instance_norm_conv1x1: true,
            fuse_conv_transpose1d_activation: true,
            norm_activ_conv_transpose1d: true,
            fuse_instance_norm_conv1d: true,
            fuse_conv1d_instance_norm: true,
            fuse_linear_layer_norm: true,
            fuse_resblock_chain: true,
            fuse_activation_conv1d: true,
        }
    }
}

/// Apply peephole optimizations to a compiled step sequence.
///
/// Runs 10 pattern-matching passes in order (see module-level docs).
/// Requires the computation graph for fan-out analysis (ensuring
/// fused operands are single-consumer).
pub(crate) fn apply_peephole(steps: &mut [CompiledStep], graph: &ComputationGraph) {
    apply_peephole_with_config(steps, graph, &PeepholeConfig::default());
}

/// Apply peephole optimizations with per-pass configuration.
///
/// Same as [`apply_peephole`] but allows disabling individual passes.
/// Useful for regression diagnosis — disable passes one at a time to
/// identify which fusion causes a performance regression.
pub(crate) fn apply_peephole_with_config(
    steps: &mut [CompiledStep],
    graph: &ComputationGraph,
    config: &PeepholeConfig,
) {
    let use_counts = build_step_use_counts(steps.len(), graph);
    // Pre-pass: Conv1d + Snake + InstanceNorm → FusedConv1dSnakeNorm (#4264).
    // Runs BEFORE FusedSnakeInstanceNorm and FusedConv1dActivation because
    // this 3-step pattern subsumes both 2-step patterns. If the 2-step passes
    // ran first they would consume the intermediate steps and prevent the
    // longer fusion.
    if config.fuse_conv1d_snake_norm {
        conv1d_snake_norm::fuse_conv1d_snake_norm(steps, &use_counts, graph);
    }
    // Pre-pass: 2× FusedConv1dSnakeNorm + add → FusedConv1dSnakeNormResBlock (#4264).
    // Runs AFTER FusedConv1dSnakeNorm (which creates the input NativeOps) and
    // BEFORE FusedSnakeInstanceNorm / FusedResBlock pass 2. Targets ResBlock
    // patterns without AdaIN style projection (plain conv → snake → norm chains).
    // 7-to-1 dispatch reduction for Kokoro Generator blocks.
    if config.fuse_conv1d_snake_norm_resblock {
        conv1d_snake_norm_resblock::fuse_conv1d_snake_norm_resblock(steps, &use_counts, graph);
    }
    // Pre-pass: Snake + InstanceNorm → FusedSnakeInstanceNorm (#4264).
    // Runs BEFORE FusedInstanceNormMulAdd and FusedAdainSnake because
    // those passes start with InstanceNorm steps. This pass consumes
    // Snake(Dispatch) + InstanceNorm(NativeOp) pairs first, leaving
    // remaining InstanceNorm steps for downstream passes.
    if config.fuse_snake_instance_norm {
        snake_instance_norm::fuse_snake_instance_norm(steps, &use_counts, graph);
    }
    // Pre-pass: InstanceNorm + Mul + Add → FusedInstanceNormMulAdd (#4252).
    // Runs BEFORE pass 0 (FusedAdainSnake) because the 3-step pattern is a
    // subset of the 4-step FusedAdainSnake pattern. The pass skips patterns
    // where snake_tensor follows (leaving those for FusedAdainSnake).
    if config.fuse_instance_norm_mul_add {
        instance_norm_mul_add::fuse_instance_norm_mul_add(steps, &use_counts, graph);
    }
    // Pass 0: InstanceNorm + Mul + Add + Snake → FusedAdainSnake (#4252).
    // Runs BEFORE pass 1 (NormActivConv1d) because FusedAdainSnake is a
    // more specific pattern: InstanceNorm+Mul+Add+Snake without a following
    // Conv1d. If NormActivConv1d ran first it would consume the AdainSnake
    // step (which this pass creates) and prevent the standalone fusion.
    if config.fuse_adain_snake {
        adain_snake::fuse_adain_snake(steps, &use_counts, graph);
    }
    // Upsample1d + Conv1d → FusedUpsampleConv1d (#4310).
    if config.fuse_upsample_conv1d {
        upsample_conv1d::fuse_upsample_conv1d(steps, &use_counts, graph);
    }
    // Pass 1: AdainSnake/LeakyRelu + Conv1d → NormActivConv1d
    if config.norm_activ_conv1d {
        fuse_norm_activ_conv1d(steps, &use_counts, graph);
    }
    // Pass 1b: AdainSnake/LeakyRelu + ConvTranspose1d → NormActivConvTranspose1d
    // Dual of pass 1 for transposed convolutions (Kokoro upsample stages).
    // Must run AFTER pass 1 which consumes AdaIN + Conv1d first.
    if config.norm_activ_conv_transpose1d {
        norm_activ_conv_transpose1d::fuse_norm_activ_conv_transpose1d(steps, &use_counts, graph);
    }
    if config.fused_resblock {
        // Pass 2: 2× NormActivConv1d + add → FusedResBlock
        resblock::fuse_resblock(steps, graph, &use_counts);
        // Pass 3: Absorb style projection Linears into FusedResBlock
        resblock::absorb_style_projections(steps, graph);
        // Pass 4: Batch style projections across FusedResBlocks (#1815 Tier 1, #2964)
        resblock::batch_style_projections(steps);
    }
    // Pass 4b: Chain 2-4 consecutive FusedResBlocks (with batched style projections)
    // into a single FusedResBlockChain. Must run AFTER pass 4 which sets
    // style_batch_offset on each FusedResBlock. Part of #4264.
    if config.fuse_resblock_chain {
        resblock_chain::fuse_resblock_chain(steps);
    }
    // Pass 5: Linear + Activation → LinearActivation (GEMM epilogue fusion)
    if config.linear_activation {
        linear_activation::fuse_linear_activation(steps, &use_counts);
    }
    // Pass 6: BinaryAdd + LayerNorm → AddLayerNorm (#1815 Tier 5 D2)
    // Must run BEFORE NormLinear so AddLayerNorm consumes the LayerNorm first.
    if config.add_layer_norm {
        add_ln::fuse_add_layer_norm(steps, &use_counts);
    }
    // Pass 7: LayerNorm + Linear → NormLinear (fused norm+GEMM, #3089)
    if config.norm_linear {
        norm_linear::fuse_norm_linear(steps, &use_counts);
    }
    // Pass 8: AddLayerNorm + Linear → AddNormLinear (#3351 T2.1).
    // Must run AFTER pass 6 (AddLayerNorm creates the NativeOp) and pass 7
    // (NormLinear consumes standalone LayerNorm+Linear first). Must run BEFORE
    // pass 12 (BatchedLinearProjection) so AddNormLinear gets priority over
    // batching for directly adjacent AddLayerNorm+Linear pairs.
    if config.add_norm_linear {
        add_norm_linear::fuse_add_norm_linear(steps, &use_counts);
    }
    // Pass 9: Absorb Transpose(1,2) around FlashAttention → SeqFirst layout (#1815).
    if config.attention_transpose {
        attention::absorb_attention_transposes(steps, graph);
    }
    // Pass 10: Absorb Flip(dim=0) around LstmSequence → reverse mode (#1815).
    // Eliminates 2 flip dispatches per backward BiLSTM layer (~192 in Kokoro).
    if config.flip_lstm {
        flip_lstm::absorb_flip_lstm(steps, &use_counts);
    }
    // Pass 11: Fwd LSTM + Rev LSTM + Cat → BiLstmCat (#4252).
    // Must run AFTER flip_lstm (pass 10) so reverse LSTMs have `reverse: true`.
    if config.bilstm_cat {
        bilstm_cat::fuse_bilstm_cat(steps, &use_counts, graph);
    }
    // Pass 12: Batch parallel Linear projections sharing one input (#3269).
    // Must run AFTER LinearActivation (pass 5) so fused linears are excluded.
    if config.batched_linear_projection {
        batched_qkv::batch_linear_projections(steps, &use_counts, graph);
    }
    // Pass 13: Transpose(1,2) + LayerNorm + Transpose(1,2) → ChannelsFirstLayerNorm (#3457).
    // Must run AFTER AddLayerNorm (pass 6) and NormLinear (pass 7) since those consume
    // the LayerNorm first. This pass absorbs the remaining standalone LayerNorms
    // bracketed by Transpose(1,2) pairs.
    if config.channels_first_layer_norm {
        conv_ln::absorb_transpose_layer_norm(steps, graph, &use_counts);
    }
    // Pass 14: Silu + Mul → SiluMul (SwiGLU MLP fusion, #3521).
    if config.silu_mul {
        silu_mul_fuse::fuse_silu_mul(steps, &use_counts, graph);
    }
    // Add + InstanceNorm + Conv1d(K=1) → FusedAddInstanceNormConv1x1 (#4264).
    // Runs AFTER FusedResBlock passes which consume deeper patterns first.
    // Catches remaining add → instance_norm → conv1d(K=1) patterns in the
    // Kokoro decoder for channel dimension changes.
    if config.fuse_add_instance_norm_conv1x1 {
        add_instance_norm_conv1x1::fuse_add_instance_norm_conv1x1(steps, &use_counts, graph);
    }
    // Conv1d + Activation → FusedConv1dActivation (#4264).
    // Must run AFTER NormActivConv1d (pass 1) and FusedResBlock (pass 2)
    // which consume AdainSnake/LeakyRelu + Conv1d sequences. This pass
    // catches remaining standalone Conv1d → Activation patterns.
    if config.fuse_conv1d_activation {
        conv1d_activation::fuse_conv1d_activation(steps, &use_counts, graph);
    }
    // Activation + Conv1d → FusedConv1dActivation (pre_activation=true) (#4264).
    // REVERSE of the above: catches Activation → Conv1d patterns (e.g., Kokoro
    // generator output stage: leaky_relu(0.01) → conv_post). Must run AFTER
    // NormActivConv1d, FusedResBlock, and Conv1d+Activation passes.
    if config.fuse_activation_conv1d {
        activation_conv1d::fuse_activation_conv1d(steps, &use_counts, graph);
    }
    // ConvTranspose1d + Activation → FusedConvTranspose1dActivation (#4264).
    // Must run AFTER FusedResBlock (pass 2) which handles pool_step
    // ConvTranspose1d. This pass catches remaining standalone
    // ConvTranspose1d → Activation patterns in upsample stages.
    if config.fuse_conv_transpose1d_activation {
        conv_transpose1d_activation::fuse_conv_transpose1d_activation(steps, &use_counts, graph);
    }
    // InstanceNorm + Conv1d → FusedInstanceNormConv1d (#4264).
    // Runs AFTER FusedResBlock, NormActivConv1d, and FusedAddInstanceNormConv1x1
    // which handle deeper norm→conv patterns with style affine or residual add.
    // This catches remaining standalone InstanceNorm → Conv1d pairs.
    if config.fuse_instance_norm_conv1d {
        instance_norm_conv1d::fuse_instance_norm_conv1d(steps, &use_counts, graph);
    }
    // Conv1d + InstanceNorm → FusedConv1dInstanceNorm (#4264).
    // Runs AFTER FusedConv1dSnakeNorm (which handles Conv1d→Snake→InstanceNorm)
    // and FusedConv1dActivation (which handles Conv1d→Activation). This catches
    // remaining Conv1d → InstanceNorm pairs without activation in between.
    if config.fuse_conv1d_instance_norm {
        conv1d_instance_norm::fuse_conv1d_instance_norm(steps, &use_counts, graph);
    }
    // Linear + LayerNorm → FusedLinearLayerNorm (#4264).
    // Runs AFTER AddLayerNorm (pass 6) and NormLinear (pass 7) to catch
    // remaining Linear → LayerNorm pairs not consumed by those passes.
    // In PlBert, post-attention and post-FFN projections feed into LayerNorm.
    if config.fuse_linear_layer_norm {
        linear_layer_norm::fuse_linear_layer_norm(steps, &use_counts);
    }
    // Pass 15: Auto-fuse remaining consecutive elementwise Dispatch chains (#3517).
    // Runs LAST so all specific named patterns (passes 1-14) match first.
    if config.auto_fuse_elementwise {
        auto_fuse::fuse_elementwise_chains(steps, &use_counts);
    }
}

/// Build a use-count map: for each graph node index, how many downstream
/// nodes consume it as input. Used for fan-out checks.
fn build_step_use_counts(num_steps: usize, graph: &ComputationGraph) -> Vec<usize> {
    let nodes = graph.nodes();
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    let mut counts = vec![0usize; num_steps];
    for node in nodes {
        for &input_id in node.inputs() {
            if let Some(&idx) = id_to_idx.get(&input_id) {
                if idx < counts.len() {
                    counts[idx] += 1;
                }
            }
        }
    }
    counts
}

/// Scan for AdainLeakyRelu/AdainSnake + Conv1d pairs and fuse them.
fn fuse_norm_activ_conv1d(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 2 {
        return;
    }

    // Scan pairs. We process left-to-right; once a pair is fused the
    // first slot becomes IdentityPassthrough (won't match again).
    let mut i = 0;
    while i + 1 < len {
        if try_fuse_pair(steps, i, use_counts, graph) {
            // Skip the fused pair — the next candidate starts at i+2.
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (AdainLeakyRelu/AdainSnake) with steps[i+1] (Conv1d).
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse_pair(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph: &ComputationGraph,
) -> bool {
    // Step i must be a NativeOp with AdainLeakyRelu or AdainSnake.
    let adain_info = match &steps[i] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AdainLeakyRelu {
                    eps,
                    slope,
                    input_shape,
                    ..
                },
            weight_data,
        } => Some(AdainInfo {
            activation: NormActivation::LeakyRelu { slope: *slope },
            eps: *eps,
            input_shape: input_shape.clone(),
            alpha_weight: None,
            adain_weight_data: weight_data.clone(),
        }),
        CompiledStep::NativeOp {
            op: NativeOpKind::AdainSnake {
                eps, input_shape, ..
            },
            weight_data,
        } => {
            let alpha = weight_data.get("alpha").cloned();
            Some(AdainInfo {
                activation: NormActivation::Snake,
                eps: *eps,
                input_shape: input_shape.clone(),
                alpha_weight: alpha,
                adain_weight_data: weight_data.clone(),
            })
        }
        _ => None,
    };

    let adain_info = match adain_info {
        Some(info) => info,
        None => return false,
    };

    // Fan-out check: the AdaIN output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step i+1 must be a Dispatch with name "conv1d".
    let conv_info = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => extract_conv1d_params(kernel, weight_data),
        _ => None,
    };

    let conv_info = match conv_info {
        Some(info) => info,
        None => return false,
    };

    // Only fuse stride=1, groups=1 Conv1d (the common case in AdainResBlk1d).
    if conv_info.stride != 1 || conv_info.groups != 1 {
        return false;
    }

    // Build the fused NativeOp.
    let mut merged_weight_data = adain_info.adain_weight_data;

    // Rename conv weights with "conv_" prefix.
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("conv_bias".to_string(), b);
    }
    // Alpha is already in adain_weight_data for Snake variant.

    // Capture the graph node IDs of the AdaIN step's inputs so the
    // edge_map builder can resolve edges generically. Part of #3261.
    let ext_ids = graph.nodes().get(i).map(|node| node.inputs().to_vec());

    let fused_op = NativeOpKind::NormActivConv1d {
        activation: adain_info.activation,
        eps: adain_info.eps,
        conv_dilation: conv_info.dilation,
        conv_padding: conv_info.padding,
        input_shape: adain_info.input_shape,
        output_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
        external_node_ids: ext_ids,
    };

    // Place NormActivConv1d at step[i] (AdaIN position) — its edge_map
    // entry is now set by external_node_ids (overriding graph topology).
    // Step[i+1] (conv position) becomes IdentityPassthrough which passes
    // through the fused output to downstream consumers.
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };

    // Replace step[i+1] with IdentityPassthrough (preserves index alignment).
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}

/// Extracted AdaIN info for pattern matching.
struct AdainInfo {
    activation: NormActivation,
    eps: f32,
    input_shape: Vec<usize>,
    #[allow(dead_code)]
    alpha_weight: Option<nn_core::dyn_tensor::trace::WeightRef>,
    adain_weight_data: HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
}

/// Extracted Conv1d parameters from a Dispatch kernel IR.
pub(crate) struct Conv1dInfo {
    pub(crate) padding: usize,
    pub(crate) dilation: usize,
    pub(crate) stride: usize,
    pub(crate) groups: usize,
    pub(crate) output_channels: usize,
    pub(crate) kernel_size: usize,
    pub(crate) weight: Option<nn_core::dyn_tensor::trace::WeightRef>,
    pub(crate) bias: Option<nn_core::dyn_tensor::trace::WeightRef>,
}

/// Extract Conv1d parameters from a CompiledKernel's IR nodes.
pub(crate) fn extract_conv1d_params(
    kernel: &super::CompiledKernel,
    weight_data: &HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
) -> Option<Conv1dInfo> {
    let def = kernel.def();

    // Find the Conv1d IR node to extract parameters.
    let conv_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Conv1d { .. }))?;

    let (padding, dilation, stride, groups) = match &conv_node.kind {
        TensorOpKind::Conv1d {
            padding,
            dilation,
            stride,
            groups,
            ..
        } => (*padding, *dilation, *stride, *groups),
        _ => return None,
    };

    // Extract output channels and kernel size from the conv weight shape.
    // Conv1d weight shape is [C_out, C_in/groups, K].
    let weight_ref = weight_data.get("weight")?;
    let weight_shape = weight_ref.shape();
    if weight_shape.len() != 3 {
        return None;
    }
    let output_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    Some(Conv1dInfo {
        padding,
        dilation,
        stride,
        groups,
        output_channels,
        kernel_size,
        weight: weight_data.get("weight").cloned(),
        bias: weight_data.get("bias").cloned(),
    })
}

#[cfg(test)]
#[path = "trace_compile_peephole_tests.rs"]
mod tests;
