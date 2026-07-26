// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! ICB + autocast composition: build-time dtype resolution for ICB replay.
//!
//! When autocast is active, the runtime dtype tracker mutates per-step dtypes
//! dynamically (F32 for accumulate ops, F16 for compute ops). This prevents
//! ICB pre-encoding because ICB commands have fixed buffer bindings.
//!
//! `IcbAutocastResolver` performs the same analysis statically at build time,
//! producing an `IcbAutocastPlan` that records the dtype decision per step
//! and identifies explicit cast points at F32/F16 boundaries. Steps with
//! matching dtypes across the plan can be pre-encoded into ICB segments
//! without runtime dtype tracking.
//!
//! Part of #3499.

use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::NativeOpKind;

use super::super::{MixedGemmInfo, StepMeta};

/// Build-time autocast resolution for ICB-compatible dtype assignment.
///
/// Walks the compiled step graph and assigns F16 or F32 to each step
/// based on operation characteristics: LSTM and reduce ops stay F32,
/// elementwise/matmul/conv use F16 for bandwidth savings.
pub(crate) struct IcbAutocastResolver<'a> {
    steps: &'a [CompiledStep],
    step_metas: &'a [StepMeta],
    mixed_gemm_infos: &'a [Option<MixedGemmInfo>],
}

impl<'a> IcbAutocastResolver<'a> {
    /// Create a resolver from a compiled model's step data.
    pub(crate) fn new(
        steps: &'a [CompiledStep],
        step_metas: &'a [StepMeta],
        mixed_gemm_infos: &'a [Option<MixedGemmInfo>],
    ) -> Self {
        Self {
            steps,
            step_metas,
            mixed_gemm_infos,
        }
    }

    /// Resolve dtype assignments for all steps, producing an ICB autocast plan.
    ///
    /// The plan records the optimal dtype per step and identifies boundary
    /// cast points where explicit F32<->F16 conversion is needed.
    pub(crate) fn resolve(&self) -> IcbAutocastPlan {
        let n = self.steps.len();
        if n == 0 {
            return IcbAutocastPlan {
                dtype_per_step: Vec::new(),
                cast_points: Vec::new(),
                total_f16_steps: 0,
                total_f32_steps: 0,
            };
        }

        // Phase 1: Assign optimal dtype per step based on op characteristics.
        let mut dtypes: Vec<DType> = Vec::with_capacity(n);
        for (i, step) in self.steps.iter().enumerate() {
            let dt = if is_f16_safe(step, self.mixed_gemm_infos.get(i).and_then(|g| g.as_ref())) {
                DType::F16
            } else {
                DType::F32
            };
            dtypes.push(dt);
        }

        // Phase 2: Propagate dtypes through passthrough/view steps.
        // These inherit from their source step, not from the op classifier.
        for i in 0..n {
            match &self.steps[i] {
                CompiledStep::Passthrough { .. }
                | CompiledStep::IdentityPassthrough
                | CompiledStep::NarrowView { .. } => {
                    if let Some(&src) = self.step_metas.get(i).and_then(|m| m.edges.first()) {
                        if let Some(&src_dt) = dtypes.get(src) {
                            dtypes[i] = src_dt;
                        }
                    }
                }
                CompiledStep::InputForward => {
                    // Inputs stay at their graph-specified dtype (F32 by default).
                    dtypes[i] = DType::F32;
                }
                _ => {}
            }
        }

        // Phase 3: Detect boundary cast points.
        let mut cast_points = Vec::new();
        for i in 0..n {
            // Only Dispatch and NativeOp steps consume typed inputs.
            if !matches!(
                &self.steps[i],
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            ) {
                continue;
            }

            let step_dt = dtypes[i];
            for &src in &self.step_metas[i].edges {
                let src_dt = dtypes.get(src).copied().unwrap_or(DType::F32);
                if src_dt != step_dt {
                    cast_points.push((i, src_dt, step_dt));
                    // One cast point per step suffices — the runtime casts all
                    // inputs of the step to the step's dtype.
                    break;
                }
            }
        }

        let total_f16 = dtypes.iter().filter(|&&d| d == DType::F16).count();
        let total_f32 = dtypes.iter().filter(|&&d| d == DType::F32).count();

        IcbAutocastPlan {
            dtype_per_step: dtypes,
            cast_points,
            total_f16_steps: total_f16,
            total_f32_steps: total_f32,
        }
    }
}

/// Build-time autocast plan recording per-step dtype decisions.
///
/// Produced by [`IcbAutocastResolver::resolve`]. Consumed by ICB
/// segment pre-encoding to determine buffer element sizes and identify
/// steps that need explicit dtype casts at boundaries.
#[derive(Debug, Clone)]
pub(crate) struct IcbAutocastPlan {
    /// Optimal dtype for each step, indexed by step index.
    pub(crate) dtype_per_step: Vec<DType>,
    /// Explicit cast points: `(step_idx, from_dtype, to_dtype)`.
    ///
    /// Each entry means the inputs to `step_idx` arrive in `from_dtype`
    /// but the step operates in `to_dtype`, requiring a runtime cast.
    pub(crate) cast_points: Vec<(usize, DType, DType)>,
    /// Number of steps assigned F16 (for diagnostics/reporting).
    pub(crate) total_f16_steps: usize,
    /// Number of steps assigned F32 (for diagnostics/reporting).
    pub(crate) total_f32_steps: usize,
}

impl IcbAutocastPlan {
    /// Returns the assigned dtype for a step, defaulting to F32.
    pub(crate) fn step_dtype(&self, step_idx: usize) -> DType {
        self.dtype_per_step
            .get(step_idx)
            .copied()
            .unwrap_or(DType::F32)
    }

    /// Returns whether a cast is needed before the given step.
    pub(crate) fn needs_cast(&self, step_idx: usize) -> bool {
        self.cast_points.iter().any(|&(idx, _, _)| idx == step_idx)
    }

    /// Returns the ScalarType for a step (for ICB codegen compatibility).
    pub(crate) fn step_scalar_type(&self, step_idx: usize) -> ScalarType {
        match self.step_dtype(step_idx) {
            DType::F16 => ScalarType::F16,
            DType::BF16 => ScalarType::BF16,
            DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::U32 | DType::U8
            | DType::Bool => ScalarType::F32,
            _ => ScalarType::F32,
        }
    }

    /// Fraction of steps using F16 (0.0 to 1.0).
    pub(crate) fn f16_ratio(&self) -> f64 {
        let total = self.dtype_per_step.len();
        if total == 0 {
            return 0.0;
        }
        self.total_f16_steps as f64 / total as f64
    }

    /// Returns step indices that are ICB-compatible under this autocast plan.
    ///
    /// A step is ICB-compatible if it is a Dispatch step and does NOT
    /// need a boundary cast (all inputs already match the step's dtype).
    pub(crate) fn icb_compatible_steps(&self, steps: &[CompiledStep]) -> Vec<bool> {
        steps
            .iter()
            .enumerate()
            .map(|(i, step)| {
                matches!(step, CompiledStep::Dispatch { .. }) && !self.needs_cast(i)
            })
            .collect()
    }
}

/// Resolve autocast dtype assignments for ICB composition.
///
/// Convenience function wrapping [`IcbAutocastResolver`]. Returns an error
/// if the step metadata vectors have mismatched lengths.
pub(crate) fn resolve_autocast(
    steps: &[CompiledStep],
    step_metas: &[StepMeta],
    mixed_gemm_infos: &[Option<MixedGemmInfo>],
) -> Result<IcbAutocastPlan, super::super::error::CompiledModelError> {
    if steps.len() != step_metas.len() {
        return Err(super::super::error::CompiledModelError::InvalidConfig {
            reason: format!(
                "steps.len() ({}) != step_metas.len() ({})",
                steps.len(),
                step_metas.len()
            ),
        });
    }
    if steps.len() != mixed_gemm_infos.len() {
        return Err(super::super::error::CompiledModelError::InvalidConfig {
            reason: format!(
                "steps.len() ({}) != mixed_gemm_infos.len() ({})",
                steps.len(),
                mixed_gemm_infos.len()
            ),
        });
    }

    let resolver = IcbAutocastResolver::new(steps, step_metas, mixed_gemm_infos);
    Ok(resolver.resolve())
}

/// Determine whether a compiled step is safe for F16 execution.
///
/// Returns `true` for elementwise ops, matmul, conv1d, and similar
/// bandwidth-bound operations that benefit from F16 without precision loss.
///
/// Returns `false` for:
/// - LSTM steps (recurrent accumulation needs F32 precision)
/// - Reduce/accumulate ops (softmax denominator, norm reductions)
/// - Steps with non-float inputs (U32 from argmax/topk)
/// - Mixed GEMM steps (already handled by separate dispatch path)
/// - RuntimeOp steps (data-dependent, always F32)
/// - ConstantValue steps (CPU-materialized, always F32)
pub(crate) fn is_f16_safe(step: &CompiledStep, mixed_gemm: Option<&MixedGemmInfo>) -> bool {
    // Mixed GEMM has its own dispatch path — not ICB/autocast compatible.
    if mixed_gemm.is_some() {
        return false;
    }

    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            let name = kernel.name();
            // Reduce/accumulate ops need F32 for numerical stability.
            if is_reduce_kernel(name) {
                return false;
            }
            // Everything else (elementwise, matmul, conv, embedding, etc.)
            // is F16-safe.
            true
        }
        CompiledStep::NativeOp { op, .. } => is_native_op_f16_safe(op),
        // Passthrough/view: inherits from source (handled in resolver).
        CompiledStep::Passthrough { .. }
        | CompiledStep::IdentityPassthrough
        | CompiledStep::NarrowView { .. } => true,
        // Input: stays F32 (handled in resolver).
        CompiledStep::InputForward => false,
        // Constants: CPU-materialized F32.
        CompiledStep::ConstantValue { .. } => false,
        // Runtime ops: data-dependent, always F32.
        CompiledStep::RuntimeOp { .. } => false,
        // Catch non-exhaustive future variants conservatively.
        _ => false,
    }
}

/// Check if a kernel name corresponds to a reduce/accumulate operation.
///
/// Reduce ops (softmax, norms, mean, sum) compute denominators or
/// statistics via summation across elements. F16 accumulation loses
/// significant precision for these operations.
fn is_reduce_kernel(name: &str) -> bool {
    matches!(
        name,
        "softmax"
            | "log_softmax"
            | "reduce_sum"
            | "reduce_mean"
            | "reduce_max"
            | "reduce_min"
            | "layer_norm"
            | "rms_norm"
            | "group_norm"
            | "instance_norm"
    )
}

/// Determine whether a NativeOp variant is safe for F16 execution.
fn is_native_op_f16_safe(op: &NativeOpKind) -> bool {
    match op {
        // LSTM: recurrent accumulation needs F32 precision.
        NativeOpKind::LstmSequence { .. } => false,
        // Cumsum: prefix sum accumulation needs F32.
        NativeOpKind::Cumsum { .. } => false,
        // Norm operations: reduce over channels, need F32 accumulators.
        NativeOpKind::InstanceNorm { .. }
        | NativeOpKind::LayerNorm { .. }
        | NativeOpKind::AddLayerNorm { .. }
        | NativeOpKind::ChannelsFirstLayerNorm { .. } => false,
        // AdaIN variants: contain InstanceNorm reduction internally.
        NativeOpKind::AdainSnake { .. }
        | NativeOpKind::AdainLeakyRelu { .. }
        | NativeOpKind::FusedAdainSnake { .. }
        | NativeOpKind::FusedInstanceNormMulAdd { .. }
        | NativeOpKind::FusedSnakeInstanceNorm { .. } => false,
        // AdaLayerNorm: contains LayerNorm reduction internally.
        NativeOpKind::AdaLayerNorm { .. } => false,
        // NormLinear: contains norm reduction internally.
        NativeOpKind::NormLinear { .. } => false,
        // Flash attention: softmax accumulation needs F32.
        NativeOpKind::FlashAttention { .. } => false,
        // Matmul/conv ops: bandwidth-bound, F16-safe.
        NativeOpKind::LinearActivation { .. }
        | NativeOpKind::BatchedLinearProjection { .. }
        | NativeOpKind::Conv1dGemm { .. }
        | NativeOpKind::Int8Gemm { .. } => true,
        // Projection slice: narrow from matmul output, inherits dtype.
        NativeOpKind::ProjectionSlice { .. } => true,
        // Elementwise: SiluMul, FusedResBlock, NormActivConv1d.
        // FusedResBlock contains internal norms → F32 for safety.
        NativeOpKind::FusedResBlock { .. } => false,
        // NormActivConv1d contains internal norm → F32 for safety.
        NativeOpKind::NormActivConv1d { .. } => false,
        // SiluMul: purely elementwise, F16-safe.
        NativeOpKind::SiluMul { .. } => true,
        // RotaryEmbedding: elementwise rotation (mul + add), F16-safe.
        NativeOpKind::RotaryEmbedding { .. } => true,
        // MaxPool1d: comparison-based, F16-safe.
        NativeOpKind::MaxPool1d { .. } => true,
        // ConstantWeight: pre-computed, stays F32 (CPU-materialized).
        NativeOpKind::ConstantWeight { .. } => false,
        // BatchedStyleProjection: single matmul, F16-safe.
        NativeOpKind::BatchedStyleProjection { .. } => true,
        // MoeGating: contains softmax reduction, needs F32 precision.
        NativeOpKind::MoeGating { .. } => false,
        // FusedUpsampleConv1d: conv1d accumulation benefits from F32.
        NativeOpKind::FusedUpsampleConv1d { .. } => false,
        // FusedConv1dActivation: conv1d accumulation benefits from F32.
        NativeOpKind::FusedConv1dActivation { .. } => false,
        // BiLstmCat: two LSTM recurrences + cat, needs F32 precision.
        NativeOpKind::BiLstmCat { .. } => false,
        // AddNormLinear: contains norm reduction internally.
        NativeOpKind::AddNormLinear { .. } => false,
        // FusedLayerNormLinear: contains LayerNorm reduction + GEMM internally.
        NativeOpKind::FusedLayerNormLinear { .. } => false,
        // Fused elementwise ops: purely elementwise, F16-safe.
        NativeOpKind::FusedMulAdd { .. }
        | NativeOpKind::FusedSiGLU { .. }
        | NativeOpKind::FusedGeGLU { .. } => true,
        // Conservative default for future variants.
        _ => false,
    }
}

#[cfg(test)]
#[path = "compiled_model_icb_autocast_tests.rs"]
mod tests;
