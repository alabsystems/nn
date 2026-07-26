// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verifiability classification for traced operations.
//!
//! Every `TraceOp` is classified by whether NY can propagate bounds
//! through it. The classification drives the compilation gate: operations in
//! learned weight paths that cannot be verified produce a compile error unless
//! explicitly annotated with `#[allow_unverifiable]`.
//!
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::TraceOp;

/// Verifiability classification for a traced operation.
///
/// Determines whether NY can propagate bounds through the op and
/// under what conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifiabilityClass {
    /// NY `LayerType` exists and propagation works at all dimensions.
    Verifiable,

    /// NY can handle this op but only with sub-block decomposition
    /// above a certain dimension threshold. Without decomposition, bounds
    /// compound multiplicatively and become vacuous.
    VerifiableBounded {
        /// Maximum dimension for direct (non-decomposed) verification.
        max_dim: usize,
    },

    /// Not verifiable by NY but explicitly annotated as safe.
    /// The op is not in a learned weight path (e.g., STFT phase computation,
    /// data preprocessing). The annotation reason is recorded in the
    /// verification certificate.
    UnverifiableSafe,

    /// In a learned weight path with no NY `LayerType`. This is
    /// a **compile error** unless the model author provides an explicit
    /// `#[allow_unverifiable]` annotation with a safety justification.
    UnverifiableLearned,

    /// Shape-only operation (reshape, transpose, squeeze, etc.) that does
    /// not affect numerical values and is always verifiable.
    ShapeOnly,

    /// Passthrough operation (dropout at inference, identity, dtype cast)
    /// that does not affect verification.
    Passthrough,
}

impl VerifiabilityClass {
    /// Whether this classification allows compilation without annotation.
    #[must_use]
    pub fn allows_compilation(&self) -> bool {
        !matches!(self, Self::UnverifiableLearned)
    }

    /// Whether NY can verify this op (possibly with decomposition).
    ///
    /// Returns `true` for all classifications except `UnverifiableLearned`.
    /// Used by `#[nn::model(verify)]` structural checks.
    #[must_use]
    pub fn is_verifiable(&self) -> bool {
        !matches!(self, Self::UnverifiableLearned)
    }

    /// Whether this classification requires sub-block decomposition for
    /// verification at a given dimension.
    #[must_use]
    pub fn needs_decomposition(&self, dim: usize) -> bool {
        match self {
            Self::VerifiableBounded { max_dim } => dim > *max_dim,
            _ => false,
        }
    }
}

/// Classify a `TraceOp` by its NY verifiability.
///
/// The classification is conservative: any op not explicitly listed as
/// `Verifiable` or `ShapeOnly` defaults to `UnverifiableLearned`.
///
/// # Maintenance
///
/// When a new `TraceOp` variant gets NY support in the NY-owned translator
/// (`ny-trace-bridge`'s `translate_node`), update this function to return
/// `Verifiable` for that variant. The consistency test
/// `test_translator_supported_ops_are_not_unverifiable_learned` catches drift.
#[must_use]
pub fn classify_op(op: &TraceOp) -> VerifiabilityClass {
    match op {
        // -- Always verifiable: direct NY LayerType mapping --

        // Unary activations
        TraceOp::Relu
        | TraceOp::Gelu
        | TraceOp::Sigmoid
        | TraceOp::Tanh
        | TraceOp::Exp
        | TraceOp::Log
        | TraceOp::Sqrt
        | TraceOp::Sqr
        | TraceOp::Abs
        | TraceOp::Neg
        | TraceOp::Recip
        | TraceOp::Sin
        | TraceOp::Cos
        | TraceOp::Silu
        | TraceOp::Floor
        | TraceOp::Round => VerifiabilityClass::Verifiable,

        // Binary element-wise
        TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Mul
        | TraceOp::Div
        | TraceOp::Maximum
        | TraceOp::Minimum => VerifiabilityClass::Verifiable,

        // Softmax
        TraceOp::Softmax { .. } | TraceOp::LogSoftmax { .. } => VerifiabilityClass::Verifiable,

        // Normalization — verifiable but bounds compound at high dimensions
        TraceOp::LayerNorm { .. } | TraceOp::RmsNorm { .. } => {
            VerifiabilityClass::VerifiableBounded { max_dim: 512 }
        }
        TraceOp::InstanceNorm { .. } | TraceOp::GroupNorm { .. } | TraceOp::BatchNorm { .. } => {
            VerifiabilityClass::VerifiableBounded { max_dim: 512 }
        }

        // Linear/Conv — always verifiable (weight is constant in verification)
        TraceOp::Linear { .. }
        | TraceOp::Conv1d { .. }
        | TraceOp::Conv2d { .. }
        | TraceOp::Conv3d { .. }
        | TraceOp::ConvTranspose1d { .. }
        | TraceOp::ConvTranspose2d { .. } => VerifiabilityClass::Verifiable,

        // MatMul — verifiable when one operand is constant weight
        TraceOp::MatMul => VerifiabilityClass::Verifiable,

        // Clamp — direct NY Clip layer
        TraceOp::Clamp { .. } => VerifiabilityClass::Verifiable,

        // Embedding — verifiable (index lookup, bounded output)
        TraceOp::Embedding { .. } => VerifiabilityClass::Verifiable,

        // Reductions — verifiable (ReduceMean, ReduceSum, etc.)
        TraceOp::ReduceSum { .. }
        | TraceOp::ReduceMean { .. }
        | TraceOp::ReduceMax { .. }
        | TraceOp::ReduceMin { .. } => VerifiabilityClass::Verifiable,

        // Pooling — verifiable
        TraceOp::AvgPool2d { .. } | TraceOp::MaxPool2d { .. } => VerifiabilityClass::Verifiable,

        // Activation variants — verifiable
        TraceOp::Elu { .. } | TraceOp::LeakyRelu { .. } => VerifiabilityClass::Verifiable,
        TraceOp::Dropout => VerifiabilityClass::Passthrough,

        // Padding — decomposed into verifiable primitives
        TraceOp::ReflectionPad1d { .. } | TraceOp::ConstantPadNd { .. } => {
            VerifiabilityClass::Verifiable
        }

        // Kokoro fused ops — decomposed into verifiable primitives
        TraceOp::KokoroFused(_) => VerifiabilityClass::VerifiableBounded { max_dim: 512 },

        // Decomposed ops — verifiable through decomposition
        TraceOp::PixelShuffle { .. }
        | TraceOp::PixelUnshuffle { .. }
        | TraceOp::Upsample1d { .. }
        | TraceOp::Upsample2d { .. }
        | TraceOp::WhereCond
        | TraceOp::Cumsum { .. }
        | TraceOp::Flip { .. }
        | TraceOp::Expand { .. }
        | TraceOp::RepeatInterleave { .. } => VerifiabilityClass::Verifiable,

        // Powf — special cases verifiable (x^1, x^2, x^0.5)
        TraceOp::Powf { exponent } => {
            if *exponent == 1.0 || *exponent == 2.0 || *exponent == 0.5 {
                VerifiabilityClass::Verifiable
            } else {
                VerifiabilityClass::UnverifiableLearned
            }
        }

        // GeluErf — not in scalar IR (requires erf() function)
        TraceOp::GeluErf => VerifiabilityClass::Verifiable,

        // Constant — always verifiable (just a value)
        TraceOp::Constant { .. } => VerifiabilityClass::Verifiable,

        // -- Shape-only operations --
        TraceOp::Reshape { .. }
        | TraceOp::Transpose { .. }
        | TraceOp::Narrow { .. }
        | TraceOp::Unsqueeze { .. }
        | TraceOp::Squeeze { .. }
        | TraceOp::Permute { .. }
        | TraceOp::Cat { .. } => VerifiabilityClass::ShapeOnly,

        // Input — structural
        TraceOp::Input => VerifiabilityClass::ShapeOnly,

        // ToDtype — passthrough (no numerical change in verification)
        TraceOp::ToDtype { .. } => VerifiabilityClass::Passthrough,

        // Atan2 — NY LayerType::Atan2 wired (W11 1f50868a, W12 884db4ee).
        TraceOp::Atan2 => VerifiabilityClass::Verifiable,

        // SDPA — decomposed to MatMul+Softmax+MatMul in translator. Bounds
        // compound at large sequence lengths, so restrict to bounded verification.
        TraceOp::Sdpa { .. } | TraceOp::SdpaCausal { .. } => {
            VerifiabilityClass::VerifiableBounded { max_dim: 512 }
        }

        // RotaryEmbedding — direct NY LayerType::RoPE.
        TraceOp::RotaryEmbedding { .. } => VerifiabilityClass::Verifiable,

        // IndexSelect/Gather — direct NY GatherLayer.
        TraceOp::IndexSelect { .. } | TraceOp::Gather { .. } => VerifiabilityClass::Verifiable,

        // MultiHeadAttention — composite op, no direct translator support.
        TraceOp::MultiHeadAttention { .. } => VerifiabilityClass::UnverifiableLearned,

        TraceOp::SwiGlu
        | TraceOp::Topk { .. }
        | TraceOp::Argmax { .. }
        | TraceOp::Argmin { .. }
        | TraceOp::ArgSort { .. }
        | TraceOp::Compare { .. }
        | TraceOp::CompareTensor { .. }
        | TraceOp::Triu { .. }
        | TraceOp::Tril { .. }
        | TraceOp::GridSample { .. }
        | TraceOp::SliceSet { .. }
        | TraceOp::Unfold { .. }
        | TraceOp::ScatterAdd { .. }
        | TraceOp::IndexAdd { .. }
        | TraceOp::IndexPut { .. } => VerifiabilityClass::UnverifiableLearned,

        // Activation by name — depends on the activation kind
        TraceOp::Activation { kind } => match kind.as_str() {
            "relu" | "gelu" | "sigmoid" | "tanh" | "silu" | "leaky_relu" | "elu" | "snake" => {
                VerifiabilityClass::Verifiable
            }
            _ => VerifiabilityClass::UnverifiableLearned,
        },

        // Recurrent — complex but decomposable
        TraceOp::Lstm { .. } => VerifiabilityClass::VerifiableBounded { max_dim: 256 },

        // Fract — no trace_to_graph translator yet. Decomposable as x - floor(x)
        // but NY translation not wired. Safe: not inherently in learned
        // weight paths. Reclassified from Verifiable (#3226).
        TraceOp::Fract => VerifiabilityClass::UnverifiableSafe,

        // Arange — produces constants at trace time, but no trace_to_graph
        // translator. Safe: output is deterministic and not learned.
        // Reclassified from Verifiable (#3226).
        TraceOp::Arange { .. } => VerifiabilityClass::UnverifiableSafe,

        // Pooling (1D/adaptive) — no NY LayerType translation yet.
        // Downgraded from Verifiable to match trace_to_graph dispatch (#2936).
        TraceOp::MaxPool1d { .. } | TraceOp::AdaptiveAvgPool2d { .. } => {
            VerifiabilityClass::UnverifiableSafe
        }

        // QLinear — quantized, verifiable with quantization bounds
        TraceOp::QLinear { .. } => VerifiabilityClass::VerifiableBounded { max_dim: 1024 },

        // Custom ops — unknown, must be annotated
        TraceOp::Custom { .. } => VerifiabilityClass::UnverifiableLearned,

        // Catch-all for future #[non_exhaustive] variants.
        // Conservative: treat unknown ops as unverifiable in learned paths.
        _ => VerifiabilityClass::UnverifiableLearned,
    }
}

/// Classify a callee function name by its NY verifiability.
///
/// Used at proc-macro time by `#[nn::model(verify)]` when only function
/// names (from `ModelDef` call graph) are available, not `TraceOp` variants.
///
/// Conservative: unknown names default to `UnverifiableLearned`.
#[must_use]
pub fn classify_callee_name(name: &str) -> VerifiabilityClass {
    match name {
        // Activations
        "relu" | "gelu" | "gelu_erf" | "sigmoid" | "tanh" | "silu" | "snake" | "elu"
        | "leaky_relu" | "exp" | "log" | "sqrt" | "sqr" | "abs" | "neg" | "recip" | "sin"
        | "cos" | "floor" | "round" | "atan2" => VerifiabilityClass::Verifiable,

        // Binary element-wise
        "add" | "sub" | "mul" | "div" | "maximum" | "minimum" => VerifiabilityClass::Verifiable,

        // Softmax
        "softmax" | "log_softmax" => VerifiabilityClass::Verifiable,

        // Linear/Conv layers
        "linear"
        | "linear_no_bias"
        | "conv1d"
        | "conv1d_no_bias"
        | "conv2d"
        | "conv2d_no_bias"
        | "conv_transpose1d"
        | "conv_transpose1d_no_bias"
        | "conv_transpose2d"
        | "conv_transpose2d_no_bias"
        | "matmul"
        | "embedding" => VerifiabilityClass::Verifiable,

        // Reductions
        "reduce_sum" | "reduce_mean" | "reduce_max" | "reduce_min" | "sum" | "mean" => {
            VerifiabilityClass::Verifiable
        }

        // Pooling
        "avg_pool2d" | "max_pool2d" => VerifiabilityClass::Verifiable,

        // Padding/spatial
        "reflection_pad1d" | "constant_pad" | "pixel_shuffle" | "pixel_unshuffle"
        | "upsample1d" | "upsample2d" | "clamp" | "where_cond" | "cumsum" | "flip" | "expand"
        | "repeat_interleave" | "index_select" | "gather" => VerifiabilityClass::Verifiable,

        // Normalization — bounded
        "layer_norm" | "rms_norm" | "instance_norm" | "group_norm" | "batch_norm" => {
            VerifiabilityClass::VerifiableBounded { max_dim: 512 }
        }

        // Attention — bounded
        "sdpa" | "sdpa_causal" | "scaled_dot_product_attention" => {
            VerifiabilityClass::VerifiableBounded { max_dim: 512 }
        }

        // LSTM — bounded
        "lstm" => VerifiabilityClass::VerifiableBounded { max_dim: 256 },

        // Shape-only
        "reshape" | "transpose" | "narrow" | "unsqueeze" | "squeeze" | "permute" | "cat"
        | "stack" | "contiguous" | "flatten" => VerifiabilityClass::ShapeOnly,

        // Passthrough
        "dropout" | "to_dtype" | "identity" => VerifiabilityClass::Passthrough,

        // Explicitly unverifiable but safe
        "max_pool1d" | "adaptive_avg_pool2d" => VerifiabilityClass::UnverifiableSafe,

        // Rotary embedding
        "rope" | "rotary_embedding" => VerifiabilityClass::Verifiable,

        // Unknown — conservative default
        _ => VerifiabilityClass::UnverifiableLearned,
    }
}

/// Summary of verifiability classification for a computation graph.
#[derive(Debug, Clone, Default)]
pub struct VerifiabilitySummary {
    /// Number of fully verifiable ops.
    pub verifiable: usize,
    /// Number of ops requiring sub-block decomposition.
    pub bounded: usize,
    /// Number of shape-only ops.
    pub shape_only: usize,
    /// Number of passthrough ops.
    pub passthrough: usize,
    /// Number of unverifiable ops annotated as safe.
    pub unverifiable_safe: usize,
    /// Number of unverifiable ops in learned paths (compile errors).
    pub unverifiable_learned: usize,
    /// Names of unverifiable learned ops (for error messages).
    pub unverifiable_learned_ops: Vec<String>,
}

impl VerifiabilitySummary {
    /// Whether the graph can be compiled without annotations.
    #[must_use]
    pub fn is_fully_compilable(&self) -> bool {
        self.unverifiable_learned == 0
    }
}

#[cfg(test)]
#[path = "verifiability_tests.rs"]
mod tests;
