// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Layer-specific validation error variants for the tensor IR.
//!
//! Extracted from `tensor_ir_error.rs` to keep that file under the 500-line
//! limit. These variants cover per-op validation (Conv1d, Conv2d, LSTM, etc.)
//! and are wrapped by `TensorIRError::Layer(TensorIRLayerError)`.
//!
//! Pattern: same as `VerifyError::Structural(StructuralError)` in nn-verify.
//!
//! Part of #837.

use thiserror::Error;

pub use tensor_ir_error_conv::TensorIRConvError;
#[path = "tensor_ir_error_conv.rs"]
mod tensor_ir_error_conv;

/// Layer-specific tensor IR validation errors.
///
/// Never matched individually by callers — only constructed and propagated via
/// `TensorIRError::Layer`. Grouped to reduce `TensorIRError` variant count.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorIRLayerError {
    // --- InstanceNorm1d ---
    #[error("InstanceNorm1d input must have at least 2 dimensions, got {rank}")]
    InstanceNormRankTooLow { rank: usize },

    #[error("InstanceNorm1d eps node must be scalar (shape [1]), got {shape:?}")]
    InstanceNormEpsNotScalar { shape: Vec<usize> },

    #[error("InstanceNorm1d axis {axis} must be the last axis (rank {rank}); normalizes over the last dimension only")]
    InstanceNormAxisNotLast { axis: usize, rank: usize },

    #[error("shape product overflow: {shape:?}")]
    ShapeProductOverflow { shape: Vec<usize> },

    #[error("InstanceNorm1d affine param {param} must have shape [{expected_channels}], got {got_shape:?}")]
    InstanceNormAffineShapeMismatch {
        param: &'static str,
        expected_channels: usize,
        got_shape: Vec<usize>,
    },

    #[error("InstanceNorm1d gamma and beta must both be present or both absent")]
    InstanceNormAffineMismatch,

    // --- Convolution (Conv1d, Conv2d, ConvTranspose1d) ---
    #[error(transparent)]
    Conv(#[from] TensorIRConvError),

    // --- Linear ---
    #[error("Linear weight must have 2 dimensions [out_features, in_features], got {shape:?}")]
    LinearWeightNotMatrix { shape: Vec<usize> },

    #[error(
        "Linear input last dimension ({input_features}) must equal weight in_features ({weight_in})"
    )]
    LinearFeatureMismatch {
        input_features: usize,
        weight_in: usize,
    },

    #[error("Linear bias must have shape [{expected}], got {got_shape:?}")]
    LinearBiasShape {
        expected: usize,
        got_shape: Vec<usize>,
    },

    #[error("Linear input must have at least 1 dimension")]
    LinearInputScalar,

    // --- MatMul ---
    #[error("MatMul {side} input must have at least 2 dimensions, got rank {rank}")]
    MatMulRankTooLow { side: String, rank: usize },

    #[error("MatMul contracted dimension mismatch: left K={left_k}, right K={right_k}")]
    MatMulDimMismatch { left_k: usize, right_k: usize },

    #[error("MatMul scale must be finite and non-zero, got {value}")]
    MatMulScaleInvalid { value: f32 },

    // --- RmsNorm ---
    #[error("RmsNorm input must have at least 2 dimensions, got {rank}")]
    RmsNormRankTooLow { rank: usize },

    #[error("RmsNorm eps node must be scalar (shape [1]), got {shape:?}")]
    RmsNormEpsNotScalar { shape: Vec<usize> },

    #[error("RmsNorm axis {axis} must be the last axis (rank {rank}); normalizes over the last dimension only")]
    RmsNormAxisNotLast { axis: usize, rank: usize },

    #[error("RmsNorm weight must have shape [{expected_hidden}], got {got_shape:?}")]
    RmsNormWeightShape {
        expected_hidden: usize,
        got_shape: Vec<usize>,
    },

    // --- BinaryAdd / BinaryMul ---
    #[error("BinaryAdd inputs have different shapes: left {left:?} vs right {right:?}")]
    BinaryAddShapeMismatch { left: Vec<usize>, right: Vec<usize> },

    #[error("BinaryMul inputs have different shapes: left {left:?} vs right {right:?}")]
    BinaryMulShapeMismatch { left: Vec<usize>, right: Vec<usize> },

    // --- AdaIN1d ---
    #[error("AdaIN1d input must have at least 2 dimensions, got {rank}")]
    AdaIN1dRankTooLow { rank: usize },

    #[error("AdaIN1d eps node must be scalar (shape [1]), got {shape:?}")]
    AdaIN1dEpsNotScalar { shape: Vec<usize> },

    #[error("AdaIN1d axis {axis} must be the last axis (rank {rank}); normalizes over the last dimension only")]
    AdaIN1dAxisNotLast { axis: usize, rank: usize },

    #[error("AdaIN1d {param} must have shape [{expected_channels}], got {got_shape:?}")]
    AdaIN1dStyleShapeMismatch {
        param: &'static str,
        expected_channels: usize,
        got_shape: Vec<usize>,
    },

    // --- Narrow ---
    #[error("narrow axis {axis} out of bounds for shape {shape:?}")]
    NarrowAxisOutOfBounds { axis: usize, shape: Vec<usize> },

    #[error("narrow length must be >= 1, got 0 at axis {axis}")]
    NarrowZeroLength { axis: usize },

    #[error(
        "narrow start ({start}) + length ({length}) = {} exceeds dimension size {dim} at axis {axis}",
        start + length
    )]
    NarrowOutOfBounds {
        start: usize,
        length: usize,
        dim: usize,
        axis: usize,
    },

    // --- Softmax ---
    #[error("Softmax input must have at least 1 dimension")]
    SoftmaxInputScalar,

    #[error(
        "Softmax axis {axis} out of bounds for rank {rank}; valid range is [{neg_rank}, {rank})"
    )]
    SoftmaxAxisOutOfBounds {
        axis: i32,
        rank: usize,
        neg_rank: i32,
    },

    // --- ZeroPad1d ---
    #[error("ZeroPad1d input must have at least 1 dimension")]
    ZeroPad1dScalarInput,

    #[error(
        "ZeroPad1d output length overflow: in_length={in_length} + pad_left={pad_left} + pad_right={pad_right}"
    )]
    ZeroPad1dOverflow {
        in_length: usize,
        pad_left: usize,
        pad_right: usize,
    },

    // --- Embedding ---
    #[error(
        "Embedding weight must have 2 dimensions [num_embeddings, embedding_dim], got {shape:?}"
    )]
    EmbeddingWeightNotMatrix { shape: Vec<usize> },

    #[error("Embedding input must have at least 1 dimension")]
    EmbeddingInputScalar,

    // --- LayerNorm ---
    #[error("LayerNorm input must have rank >= 2, got rank {rank}")]
    LayerNormRankTooLow { rank: usize },

    #[error("LayerNorm eps must be scalar [1], got shape {shape:?}")]
    LayerNormEpsNotScalar { shape: Vec<usize> },

    #[error("LayerNorm axis must be the last axis ({axis} + 1 != rank {rank})")]
    LayerNormAxisNotLast { axis: usize, rank: usize },

    #[error("LayerNorm weight shape mismatch: expected [{expected_hidden}], got {got_shape:?}")]
    LayerNormWeightShape {
        expected_hidden: usize,
        got_shape: Vec<usize>,
    },

    #[error("LayerNorm bias shape mismatch: expected [{expected_hidden}], got {got_shape:?}")]
    LayerNormBiasShape {
        expected_hidden: usize,
        got_shape: Vec<usize>,
    },

    // --- Attention ---
    #[error("Attention {side} input must have at least 2 dimensions, got rank {rank}")]
    AttentionRankTooLow { side: &'static str, rank: usize },

    #[error("Attention Q/K head dimension mismatch: Q D={q_d}, K D={k_d}")]
    AttentionHeadDimMismatch { q_d: usize, k_d: usize },

    #[error("Attention K/V sequence length mismatch: K T_kv={k_t}, V T_kv={v_t}")]
    AttentionKvSeqMismatch { k_t: usize, v_t: usize },

    #[error("Attention scale must be finite and positive, got {value}")]
    AttentionScaleInvalid { value: f32 },

    // --- LeakyRelu ---
    #[error("LeakyRelu negative_slope must be finite, got {value}")]
    LeakyReluSlopeInvalid { value: f32 },

    // --- Elu ---
    #[error("Elu alpha must be finite, got {value}")]
    EluAlphaInvalid { value: f32 },

    // --- LSTM ---
    #[error("LSTM input must have at least 1 dimension (scalar inputs not supported)")]
    LstmInputScalar,

    #[error(
        "LSTM hidden_state and cell_state shapes must match: hidden={hidden:?}, cell={cell:?}"
    )]
    LstmHiddenCellMismatch {
        hidden: Vec<usize>,
        cell: Vec<usize>,
    },

    #[error("LSTM weight_ih must have shape [4*H, I]=[{expected_rows}, {expected_cols}], got {got_shape:?}")]
    LstmWeightIhShape {
        expected_rows: usize,
        expected_cols: usize,
        got_shape: Vec<usize>,
    },

    #[error("LSTM weight_hh must have shape [4*H, H]=[{expected_rows}, {expected_cols}], got {got_shape:?}")]
    LstmWeightHhShape {
        expected_rows: usize,
        expected_cols: usize,
        got_shape: Vec<usize>,
    },

    #[error("LSTM bias must have shape [4*H]=[{expected}], got {got_shape:?}")]
    LstmBiasShape {
        expected: usize,
        got_shape: Vec<usize>,
    },

    #[error("LSTM weight_ih must have 2 dimensions [4*H, I], got {shape:?}")]
    LstmWeightIhNotMatrix { shape: Vec<usize> },

    #[error("LSTM weight_hh must have 2 dimensions [4*H, H], got {shape:?}")]
    LstmWeightHhNotMatrix { shape: Vec<usize> },

    #[error("LSTM {param} must be >= 1, got 0")]
    LstmZeroDimension { param: &'static str },

    // --- GLU ---
    #[error("GLU requires even dimension along axis {axis}, got {dim}")]
    GluOddDimension { axis: usize, dim: usize },

    // --- Concat ---
    #[error("concat must have at least two inputs")]
    EmptyConcat,

    #[error("concat inputs have different shapes at non-concat axis {axis}: expected {expected}, found {found}")]
    ConcatShapeMismatch {
        axis: usize,
        expected: usize,
        found: usize,
    },

    #[error("concat axis {axis} out of bounds for rank {rank}")]
    ConcatAxisOutOfBounds { axis: usize, rank: usize },

    #[error("concat inputs have different ranks: expected {expected}, found {found}")]
    ConcatRankMismatch { expected: usize, found: usize },

    // --- Transpose ---
    #[error("Transpose axes length ({axes_len}) must equal input rank ({rank})")]
    TransposeAxesLengthMismatch { axes_len: usize, rank: usize },

    #[error("Transpose axis {axis} out of bounds for rank {rank}")]
    TransposeAxisOutOfBounds { axis: usize, rank: usize },

    #[error("Transpose axes contain duplicate axis {axis}")]
    TransposeDuplicateAxis { axis: usize },

    // --- MHA (Multi-Head Attention) ---
    #[error("MHA model dimension ({model_dim}) must be divisible by num_heads ({num_heads})")]
    MhaHeadDimNotDivisible { model_dim: usize, num_heads: usize },

    #[error("MHA input must have exactly 2 dimensions [T, D], got rank {rank}")]
    MhaInputRankInvalid { rank: usize },

    #[error("MHA num_heads must be >= 1, got 0")]
    MhaZeroHeads,

    // --- Transformer ---
    #[error("Transformer block num_heads must be >= 1, got 0")]
    TransformerZeroHeads,

    #[error("Transformer block ffn_hidden_dim must be >= 1, got 0")]
    TransformerZeroFfnDim,

    #[error("Transformer block input must have exactly 2 dimensions [T, D], got rank {rank}")]
    TransformerInputRankInvalid { rank: usize },

    // --- BatchNorm ---
    #[error("BatchNorm input must have at least 2 dimensions, got {rank}")]
    BatchNormRankTooLow { rank: usize },

    #[error("BatchNorm eps node must be scalar (shape [1]), got {shape:?}")]
    BatchNormEpsNotScalar { shape: Vec<usize> },

    #[error("BatchNorm {param} must have shape [{expected_channels}], got {got_shape:?}")]
    BatchNormParamShape {
        param: &'static str,
        expected_channels: usize,
        got_shape: Vec<usize>,
    },

    // --- GatedDeltaNet ---
    #[error("GatedDeltaNet {param} must be >= 1, got 0")]
    GatedDeltaNetZeroDimension { param: &'static str },

    #[error("GatedDeltaNet Q/K head dimension mismatch: Q last={q_dim}, K last={k_dim}")]
    GatedDeltaNetQkDimMismatch { q_dim: usize, k_dim: usize },

    #[error("GatedDeltaNet state shape must be [*, H, K, V], got {shape:?}")]
    GatedDeltaNetStateShape { shape: Vec<usize> },

    #[error("GatedDeltaNet scale must be finite and positive, got {value}")]
    GatedDeltaNetScaleInvalid { value: f32 },
}
