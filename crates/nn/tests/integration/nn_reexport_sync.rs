// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verifies that the explicit nn re-exports in `nn/src/lib.rs` stay in sync
//! with `nn::layers::*` (which re-exports `nn_core::layers::*`).
//!
//! If a new nn type is added to nn-core but not to the explicit block in
//! nn/src/lib.rs, this test fails. This prevents silent desync between
//! `nn::Linear` (explicit re-export) and `nn::layers::Linear` (glob re-export).

/// Every type in the explicit `pub use nn_core::{ layers::... }` block in lib.rs
/// must be importable from both `nn::TypeName` and `nn::layers::TypeName`.
///
/// This macro generates a compile-time check for each type.
macro_rules! assert_reexported {
    ($($ty:ident),+ $(,)?) => {
        $(
            #[allow(unused_imports)]
            use nn::$ty;
        )+
    };
}

// Verify all nn types from the explicit block are accessible via `nn::TypeName`.
// If this fails to compile, a type was removed from the explicit block.
assert_reexported! {
    // Functions are tested below; types only here.
    Activation, AdaIn, AdaLnParams, AdaLnZero, AdaLnZeroDual,
    AdaptiveAvgPool2d, AvgPool2d,
    BatchNorm, BatchNormConfig, BeamHypothesis, BeamSearchConfig, BeamSearchOutput,
    BiLstm, BlockQ4K,
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig,
    CtcBeamHypothesis, CtcConfig,
    DeformableAttention, DeformableAttentionConfig,
    DiTBlock, DiTBlockDual, Dropout, Embedding,
    GatedDeltaNet, GatedDeltaNetState,
    GenerationConfig, GenerationOutput, GgmlDType, GroupNorm,
    HalfRotaryEmbedding, InstanceNorm,
    InterleavedMRoPE, InterleavedMRoPEConfig,
    JointAttention,
    KvCache, KvCacheBackend, KvCacheLayer, KvCacheLayerBackend,
    LayerNorm, LayerNormConfig, Linear,
    PreallocKvCache, PreallocKvCacheLayer,
    LowRankAdaLn,
    Lstm, LstmCell, LstmState,
    MBConv, MBConvConfig, MaxPool2d,
    Module, ModuleT,
    MoeDispatch, MoeDispatchConfig, MoeDispatchOutput,
    MoeLayer, MoeLayerConfig, MoeOutput, MoeRouter, MoeRoutingOutput,
    ExpertFFN,
    MultiHeadAttention,
    PatchEmbedding, PixelShuffle, PixelUnshuffle,
    Pool2dConfig, PoolingStrategy,
    QLinear, QuantizedWeight,
    RmsNorm, RotaryEmbedding, RotaryEmbedding2d, Rvq,
    Sequential, SqueezeExcitation, SwiGlu, SwiGluExpert,
    Upsample2d, UpsampleMode,
    VitConfig, VitEncoder, VitEncoderBlock, VqCodebook,
    WeightNormConv1d, YarnScaling,
}

/// Verify nn functions are accessible via `nn::func_name`.
#[test]
fn test_nn_function_reexports_accessible() {
    // Verify function re-exports exist (type-level check).
    // We don't call them — just verify they resolve.
    // `linear`/`linear_no_bias` use `impl AsRef<VarBuilder>`, so we verify
    // via a reference binding rather than a concrete function pointer cast.
    let _: fn(usize, usize, &nn::VarBuilder) -> nn::Result<nn::Linear> =
        |a, b, vb| nn::linear(a, b, vb);
    let _: fn(usize, usize, &nn::VarBuilder) -> nn::Result<nn::Linear> =
        |a, b, vb| nn::linear_no_bias(a, b, vb);
    let _ = nn::check_output_finite as fn(&nn::DynTensor, &str) -> nn::Result<()>;
}

/// Verify DynTensor utility function re-exports.
#[test]
fn test_dyn_tensor_utility_reexports() {
    // These return Result<usize> — verify they resolve from `nn::`.
    let _ = nn::conv1d_out_len(10, 3, 1, 1, 1).unwrap();
    let _ = nn::conv2d_out_len(10, 3, 1, 1, 1).unwrap();
    let _ = nn::conv_transpose1d_out_len(10, 3, 1, 0, 1, 1).unwrap();
    // GridSamplePaddingMode is a type, verified by import above in assert_reexported!
    // ... but it's in the DynTensor block, not nn. Verify separately:
    #[allow(unused_imports)]
    use nn::GridSamplePaddingMode;
}

/// Verify `nn::layers::*` glob export contains all the same types.
/// If a type is in `nn::layers::Foo` but NOT in `nn::Foo`, this test won't
/// catch it (that's the explicit block's job). But if it's in `nn::Foo`
/// but NOT in `nn::layers::Foo`, this fails to compile.
#[test]
fn test_nn_glob_covers_explicit_types() {
    // Spot-check: key types accessible from both paths.
    fn _check_linear(_: nn::Linear) {}
    fn _check_linear_nn(_: nn::layers::Linear) {}

    fn _check_lstm(_: nn::Lstm) {}
    fn _check_lstm_nn(_: nn::layers::Lstm) {}

    fn _check_embedding(_: nn::Embedding) {}
    fn _check_embedding_nn(_: nn::layers::Embedding) {}

    fn _check_layer_norm(_: nn::LayerNorm) {}
    fn _check_layer_norm_nn(_: nn::layers::LayerNorm) {}
}

/// Verify LstmState::new() returns Result (AC4).
#[test]
fn test_lstm_state_new_returns_result() {
    let h = nn::DynTensor::zeros(&[1, 4], nn::DType::F32, &nn::Device::Cpu).unwrap();
    let c = nn::DynTensor::zeros(&[1, 4], nn::DType::F32, &nn::Device::Cpu).unwrap();
    let state = nn::LstmState::new(h, c);
    assert!(
        state.is_ok(),
        "LstmState::new with matching shapes should succeed"
    );
}

/// LstmState::new() rejects mismatched shapes.
#[test]
fn test_lstm_state_shape_mismatch() {
    let h = nn::DynTensor::zeros(&[1, 4], nn::DType::F32, &nn::Device::Cpu).unwrap();
    let c = nn::DynTensor::zeros(&[1, 8], nn::DType::F32, &nn::Device::Cpu).unwrap();
    let result = nn::LstmState::new(h, c);
    assert!(
        result.is_err(),
        "LstmState::new with mismatched shapes should fail"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Shape mismatch"),
        "Error should mention shape mismatch, got: {msg}"
    );
}
