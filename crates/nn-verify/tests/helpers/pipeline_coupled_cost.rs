// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper for coupled CROWN + cost propagation tests.
//!
//! Builds layers in the format required by `verify_layerwise_coupled()`:
//! each layer is a `(TensorKernelDef, Vec<TensorParamBinding>)` pair.
//!
//! The coupled test uses the *same* `TensorKernelDef` for both CROWN
//! propagation and dispatch plan generation, guaranteeing that the
//! cost profile accurately reflects the verified computation.
//!
//! Layer chain: SpectralEncoderBlock → KokoroDecoder
//!   - SpectralEncoderBlock: [in_channels=4, spatial_in=16] → [out_channels=8, conv_f_out=4]
//!   - KokoroDecoder:        [IN_CHANNELS=8, TIME_IN=4] → [OUT_CHANNELS=4, TIME_UP=8]
//!
//! The output dimension `[8, 4]` of the spectral encoder exactly matches the
//! input dimension `[8, 4]` of the Kokoro decoder, enabling sequential
//! chaining through `verify_layerwise_coupled()`.
//!
//! Part of #1739: Provable Computational Boundedness.
//! Part of #1741: THE MOONSHOT — Property 5 (Temporal Boundedness).

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;

/// A layer specification for coupled verification.
pub(crate) struct CoupledLayer {
    pub(super) def: TensorKernelDef,
    pub(super) bindings: Vec<TensorParamBinding>,
    pub(super) name: String,
}

/// Build layers for coupled CROWN + cost verification.
///
/// Returns the spectral encoder block and Kokoro decoder as two layers
/// that chain sequentially: spectral encoder output `[8, 4]` feeds
/// directly into Kokoro decoder input `[8, 4]`.
///
/// Each layer's `TensorKernelDef` is used for both:
/// 1. `tensor_kernel_to_graph()` → CROWN propagation (numerical bounds)
/// 2. `build_dispatch_plan()` → cost profiling (FLOPs, memory, timing)
pub(crate) fn build_coupled_layers() -> Vec<CoupledLayer> {
    let (spec_def, _conv_f_out, _out_channels) =
        super::enc_helpers::build_encoder_block(&super::enc_helpers::SPECTRAL_CONFIG);
    let spec_bindings =
        super::enc_helpers::encoder_block_bindings(&super::enc_helpers::SPECTRAL_CONFIG);

    let (dec_def, _out_shape) = super::kokoro_decoder::build_kokoro_decoder();
    let dec_bindings = super::kokoro_decoder::kokoro_decoder_bindings();

    vec![
        CoupledLayer {
            def: spec_def,
            bindings: spec_bindings,
            name: "spectral_encoder_block".to_string(),
        },
        CoupledLayer {
            def: dec_def,
            bindings: dec_bindings,
            name: "kokoro_decoder".to_string(),
        },
    ]
}

/// Convert `CoupledLayer` list to the tuple format expected by
/// `verify_layerwise_coupled()`.
pub(crate) fn layers_to_tuples(
    layers: &[CoupledLayer],
) -> Vec<(TensorKernelDef, Vec<TensorParamBinding>)> {
    layers
        .iter()
        .map(|l| (l.def.clone(), l.bindings.clone()))
        .collect()
}

/// Input shape for the first layer (spectral encoder).
pub(crate) const SPECTRAL_INPUT_SHAPE: [usize; 2] = [
    super::enc_helpers::SPECTRAL_CONFIG.in_channels,
    super::enc_helpers::SPECTRAL_CONFIG.spatial_in,
];
