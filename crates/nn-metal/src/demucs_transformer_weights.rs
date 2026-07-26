// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thin re-export wrapper — implementation lives in nn-models.
//!
//! Part of #860: extract backend-agnostic weight types to nn-models.

pub use nn_models::demucs_transformer_weights::DemucsTransformerWeights;
pub(crate) use nn_models::demucs_transformer_weights::{
    CrossAttentionLayerWeights, LayerNormWeights, SelfAttentionLayerWeights,
    TransformerLayerWeights,
};
