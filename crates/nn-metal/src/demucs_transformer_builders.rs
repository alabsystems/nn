// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thin re-export wrapper — canonical implementations live in nn-models.
//!
//! Part of #860 (model code extraction from nn-metal).

pub(super) use nn_models::demucs_transformer_builders::{
    build_channel_bridge_def, build_conv1d_weight_map, build_cross_attention_layer_def,
    build_layer_norm_def, build_self_attention_layer_def,
};
pub(super) use nn_models::demucs_transformer_validate::validate_all_weights;
