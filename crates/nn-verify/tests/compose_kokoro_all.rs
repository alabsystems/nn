// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Kokoro TTS composition tests.
//!
//! Merges 42 `compose_kokoro_*.rs` test modules into one binary to reduce
//! link-time overhead from redundant NY linkage.
//!
//! Part of #1982: Test binary consolidation.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_production_weights.rs"]
mod kokoro_production_weights;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_production_segments.rs"]
mod kokoro_production_segments;

#[path = "helpers/compose_kokoro_decoder.rs"]
mod compose_kokoro_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_decoder_scaled.rs"]
mod compose_kokoro_decoder_scaled;

#[path = "helpers/compose_kokoro_disentanglement.rs"]
mod compose_kokoro_disentanglement;

#[path = "helpers/compose_kokoro_duration_parametric.rs"]
mod compose_kokoro_duration_parametric;

#[path = "helpers/compose_kokoro_duration_scaled_parametric.rs"]
mod compose_kokoro_duration_scaled_parametric;

#[path = "helpers/compose_kokoro_f0_disentanglement.rs"]
mod compose_kokoro_f0_disentanglement;

#[path = "helpers/compose_kokoro_full_pipeline.rs"]
mod compose_kokoro_full_pipeline;

#[path = "helpers/compose_kokoro_layerwise_d128.rs"]
mod compose_kokoro_layerwise_d128;

#[path = "helpers/compose_kokoro_layerwise_d512.rs"]
mod compose_kokoro_layerwise_d512;

#[path = "helpers/compose_kokoro_layerwise_deep.rs"]
mod compose_kokoro_layerwise_deep;

#[path = "helpers/compose_kokoro_scaled_pipeline.rs"]
mod compose_kokoro_scaled_pipeline;

#[path = "helpers/compose_kokoro_speaker_pipeline.rs"]
mod compose_kokoro_speaker_pipeline;

#[path = "helpers/compose_kokoro_temporal_bounds.rs"]
mod compose_kokoro_temporal_bounds;

#[path = "helpers/compose_kokoro_forward_mode.rs"]
mod compose_kokoro_forward_mode;

#[path = "helpers/compose_kokoro_trace_full.rs"]
mod compose_kokoro_trace_full;

#[path = "helpers/compose_kokoro_trace_full_generator.rs"]
mod compose_kokoro_trace_full_generator;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_traced.rs"]
mod compose_kokoro_traced;

#[path = "helpers/compose_kokoro_traced_structure.rs"]
mod compose_kokoro_traced_structure;

#[path = "helpers/compose_kokoro_f0_plbert.rs"]
mod compose_kokoro_f0_plbert;

#[path = "helpers/compose_kokoro_ibp_scaling.rs"]
mod compose_kokoro_ibp_scaling;

#[allow(dead_code)]
#[path = "helpers/compose_kokoro_alpha_crown.rs"]
mod compose_kokoro_alpha_crown;

#[path = "helpers/compose_kokoro_layerwise_traced.rs"]
mod compose_kokoro_layerwise_traced;

#[path = "helpers/compose_kokoro_layerwise_grouped.rs"]
mod compose_kokoro_layerwise_grouped;

#[path = "helpers/compose_kokoro_layerwise_grouped_he.rs"]
mod compose_kokoro_layerwise_grouped_he;

#[path = "helpers/compose_kokoro_layerwise_mixed.rs"]
mod compose_kokoro_layerwise_mixed;

#[path = "helpers/compose_kokoro_full_decoder.rs"]
mod compose_kokoro_full_decoder;

#[path = "helpers/compose_kokoro_decoder_traced.rs"]
mod compose_kokoro_decoder_traced;

#[path = "helpers/compose_kokoro_prosody_traced.rs"]
mod compose_kokoro_prosody_traced;

#[path = "helpers/compose_kokoro_scaled_traced.rs"]
mod compose_kokoro_scaled_traced;

#[path = "helpers/compose_kokoro_generator_subblock.rs"]
mod compose_kokoro_generator_subblock;

#[path = "helpers/compose_kokoro_generator_subblock_contracts.rs"]
mod compose_kokoro_generator_subblock_contracts;

#[path = "helpers/compose_kokoro_generator_d512_ibp.rs"]
mod compose_kokoro_generator_d512_ibp;

#[path = "helpers/compose_kokoro_pipeline_traced.rs"]
mod compose_kokoro_pipeline_traced;

#[path = "helpers/compose_kokoro_plbert_traced.rs"]
mod compose_kokoro_plbert_traced;

#[path = "helpers/compose_kokoro_generator_d512_crown.rs"]
mod compose_kokoro_generator_d512_crown;

#[path = "helpers/compose_kokoro_generator_d512_mixed.rs"]
mod compose_kokoro_generator_d512_mixed;

#[path = "helpers/compose_kokoro_istft.rs"]
mod compose_kokoro_istft;

#[path = "helpers/compose_kokoro_forward_stft.rs"]
mod compose_kokoro_forward_stft;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production.rs"]
mod compose_kokoro_production;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_crown.rs"]
mod compose_kokoro_production_crown;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_moonshot.rs"]
mod compose_kokoro_production_moonshot;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_segments.rs"]
mod compose_kokoro_production_segments;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_conv_transpose_layernorm.rs"]
mod compose_kokoro_conv_transpose_layernorm;

#[path = "helpers/kokoro_segment_certification.rs"]
mod kokoro_segment_certification;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_sound_reverify.rs"]
mod compose_kokoro_sound_reverify;

#[path = "helpers/compose_kokoro_fused_resblock.rs"]
mod compose_kokoro_fused_resblock;

#[path = "helpers/compose_kokoro_multi_stage.rs"]
mod compose_kokoro_multi_stage;

#[path = "helpers/compose_kokoro_sound_promotion.rs"]
mod compose_kokoro_sound_promotion;

#[path = "helpers/compose_kokoro_heuristic_promotion.rs"]
mod compose_kokoro_heuristic_promotion;

#[path = "helpers/compose_kokoro_stale_promotion.rs"]
mod compose_kokoro_stale_promotion;

#[path = "helpers/compose_kokoro_cross_stage_sound.rs"]
mod compose_kokoro_cross_stage_sound;

#[path = "helpers/compose_kokoro_deep_chain.rs"]
mod compose_kokoro_deep_chain;

#[path = "helpers/compose_kokoro_sinegen.rs"]
mod compose_kokoro_sinegen;

#[path = "helpers/compose_kokoro_sinegen_sound.rs"]
mod compose_kokoro_sinegen_sound;

#[path = "helpers/compose_kokoro_chorus_blend.rs"]
mod compose_kokoro_chorus_blend;

#[path = "helpers/compose_kokoro_resblock_bounds.rs"]
mod compose_kokoro_resblock_bounds;

#[path = "helpers/compose_kokoro_resblock_equivalence.rs"]
mod compose_kokoro_resblock_equivalence;

#[path = "helpers/compose_kokoro_duration_bounds.rs"]
mod compose_kokoro_duration_bounds;

#[path = "helpers/compose_kokoro_encoder_bounds.rs"]
mod compose_kokoro_encoder_bounds;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_generator_bounds.rs"]
mod compose_kokoro_generator_bounds;

#[path = "helpers/compose_kokoro_gap_fill.rs"]
mod compose_kokoro_gap_fill;
