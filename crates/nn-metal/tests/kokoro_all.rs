// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated Kokoro tests: compiled dispatch, synthesis, AdaLN/ResBlock
//! equivalence, and L3 parity.

mod test_utils;

#[path = "compiled_model_test_helpers.rs"]
mod helpers;

mod kokoro_test_env;
mod kokoro_test_weights;

#[path = "kokoro/compiled_adaln_equivalence.rs"]
mod compiled_adaln_equivalence;
#[path = "kokoro/compiled_kokoro_autocast.rs"]
mod compiled_kokoro_autocast;
#[path = "kokoro/compiled_kokoro_autocast_chorus.rs"]
mod compiled_kokoro_autocast_chorus;
#[path = "kokoro/compiled_kokoro_autocast_pipeline.rs"]
mod compiled_kokoro_autocast_pipeline;
#[path = "kokoro/compiled_kokoro_autocast_segments.rs"]
mod compiled_kokoro_autocast_segments;
#[path = "kokoro/compiled_kokoro_chorus.rs"]
mod compiled_kokoro_chorus;
#[path = "kokoro/compiled_kokoro_clone_dispatch.rs"]
mod compiled_kokoro_clone_dispatch;
#[path = "kokoro/compiled_kokoro_hard_bounds.rs"]
mod compiled_kokoro_hard_bounds;
#[path = "kokoro/compiled_kokoro_pipelined.rs"]
mod compiled_kokoro_pipelined;
#[path = "kokoro/compiled_kokoro_streaming.rs"]
mod compiled_kokoro_streaming;
#[path = "kokoro/compiled_kokoro_synthesize.rs"]
mod compiled_kokoro_synthesize;
#[path = "kokoro/compiled_resblock_equivalence.rs"]
mod compiled_resblock_equivalence;
#[path = "kokoro/kokoro_arena_presizing.rs"]
mod kokoro_arena_presizing;
#[path = "kokoro/kokoro_audio_quality.rs"]
mod kokoro_audio_quality;
#[path = "kokoro/kokoro_auto_convert_parity.rs"]
mod kokoro_auto_convert_parity;
#[path = "kokoro/kokoro_benchmark.rs"]
mod kokoro_benchmark;
#[path = "kokoro/kokoro_benchmark_d512.rs"]
mod kokoro_benchmark_d512;
#[path = "kokoro/kokoro_chorus_gates.rs"]
mod kokoro_chorus_gates;
#[path = "kokoro/kokoro_chorus_streaming.rs"]
mod kokoro_chorus_streaming;
#[path = "kokoro/kokoro_dispatch_audit.rs"]
mod kokoro_dispatch_audit;
#[path = "kokoro/kokoro_dispatch_census.rs"]
mod kokoro_dispatch_census;
#[path = "kokoro/kokoro_dispatch_decomposition.rs"]
mod kokoro_dispatch_decomposition;
#[path = "kokoro/kokoro_dispatch_profiler.rs"]
mod kokoro_dispatch_profiler;
#[path = "kokoro/kokoro_gap_analysis.rs"]
mod kokoro_gap_analysis;
#[path = "kokoro/kokoro_gates.rs"]
mod kokoro_gates;
#[path = "kokoro/kokoro_generator_census.rs"]
mod kokoro_generator_census;
#[path = "kokoro/kokoro_gpu_profile.rs"]
mod kokoro_gpu_profile;
#[path = "kokoro/kokoro_l3_parity.rs"]
mod kokoro_l3_parity;
#[path = "kokoro/kokoro_metal_forward.rs"]
mod kokoro_metal_forward;
#[path = "kokoro/kokoro_model_serialization.rs"]
mod kokoro_model_serialization;
#[path = "kokoro/kokoro_optimize.rs"]
mod kokoro_optimize;
#[path = "kokoro/kokoro_optimize_production.rs"]
mod kokoro_optimize_production;
#[path = "kokoro/kokoro_partition_impact.rs"]
mod kokoro_partition_impact;
#[path = "kokoro/kokoro_pass_impact_production.rs"]
mod kokoro_pass_impact_production;
#[path = "kokoro/kokoro_production_chorus.rs"]
mod kokoro_production_chorus;
#[path = "kokoro/kokoro_profiler_feedback.rs"]
mod kokoro_profiler_feedback;
#[path = "kokoro/kokoro_release_weights.rs"]
mod kokoro_release_weights;
#[path = "kokoro/kokoro_resblock_chain.rs"]
mod kokoro_resblock_chain;
#[path = "kokoro/kokoro_roofline_analysis.rs"]
mod kokoro_roofline_analysis;
#[path = "kokoro/kokoro_rtf_production.rs"]
mod kokoro_rtf_production;
#[path = "kokoro/kokoro_segment_dispatch_profile.rs"]
mod kokoro_segment_dispatch_profile;
#[path = "kokoro/kokoro_streaming_chorus_gates.rs"]
mod kokoro_streaming_chorus_gates;
#[path = "kokoro/kokoro_streaming_chorus_production.rs"]
mod kokoro_streaming_chorus_production;
