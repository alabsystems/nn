// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated compiled model tests: end-to-end Metal dispatch plan
//! construction, execution, NativeOp dispatch, and validation.

mod test_utils;

#[path = "compiled_model_test_helpers.rs"]
mod helpers;

mod kokoro_test_env;
mod kokoro_test_weights;

#[path = "compiled_model/adain_edge_map_test.rs"]
mod compiled_model_adain_edge_map;
#[path = "compiled_model/adain_nativeop_test.rs"]
mod compiled_model_adain_nativeop;
#[path = "compiled_model/adaln_nativeop_test.rs"]
mod compiled_model_adaln_nativeop;
#[path = "compiled_model/autocast_test.rs"]
mod compiled_model_autocast;
#[path = "compiled_model/batched_qkv_e2e.rs"]
mod compiled_model_batched_qkv_e2e;
#[path = "compiled_model/blit_elimination_test.rs"]
mod compiled_model_blit_elimination;
#[path = "compiled_model/e2e.rs"]
mod compiled_model_e2e;
#[path = "compiled_model/f16_mixed_precision_test.rs"]
mod compiled_model_f16_mixed_precision;
#[path = "compiled_model/flash_attn_nativeop_test.rs"]
mod compiled_model_flash_attn_nativeop;
#[path = "compiled_model/flash_attn_seq_first_test.rs"]
mod compiled_model_flash_attn_seq_first;
#[path = "compiled_model/fused_resblock_autocast_test.rs"]
mod compiled_model_fused_resblock_autocast;
#[path = "compiled_model/glm5_decoder.rs"]
mod compiled_model_glm5_decoder;
#[path = "compiled_model/icb_frame_bucket_test.rs"]
mod compiled_model_icb_frame_bucket;
#[path = "compiled_model/icb_replay_test.rs"]
mod compiled_model_icb_replay;
#[path = "compiled_model/kokoro_cross_path_parity.rs"]
mod compiled_model_kokoro_cross_path_parity;
#[path = "compiled_model/kokoro_e2e.rs"]
mod compiled_model_kokoro_e2e;
#[path = "compiled_model/kokoro_f0_precision.rs"]
mod compiled_model_kokoro_f0_precision;
#[path = "compiled_model/kokoro_pipeline.rs"]
mod compiled_model_kokoro_pipeline;
#[path = "compiled_model/lstm_bounds.rs"]
mod compiled_model_lstm_bounds;
#[path = "compiled_model/mixed_gemm_test.rs"]
mod compiled_model_mixed_gemm;
#[path = "compiled_model/mlp_test.rs"]
mod compiled_model_mlp_test;
#[path = "compiled_model/multi_output_test.rs"]
mod compiled_model_multi_output_test;
#[path = "compiled_model/narrow_view_validation_test.rs"]
mod compiled_model_narrow_view_validation;
#[path = "compiled_model/nativeop_bounds.rs"]
mod compiled_model_nativeop_bounds;
#[path = "compiled_model/norm_activ_conv1d_nativeop_test.rs"]
mod compiled_model_norm_activ_conv1d_nativeop;
#[path = "compiled_model/ops_e2e.rs"]
mod compiled_model_ops_e2e;
#[path = "compiled_model/ops_e2e_10.rs"]
mod compiled_model_ops_e2e_10;
#[path = "compiled_model/ops_e2e_11.rs"]
mod compiled_model_ops_e2e_11;
#[path = "compiled_model/ops_e2e_12.rs"]
mod compiled_model_ops_e2e_12;
#[path = "compiled_model/ops_e2e_2.rs"]
mod compiled_model_ops_e2e_2;
#[path = "compiled_model/ops_e2e_3.rs"]
mod compiled_model_ops_e2e_3;
#[path = "compiled_model/ops_e2e_4.rs"]
mod compiled_model_ops_e2e_4;
#[path = "compiled_model/ops_e2e_5.rs"]
mod compiled_model_ops_e2e_5;
#[path = "compiled_model/ops_e2e_6.rs"]
mod compiled_model_ops_e2e_6;
#[path = "compiled_model/ops_e2e_7.rs"]
mod compiled_model_ops_e2e_7;
#[path = "compiled_model/ops_e2e_8.rs"]
mod compiled_model_ops_e2e_8;
#[path = "compiled_model/ops_e2e_9.rs"]
mod compiled_model_ops_e2e_9;
#[path = "compiled_model/profiled_test.rs"]
mod compiled_model_profiled;
#[path = "compiled_model/qwen3_decoder.rs"]
mod compiled_model_qwen3_decoder;
#[path = "compiled_model/runtime_op.rs"]
mod compiled_model_runtime_op;
#[path = "compiled_model/simdgroup_e2e.rs"]
mod compiled_model_simdgroup_e2e;
#[path = "compiled_model/simple_nativeop_test.rs"]
mod compiled_model_simple_nativeop;
#[path = "compiled_model/test.rs"]
mod compiled_model_test;
#[path = "compiled_model/tiled_gemm_e2e.rs"]
mod compiled_model_tiled_gemm_e2e;
#[path = "compiled_model/trace_e2e.rs"]
mod compiled_model_trace_e2e;
#[path = "compiled_model/upsample_conv1d_nativeop_test.rs"]
mod compiled_model_upsample_conv1d_nativeop;
#[path = "compiled_model/validation_test.rs"]
mod compiled_model_validation_test;
#[cfg(feature = "verify")]
#[path = "compiled_model/verify_test.rs"]
mod compiled_model_verify;
#[path = "compiled_model/vit_encoder.rs"]
mod compiled_model_vit_encoder;
#[path = "compiled_model/whisper_decoder.rs"]
mod compiled_model_whisper_decoder;
#[path = "compiled_model/whisper_encoder.rs"]
mod compiled_model_whisper_encoder;
