#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code, unreachable_pub)]

//! Consolidated training tests: ops, pipeline, checkpoint, GPU, conv, MHA+SwiGLU, LSTM.

mod common;

#[path = "training/training_conv_gpu.rs"]
mod training_conv_gpu;
#[path = "training/training_e2e.rs"]
mod training_e2e;
#[path = "training/training_mha_swiglu.rs"]
mod training_mha_swiglu;
#[path = "training/training_ops.rs"]
mod training_ops;
#[path = "training/training_ops_extra.rs"]
mod training_ops_extra;
#[path = "training/training_ops_extra_lstm.rs"]
mod training_ops_extra_lstm;
#[path = "training/training_ops_pool.rs"]
mod training_ops_pool;
#[path = "training/training_pipeline.rs"]
mod training_pipeline;
#[path = "training/training_pipeline_checkpoint.rs"]
mod training_pipeline_checkpoint;
#[path = "training/training_pipeline_gpu.rs"]
mod training_pipeline_gpu;
