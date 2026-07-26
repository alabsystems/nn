// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated pipeline tests: end-to-end, kernel pipeline, MSL pipeline,
//! metallib precompilation, and safetensors loading.

#[cfg(feature = "verify")]
#[path = "pipeline/end_to_end_pipeline.rs"]
mod end_to_end_pipeline;

#[path = "pipeline/kernel_pipeline.rs"]
mod kernel_pipeline;

#[path = "pipeline/kernel_pipeline_from_msl.rs"]
mod kernel_pipeline_from_msl;

#[path = "pipeline/precompile_metallib.rs"]
mod precompile_metallib;

#[path = "pipeline/safetensors_load.rs"]
mod safetensors_load;
