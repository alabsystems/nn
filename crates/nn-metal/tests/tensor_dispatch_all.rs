// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated tensor dispatch tests: MSL codegen, dispatch plans, and
//! GPU execution across all dispatch modes.

mod test_utils;

#[path = "tensor_dispatch/activations.rs"]
mod tensor_dispatch_activations;
#[path = "tensor_dispatch/batched.rs"]
mod tensor_dispatch_batched;
#[path = "tensor_dispatch/buffer.rs"]
mod tensor_dispatch_buffer;
#[path = "tensor_dispatch/causal_conv1d.rs"]
mod tensor_dispatch_causal_conv1d;
#[path = "tensor_dispatch/core.rs"]
mod tensor_dispatch_core;
#[path = "tensor_dispatch/f16.rs"]
mod tensor_dispatch_f16;
#[path = "tensor_dispatch/packed.rs"]
mod tensor_dispatch_packed;
#[path = "tensor_dispatch/packed_elementwise.rs"]
mod tensor_dispatch_packed_elementwise;
#[path = "tensor_dispatch/simdgroup.rs"]
mod tensor_dispatch_simdgroup;
