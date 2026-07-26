// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated INT8 quantization NY composition verification tests.
//!
//! Proves that INT8 W8A16 (weight-only) quantization preserves output bounds
//! for Linear layers via NY IBP and CROWN propagation.
//!
//! Tests cover:
//! - Small (64->64) and medium (256->256) Linear layers
//! - F32 vs INT8-quantized weight bounds propagation
//! - Quantization drift bounded by theoretical INT8 error
//! - CROWN tightness verification
//!
//! Part of #3533: INT8 quantization NY soundness proof.
//! Part of #3525.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_int8_quantization.rs"]
mod int8_quantization;
