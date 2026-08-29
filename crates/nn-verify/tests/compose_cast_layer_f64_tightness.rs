// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ToDtype compose tests with f64 tightness validation.
//!
//! Posture (ny hardening 25dee0c5e): only the F32 -> F32 identity ToDtype
//! lowers (as an Add(x, 0) identity); every non-F32 cast — F16/BF16/F64
//! target or non-F32 source — is refused fail-closed because the non-F32
//! endpoint's rounding semantics are unmodeled. These modules cover the
//! restored identity lowering's IBP/CROWN behavior, one representative
//! refusal per non-F32 dtype, and f64-evaluation tightness baselines.
//!
//! Historical: #4316 (CastLayer for ToDtype verification + f64 evaluation).

#![allow(clippy::duplicate_mod)]

mod common;

// IBP/CROWN propagation tests through CastLayer and Clamp.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_cast_layer_f64_tightness.rs"]
mod compose_cast_layer_ibp_crown;

// f64 tightness comparison and CastLayer zero-impact validation.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_cast_layer_f64_comparison.rs"]
mod compose_cast_layer_f64_comparison;
