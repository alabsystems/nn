// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated HTDemucs certificate/full/deep/separator/encoder composition tests.
//!
//! Combines 5 HTDemucs test files into a single test binary to reduce
//! compilation overhead (5 NY link steps -> 1).
//!
//! - `htdemucs_certificate`: Proof certificate generation + validation
//! - `htdemucs_full`: Full model (encoder + cross-domain transformer + decoder)
//! - `htdemucs_deep`: Deep encoder blocks (depth 4-5 DConv with LSTM)
//! - `htdemucs_separator`: Separator block composition (two-stage encoder,
//!   enc+transformer+dec Conservative, decoder DConv, multi-step LSTM)
//! - `htdemucs_encoder`: Encoder sub-stage IBP bounds (Conv1d stride, GroupNorm,
//!   GELU, DConv, two-stage stacking, spectral path)
//!
//! Part of #1982, #4186, #4278, #4314.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_certificate.rs"]
mod htdemucs_certificate;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_full.rs"]
mod htdemucs_full;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_deep.rs"]
mod htdemucs_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_separator.rs"]
mod htdemucs_separator;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_encoder.rs"]
mod htdemucs_encoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_transformer.rs"]
mod htdemucs_transformer;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_gelu_reverify.rs"]
mod htdemucs_gelu_reverify;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_htdemucs_gelu_reverify_deep.rs"]
mod htdemucs_gelu_reverify_deep;
