// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated HTDemucs tests: DConv debug, step traces, e2e contracts,
//! spectral path, and encoder smoke tests.

mod test_utils;

#[path = "demucs_test_utils.rs"]
mod demucs_test_utils;

#[path = "demucs_e2e_spectral_helpers.rs"]
mod demucs_e2e_spectral_helpers;

#[path = "demucs/demucs_dconv_debug.rs"]
mod demucs_dconv_debug;
#[path = "demucs/demucs_dconv_step_trace.rs"]
mod demucs_dconv_step_trace;
#[path = "demucs/demucs_e2e.rs"]
mod demucs_e2e;
#[path = "demucs/demucs_e2e_contract.rs"]
mod demucs_e2e_contract;
#[path = "demucs/demucs_e2e_spectral_contract.rs"]
mod demucs_e2e_spectral_contract;
#[cfg(feature = "verify")]
#[path = "demucs/integration_smoke_demucs_encoder.rs"]
mod integration_smoke_demucs_encoder;
