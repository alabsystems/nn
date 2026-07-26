// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thin re-export wrapper — implementation lives in nn-models.
//!
//! Part of #860: extract backend-agnostic helpers to nn-models.

pub(super) use nn_models::demucs_transformer_helpers::{
    add_sinusoidal_1d, build_sinusoidal_table, transpose_ct_to_tc, transpose_tc_to_ct,
};
