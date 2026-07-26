// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated speaker embedding composition and verification tests.
//!
//! ECAPA-TDNN speaker encoder with SE blocks, Res2Net, BatchNorm,
//! dilated Conv1d, and mean pooling verified through NY.
//!
//! Part of #2079.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_speaker_embedding.rs"]
mod speaker_embedding;
