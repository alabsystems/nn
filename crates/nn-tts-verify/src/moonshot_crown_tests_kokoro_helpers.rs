// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro-architecture helper functions for CROWN timing certificate tests.
//!
//! Provides synthetic `VerifiedStage` pipeline stages for Kokoro's
//! text_encoder → prosody_predictor → vocoder architecture.
//! The dispatch plan itself is now in `kokoro_dispatch.rs`.

use crate::pipeline::VerifiedStage;

/// Build VerifiedStage pipeline stages matching the Kokoro architecture.
pub(super) fn kokoro_verified_stages(dim: usize) -> Vec<VerifiedStage> {
    vec![
        VerifiedStage {
            name: "text_encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.8; dim],
            output_upper: vec![0.8; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "prosody_predictor".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.5; dim],
            output_upper: vec![0.5; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "vocoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.3; dim],
            output_upper: vec![0.3; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ]
}
