// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion equivalence verification configurations for `verify_all`.
//!
//! Split from `configs.rs` (#1544 D7) for 500-line compliance.
//! Contains multi-variable fusion configs (AdaIN+Snake, RMSNorm+SiLU-Mul,
//! LayerNorm+GELU).
//!
//! All configs use `NormBoundsMode::ForwardMode` (#2225) to enable tighter
//! CROWN linearization through normalization decompositions.

use nn_verify::{NormBoundsMode, VerifyConfig};

/// A fusion equivalence verification configuration.
///
/// Unlike scalar `KernelConfig`, fusion configs use multi-variable bounds
/// and verify that a fused kernel matches its sequential components within
/// epsilon (#803 AC3).
pub(super) struct FusionConfig {
    pub(super) config_name: &'static str,
    pub(super) variable_bounds: Vec<(f32, f32)>,
    pub(super) epsilon: f32,
    pub(super) verify_config: VerifyConfig,
    pub(super) verify_fn: fn(
        &[(f32, f32)],
        f32,
        &VerifyConfig,
    )
        -> Result<nn_verify::FusionVerification, nn_verify::VerifyError>,
}

/// Build all fusion equivalence verification configurations.
///
/// Each config exercises one of the convenience wrappers in `fusion_adain.rs`:
/// AdaIN+Snake (dvoice Kokoro), RMSNorm+SiLU-Mul (LLaMA SwiGLU),
/// LayerNorm+GELU (Transformer FFN).
///
/// All configs use `ForwardMode` norm bounds (#2225): enables forward-mode IBP
/// through normalization decompositions and `IbpValidated` CROWN linearization.
pub(super) fn build_fusion_configs() -> Vec<FusionConfig> {
    let forward_config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);

    vec![
        // AdaIN+Snake (K4): 7 shared inputs — dvoice Kokoro decoder pattern.
        // Tightened bounds from #2225: var [0.1, 3.0], alpha [0.5, 5.0].
        FusionConfig {
            config_name: "fusion_adain_snake",
            variable_bounds: vec![
                (-3.0, 3.0),  // x (tightened from ±5)
                (-2.0, 2.0),  // mu (tightened from ±3)
                (0.1, 3.0),   // var (positive, tightened from [0.01, 5.0])
                (0.5, 2.0),   // gamma (tightened from [0.5, 3.0])
                (-1.0, 1.0),  // beta (tightened from ±2)
                (0.5, 5.0),   // alpha (positive, tightened from [0.1, 10.0])
                (1e-5, 1e-5), // eps (point)
            ],
            epsilon: 1e-4,
            verify_config: forward_config.clone(),
            verify_fn: nn_verify::verify_adain_snake_fusion_with_config,
        },
        // RMSNorm+SiLU-Mul: 4 shared inputs — LLaMA/Mistral SwiGLU pattern.
        // Tightened bounds from #2225: max |normed| well within exp threshold.
        FusionConfig {
            config_name: "fusion_rms_norm_silu_mul",
            variable_bounds: vec![
                (-3.0, 3.0), // x (tightened from ±5)
                (0.2, 2.0),  // rms_inv (tightened from [0.1, 3.0])
                (-2.0, 2.0), // weight (tightened from ±3)
                (-3.0, 3.0), // up (tightened from ±5)
            ],
            epsilon: 1e-4,
            verify_config: forward_config.clone(),
            verify_fn: nn_verify::verify_rms_norm_silu_mul_fusion_with_config,
        },
        // LayerNorm+GELU: 6 shared inputs — Transformer FFN pattern.
        // Bounds kept tight to avoid GELU inner exp overflow.
        FusionConfig {
            config_name: "fusion_layer_norm_gelu",
            variable_bounds: vec![
                (-1.5, 1.5),  // x (tightened from ±2)
                (-0.5, 0.5),  // mean (tightened from ±1)
                (0.5, 3.0),   // var_val (tightened from [0.5, 5.0])
                (1e-5, 1e-5), // eps (point)
                (0.5, 1.5),   // gamma (tightened from [0.5, 2.0])
                (-0.5, 0.5),  // beta (tightened from ±1)
            ],
            epsilon: 1e-4,
            verify_config: forward_config,
            verify_fn: nn_verify::verify_layer_norm_gelu_fusion_with_config,
        },
    ]
}
