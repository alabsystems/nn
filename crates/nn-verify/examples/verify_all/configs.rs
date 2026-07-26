// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kernel configuration builders for `verify_all`.
//!
//! Extracted from `verify_all.rs` to keep the main example under 500 lines (#571 AC1).

use nn_dsl::ir::KernelDef;
use nn_dsl::LowerError;
use nn_dsl::{
    build_adain_scalar_kernel, build_adain_snake_fused_kernel, build_gelu_kernel,
    build_instance_norm_affine_scalar_kernel, build_instance_norm_scalar_kernel,
    build_layer_norm_scalar_kernel, build_relu_kernel, build_rms_norm_scalar_kernel,
    build_rope_cos_kernel, build_rope_sin_kernel, build_sigmoid_kernel, build_silu_mul_kernel,
    build_snake_scalar_kernel, build_tanh_kernel,
};

/// A kernel with its standard verification configuration.
///
/// `config_name` provides a distinct status key for each configuration,
/// preventing same-named kernel configs from overwriting each other (#513).
pub(super) struct KernelConfig {
    pub(super) config_name: &'static str,
    pub(super) kernel: KernelDef,
    pub(super) constant_params: Vec<f32>,
    pub(super) input_lower: f32,
    pub(super) input_upper: f32,
}

/// Pending kernel configuration before the builder is invoked.
///
/// Separates config metadata from the fallible builder call (#558).
struct PendingConfig {
    config_name: &'static str,
    builder: fn() -> Result<KernelDef, LowerError>,
    constant_params: Vec<f32>,
    input_lower: f32,
    input_upper: f32,
}

/// Metadata for a builder that failed, retained for status-file recording.
pub(super) struct BuilderFailure {
    pub(super) config_name: &'static str,
    pub(super) constant_params: Vec<f32>,
    pub(super) input_lower: f32,
    pub(super) input_upper: f32,
    pub(super) error: LowerError,
}

/// Build all kernel configurations with standard and non-trivial parameters.
///
/// Non-trivial configs (#483) exercise the verification machinery under
/// realistic parameter ranges instead of identity-degenerate constants.
///
/// Builder failures are logged to stderr and skipped (#558) — a single
/// failing kernel builder no longer panics the entire pipeline. Returns
/// the successfully-built configs and the collected failure metadata.
pub(super) fn build_kernel_configs() -> (Vec<KernelConfig>, Vec<BuilderFailure>) {
    let pending: Vec<PendingConfig> = [
        activation_pending(),
        gelu_pending(),
        sigmoid_pending(),
        relu_pending(),
        tanh_pending(),
        rope_pending(),
        layer_rms_pending(),
        instance_norm_pending(),
    ]
    .into_iter()
    .flatten()
    .collect();

    let mut configs = Vec::with_capacity(pending.len());
    let mut failures = Vec::new();

    for p in pending {
        match (p.builder)() {
            Ok(kernel) => configs.push(KernelConfig {
                config_name: p.config_name,
                kernel,
                constant_params: p.constant_params,
                input_lower: p.input_lower,
                input_upper: p.input_upper,
            }),
            Err(e) => {
                eprintln!("{:<30} BUILD_ERR kernel builder failed: {e}", p.config_name);
                failures.push(BuilderFailure {
                    config_name: p.config_name,
                    constant_params: p.constant_params,
                    input_lower: p.input_lower,
                    input_upper: p.input_upper,
                    error: e,
                });
            }
        }
    }

    (configs, failures)
}

/// Snake, SiLU-Mul, AdaIN, AdaIN+Snake pending configs.
fn activation_pending() -> Vec<PendingConfig> {
    vec![
        PendingConfig {
            config_name: "snake",
            builder: build_snake_scalar_kernel,
            constant_params: vec![1.0],
            input_lower: -10.0,
            input_upper: 10.0,
        },
        PendingConfig {
            config_name: "silu_mul",
            builder: build_silu_mul_kernel,
            constant_params: vec![2.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        // AdaIN: identity then non-trivial (#483)
        PendingConfig {
            config_name: "adain_identity",
            builder: build_adain_scalar_kernel,
            constant_params: vec![0.0, 1.0, 1.0, 0.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "adain_scaled",
            builder: build_adain_scalar_kernel,
            constant_params: vec![2.0, 0.5, 5.0, 3.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "adain_wide",
            builder: build_adain_scalar_kernel,
            constant_params: vec![-5.0, 100.0, 0.01, -10.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        // AdaIN+Snake: identity then non-trivial (#483)
        PendingConfig {
            config_name: "adain_snake_identity",
            builder: build_adain_snake_fused_kernel,
            constant_params: vec![0.0, 1.0, 1.0, 0.0, 1.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "adain_snake_scaled",
            builder: build_adain_snake_fused_kernel,
            constant_params: vec![2.0, 0.5, 5.0, -3.0, 0.5, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "adain_snake_wide",
            builder: build_adain_snake_fused_kernel,
            constant_params: vec![-5.0, 100.0, 0.01, -10.0, 10.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
    ]
}

/// GELU pending configs — standard and wide-range (#639).
///
/// GELU has 0 constant params (single-variable activation), so constant_params is empty.
fn gelu_pending() -> Vec<PendingConfig> {
    vec![
        PendingConfig {
            config_name: "gelu",
            builder: build_gelu_kernel,
            constant_params: vec![],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "gelu_wide",
            builder: build_gelu_kernel,
            constant_params: vec![],
            input_lower: -10.0,
            input_upper: 10.0,
        },
    ]
}

/// Sigmoid pending configs — standard and wide-range (#659 AC1).
///
/// Sigmoid has 0 constant params (single-variable activation), so constant_params is empty.
fn sigmoid_pending() -> Vec<PendingConfig> {
    vec![
        PendingConfig {
            config_name: "sigmoid",
            builder: build_sigmoid_kernel,
            constant_params: vec![],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "sigmoid_wide",
            builder: build_sigmoid_kernel,
            constant_params: vec![],
            input_lower: -10.0,
            input_upper: 10.0,
        },
    ]
}

/// ReLU pending configs — standard and wide-range (#761 D1).
///
/// ReLU has 0 constant params (single-variable activation), so constant_params is empty.
fn relu_pending() -> Vec<PendingConfig> {
    vec![
        PendingConfig {
            config_name: "relu",
            builder: build_relu_kernel,
            constant_params: vec![],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "relu_wide",
            builder: build_relu_kernel,
            constant_params: vec![],
            input_lower: -10.0,
            input_upper: 10.0,
        },
    ]
}

/// Tanh pending configs — standard and wide-range (#761 D1).
///
/// Tanh has 0 constant params (single-variable activation), so constant_params is empty.
fn tanh_pending() -> Vec<PendingConfig> {
    vec![
        PendingConfig {
            config_name: "tanh_act",
            builder: build_tanh_kernel,
            constant_params: vec![],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "tanh_act_wide",
            builder: build_tanh_kernel,
            constant_params: vec![],
            input_lower: -10.0,
            input_upper: 10.0,
        },
    ]
}

/// RoPE cos/sin pending configs.
fn rope_pending() -> Vec<PendingConfig> {
    vec![
        PendingConfig {
            config_name: "rope_cos",
            builder: build_rope_cos_kernel,
            constant_params: vec![1.0, 0.5],
            input_lower: -10.0,
            input_upper: 10.0,
        },
        PendingConfig {
            config_name: "rope_sin",
            builder: build_rope_sin_kernel,
            constant_params: vec![1.0, 0.5],
            input_lower: -10.0,
            input_upper: 10.0,
        },
    ]
}

/// LayerNorm and RMSNorm pending configs — each with identity + 2 non-trivial (#483).
fn layer_rms_pending() -> Vec<PendingConfig> {
    vec![
        // LayerNorm: identity, scaled, wide
        PendingConfig {
            config_name: "layer_norm_identity",
            builder: build_layer_norm_scalar_kernel,
            constant_params: vec![0.0, 1.0, 1e-5, 1.0, 0.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "layer_norm_scaled",
            builder: build_layer_norm_scalar_kernel,
            constant_params: vec![-3.0, 0.25, 1e-5, 8.0, -2.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "layer_norm_wide",
            builder: build_layer_norm_scalar_kernel,
            constant_params: vec![5.0, 100.0, 1e-5, 0.01, -10.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        // RMSNorm: identity, scaled (product=1.5), inv (product=0.05)
        PendingConfig {
            config_name: "rms_norm_identity",
            builder: build_rms_norm_scalar_kernel,
            constant_params: vec![1.0, 1.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "rms_norm_scaled",
            builder: build_rms_norm_scalar_kernel,
            constant_params: vec![0.5, 3.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "rms_norm_inv",
            builder: build_rms_norm_scalar_kernel,
            constant_params: vec![5.0, 0.01],
            input_lower: -5.0,
            input_upper: 5.0,
        },
    ]
}

/// InstanceNorm and InstanceNorm+Affine pending configs — each with identity + 2 non-trivial (#483).
fn instance_norm_pending() -> Vec<PendingConfig> {
    vec![
        // InstanceNorm: identity, shifted, wide
        PendingConfig {
            config_name: "instance_norm_identity",
            builder: build_instance_norm_scalar_kernel,
            constant_params: vec![0.0, 1.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "instance_norm_shifted",
            builder: build_instance_norm_scalar_kernel,
            constant_params: vec![5.0, 0.25, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "instance_norm_wide",
            builder: build_instance_norm_scalar_kernel,
            constant_params: vec![-5.0, 100.0, 1e-5],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        // InstanceNorm+Affine: identity, scaled, amplified
        PendingConfig {
            config_name: "instance_norm_affine_identity",
            builder: build_instance_norm_affine_scalar_kernel,
            constant_params: vec![0.0, 1.0, 1e-5, 1.0, 0.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "instance_norm_affine_scaled",
            builder: build_instance_norm_affine_scalar_kernel,
            constant_params: vec![-2.0, 4.0, 1e-5, 0.01, -10.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
        PendingConfig {
            config_name: "instance_norm_affine_amplified",
            builder: build_instance_norm_affine_scalar_kernel,
            constant_params: vec![3.0, 0.1, 1e-5, 10.0, 5.0],
            input_lower: -5.0,
            input_upper: 5.0,
        },
    ]
}
