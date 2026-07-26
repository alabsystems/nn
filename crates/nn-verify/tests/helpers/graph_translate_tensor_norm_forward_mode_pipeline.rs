// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NormBoundsMode config wiring and full-pipeline forward-mode tests.
//!
//! Extracted from `graph_translate_tensor_norm_forward_mode.rs` for file-size
//! compliance (#1402).

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{
    tensor_kernel_to_graph, tensor_kernel_to_graph_with_norm_mode, BoundedTensor, NormBoundsMode,
    TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

/// Build an InstanceNorm1d kernel def: input [C=2, T=4], eps=1e-5, axis=last.
/// Duplicated from parent module to keep this file standalone-compilable.
fn instance_norm_kernel() -> TensorKernelDef {
    TensorKernelDef::new(
        "instance_norm_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    gamma: None,
                    beta: None,
                    axis: 1,
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(2),
    )
}

/// Create high-variance input bounds: [C=2, T=4].
/// Duplicated from parent module.
fn high_variance_instance_norm_input() -> BoundedTensor {
    let r = 0.05;
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            0.0 - r,
            5.0 - r,
            10.0 - r,
            15.0 - r,
            -8.0 - r,
            -3.0 - r,
            3.0 - r,
            8.0 - r,
        ],
    )
    .expect("valid lower");
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            0.0 + r,
            5.0 + r,
            10.0 + r,
            15.0 + r,
            -8.0 + r,
            -3.0 + r,
            3.0 + r,
            8.0 + r,
        ],
    )
    .expect("valid upper");
    BoundedTensor::new(lower, upper).expect("valid bounded tensor")
}

/// Propagate bounds through a graph and return the max output width.
/// Duplicated from parent module.
fn propagate_width(graph: &nn_verify::GraphNetwork, input: &BoundedTensor) -> f32 {
    let output = graph.propagate_ibp(input).expect("IBP propagation");
    output.max_width()
}

#[test]
fn test_forward_mode_is_default() {
    // tensor_kernel_to_graph (no config) should produce the same graph as
    // explicit NormBoundsMode::ForwardMode.
    let kernel = instance_norm_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let input = high_variance_instance_norm_input();

    let graph_default = tensor_kernel_to_graph(&kernel, &bindings).expect("default graph");
    let graph_forward =
        tensor_kernel_to_graph_with_norm_mode(&kernel, &bindings, NormBoundsMode::ForwardMode)
            .expect("explicit forward mode graph");

    let width_default = propagate_width(&graph_default, &input);
    let width_forward = propagate_width(&graph_forward, &input);

    // Widths should be identical since default IS ForwardMode.
    let diff = (width_default - width_forward).abs();
    assert!(
        diff < 1e-6,
        "default ({width_default}) and forward mode ({width_forward}) should match, diff={diff}"
    );
}

#[test]
fn test_norm_bounds_mode_config_wiring() {
    // Verify the NormBoundsMode enum methods return correct values.
    assert!(!NormBoundsMode::Conservative.forward_mode());
    assert!(NormBoundsMode::ForwardMode.forward_mode());
    assert!(NormBoundsMode::CrownSampling.forward_mode());

    // Conservative and ForwardMode use IbpValidated crown_mode (sound Jacobian
    // linearization with IBP-validated error margins); CrownSampling uses Sampling.
    use ny_propagate::layers::LayerNormCrownMode;
    assert_eq!(
        NormBoundsMode::Conservative.crown_mode(),
        LayerNormCrownMode::IbpValidated
    );
    assert_eq!(
        NormBoundsMode::ForwardMode.crown_mode(),
        LayerNormCrownMode::IbpValidated
    );
    assert_eq!(
        NormBoundsMode::CrownSampling.crown_mode(),
        LayerNormCrownMode::Sampling
    );
}

#[test]
fn test_verify_config_norm_mode_setter() {
    use nn_verify::VerifyConfig;

    let config = VerifyConfig::default();
    assert_eq!(config.norm_mode(), NormBoundsMode::ForwardMode);

    let config_conservative = config.with_norm_mode(NormBoundsMode::Conservative);
    assert_eq!(
        config_conservative.norm_mode(),
        NormBoundsMode::Conservative
    );
}

/// Run the tensor pipeline with a given config and return per-element max_width
/// (from the output BoundedTensor) and whether the status entry was recorded.
///
/// Uses `output_bounds.max_width()` instead of `verification.output_width`
/// because the scalar summary (global min of lower, global max of upper)
/// compresses per-element variation and hides the forward-mode improvement.
fn run_pipeline(
    kernel: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
    key: &str,
    config: &nn_verify::VerifyConfig,
) -> (f32, bool) {
    let mut status = nn_verify::VerifyStatus::default();
    let result = nn_verify::verify_tensor_and_record_with_config(
        &mut status,
        kernel,
        bindings,
        input,
        Some(key),
        config,
    )
    .expect("pipeline should succeed");
    assert!(
        result.verification.is_finite,
        "{key}: bounds must be finite"
    );
    let recorded = status.has_kernel(key) && status.kernel(key).unwrap().output_width.is_finite();
    (result.output_bounds.max_width(), recorded)
}

/// Full pipeline test: verify_tensor_and_record_with_config exercises the
/// complete path from TensorKernelDef through NY to status recording.
#[test]
fn test_pipeline_forward_mode_produces_tighter_verification() {
    let kernel = instance_norm_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let input = high_variance_instance_norm_input();

    let (width_conservative, rec_c) = run_pipeline(
        &kernel,
        &bindings,
        &input,
        "instance_norm_conservative",
        &nn_verify::VerifyConfig::default(),
    );
    let config_fwd =
        nn_verify::VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let (width_forward, rec_f) = run_pipeline(
        &kernel,
        &bindings,
        &input,
        "instance_norm_forward",
        &config_fwd,
    );

    assert!(rec_c, "conservative entry should be recorded");
    assert!(rec_f, "forward entry should be recorded");
    assert!(
        width_forward <= width_conservative + 1e-3,
        "forward ({width_forward}) should be no wider than conservative ({width_conservative})"
    );
    if width_conservative > 1.0 && width_forward > 0.0 {
        let ratio = width_conservative / width_forward;
        assert!(
            ratio >= 10.0,
            "pipeline forward-mode should be >=10x tighter, \
             got {ratio:.1}x (cons={width_conservative:.2}, fwd={width_forward:.2})"
        );
    }
}
