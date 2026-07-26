// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the parallel verification wrapper.
//!
//! Part of #813.

use super::*;
use ny_api::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

/// Helper: build a simple Linear graph for parallel testing.
fn linear_graph() -> GraphNetwork {
    use ny_propagate::layers::LinearLayer;
    use ny_propagate::{GraphNode, Layer};

    let mut graph = GraphNetwork::new();
    let eye = ndarray::Array2::<f32>::eye(4);
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));
    graph.set_output("linear");
    graph
}

#[test]
fn test_parallel_verify_positions_ibp() {
    let graph = linear_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[4, 4]), 1.0f32),
    )
    .unwrap();

    let result = parallel_verify_positions(&graph, &input, 0, None).unwrap();
    assert_eq!(result.num_positions, 4);
    let (lo, hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[4, 4]);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite());
        assert!(u.is_finite());
        assert!(l <= u);
    }
}

#[test]
fn test_parallel_verify_with_config() {
    let graph = linear_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[6, 4]), -0.5f32),
        ArrayD::from_elem(IxDyn(&[6, 4]), 0.5f32),
    )
    .unwrap();

    let config = ParallelVerifyConfig::default().with_max_threads(2);
    let result = parallel_verify_positions(&graph, &input, 0, Some(&config)).unwrap();
    assert_eq!(result.num_positions, 6);
}

#[test]
fn test_parallel_verify_with_method_crown() {
    let graph = linear_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[3, 4]), 1.0f32),
    )
    .unwrap();

    let output = parallel_verify_with_method(&graph, &input, 0, PropMethod::Crown).unwrap();
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[3, 4]);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite());
        assert!(u.is_finite());
        assert!(l <= u);
    }
}

#[test]
fn test_parallel_verify_config_crown_constructor() {
    let config = ParallelVerifyConfig::crown();
    assert_eq!(config.method, PropMethod::Crown);
    assert_eq!(config.min_positions, 4);
    assert!(config.max_threads.is_none());
}

#[test]
fn test_parallel_verify_backend_default_is_cpu() {
    let config = ParallelVerifyConfig::default();
    assert!(matches!(config.backend, ParallelVerifyBackend::Cpu));
    assert!(config.engine().is_none());
}

#[test]
fn test_parallel_verify_crown_with_naive_engine() {
    use ny_core::NaiveCpuGemmEngine;

    let graph = linear_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[4, 4]), 1.0f32),
    )
    .unwrap();

    let engine: Arc<dyn GemmEngine> = Arc::new(NaiveCpuGemmEngine);
    let config =
        ParallelVerifyConfig::crown().with_backend(ParallelVerifyBackend::GpuEngine(engine));

    let result = parallel_verify_positions(&graph, &input, 0, Some(&config)).unwrap();
    assert_eq!(result.num_positions, 4);
    let (lo, hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[4, 4]);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite());
        assert!(u.is_finite());
        assert!(l <= u);
    }
}

#[test]
fn test_parallel_verify_engine_vs_cpu_match() {
    use ny_core::NaiveCpuGemmEngine;

    let graph = linear_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[3, 4]), 1.0f32),
    )
    .unwrap();

    // CPU-default path (delegates to NY ParallelVerifier)
    let cpu_config = ParallelVerifyConfig::crown();
    let cpu_result = parallel_verify_positions(&graph, &input, 0, Some(&cpu_config)).unwrap();

    // Engine path (uses our engine-aware loop with NaiveCpuGemmEngine)
    let engine: Arc<dyn GemmEngine> = Arc::new(NaiveCpuGemmEngine);
    let engine_config =
        ParallelVerifyConfig::crown().with_backend(ParallelVerifyBackend::GpuEngine(engine));
    let engine_result = parallel_verify_positions(&graph, &input, 0, Some(&engine_config)).unwrap();

    assert_eq!(cpu_result.num_positions, engine_result.num_positions);

    let (cpu_lo, cpu_hi) = cpu_result.output_bounds.lower_upper();
    let (eng_lo, eng_hi) = engine_result.output_bounds.lower_upper();
    assert_eq!(cpu_lo.shape(), eng_lo.shape());

    for (cl, el) in cpu_lo.iter().zip(eng_lo.iter()) {
        assert!((cl - el).abs() < 1e-6, "lower mismatch: {cl} vs {el}");
    }
    for (ch, eh) in cpu_hi.iter().zip(eng_hi.iter()) {
        assert!((ch - eh).abs() < 1e-6, "upper mismatch: {ch} vs {eh}");
    }
}

#[test]
fn test_parallel_verify_with_backend_builder() {
    use ny_core::NaiveCpuGemmEngine;

    let engine: Arc<dyn GemmEngine> = Arc::new(NaiveCpuGemmEngine);
    let config = ParallelVerifyConfig::crown()
        .with_max_threads(2)
        .with_backend(ParallelVerifyBackend::GpuEngine(engine));

    assert!(matches!(
        config.backend,
        ParallelVerifyBackend::GpuEngine(_)
    ));
    assert_eq!(config.max_threads, Some(2));
    assert!(config.engine().is_some());
}
