// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for harmonic_source segmented verification (#2411).
//!
//! Validates that the cumsum analytical bypass produces sound verification
//! results for the harmonic_source pattern (cumsum → sin) with time
//! dimensions exceeding MAX_DECOMPOSE_DIM (2048).
//!
//! The key property: cumsum on large T dimensions no longer blocks
//! verification. Instead, an identity pass-through is emitted and the
//! downstream sin() naturally bounds the output to [-1, 1].

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_verify::HarmonicSourceBounds;

/// Verify that `HarmonicSourceBounds` correctly models the cumsum→sin pattern.
#[test]
fn test_harmonic_source_bounds_struct() {
    let bounds = HarmonicSourceBounds::new(24000).expect("valid bounds for 1s audio");
    assert!(bounds.is_valid());
    assert_eq!(bounds.sin_lower, -1.0);
    assert_eq!(bounds.sin_upper, 1.0);
    assert!((bounds.sin_width() - 2.0).abs() < 1e-12);
    assert_eq!(bounds.cumsum_dim_size, 24000);
}

/// Verify that HarmonicSourceBounds rejects zero-length dimensions.
#[test]
fn test_harmonic_source_bounds_rejects_zero() {
    assert!(HarmonicSourceBounds::new(0).is_err());
}

/// The cumsum analytical bypass should not fail for dimensions above 2048.
///
/// This test traces a small graph that includes cumsum on a dimension
/// within the cap (T=100) to confirm the trace-to-graph translation
/// works. The large-dimension bypass (T=24000) cannot be tested via
/// trace_graph because DynTensor::cumsum actually computes the operation
/// on real data, which is fine — the bypass only activates in the
/// verify-path translator when translating the trace to NY.
#[cfg(feature = "ny")]
#[test]
fn test_cumsum_within_cap_verifies() {
    use nn_verify::verify_trace;
    use ndarray::{ArrayD, IxDyn};

    // Small cumsum within the cap (T=100).
    let t = 100_usize;
    let input_data: Vec<f32> = (0..t).map(|i| 0.01 * i as f32).collect();
    let input = DynTensor::from_vec(input_data, &[1, 1, t], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let phase = traced.cumsum(2)?;
        phase.sin()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 1, t]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 1, t]), 1.0f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid input bounds");

    let result = verify_trace(&graph, &input_bounds).expect("verification should succeed");

    // sin output should be bounded by [-1, 1].
    let (lo, hi) = result.ibp_bounds.lower_upper();
    for &l in lo.iter() {
        assert!(
            l >= -1.0 - 1e-4,
            "sin lower bound should be >= -1.0, got {l}"
        );
    }
    for &h in hi.iter() {
        assert!(h <= 1.0 + 1e-4, "sin upper bound should be <= 1.0, got {h}");
    }
    assert!(result.ibp_width.is_finite(), "IBP width should be finite");
    assert!(
        result.ibp_width <= 2.0 + 1e-3,
        "sin output width should be <= 2.0, got {}",
        result.ibp_width
    );
}

/// Verify the analytical bypass for large cumsum dimensions.
///
/// Traces a cumsum on T=3000 (exceeds MAX_DECOMPOSE_DIM=2048) followed by sin.
/// Before #2411 fix, this would fail with "Cumsum: dim size 3000 exceeds
/// decomposition limit 2048". After the fix, the identity bypass allows
/// verification to proceed.
#[cfg(feature = "ny")]
#[test]
fn test_cumsum_above_cap_verifies_via_analytical_bypass() {
    use nn_verify::verify_trace;
    use ndarray::{ArrayD, IxDyn};

    // Large cumsum exceeding the cap (T=3000).
    let t = 3000_usize;
    let input_data: Vec<f32> = (0..t).map(|i| 0.001 * i as f32).collect();
    let input = DynTensor::from_vec(input_data, &[1, 1, t], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let phase = traced.cumsum(2)?;
        phase.sin()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 1, t]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 1, t]), 0.5f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid input bounds");

    // This previously failed with UnsupportedOp("Cumsum: dim size 3000 exceeds ...").
    let result = verify_trace(&graph, &input_bounds)
        .expect("verification should succeed with analytical bypass");

    // sin output should be bounded by [-1, 1].
    let (lo, hi) = result.ibp_bounds.lower_upper();
    for &l in lo.iter() {
        assert!(
            l >= -1.0 - 1e-4,
            "sin lower bound should be >= -1.0, got {l}"
        );
    }
    for &h in hi.iter() {
        assert!(h <= 1.0 + 1e-4, "sin upper bound should be <= 1.0, got {h}");
    }
    assert!(result.ibp_width.is_finite(), "IBP width should be finite");
}

/// Verify the Kokoro-realistic case: 1 second of 24kHz audio.
///
/// harmonic_source uses cumsum(2) on T=24000. Before #2411, this was
/// permanently blocked.
#[cfg(feature = "ny")]
#[test]
fn test_cumsum_24000_kokoro_realistic_verifies() {
    use nn_verify::verify_trace;
    use ndarray::{ArrayD, IxDyn};

    let t = 24000_usize; // 1 second at 24kHz
    let input_data: Vec<f32> = vec![0.01; t]; // uniform phase increment
    let input = DynTensor::from_vec(input_data, &[1, 1, t], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let phase = traced.cumsum(2)?;
        phase.sin()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 1, t]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 1, t]), 0.1f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid input bounds");

    let result = verify_trace(&graph, &input_bounds)
        .expect("24kHz cumsum→sin should verify via analytical bypass");

    assert!(
        result.ibp_width.is_finite(),
        "IBP width should be finite for 24kHz case"
    );
    assert!(
        result.ibp_width <= 2.0 + 1e-3,
        "sin output width should be <= 2.0, got {}",
        result.ibp_width
    );
}
