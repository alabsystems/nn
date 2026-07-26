// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CROWN-verified fairness bounds (Phase 2).

use super::*;
use crate::fairness::Group;

#[test]
fn test_validate_empty_regions() {
    let regions: Vec<GroupInputRegion> = Vec::new();
    let result = validate_fairness_regions(&regions, &[3]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("empty") || msg.contains("Empty"),
        "Expected empty error, got: {msg}"
    );
}

#[test]
fn test_validate_mismatched_lengths() {
    let regions = vec![GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "en".to_string(),
        },
        lower: vec![0.0, 0.0, 0.0],
        upper: vec![1.0, 1.0], // mismatched length
    }];
    let result = validate_fairness_regions(&regions, &[3]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("must match"),
        "Expected length mismatch error, got: {msg}"
    );
}

#[test]
fn test_validate_nan_rejected() {
    let regions = vec![GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "ja".to_string(),
        },
        lower: vec![0.0, f64::NAN, 0.0],
        upper: vec![1.0, 1.0, 1.0],
    }];
    let result = validate_fairness_regions(&regions, &[3]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("must be finite"),
        "Expected finiteness error, got: {msg}"
    );
}

#[test]
fn test_validate_lower_greater_than_upper_rejected() {
    let regions = vec![GroupInputRegion {
        group: Group {
            dimension: "gender".to_string(),
            value: "female".to_string(),
        },
        lower: vec![0.0, 2.0, 0.0], // lower > upper at index 1
        upper: vec![1.0, 1.0, 1.0],
    }];
    let result = validate_fairness_regions(&regions, &[3]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("lower bound must be less than upper"),
        "Expected inverted range error, got: {msg}"
    );
}

#[test]
fn test_compute_max_width_ratio_single_group() {
    let results = vec![GroupBoundsResult {
        group: Group {
            dimension: "language".to_string(),
            value: "en".to_string(),
        },
        mean_output_width: 0.5,
        max_output_width: 1.0,
        output_lower: vec![0.0],
        output_upper: vec![1.0],
        propagation_mode: "Crown".to_string(),
    }];
    assert_eq!(compute_max_width_ratio(&results), 1.0);
}

#[test]
fn test_compute_max_width_ratio_equal_groups() {
    let results = vec![
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            mean_output_width: 0.5,
            max_output_width: 0.8,
            output_lower: vec![0.1],
            output_upper: vec![0.6],
            propagation_mode: "Crown".to_string(),
        },
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "ja".to_string(),
            },
            mean_output_width: 0.5,
            max_output_width: 0.9,
            output_lower: vec![0.0],
            output_upper: vec![0.5],
            propagation_mode: "Crown".to_string(),
        },
    ];
    let ratio = compute_max_width_ratio(&results);
    assert!(
        (ratio - 1.0).abs() < 1e-10,
        "Equal widths should give ratio 1.0, got {ratio}"
    );
}

#[test]
fn test_compute_max_width_ratio_asymmetric() {
    let results = vec![
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            mean_output_width: 0.5,
            max_output_width: 0.8,
            output_lower: vec![0.1],
            output_upper: vec![0.6],
            propagation_mode: "Crown".to_string(),
        },
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "ko".to_string(),
            },
            mean_output_width: 1.5, // 3x wider than en
            max_output_width: 2.0,
            output_lower: vec![-0.5],
            output_upper: vec![1.0],
            propagation_mode: "Crown".to_string(),
        },
    ];
    let ratio = compute_max_width_ratio(&results);
    assert!(
        (ratio - 3.0).abs() < 1e-10,
        "3x width difference should give ratio 3.0, got {ratio}"
    );
}

#[test]
fn test_compute_max_width_ratio_three_groups() {
    // en: 0.5 width, ja: 1.0 width, ko: 2.0 width
    // max ratio should be ko/en = 4.0
    let results = vec![
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            mean_output_width: 0.5,
            max_output_width: 0.8,
            output_lower: vec![0.0],
            output_upper: vec![0.5],
            propagation_mode: "Crown".to_string(),
        },
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "ja".to_string(),
            },
            mean_output_width: 1.0,
            max_output_width: 1.5,
            output_lower: vec![-0.5],
            output_upper: vec![0.5],
            propagation_mode: "Crown".to_string(),
        },
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "ko".to_string(),
            },
            mean_output_width: 2.0,
            max_output_width: 3.0,
            output_lower: vec![-1.0],
            output_upper: vec![1.0],
            propagation_mode: "Crown".to_string(),
        },
    ];
    let ratio = compute_max_width_ratio(&results);
    assert!(
        (ratio - 4.0).abs() < 1e-10,
        "ko/en should give ratio 4.0, got {ratio}"
    );
}

#[test]
fn test_compute_max_width_ratio_zero_width() {
    // If one group has zero width (constant output), ratio should be infinity.
    let results = vec![
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            mean_output_width: 0.5,
            max_output_width: 0.8,
            output_lower: vec![0.0],
            output_upper: vec![0.5],
            propagation_mode: "Crown".to_string(),
        },
        GroupBoundsResult {
            group: Group {
                dimension: "language".to_string(),
                value: "zh".to_string(),
            },
            mean_output_width: 0.0, // constant output for this group
            max_output_width: 0.0,
            output_lower: vec![0.5],
            output_upper: vec![0.5],
            propagation_mode: "Crown".to_string(),
        },
    ];
    let ratio = compute_max_width_ratio(&results);
    assert!(
        ratio.is_infinite(),
        "Zero-width group should give infinite ratio, got {ratio}"
    );
}

#[test]
fn test_validate_valid_regions_accepted() {
    let regions = vec![
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            lower: vec![0.0, -1.0, 0.5],
            upper: vec![1.0, 1.0, 1.5],
        },
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "ja".to_string(),
            },
            lower: vec![-0.5, -0.5, 0.0],
            upper: vec![0.5, 0.5, 1.0],
        },
    ];
    assert!(validate_fairness_regions(&regions, &[3]).is_ok());
}

#[test]
fn test_validate_shape_mismatch_rejected() {
    let regions = vec![GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "en".to_string(),
        },
        lower: vec![0.0, 0.0, 0.0],
        upper: vec![1.0, 1.0, 1.0],
    }];
    // Region has 3 elements but shape says 4
    let result = validate_fairness_regions(&regions, &[4]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("must match input shape"),
        "Expected shape mismatch error, got: {msg}"
    );
}

#[test]
fn test_validate_infinity_rejected() {
    let regions = vec![GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "en".to_string(),
        },
        lower: vec![f64::NEG_INFINITY, 0.0],
        upper: vec![1.0, 1.0],
    }];
    let result = validate_fairness_regions(&regions, &[2]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("must be finite"),
        "Expected finiteness error, got: {msg}"
    );
}
