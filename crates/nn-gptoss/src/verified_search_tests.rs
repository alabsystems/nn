// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for verified search pipeline types.

use super::*;
use ndarray::ArrayD;

#[test]
fn test_search_query_new_valid() {
    let q = SearchQuery::new("what is formal verification?").expect("valid query");
    assert_eq!(q.query, "what is formal verification?");
    assert_eq!(q.top_k, 10);
    assert!(q.perturbation_eps.is_none());
    assert!(q.min_confidence.is_none());
}

#[test]
fn test_search_query_empty_rejected() {
    let result = SearchQuery::new("");
    assert!(result.is_err());
}

#[test]
fn test_search_query_builder() {
    let q = SearchQuery::new("test query")
        .expect("valid")
        .with_top_k(5)
        .with_perturbation_eps(0.01)
        .expect("valid eps")
        .with_min_confidence(0.8)
        .expect("valid confidence");
    assert_eq!(q.top_k, 5);
    assert_eq!(q.perturbation_eps, Some(0.01));
    assert_eq!(q.min_confidence, Some(0.8));
}

#[test]
fn test_search_query_negative_eps_rejected() {
    let result = SearchQuery::new("test")
        .expect("valid")
        .with_perturbation_eps(-1.0);
    assert!(result.is_err());
}

#[test]
fn test_search_query_nan_eps_rejected() {
    let result = SearchQuery::new("test")
        .expect("valid")
        .with_perturbation_eps(f32::NAN);
    assert!(result.is_err());
}

#[test]
fn test_search_query_confidence_out_of_range() {
    let too_high = SearchQuery::new("test")
        .expect("valid")
        .with_min_confidence(1.5);
    assert!(too_high.is_err());

    let too_low = SearchQuery::new("test")
        .expect("valid")
        .with_min_confidence(-0.1);
    assert!(too_low.is_err());
}

#[test]
fn test_verified_search_result_unverified() {
    let r = VerifiedSearchResult::unverified("doc1".into(), "Title".into(), "Snippet".into(), 0.95);
    assert!(!r.is_sound());
    assert!(!r.is_verified());
    assert!(r.logit_bounds.is_none());
}

#[test]
fn test_verified_search_result_with_sound_verification() {
    let r = VerifiedSearchResult::unverified("doc1".into(), "Title".into(), "Snippet".into(), 0.95)
        .with_verification(
            VerificationStatus::Sound {
                method: "crown".into(),
                max_bound_width: 0.5,
            },
            None,
        );
    assert!(r.is_sound());
    assert!(r.is_verified());
}

#[test]
fn test_verified_search_result_heuristic() {
    let r = VerifiedSearchResult::unverified("doc1".into(), "Title".into(), "Snippet".into(), 0.8)
        .with_verification(
            VerificationStatus::Heuristic {
                method: "sampling".into(),
                max_bound_width: 1.2,
            },
            None,
        );
    assert!(!r.is_sound());
    assert!(r.is_verified());
}

#[test]
fn test_logit_bounds_max_width() {
    let lower = ArrayD::from_shape_vec(vec![1, 2], vec![0.0, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![1, 2], vec![0.5, 3.0]).unwrap();
    let bounds = IntervalBounds::new(lower, upper).expect("valid bounds");
    let lb = LogitBounds::new(bounds, 0.01, vec![0]).expect("valid");
    assert_eq!(lb.max_width(), Some(2.0));
}

#[test]
fn test_logit_bounds_invalid_eps() {
    let lower = ArrayD::from_shape_vec(vec![1], vec![0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap();
    let bounds = IntervalBounds::new(lower, upper).expect("valid bounds");
    assert!(LogitBounds::new(bounds.clone(), -1.0, vec![0]).is_err());
    assert!(LogitBounds::new(bounds, f32::INFINITY, vec![0]).is_err());
}

#[test]
fn test_logit_bounds_empty_positions() {
    let lower = ArrayD::from_shape_vec(vec![1], vec![0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap();
    let bounds = IntervalBounds::new(lower, upper).expect("valid bounds");
    let lb = LogitBounds::new(bounds, 0.01, vec![]).expect("valid");
    assert_eq!(lb.max_width(), None);
}

#[test]
fn test_report_from_empty_results() {
    let report = SearchVerificationReport::from_results(&[]);
    assert_eq!(report.total_results, 0);
    assert_eq!(report.sound_count, 0);
    assert_eq!(report.sound_fraction(), 0.0);
    assert_eq!(report.verified_fraction(), 0.0);
}

#[test]
fn test_report_from_mixed_results() {
    let results = vec![
        VerifiedSearchResult::unverified("a".into(), "A".into(), "s".into(), 0.9)
            .with_verification(
                VerificationStatus::Sound {
                    method: "ibp".into(),
                    max_bound_width: 0.3,
                },
                None,
            ),
        VerifiedSearchResult::unverified("b".into(), "B".into(), "s".into(), 0.8)
            .with_verification(
                VerificationStatus::Heuristic {
                    method: "sampling".into(),
                    max_bound_width: 1.5,
                },
                None,
            ),
        VerifiedSearchResult::unverified("c".into(), "C".into(), "s".into(), 0.7),
    ];
    let report = SearchVerificationReport::from_results(&results);
    assert_eq!(report.total_results, 3);
    assert_eq!(report.sound_count, 1);
    assert_eq!(report.heuristic_count, 1);
    assert_eq!(report.unverified_count, 1);
    assert_eq!(report.tightest_bound_width, Some(0.3));
    assert_eq!(report.widest_bound_width, Some(1.5));
    // sound_fraction = 1/3
    assert!((report.sound_fraction() - 1.0 / 3.0).abs() < 1e-10);
    // verified_fraction = 2/3
    assert!((report.verified_fraction() - 2.0 / 3.0).abs() < 1e-10);
}
