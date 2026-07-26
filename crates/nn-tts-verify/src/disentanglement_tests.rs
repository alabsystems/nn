// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_control_dimension_dim() {
    let cd = ControlDimension::new("test", 0, 4, 12);
    assert_eq!(cd.dim(), 8);
}

#[test]
fn test_acoustic_property_dim() {
    let ap = AcousticProperty::new("f0", 0, 16);
    assert_eq!(ap.dim(), 16);
}

#[test]
fn test_control_dimension_zero_width() {
    let cd = ControlDimension::new("zero", 0, 5, 5);
    assert_eq!(cd.dim(), 0);
}
