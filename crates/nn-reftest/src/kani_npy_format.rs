// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for NPY format parsing and payload reconstruction.
//!
//! These proofs focus on the uncovered format-level invariants in `npy.rs`:
//! supported dtype descriptor parsing, 3-D shape reconstruction from headers,
//! and full-file endianness handling through `parse_npy`.
//!
//! Issue: #3726

#[cfg(kani)]
mod proofs {
    use crate::npy::{parse_npy, parse_npy_header};

    fn assume_finite_bounded(value: f32) {
        kani::assume(value.is_finite());
        kani::assume(value >= -1.0e6 && value <= 1.0e6);
    }

    fn build_npy_v1(header: &str, raw: &[u8]) -> Vec<u8> {
        let header_len = u16::try_from(header.len()).expect("small proof headers fit in v1");

        let mut bytes = Vec::with_capacity(10 + header.len() + raw.len());
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(raw);
        bytes
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn supported_dtype_descriptor_roundtrips_from_header() {
        let idx: u8 = kani::any();
        kani::assume(idx < 6);

        let dtype = match idx {
            0 => "<f4",
            1 => ">f4",
            2 => "<f2",
            3 => ">f2",
            4 => "<f8",
            _ => ">f8",
        };

        let header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': (3,), }}");
        let (parsed_dtype, shape, fortran_order) =
            parse_npy_header(&header).expect("well-formed header must parse");

        assert!(parsed_dtype == dtype, "dtype descriptor must be preserved");
        assert!(
            shape == vec![3],
            "shape must be reconstructed from the header"
        );
        assert!(!fortran_order, "fortran_order must stay false");
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn three_dimensional_shape_reconstruction_from_header() {
        let d0: usize = kani::any();
        let d1: usize = kani::any();
        let d2: usize = kani::any();

        kani::assume(d0 <= 3);
        kani::assume(d1 <= 3);
        kani::assume(d2 <= 3);

        let header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({d0}, {d1}, {d2}), }}");
        let (_, shape, fortran_order) = parse_npy_header(&header).expect("3-D header must parse");

        assert!(
            shape == vec![d0, d1, d2],
            "parsed 3-D shape must match the serialized header tuple"
        );
        assert!(!fortran_order, "3-D reconstruction must preserve C order");
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn parse_npy_little_endian_payload_roundtrips_value() {
        let value: f32 = kani::any();
        assume_finite_bounded(value);

        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }";
        let bytes = build_npy_v1(header, &value.to_le_bytes());
        let tensor = parse_npy(&bytes, "little".into()).expect("valid little-endian NPY");

        assert!(tensor.shape == vec![1], "shape must come from the header");
        assert!(
            tensor.data.len() == 1,
            "single-element payload must stay single-element"
        );
        assert!(
            tensor.data[0] == value,
            "little-endian f32 payload must decode to the original value"
        );
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn parse_npy_big_endian_payload_roundtrips_value() {
        let value: f32 = kani::any();
        assume_finite_bounded(value);

        let header = "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }";
        let bytes = build_npy_v1(header, &value.to_be_bytes());
        let tensor = parse_npy(&bytes, "big".into()).expect("valid big-endian NPY");

        assert!(tensor.shape == vec![1], "shape must come from the header");
        assert!(
            tensor.data.len() == 1,
            "single-element payload must stay single-element"
        );
        assert!(
            tensor.data[0] == value,
            "big-endian f32 payload must decode to the original value"
        );
    }
}
