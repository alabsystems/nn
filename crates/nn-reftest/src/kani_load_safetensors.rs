// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for safetensors load bounds and alignment safety.
//!
//! These proofs focus on uncovered loader invariants in `load.rs`:
//! deserialized tensor data stays within the backing buffer, truncated
//! safetensors payloads are rejected, and byte-wise decoding works even when
//! tensor bytes are not naturally aligned for the element type.
//!
//! Issue: #3726

#[cfg(kani)]
mod proofs {
    use crate::load::{convert_to_f32, load_safetensors_from_bytes};

    #[repr(align(4))]
    struct AlignedBytes([u8; 1 + std::mem::size_of::<f32>()]);

    fn build_single_tensor_buffer(values: &[f32]) -> Vec<u8> {
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![values.len()], &raw)
                .expect("valid tensor view");

        safetensors::tensor::serialize(vec![("tensor".to_string(), view)], None)
            .expect("serialization should succeed")
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn deserialized_tensor_offsets_stay_within_backing_buffer() {
        let values = [1.0f32, -2.5];
        let serialized = build_single_tensor_buffer(&values);

        let tensors =
            safetensors::SafeTensors::deserialize(&serialized).expect("valid safetensors buffer");
        let view = tensors.tensor("tensor").expect("tensor must exist");
        let raw = view.data();

        let backing_start = serialized.as_ptr() as usize;
        let backing_end = backing_start + serialized.len();
        let raw_start = raw.as_ptr() as usize;
        let raw_end = raw_start + raw.len();

        assert!(
            raw_start >= backing_start,
            "tensor data start must lie inside the backing buffer"
        );
        assert!(
            raw_end <= backing_end,
            "tensor data end must lie inside the backing buffer"
        );
        assert!(
            raw.len() == values.len() * std::mem::size_of::<f32>(),
            "deserialized raw bytes must match numel * bytes_per_element"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn truncated_safetensors_payload_is_rejected() {
        let serialized = build_single_tensor_buffer(&[3.25f32]);
        let truncated = &serialized[..serialized.len() - 1];

        let result = load_safetensors_from_bytes(truncated);

        assert!(
            result.is_err(),
            "truncating the backing buffer must reject out-of-bounds tensor offsets"
        );
    }

    #[kani::unwind(5)]
    #[kani::proof]
    fn misaligned_f32_raw_slice_still_decodes_correctly() {
        let value: f32 = 42.5;
        let mut backing = AlignedBytes([0u8; 1 + std::mem::size_of::<f32>()]);
        backing.0[0] = 0xAA;
        backing.0[1..].copy_from_slice(&value.to_le_bytes());

        let raw = &backing.0[1..];
        let decoded =
            convert_to_f32(raw, safetensors::Dtype::F32, &[1], "misaligned").expect("decode");

        assert!(
            (raw.as_ptr() as usize) % std::mem::align_of::<f32>() != 0,
            "proof slice must actually be misaligned for f32"
        );
        assert!(
            decoded.len() == 1,
            "one element must decode from four raw bytes"
        );
        assert!(
            decoded[0] == value,
            "byte-wise decode must not depend on pointer alignment"
        );
    }
}
