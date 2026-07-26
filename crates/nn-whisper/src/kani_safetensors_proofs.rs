// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for safetensors load-path shape validation.
//!
//! Covers:
//! - Valid F32 safetensors shape/data pairs survive conversion into `DynTensor`
//! - `VarBuilder` preserves the loaded tensor shape for later model load lookups
//! - Wrong requested shapes are rejected against safetensors-loaded tensors
//!
//! Issue: #3724

#[cfg(kani)]
mod proofs {
    use super::super::convert_tensor_bytes;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::{DType, Device, VarBuilder};
    use std::collections::HashMap;

    fn f32_bytes(data: &[f32]) -> &[u8] {
        unsafe {
            // SAFETY: `f32` is plain old data, and we preserve the exact backing length.
            std::slice::from_raw_parts(
                data.as_ptr().cast::<u8>(),
                data.len() * std::mem::size_of::<f32>(),
            )
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn safetensors_valid_f32_shape_builds_tensor() {
        let rows: usize = kani::any();
        let cols: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 3);
        kani::assume(cols >= 1 && cols <= 3);

        let shape = vec![rows, cols];
        let float_data = vec![0.0f32; rows * cols];
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.clone(),
            f32_bytes(&float_data),
        )
        .expect("shape/data pair must form a valid safetensors view");

        let converted = convert_tensor_bytes("weight", view.data(), view.dtype())
            .expect("byte conversion")
            .expect("float tensor should not be skipped");
        let tensor =
            DynTensor::new(&converted, view.shape(), &Device::Cpu).expect("shape must match data");

        assert_eq!(
            tensor.dims(),
            view.shape(),
            "tensor dims must match safetensors shape"
        );
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn safetensors_loaded_shape_roundtrips_through_var_builder() {
        let rows: usize = kani::any();
        let cols: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 3);
        kani::assume(cols >= 1 && cols <= 3);

        let shape = vec![rows, cols];
        let float_data = vec![1.0f32; rows * cols];
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.clone(),
            f32_bytes(&float_data),
        )
        .expect("shape/data pair must form a valid safetensors view");

        let converted = convert_tensor_bytes("weight", view.data(), view.dtype())
            .expect("byte conversion")
            .expect("float tensor should not be skipped");
        let tensor = DynTensor::new(&converted, view.shape(), &Device::Cpu).expect("tensor build");

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), tensor);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

        let loaded = vb
            .get(view.shape(), "weight")
            .expect("exact shape lookup must succeed");
        assert_eq!(
            loaded.dims(),
            view.shape(),
            "VarBuilder lookup must preserve the original shape"
        );
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn safetensors_loaded_tensor_rejects_wrong_shape() {
        let rows: usize = kani::any();
        let cols: usize = kani::any();
        let wrong_cols: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 3);
        kani::assume(cols >= 1 && cols <= 3);
        kani::assume(wrong_cols >= 1 && wrong_cols <= 4);
        kani::assume(wrong_cols != cols);

        let correct_shape = vec![rows, cols];
        let wrong_shape = vec![rows, wrong_cols];
        let float_data = vec![2.0f32; rows * cols];
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            correct_shape.clone(),
            f32_bytes(&float_data),
        )
        .expect("shape/data pair must form a valid safetensors view");

        let converted = convert_tensor_bytes("weight", view.data(), view.dtype())
            .expect("byte conversion")
            .expect("float tensor should not be skipped");
        let tensor = DynTensor::new(&converted, view.shape(), &Device::Cpu).expect("tensor build");

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), tensor);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

        let result = vb.get(&wrong_shape, "weight");
        assert!(
            result.is_err(),
            "mismatched shape requests must be rejected"
        );
    }
}
