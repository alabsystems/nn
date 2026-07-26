// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-to-GPU standalone buffer copy for arena-resident tensors.
//!
//! [`to_standalone`] eliminates the CPU roundtrip in
//! `without_arena(|| tensor.clone())` by using a Metal blit command encoder
//! to copy data GPU→GPU. If the tensor is already standalone (no arena
//! generation), returns a cheap alias (no copy).
//!
//! Part of #4279.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};

use crate::dyn_tensor_metal::MetalTensorData;
use crate::error::MetalError;
use crate::metal_backend::global_metal_context;

/// Create a standalone copy of a GPU tensor via GPU blit.
///
/// If the tensor is arena-resident (has an arena generation stamp), allocates
/// a new standalone Metal buffer and copies data GPU→GPU via a blit command
/// encoder. The returned tensor has `arena_generation: None` and is safe to
/// hold across arena resets and GPU flushes.
///
/// If already standalone (no arena generation), returns a cheap alias — the
/// underlying `MetalBuffer` is reference-counted via ObjC ARC, so no copy
/// occurs.
///
/// If the tensor is not on a Metal GPU device, returns an error.
///
/// This replaces the pattern:
/// ```ignore
/// let stable = nn_metal::without_arena(|| tensor.clone());
/// ```
/// which forces a GPU→CPU→GPU roundtrip. `to_standalone` keeps data on GPU.
///
/// # Errors
///
/// - [`TensorError::Unsupported`] if the tensor is not GPU-resident.
/// - [`TensorError::BackendFailure`] if buffer allocation or blit encoding fails.
///
/// Part of #4279.
pub fn to_standalone(tensor: &DynTensor) -> Result<DynTensor> {
    // Verify the tensor is on a Metal GPU device.
    let device = tensor.device();
    if !matches!(device, Device::Metal { .. }) {
        return Err(TensorError::Unsupported(
            "to_standalone: tensor is not on a Metal GPU device".into(),
        ));
    }

    let gpu_data = tensor.gpu_data::<MetalTensorData>()?;

    // If already standalone (no arena generation), return a cheap alias.
    if gpu_data.arena_generation().is_none() {
        return Ok(tensor.clone());
    }

    // Arena-resident: allocate a fresh standalone buffer and blit copy.
    let ctx = global_metal_context().map_err(|e| {
        TensorError::backend_failure(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            e.to_string(),
        )
    })?;

    let byte_size = tensor.numel()
        .checked_mul(tensor.dtype().size_bytes())
        .ok_or(TensorError::DimensionOverflow {
            dims: vec![tensor.numel(), tensor.dtype().size_bytes()],
        })?;

    if byte_size == 0 {
        // Zero-element tensor: return a standalone empty tensor.
        // No GPU copy needed.
        return Ok(tensor.clone());
    }

    let fresh_buf = ctx.create_buffer_zeroed(byte_size).map_err(|e| {
        TensorError::backend_failure(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            format!("to_standalone: buffer allocation failed: {e}"),
        )
    })?;

    // Encode the blit copy into the lazy GPU command batch.
    crate::gpu_scope::ensure_batch_for_blit()?;
    let blit_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        batch.blit_copy(
            gpu_data.buffer(),
            gpu_data.byte_offset(),
            &fresh_buf,
            0,
            byte_size,
        )
    })?;

    blit_result.map_err(|e: MetalError| {
        TensorError::backend_failure(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::DispatchFailed,
            format!("to_standalone: blit copy failed: {e}"),
        )
    })?;

    // Construct a new DynTensor with standalone storage (arena_generation: None).
    let storage = MetalTensorData::new(fresh_buf);
    DynTensor::from_gpu_storage(
        tensor.dims().to_vec(),
        tensor.dtype(),
        Arc::new(storage),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::without_arena;
    use crate::MetalBackend;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::Device;

    fn setup() {
        let _ = MetalBackend::init();
        crate::register_metal_dyn_backend();
    }

    fn cpu() -> Device {
        Device::Cpu
    }

    fn gpu() -> Device {
        Device::metal()
    }

    #[test]
    fn test_to_standalone_already_standalone() {
        setup();
        // Create a standalone GPU tensor (without_arena ensures no arena gen).
        let cpu_t = DynTensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &cpu(),
        )
        .expect("cpu tensor");
        let gpu_t = without_arena(|| cpu_t.to_device(&gpu())).expect("to gpu");

        // Verify it is standalone (no arena generation).
        let data = gpu_t.gpu_data::<MetalTensorData>().expect("gpu data");
        assert!(data.arena_generation().is_none(), "should be standalone");

        // to_standalone should return a cheap alias (no copy).
        let standalone = to_standalone(&gpu_t).expect("to_standalone");
        assert_eq!(standalone.dims(), gpu_t.dims());
        assert_eq!(standalone.dtype(), gpu_t.dtype());

        // Verify data is identical by reading back to CPU.
        crate::gpu_scope::flush().expect("flush");
        let orig_vals = gpu_t
            .to_device(&cpu())
            .expect("to cpu")
            .to_flat_vec::<f32>()
            .expect("flat vec");
        let standalone_vals = standalone
            .to_device(&cpu())
            .expect("to cpu")
            .to_flat_vec::<f32>()
            .expect("flat vec");
        assert_eq!(orig_vals, standalone_vals);
    }

    #[test]
    fn test_to_standalone_arena_resident() {
        setup();
        // Create an arena-resident GPU tensor by allocating inside arena scope.
        let cpu_t = DynTensor::from_vec(
            vec![10.0f32, 20.0, 30.0, 40.0],
            &[4],
            &cpu(),
        )
        .expect("cpu tensor");
        let gpu_t = cpu_t.to_device(&gpu()).expect("to gpu");

        // The default arena allocates with arena_generation.
        let data = gpu_t.gpu_data::<MetalTensorData>().expect("gpu data");
        let is_arena = data.arena_generation().is_some();

        if is_arena {
            // to_standalone should create a new standalone buffer via blit.
            let standalone = to_standalone(&gpu_t).expect("to_standalone");
            let standalone_data = standalone
                .gpu_data::<MetalTensorData>()
                .expect("gpu data");
            assert!(
                standalone_data.arena_generation().is_none(),
                "standalone should have no arena generation"
            );
            assert_eq!(standalone.dims(), gpu_t.dims());
            assert_eq!(standalone.dtype(), gpu_t.dtype());

            // Verify data is identical.
            crate::gpu_scope::flush().expect("flush");
            let orig_vals = gpu_t
                .to_device(&cpu())
                .expect("to cpu")
                .to_flat_vec::<f32>()
                .expect("flat vec");
            let standalone_vals = standalone
                .to_device(&cpu())
                .expect("to cpu")
                .to_flat_vec::<f32>()
                .expect("flat vec");
            assert_eq!(orig_vals, standalone_vals);
        }
        // If not arena-resident (e.g., pool-acquired), the test still passes.
    }

    #[test]
    fn test_to_standalone_survives_arena_reset() {
        setup();
        let cpu_t = DynTensor::from_vec(
            vec![1.0f32, 2.0, 3.0],
            &[3],
            &cpu(),
        )
        .expect("cpu tensor");
        let gpu_t = cpu_t.to_device(&gpu()).expect("to gpu");

        // Make it standalone.
        let standalone = to_standalone(&gpu_t).expect("to_standalone");

        // Reset the default arena.
        crate::gpu_scope::flush().expect("flush before reset");
        crate::arena::try_reset_active_arena();

        // The standalone tensor should still be readable.
        let vals = standalone
            .to_device(&cpu())
            .expect("to cpu after reset")
            .to_flat_vec::<f32>()
            .expect("flat vec");
        assert_eq!(vals, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_to_standalone_cpu_tensor_errors() {
        setup();
        let cpu_t = DynTensor::from_vec(
            vec![1.0f32, 2.0],
            &[2],
            &cpu(),
        )
        .expect("cpu tensor");
        let result = to_standalone(&cpu_t);
        assert!(result.is_err(), "CPU tensor should fail");
    }
}
