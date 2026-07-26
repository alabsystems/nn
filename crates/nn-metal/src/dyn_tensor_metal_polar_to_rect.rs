// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Fused polar-to-rectangular GPU kernel (#2491).
//!
//! Converts (magnitude, phase) to (real, imag) in a single Metal compute
//! dispatch using `sincos()`, replacing 4 separate dispatches
//! (cos, sin, mul, mul) in the Kokoro iSTFT path.
//!
//! Input: magnitude `[*shape]`, phase `[*shape]` (same shape, F32).
//! Output: (real `[*shape]`, imag `[*shape]`).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;

use super::MetalTensorData;

#[path = "dyn_tensor_metal_polar_to_rect_msl.rs"]
mod msl;

/// Threads per threadgroup for the elementwise polar_to_rect kernel.
const TG_SIZE: u32 = 256;

impl super::MetalDynBackend {
    /// Fused polar-to-rectangular conversion in a single Metal dispatch.
    ///
    /// Computes `real = magnitude * cos(phase)` and `imag = magnitude * sin(phase)`
    /// using the Metal `sincos()` intrinsic.
    ///
    /// - magnitude: any shape, F32
    /// - phase: same shape as magnitude, F32
    ///
    /// Returns `(real, imag)` with the same shape as inputs.
    pub(in super::super) fn gpu_polar_to_rect(
        magnitude: &DynTensor,
        phase: &DynTensor,
    ) -> Result<(DynTensor, DynTensor)> {
        // MSL kernel uses `float*` (4-byte) — reject BF16/F16 which have 2-byte buffers.
        Self::validate_f32_buffer(magnitude, "gpu_polar_to_rect")?;
        Self::validate_f32_buffer(phase, "gpu_polar_to_rect")?;

        let dims = magnitude.dims();
        if dims != phase.dims() {
            return Err(TensorError::InvalidShape(format!(
                "gpu_polar_to_rect: shape mismatch: magnitude {:?} vs phase {:?}",
                dims,
                phase.dims()
            )));
        }

        let total_elems: usize = dims.iter().product();
        if total_elems == 0 {
            let z = DynTensor::zeros(dims, DType::F32, &Device::metal())?;
            return Ok((z.clone(), z));
        }

        let mag_data = magnitude.gpu_data::<MetalTensorData>()?;
        let phase_data = phase.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let msl_src = msl::fused_polar_to_rect_msl();
            let pipeline = KernelPipeline::from_msl(
                cache,
                msl_src,
                "fused_polar_to_rect_f32",
                2, // 2 input buffers: magnitude, phase
                false,
            )
            .map_err(metal_err)?;

            let out_bytes = total_elems.checked_mul(size_of::<f32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                }
            })?;

            // Allocate two standalone output buffers (real and imag).
            // Freshly created buffers always have byte_offset 0, so
            // set_buffer (without _with_offset) is correct below.
            let real_buf = ctx.create_buffer_zeroed(out_bytes).map_err(metal_err)?;
            let imag_buf = ctx.create_buffer_zeroed(out_bytes).map_err(metal_err)?;

            let count_u32 = crate::to_u32(total_elems, "polar_to_rect count")?;
            let num_tgs = count_u32.div_ceil(TG_SIZE);

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &mag_data.buffer, mag_data.byte_offset);
                    enc.set_buffer_with_offset(1, &phase_data.buffer, phase_data.byte_offset);
                    enc.set_buffer(2, &real_buf);
                    enc.set_buffer(3, &imag_buf);
                    enc.set_bytes(4, &count_u32);
                    enc.encode_threadgroups(
                        pipeline.pipeline(),
                        [num_tgs, 1, 1],
                        [TG_SIZE, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };

            crate::gpu_scope::get_or_create_batch()?;
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| encode(batch));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            // Standalone buffers — no arena, offset always 0.
            let real_storage = MetalTensorData::new(real_buf);
            let imag_storage = MetalTensorData::new(imag_buf);

            let real = DynTensor::from_gpu_storage(
                dims.to_vec(),
                DType::F32,
                Arc::new(real_storage),
                Device::metal(),
            )?;
            let imag = DynTensor::from_gpu_storage(
                dims.to_vec(),
                DType::F32,
                Arc::new(imag_storage),
                Device::metal(),
            )?;
            Ok((real, imag))
        })
    }
}

/// MSL source for pre-compilation: fused polar-to-rect kernel.
pub(crate) fn polar_to_rect_msl_source() -> &'static str {
    msl::fused_polar_to_rect_msl()
}

#[cfg(test)]
mod tests {
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::Device;

    fn init() {
        let _ = crate::MetalBackend::init();
        crate::register_metal_dyn_backend();
    }

    /// Fused sincos kernel vs decomposed cos()+sin()+mul()+mul() on GPU.
    #[test]
    fn test_fused_vs_decomposed_gpu() {
        init();

        let shape = [1, 513, 64];
        let total: usize = shape.iter().product();
        let mag_data = nn_core::test_prng::rand_f32_vec(42, total, 0.0, 5.0);
        let phase_data = nn_core::test_prng::rand_f32_vec(
            99,
            total,
            -std::f32::consts::PI,
            std::f32::consts::PI,
        );

        let mag = DynTensor::from_vec(mag_data, &shape, &Device::metal()).unwrap();
        let phase = DynTensor::from_vec(phase_data, &shape, &Device::metal()).unwrap();

        // Decomposed GPU path: 4 dispatches.
        let dec_real = mag.mul(&phase.cos().unwrap()).unwrap();
        let dec_imag = mag.mul(&phase.sin().unwrap()).unwrap();

        // Fused path: 1 dispatch.
        let (fused_real, fused_imag) =
            super::super::MetalDynBackend::gpu_polar_to_rect(&mag, &phase).unwrap();

        let cpu = Device::Cpu;
        let dr = dec_real
            .to_device(&cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        let di = dec_imag
            .to_device(&cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        let fr = fused_real
            .to_device(&cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        let fi = fused_imag
            .to_device(&cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();

        let max_real = dr
            .iter()
            .zip(fr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_imag = di
            .iter()
            .zip(fi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!("fused vs decomposed GPU: max_real={max_real:.6e}, max_imag={max_imag:.6e}");

        assert!(max_real < 1e-6, "real diff {max_real:.6e}");
        assert!(max_imag < 1e-6, "imag diff {max_imag:.6e}");
    }
}
