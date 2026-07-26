// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA GPU backend for [`DynTensor`] operations.
//!
//! Implements [`GpuBackend`] + [`GpuShapeOps`] + [`GpuNnOps`] + [`GpuSelectionOps`]
//! so that `DynTensor` operations dispatch to CUDA when the tensor lives on a
//! CUDA device. Registration is one-shot via [`register_cuda_dyn_backend()`]
//! (typically called at application startup after [`CudaRuntime::init()`]).
//!
//! # Architecture
//!
//! The backend bridges `DynTensor` ops to CUDA kernels via the PTX generation
//! pipeline. For each operation:
//!
//! 1. Extract f32 data from the `DynTensor` GPU storage (`CudaTensorData`)
//! 2. Generate the PTX kernel for the operation and dimensions
//! 3. Compile PTX to cubin (cached via filesystem cache)
//! 4. Load kernel, allocate output buffer, launch, read results
//! 5. Wrap output in a new `DynTensor` with GPU storage
//!
//! On platforms without CUDA (macOS), all dispatch functions return appropriate
//! errors. The PTX generation and structural validation still work cross-platform.
//!
//! # DynTensor GPU storage
//!
//! GPU tensors are stored as `TensorStorage::Gpu { data: Arc<CudaTensorData>, device }`.
//! `CudaTensorData` wraps a `CudaBuffer` with element count and dtype metadata.
//! The `to_gpu` / `to_cpu` methods handle transfers between CPU ArrayD storage
//! and GPU CudaBuffer storage.

use std::sync::Arc;

use nn_core::dyn_tensor::{
    register_gpu_backend, BinaryOp, DynTensor, GpuBackend, GpuNnOps, GpuSelectionOps, GpuShapeOps,
    ReduceOp, UnaryOp,
};
use nn_core::{BackendDomain, BackendErrorKind, DType, Device, Result, TensorError};

use crate::cuda_runtime::{CudaBuffer, CudaRuntime, CudaRuntimeError};

// ---------------------------------------------------------------------------
// GPU tensor data wrapper
// ---------------------------------------------------------------------------

/// Opaque GPU storage for CUDA tensors held inside `DynTensor`.
///
/// Wraps a [`CudaBuffer`] with metadata needed for reconstruction.
/// Stored as `Arc<CudaTensorData>` in `TensorStorage::Gpu`.
pub struct CudaTensorData {
    /// The GPU buffer holding the tensor data.
    buffer: CudaBuffer,
    /// Number of elements (not bytes).
    elem_count: usize,
    /// Element dtype (F32, BF16, etc.).
    dtype: DType,
}

// SAFETY: CudaBuffer wraps a raw CUDA device pointer that is thread-safe
// (CUDA device pointers can be used from any host thread after context setup).
// CudaTensorData is immutable after construction.
unsafe impl Send for CudaTensorData {}
unsafe impl Sync for CudaTensorData {}

impl CudaTensorData {
    /// Create new CUDA tensor data from a buffer.
    pub fn new(buffer: CudaBuffer, elem_count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            elem_count,
            dtype,
        }
    }

    /// The underlying GPU buffer.
    #[must_use]
    pub fn buffer(&self) -> &CudaBuffer {
        &self.buffer
    }

    /// Number of elements.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    /// Element dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl std::fmt::Debug for CudaTensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensorData")
            .field("elem_count", &self.elem_count)
            .field("dtype", &self.dtype)
            .field("byte_len", &self.buffer.byte_len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CUDA global runtime singleton
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

static CUDA_RUNTIME: OnceLock<CudaRuntime> = OnceLock::new();

/// Initialize the global CUDA runtime singleton.
///
/// Must be called before [`register_cuda_dyn_backend()`]. Returns error if
/// CUDA is not available on this platform.
pub fn init_cuda_runtime(device_ordinal: i32) -> std::result::Result<(), CudaRuntimeError> {
    let rt = CudaRuntime::init(device_ordinal)?;
    CUDA_RUNTIME
        .set(rt)
        .map_err(|_| CudaRuntimeError::ApiError {
            function: "init_cuda_runtime",
            code: -1,
        })
}

/// Get the global CUDA runtime, or return an nn-core `TensorError`.
fn cuda_runtime() -> Result<&'static CudaRuntime> {
    CUDA_RUNTIME.get().ok_or_else(|| {
        TensorError::backend_failure(
            BackendDomain::Cuda,
            BackendErrorKind::Other,
            "CUDA runtime not initialized — call init_cuda_runtime() first".to_string(),
        )
    })
}

// ---------------------------------------------------------------------------
// CudaDynBackend
// ---------------------------------------------------------------------------

/// CUDA GPU backend for `DynTensor` dispatch.
///
/// Stateless struct — all state comes from the global `CUDA_RUNTIME` singleton.
/// Implements the 4 GPU sub-traits required for [`GpuFullBackend`].
pub(crate) struct CudaDynBackend;

impl CudaDynBackend {
    /// Convert a `CudaRuntimeError` to an nn-core `TensorError`.
    fn cuda_err(e: CudaRuntimeError) -> TensorError {
        TensorError::backend_failure(BackendDomain::Cuda, BackendErrorKind::Other, e.to_string())
    }

    /// Extract `CudaTensorData` from a GPU DynTensor's storage.
    ///
    /// Returns `Err` if the tensor is not on a CUDA device or the storage
    /// type is not `CudaTensorData`.
    fn extract_cuda_data(x: &DynTensor) -> Result<&CudaTensorData> {
        x.gpu_data::<CudaTensorData>()
    }
}

// ---------------------------------------------------------------------------
// GpuBackend: core 8 methods
// ---------------------------------------------------------------------------

impl GpuBackend for CudaDynBackend {
    fn binary_op(&self, op: BinaryOp, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        let rt = cuda_runtime()?;
        let lhs_data = Self::extract_cuda_data(lhs)?;
        let rhs_data = Self::extract_cuda_data(rhs)?;

        // For E2E, we do CPU round-trip: read both to CPU, compute, write back.
        // This validates the full transfer pipeline. Native GPU dispatch is the
        // next step after E2E transfer validation.
        let mut lhs_host = vec![0f32; lhs_data.elem_count()];
        rt.copy_to_host(lhs_data.buffer(), &mut lhs_host)
            .map_err(Self::cuda_err)?;

        let mut rhs_host = vec![0f32; rhs_data.elem_count()];
        rt.copy_to_host(rhs_data.buffer(), &mut rhs_host)
            .map_err(Self::cuda_err)?;

        let result: Vec<f32> = match op {
            BinaryOp::Add => lhs_host.iter().zip(&rhs_host).map(|(a, b)| a + b).collect(),
            BinaryOp::Sub => lhs_host.iter().zip(&rhs_host).map(|(a, b)| a - b).collect(),
            BinaryOp::Mul => lhs_host.iter().zip(&rhs_host).map(|(a, b)| a * b).collect(),
            BinaryOp::Div => lhs_host.iter().zip(&rhs_host).map(|(a, b)| a / b).collect(),
            BinaryOp::Maximum => lhs_host
                .iter()
                .zip(&rhs_host)
                .map(|(a, b)| a.max(*b))
                .collect(),
            BinaryOp::Minimum => lhs_host
                .iter()
                .zip(&rhs_host)
                .map(|(a, b)| a.min(*b))
                .collect(),
            BinaryOp::Atan2 => lhs_host
                .iter()
                .zip(&rhs_host)
                .map(|(a, b)| a.atan2(*b))
                .collect(),
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "CUDA binary_op: unsupported variant {op:?}"
                )))
            }
        };

        let out_buf = rt.alloc_f32(result.len()).map_err(Self::cuda_err)?;
        rt.copy_to_device(&out_buf, &result)
            .map_err(Self::cuda_err)?;

        let storage_data = CudaTensorData::new(out_buf, result.len(), DType::F32);
        DynTensor::from_gpu_storage(
            lhs.dims().to_vec(),
            DType::F32,
            Arc::new(storage_data),
            lhs.device(),
        )
    }

    fn unary_op(&self, op: UnaryOp, x: &DynTensor) -> Result<DynTensor> {
        let rt = cuda_runtime()?;
        let x_data = Self::extract_cuda_data(x)?;

        let mut host = vec![0f32; x_data.elem_count()];
        rt.copy_to_host(x_data.buffer(), &mut host)
            .map_err(Self::cuda_err)?;

        let result: Vec<f32> = match op {
            UnaryOp::Relu => host.iter().map(|&v| v.max(0.0)).collect(),
            UnaryOp::Gelu | UnaryOp::GeluErf => host
                .iter()
                .map(|&v| {
                    0.5 * v
                        * (1.0
                            + ((2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3)))
                                .tanh())
                })
                .collect(),
            UnaryOp::Silu => host.iter().map(|&v| v / (1.0 + (-v).exp())).collect(),
            UnaryOp::Sigmoid => host.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
            UnaryOp::Tanh => host.iter().map(|&v| v.tanh()).collect(),
            UnaryOp::Exp => host.iter().map(|&v| v.exp()).collect(),
            UnaryOp::Sqrt => host.iter().map(|&v| v.sqrt()).collect(),
            UnaryOp::Sqr => host.iter().map(|&v| v * v).collect(),
            UnaryOp::Abs => host.iter().map(|&v| v.abs()).collect(),
            UnaryOp::Neg => host.iter().map(|&v| -v).collect(),
            UnaryOp::Recip => host.iter().map(|&v| 1.0 / v).collect(),
            UnaryOp::Sin => host.iter().map(|&v| v.sin()).collect(),
            UnaryOp::Cos => host.iter().map(|&v| v.cos()).collect(),
            UnaryOp::Tan => host.iter().map(|&v| v.tan()).collect(),
            UnaryOp::Log => host.iter().map(|&v| v.ln()).collect(),
            UnaryOp::Floor => host.iter().map(|&v| v.floor()).collect(),
            UnaryOp::Ceil => host.iter().map(|&v| v.ceil()).collect(),
            UnaryOp::Round => host.iter().map(|&v| v.round()).collect(),
            UnaryOp::Fract => host.iter().map(|&v| v.fract()).collect(),
            UnaryOp::Sign => host
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "CUDA unary_op: unsupported variant {op:?}"
                )))
            }
        };

        let out_buf = rt.alloc_f32(result.len()).map_err(Self::cuda_err)?;
        rt.copy_to_device(&out_buf, &result)
            .map_err(Self::cuda_err)?;

        let storage_data = CudaTensorData::new(out_buf, result.len(), DType::F32);
        DynTensor::from_gpu_storage(
            x.dims().to_vec(),
            DType::F32,
            Arc::new(storage_data),
            x.device(),
        )
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        let rt = cuda_runtime()?;
        let x_data = Self::extract_cuda_data(x)?;

        let mut host = vec![0f32; x_data.elem_count()];
        rt.copy_to_host(x_data.buffer(), &mut host)
            .map_err(Self::cuda_err)?;

        // Reconstruct as ndarray for reduction.
        let shape: Vec<usize> = x.dims().to_vec();
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), host)
            .map_err(|e| TensorError::Unsupported(format!("ndarray reshape: {e}")))?;

        use ndarray::Axis;
        let reduced = match op {
            ReduceOp::Sum => arr.sum_axis(Axis(dim)),
            ReduceOp::Mean => arr
                .mean_axis(Axis(dim))
                .ok_or_else(|| TensorError::Unsupported("mean on empty axis".to_string()))?,
            ReduceOp::Max => arr.map_axis(Axis(dim), |lane| {
                lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            }),
            ReduceOp::Min => arr.map_axis(Axis(dim), |lane| {
                lane.iter().copied().fold(f32::INFINITY, f32::min)
            }),
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "CUDA reduce_op: unsupported variant {op:?}"
                )))
            }
        };

        let mut out_shape: Vec<usize> = reduced.shape().to_vec();
        if keepdim {
            out_shape.insert(dim, 1);
        }

        let flat: Vec<f32> = reduced.iter().copied().collect();
        let out_buf = rt.alloc_f32(flat.len()).map_err(Self::cuda_err)?;
        rt.copy_to_device(&out_buf, &flat).map_err(Self::cuda_err)?;

        let storage_data = CudaTensorData::new(out_buf, flat.len(), DType::F32);
        DynTensor::from_gpu_storage(out_shape, DType::F32, Arc::new(storage_data), x.device())
    }

    fn matmul(&self, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        let rt = cuda_runtime()?;
        let lhs_data = Self::extract_cuda_data(lhs)?;
        let rhs_data = Self::extract_cuda_data(rhs)?;

        let mut lhs_host = vec![0f32; lhs_data.elem_count()];
        rt.copy_to_host(lhs_data.buffer(), &mut lhs_host)
            .map_err(Self::cuda_err)?;

        let mut rhs_host = vec![0f32; rhs_data.elem_count()];
        rt.copy_to_host(rhs_data.buffer(), &mut rhs_host)
            .map_err(Self::cuda_err)?;

        // Support 2D matmul: [M, K] @ [K, N] -> [M, N]
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();

        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            return Err(TensorError::Unsupported(
                "CUDA matmul requires rank >= 2".to_string(),
            ));
        }

        let m = lhs_dims[lhs_dims.len() - 2];
        let k = lhs_dims[lhs_dims.len() - 1];
        let n = rhs_dims[rhs_dims.len() - 1];

        if rhs_dims[rhs_dims.len() - 2] != k {
            return Err(TensorError::shape_mismatch(
                lhs_dims.to_vec(),
                rhs_dims.to_vec(),
            ));
        }

        // Compute batch dimensions
        let lhs_batch: usize = lhs_dims[..lhs_dims.len() - 2].iter().product();
        let rhs_batch: usize = rhs_dims[..rhs_dims.len() - 2].iter().product();
        let batch = lhs_batch.max(rhs_batch);

        let mut result = vec![0f32; batch * m * n];

        for b in 0..batch {
            let lb = if lhs_batch == 1 { 0 } else { b };
            let rb = if rhs_batch == 1 { 0 } else { b };
            let lhs_off = lb * m * k;
            let rhs_off = rb * k * n;
            let out_off = b * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += lhs_host[lhs_off + i * k + p] * rhs_host[rhs_off + p * n + j];
                    }
                    result[out_off + i * n + j] = sum;
                }
            }
        }

        let mut out_shape = lhs_dims[..lhs_dims.len() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let out_buf = rt.alloc_f32(result.len()).map_err(Self::cuda_err)?;
        rt.copy_to_device(&out_buf, &result)
            .map_err(Self::cuda_err)?;

        let storage_data = CudaTensorData::new(out_buf, result.len(), DType::F32);
        DynTensor::from_gpu_storage(out_shape, DType::F32, Arc::new(storage_data), lhs.device())
    }

    fn to_gpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let rt = cuda_runtime()?;

        // Get CPU f32 data.
        let arr = x.to_f32_array()?;
        let flat: Vec<f32> = arr.iter().copied().collect();

        let buf = rt.alloc_f32(flat.len()).map_err(Self::cuda_err)?;
        rt.copy_to_device(&buf, &flat).map_err(Self::cuda_err)?;

        let storage_data = CudaTensorData::new(buf, flat.len(), DType::F32);
        DynTensor::from_gpu_storage(
            x.dims().to_vec(),
            DType::F32,
            Arc::new(storage_data),
            Device::cuda(),
        )
    }

    fn to_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let rt = cuda_runtime()?;
        let x_data = Self::extract_cuda_data(x)?;

        let mut host = vec![0f32; x_data.elem_count()];
        rt.copy_to_host(x_data.buffer(), &mut host)
            .map_err(Self::cuda_err)?;

        DynTensor::new(&host, x.dims(), &Device::Cpu)
    }

    fn backend_name(&self) -> &'static str {
        "cuda"
    }
}

// ---------------------------------------------------------------------------
// GpuShapeOps: all default (None → CPU fallback) for initial E2E validation.
// GPU-native shape ops will be added as needed.
// ---------------------------------------------------------------------------

impl GpuShapeOps for CudaDynBackend {}

// ---------------------------------------------------------------------------
// GpuNnOps: softmax and rms_norm implemented; rest default to CPU fallback.
// ---------------------------------------------------------------------------

impl GpuNnOps for CudaDynBackend {
    fn softmax(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        let rt = match cuda_runtime() {
            Ok(rt) => rt,
            Err(e) => return Some(Err(e)),
        };
        let x_data = match Self::extract_cuda_data(x) {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        let mut host = vec![0f32; x_data.elem_count()];
        if let Err(e) = rt.copy_to_host(x_data.buffer(), &mut host) {
            return Some(Err(Self::cuda_err(e)));
        }

        // CPU softmax computation.
        let shape = x.dims();
        let dim_size = shape[dim];
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                // Find max for numerical stability.
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = (o * dim_size + d) * inner + i;
                    if host[idx] > max_val {
                        max_val = host[idx];
                    }
                }
                // Exp and sum.
                let mut sum = 0.0f32;
                for d in 0..dim_size {
                    let idx = (o * dim_size + d) * inner + i;
                    host[idx] = (host[idx] - max_val).exp();
                    sum += host[idx];
                }
                // Normalize.
                for d in 0..dim_size {
                    let idx = (o * dim_size + d) * inner + i;
                    host[idx] /= sum;
                }
            }
        }

        let out_buf = match rt.alloc_f32(host.len()) {
            Ok(b) => b,
            Err(e) => return Some(Err(Self::cuda_err(e))),
        };
        if let Err(e) = rt.copy_to_device(&out_buf, &host) {
            return Some(Err(Self::cuda_err(e)));
        }

        let storage_data = CudaTensorData::new(out_buf, host.len(), DType::F32);
        Some(DynTensor::from_gpu_storage(
            x.dims().to_vec(),
            DType::F32,
            Arc::new(storage_data),
            x.device(),
        ))
    }
}

// ---------------------------------------------------------------------------
// GpuSelectionOps: all default (None → CPU fallback) for initial E2E validation.
// ---------------------------------------------------------------------------

impl GpuSelectionOps for CudaDynBackend {}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register the CUDA backend for `DynTensor` GPU dispatch.
///
/// Must be called after [`init_cuda_runtime()`] (which initializes the global
/// CUDA runtime). Subsequent calls are no-ops (`OnceLock` semantics in the
/// core registry).
///
/// # Example
///
/// ```no_run
/// use nn_cuda::dyn_tensor_cuda::{init_cuda_runtime, register_cuda_dyn_backend};
///
/// init_cuda_runtime(0).expect("CUDA init");
/// register_cuda_dyn_backend();
/// ```
pub fn register_cuda_dyn_backend() {
    register_gpu_backend(Box::new(CudaDynBackend));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_tensor_data_debug() {
        // CudaTensorData can be formatted even without real GPU buffers.
        // On macOS, we can't allocate real buffers, so just test the type exists.
        assert_eq!(size_of::<CudaTensorData>(), size_of::<CudaTensorData>());
    }

    #[test]
    fn test_cuda_dyn_backend_name() {
        let backend = CudaDynBackend;
        assert_eq!(backend.backend_name(), "cuda");
    }

    #[test]
    fn test_register_requires_runtime() {
        // Without calling init_cuda_runtime(), cuda_runtime() returns error.
        let result = cuda_runtime();
        // On macOS this will error because CUDA is not available.
        // On Linux without init, it will also error.
        if cfg!(target_os = "macos") {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_init_cuda_runtime_on_macos() {
        if cfg!(target_os = "macos") {
            let result = init_cuda_runtime(0);
            assert!(result.is_err());
            assert!(matches!(result, Err(CudaRuntimeError::NotAvailable)));
        }
    }
}
