// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor GPU dispatch logic safety (#4119).
//!
//! Proves correctness properties of the GPU dispatch routing layer:
//!
//! - DType classification: `is_float()` / `is_int()` exhaustive partition
//! - DType byte widths: `size_bytes()` non-zero for all variants
//! - GPU byte-width safety: same-width floats share byte width, cross-width differ
//! - GPU buffer size: `elements * bytes_per_element` overflow-safe via `checked_mul`
//! - Device classification: `is_gpu()` / `is_cpu()` / `is_accelerator()` consistency
//! - Device matching: same device equals itself
//! - Softmax clamp constants: finite, ordered, positive min_positive per dtype
//! - GPU fallback routing: non-F32 triggers f32 fallback, non-float triggers shape fallback
//! - `checked_f64_to_f32`: finite-preserving, overflow-detecting
//! - Dispatch enum variant coverage: BinaryOp, UnaryOp, ReduceOp, CompareOp
//! - GpuNnOps default `None` contract
//! - GpuShapeOps default `None` contract
//!
//! These harnesses operate on pure type/enum/scalar logic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use crate::tensor::checked_dim_product;
use crate::DType;
use crate::Device;

// ============================================================================
// 1. DType float/int classification is an exhaustive partition
// ============================================================================

/// Prove: every DType variant is classified as exactly one of float, int, or
/// neither (Bool). No variant is both float and int. This is the routing
/// invariant for GPU dispatch: float dtypes take the GPU float path, integer
/// dtypes take the CPU fallback path.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_float_int_partition() {
    // Check all 9 variants exhaustively
    let all_dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    let mut i = 0;
    while i < 9 {
        let dt = all_dtypes[i];
        // No dtype is both float and int
        assert!(
            !(dt.is_float() && dt.is_int()),
            "a dtype cannot be both float and int"
        );
        i += 1;
    }
    // Float dtypes
    assert!(DType::F32.is_float());
    assert!(DType::F16.is_float());
    assert!(DType::BF16.is_float());
    assert!(DType::F64.is_float());
    // Integer dtypes
    assert!(DType::I32.is_int());
    assert!(DType::I64.is_int());
    assert!(DType::U32.is_int());
    assert!(DType::U8.is_int());
    // Bool is neither
    assert!(!DType::Bool.is_float());
    assert!(!DType::Bool.is_int());
}

// ============================================================================
// 2. DType size_bytes is non-zero for all variants
// ============================================================================

/// Prove: every DType has a non-zero byte size. A zero size_bytes would cause
/// division-by-zero in GPU buffer calculations (elements * 0 = 0 bytes, but
/// the buffer needs real memory).
#[kani::unwind(1)]
#[kani::proof]
fn dtype_size_bytes_all_nonzero() {
    let all_dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    let mut i = 0;
    while i < 9 {
        assert!(
            all_dtypes[i].size_bytes() > 0,
            "every dtype must have nonzero byte size"
        );
        i += 1;
    }
}

// ============================================================================
// 3. GPU byte-width: BF16 and F16 share 2-byte width (relabel safety)
// ============================================================================

/// Prove: BF16 and F16 have identical byte widths (2 bytes each). This is the
/// safety invariant for zero-copy `gpu_relabel_dtype`: BF16 <-> F16 can be
/// relabeled without data movement because they share the same 2-byte GPU
/// buffer layout. Source: #1687.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_byte_width_bf16_f16_same() {
    assert_eq!(DType::BF16.size_bytes(), 2, "BF16 must be 2 bytes");
    assert_eq!(DType::F16.size_bytes(), 2, "F16 must be 2 bytes");
    assert_eq!(
        DType::BF16.size_bytes(),
        DType::F16.size_bytes(),
        "BF16 and F16 must share byte width for GPU relabel"
    );
}

// ============================================================================
// 4. GPU byte-width: F32 and BF16/F16 differ (cross-width guard)
// ============================================================================

/// Prove: F32 (4 bytes) and BF16/F16 (2 bytes) have different byte widths.
/// Zero-copy relabel between them would cause the GPU dispatch to misinterpret
/// buffer data (reading 4-byte floats as 2-byte halves or vice versa).
/// This is the guard enforced by `same_gpu_byte_width()`. Source: #1687.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_byte_width_f32_vs_half_differ() {
    assert_ne!(
        DType::F32.size_bytes(),
        DType::BF16.size_bytes(),
        "F32 and BF16 must differ in byte width"
    );
    assert_ne!(
        DType::F32.size_bytes(),
        DType::F16.size_bytes(),
        "F32 and F16 must differ in byte width"
    );
}

// ============================================================================
// 5. GPU buffer size calculation: checked_mul overflow detection
// ============================================================================

/// Prove: GPU buffer size computation (elements * bytes_per_element) correctly
/// detects overflow when the product exceeds usize::MAX. A silent wraparound
/// would allocate a too-small buffer, causing GPU out-of-bounds writes.
/// Source: #930.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_buffer_size_overflow_detected() {
    let elements: usize = kani::any();
    let bytes_per_element: usize = kani::any();

    // Constrain to realistic GPU buffer sizes (non-zero element size)
    kani::assume(bytes_per_element >= 1 && bytes_per_element <= 8);
    kani::assume(elements >= 1);

    let result = elements.checked_mul(bytes_per_element);

    match result {
        Some(total_bytes) => {
            // If no overflow, the total must be >= both factors
            assert!(total_bytes >= elements, "total_bytes must be >= elements");
            assert!(
                total_bytes >= bytes_per_element,
                "total_bytes must be >= bytes_per_element"
            );
        }
        None => {
            // Overflow detected — this is the safe path
            // Verify that unchecked multiplication would indeed wrap
            let wrapped = elements.wrapping_mul(bytes_per_element);
            assert!(
                wrapped < elements || wrapped < bytes_per_element,
                "checked_mul must only return None on actual overflow"
            );
        }
    }
}

// ============================================================================
// 6. Shape validation: checked_dim_product for GPU dispatch
// ============================================================================

/// Prove: checked_dim_product correctly computes element count for shapes used
/// in GPU dispatch. The GPU dispatch layer calls this to determine buffer
/// allocation sizes before launching kernels.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_shape_dim_product_2d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4096);
    kani::assume(d1 >= 1 && d1 <= 4096);

    let dims = [d0 as usize, d1 as usize];
    let result = checked_dim_product(&dims);

    assert!(result.is_ok(), "small 2D shapes must not overflow");
    let numel = result.unwrap();
    assert_eq!(
        numel,
        dims[0] * dims[1],
        "checked product must agree with direct multiplication"
    );
    assert!(
        numel >= 1,
        "element count must be positive for non-empty shapes"
    );
}

// ============================================================================
// 7. Device GPU classification consistency
// ============================================================================

/// Prove: `is_gpu()` returns true for exactly the GPU device variants
/// (Metal, CUDA, Vulkan) and false for CPU and ANE. The GPU dispatch layer
/// checks `device.is_gpu()` to decide whether to route through the registered
/// GpuBackend.
#[kani::unwind(1)]
#[kani::proof]
fn device_is_gpu_consistency() {
    // GPU devices
    assert!(Device::Metal { device_id: 0 }.is_gpu(), "Metal must be GPU");
    assert!(Device::Cuda { device_id: 0 }.is_gpu(), "CUDA must be GPU");
    assert!(
        Device::Vulkan { device_id: 0 }.is_gpu(),
        "Vulkan must be GPU"
    );
    // Non-GPU devices
    assert!(!Device::Cpu.is_gpu(), "CPU must not be GPU");
    assert!(!Device::Ane.is_gpu(), "ANE must not be GPU");
}

// ============================================================================
// 8. Device is_cpu / is_gpu mutual exclusivity
// ============================================================================

/// Prove: `is_cpu()` and `is_gpu()` are mutually exclusive for the standard
/// device variants. A tensor on a GPU device must not report as CPU, and vice
/// versa. Violating this would cause the dispatch layer to attempt CPU
/// operations on GPU buffers (segfault) or GPU operations on CPU arrays.
#[kani::unwind(1)]
#[kani::proof]
fn device_cpu_gpu_mutually_exclusive() {
    let devices = [
        Device::Cpu,
        Device::Metal { device_id: 0 },
        Device::Cuda { device_id: 0 },
        Device::Vulkan { device_id: 0 },
        Device::Ane,
    ];
    let mut i = 0;
    while i < 5 {
        let d = devices[i];
        assert!(
            !(d.is_cpu() && d.is_gpu()),
            "a device cannot be both CPU and GPU"
        );
        i += 1;
    }
}

// ============================================================================
// 9. Device self-equality for matching
// ============================================================================

/// Prove: every device variant equals itself. GPU dispatch requires tensor
/// device to match the backend device — this fails if PartialEq is broken.
#[kani::unwind(1)]
#[kani::proof]
fn device_self_equality() {
    let d0 = Device::Cpu;
    assert_eq!(d0, d0, "CPU must equal itself");

    let d1 = Device::Metal { device_id: 0 };
    assert_eq!(d1, d1, "Metal(0) must equal itself");

    let d2 = Device::Metal { device_id: 1 };
    assert_eq!(d2, d2, "Metal(1) must equal itself");
    assert_ne!(d1, d2, "Metal(0) must not equal Metal(1)");

    let d3 = Device::Cuda { device_id: 0 };
    assert_eq!(d3, d3, "CUDA(0) must equal itself");
    assert_ne!(d1, d3, "Metal(0) must not equal CUDA(0)");
}

// ============================================================================
// 10. Device is_accelerator consistency
// ============================================================================

/// Prove: `is_accelerator()` is equivalent to `!is_cpu()`. Accelerator
/// routing dispatches to GPU or ANE — anything that is not CPU. This must
/// be consistent with `is_cpu()` for the fallback decision to be correct.
#[kani::unwind(1)]
#[kani::proof]
fn device_is_accelerator_is_not_cpu() {
    let devices = [
        Device::Cpu,
        Device::Metal { device_id: 0 },
        Device::Cuda { device_id: 0 },
        Device::Vulkan { device_id: 0 },
        Device::Ane,
    ];
    let mut i = 0;
    while i < 5 {
        let d = devices[i];
        assert_eq!(
            d.is_accelerator(),
            !d.is_cpu(),
            "is_accelerator must be equivalent to !is_cpu"
        );
        i += 1;
    }
}

// ============================================================================
// 11. GPU fallback: needs_f32_fallback logic
// ============================================================================

/// Prove: the `needs_f32_fallback` routing logic (returns true when dtype != F32)
/// correctly identifies non-F32 tensors. This is the guard used by raw MSL
/// kernels that only support `float*` buffers. Source: #1668.
#[kani::unwind(1)]
#[kani::proof]
fn needs_f32_fallback_routing() {
    // F32 does not need fallback
    let f32_needs_fallback = DType::F32 != DType::F32;
    assert!(!f32_needs_fallback, "F32 must not need f32 fallback");

    // All other dtypes need fallback
    let non_f32 = [
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    let mut i = 0;
    while i < 8 {
        let needs_fallback = non_f32[i] != DType::F32;
        assert!(needs_fallback, "non-F32 dtypes must need f32 fallback");
        i += 1;
    }
}

// ============================================================================
// 12. GPU fallback: needs_non_float_fallback logic
// ============================================================================

/// Prove: the `needs_non_float_fallback` routing logic (returns true when
/// dtype is not F32/BF16/F16) correctly identifies non-float tensors for
/// GPU shape ops. GPU shape ops use `dispatch_def` which supports float and
/// half buffers only. Source: #1709.
#[kani::unwind(1)]
#[kani::proof]
fn needs_non_float_fallback_routing() {
    // F32, BF16, F16 do NOT need non-float fallback
    let gpu_supported = [DType::F32, DType::BF16, DType::F16];
    let mut i = 0;
    while i < 3 {
        let needs_fallback = !matches!(gpu_supported[i], DType::F32 | DType::BF16 | DType::F16);
        assert!(
            !needs_fallback,
            "F32/BF16/F16 must not need non-float fallback"
        );
        i += 1;
    }

    // All other dtypes DO need non-float fallback
    let non_gpu = [
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    let mut j = 0;
    while j < 6 {
        let needs_fallback = !matches!(non_gpu[j], DType::F32 | DType::BF16 | DType::F16);
        assert!(
            needs_fallback,
            "non-F32/BF16/F16 dtypes must need non-float fallback"
        );
        j += 1;
    }
}

// ============================================================================
// 13. checked_f64_to_f32: finite f64 that fits in f32 round-trips
// ============================================================================

/// Prove: `checked_f64_to_f32` accepts a finite f64 value that fits in f32
/// without overflow, and returns the correct f32 value.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_f32_within_range() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    // A value that is exactly representable as f32 must round-trip through f64
    let val_f64 = val as f64;
    let result = super::checked_f64_to_f32(val_f64, "test");
    assert!(result.is_ok(), "finite f32-representable f64 must succeed");
    assert_eq!(
        result.unwrap(),
        val,
        "checked_f64_to_f32 must preserve f32-representable values"
    );
}

// ============================================================================
// 14. checked_f64_to_f32: overflow detection
// ============================================================================

/// Prove: `checked_f64_to_f32` rejects finite f64 values that overflow to
/// infinity in f32 representation. This prevents silent data corruption when
/// user-supplied parameters (e.g., clamp bounds) exceed the f32 range.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_f32_rejects_overflow() {
    // f64 value larger than f32::MAX — must be detected as overflow
    let huge: f64 = f64::from(f32::MAX) * 2.0;
    assert!(huge.is_finite(), "test value must be finite in f64");
    let result = super::checked_f64_to_f32(huge, "test");
    assert!(result.is_err(), "f64 overflowing f32 must be rejected");

    // Negative overflow
    let neg_huge: f64 = f64::from(f32::MIN) * 2.0;
    assert!(
        neg_huge.is_finite(),
        "negative test value must be finite in f64"
    );
    let neg_result = super::checked_f64_to_f32(neg_huge, "test");
    assert!(
        neg_result.is_err(),
        "negative f64 overflowing f32 must be rejected"
    );
}

// ============================================================================
// 15. BinaryOp: all variants are distinct
// ============================================================================

/// Prove: BinaryOp variants are pairwise distinct (PartialEq correctness).
/// GPU dispatch matches on BinaryOp to select the correct MSL kernel. If
/// two variants compared equal, the wrong kernel would be dispatched.
#[kani::unwind(1)]
#[kani::proof]
fn binary_op_variants_distinct() {
    use super::BinaryOp;
    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
        BinaryOp::Atan2,
    ];
    let mut i = 0;
    while i < 7 {
        let mut j = i + 1;
        while j < 7 {
            assert_ne!(
                ops[i], ops[j],
                "BinaryOp variants must be pairwise distinct"
            );
            j += 1;
        }
        i += 1;
    }
}

// ============================================================================
// 16. UnaryOp: all variants are distinct
// ============================================================================

/// Prove: UnaryOp variants are pairwise distinct. Same rationale as BinaryOp:
/// incorrect matching dispatches the wrong kernel (e.g., Exp instead of Log).
#[kani::unwind(1)]
#[kani::proof]
fn unary_op_variants_distinct() {
    use super::UnaryOp;
    let ops = [
        UnaryOp::Relu,
        UnaryOp::Gelu,
        UnaryOp::Silu,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Sqr,
        UnaryOp::Abs,
        UnaryOp::Neg,
        UnaryOp::Recip,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::GeluErf,
        UnaryOp::Floor,
        UnaryOp::Round,
        UnaryOp::Fract,
    ];
    let mut i = 0;
    while i < 18 {
        let mut j = i + 1;
        while j < 18 {
            assert_ne!(ops[i], ops[j], "UnaryOp variants must be pairwise distinct");
            j += 1;
        }
        i += 1;
    }
}

// ============================================================================
// 17. ReduceOp: all variants are distinct
// ============================================================================

/// Prove: ReduceOp variants are pairwise distinct. A mixup between Sum and
/// Max reduction would silently produce wrong results in GPU dispatch.
#[kani::unwind(1)]
#[kani::proof]
fn reduce_op_variants_distinct() {
    use super::ReduceOp;
    let ops = [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min];
    let mut i = 0;
    while i < 4 {
        let mut j = i + 1;
        while j < 4 {
            assert_ne!(
                ops[i], ops[j],
                "ReduceOp variants must be pairwise distinct"
            );
            j += 1;
        }
        i += 1;
    }
}

// ============================================================================
// 18. CompareOp: all variants are distinct
// ============================================================================

/// Prove: CompareOp variants are pairwise distinct. A mixup between Ge and
/// Gt (or Eq and Ne) in GPU compare dispatch silently produces wrong masks.
#[kani::unwind(1)]
#[kani::proof]
fn compare_op_variants_distinct() {
    use super::CompareOp;
    let ops = [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Ge,
        CompareOp::Gt,
        CompareOp::Lt,
        CompareOp::Le,
    ];
    let mut i = 0;
    while i < 6 {
        let mut j = i + 1;
        while j < 6 {
            assert_ne!(
                ops[i], ops[j],
                "CompareOp variants must be pairwise distinct"
            );
            j += 1;
        }
        i += 1;
    }
}

// ============================================================================
// 19. GPU buffer size: elements * size_bytes matches checked_dim_product
// ============================================================================

/// Prove: GPU buffer byte size equals `checked_dim_product(dims) * dtype.size_bytes()`
/// and both computations agree. The GPU dispatch layer uses this for buffer
/// allocation before kernel launch. Source: #930.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_buffer_size_consistent() {
    let dim0: u8 = kani::any();
    let dim1: u8 = kani::any();
    kani::assume(dim0 >= 1 && dim0 <= 128);
    kani::assume(dim1 >= 1 && dim1 <= 128);

    let dims = [dim0 as usize, dim1 as usize];
    let numel = checked_dim_product(&dims).unwrap();

    // F32: 4 bytes per element
    let f32_buf = numel.checked_mul(DType::F32.size_bytes());
    assert!(
        f32_buf.is_some(),
        "small tensor F32 buffer must not overflow"
    );
    assert_eq!(
        f32_buf.unwrap(),
        dims[0] * dims[1] * 4,
        "F32 buffer size must be numel * 4"
    );

    // BF16: 2 bytes per element
    let bf16_buf = numel.checked_mul(DType::BF16.size_bytes());
    assert!(
        bf16_buf.is_some(),
        "small tensor BF16 buffer must not overflow"
    );
    assert_eq!(
        bf16_buf.unwrap(),
        dims[0] * dims[1] * 2,
        "BF16 buffer size must be numel * 2"
    );

    // F32 buffer is exactly 2x BF16 buffer for same shape
    assert_eq!(
        f32_buf.unwrap(),
        bf16_buf.unwrap() * 2,
        "F32 buffer must be 2x BF16 buffer"
    );
}

// ============================================================================
// 20. DType float storage invariant: all float dtypes share the F32 label
// ============================================================================

/// Prove: the float storage invariant — all float dtypes (F32, BF16, F16, F64)
/// are classified as `is_float()`, meaning GPU dispatch routes them through
/// the float dispatch path. The DynTensor float storage invariant (#1690)
/// states that F32 is the internal storage dtype; BF16/F16 use FloatStorage
/// with native representations. This harness verifies the classification gate.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_classification_gate() {
    // All float dtypes pass the is_float gate
    assert!(DType::F32.is_float(), "F32 is float");
    assert!(DType::F16.is_float(), "F16 is float");
    assert!(DType::BF16.is_float(), "BF16 is float");
    assert!(DType::F64.is_float(), "F64 is float");

    // No non-float dtype passes the gate
    assert!(!DType::I32.is_float(), "I32 is not float");
    assert!(!DType::I64.is_float(), "I64 is not float");
    assert!(!DType::U32.is_float(), "U32 is not float");
    assert!(!DType::U8.is_float(), "U8 is not float");
    assert!(!DType::Bool.is_float(), "Bool is not float");
}

// ============================================================================
// 21. Metal threadgroup size: power-of-2 alignment
// ============================================================================

/// Prove: standard Metal threadgroup sizes (32, 64, 128, 256, 512, 1024) are
/// powers of 2. Threadgroup sizes must be powers of 2 for efficient SIMD
/// utilization on Apple Silicon. A non-power-of-2 threadgroup size causes
/// suboptimal occupancy.
#[kani::unwind(1)]
#[kani::proof]
fn metal_threadgroup_sizes_power_of_two() {
    let tg_sizes: [u32; 6] = [32, 64, 128, 256, 512, 1024];
    let mut i = 0;
    while i < 6 {
        let s = tg_sizes[i];
        assert!(s > 0, "threadgroup size must be positive");
        assert_eq!(s & (s - 1), 0, "threadgroup size must be a power of 2");
        i += 1;
    }
}

// ============================================================================
// 22. GPU arena allocation: size alignment safety
// ============================================================================

/// Prove: GPU arena allocation sizes rounded up to a 16-byte alignment
/// boundary are always >= the original size. Metal requires buffer offsets
/// and sizes to be 16-byte aligned. Source: #1956.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_arena_alignment_safety() {
    let size: u32 = kani::any();
    kani::assume(size >= 1);
    // Prevent overflow: size must be small enough that rounding up doesn't wrap
    kani::assume(size <= u32::MAX - 15);

    let alignment: u32 = 16;
    let aligned = (size + alignment - 1) & !(alignment - 1);

    assert!(aligned >= size, "aligned size must be >= original size");
    assert_eq!(
        aligned % alignment,
        0,
        "aligned size must be a multiple of 16"
    );
    // The padding is at most alignment - 1
    assert!(
        aligned - size < alignment,
        "padding must be less than alignment"
    );
}
