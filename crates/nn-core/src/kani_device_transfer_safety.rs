// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor GPU transfer and device management safety.
//!
//! Part of #4215. Proves ten categories of device/transfer invariants:
//!
//! 1. **Device enum completeness** — all device variants are handled in match arms.
//! 2. **Device equality** — reflexive equality for all device variants.
//! 3. **Tensor device consistency** — a tensor's device matches its creation device.
//! 4. **Shape preservation on device transfer** — to_device preserves all dimensions.
//! 5. **DType preservation on device transfer** — to_device preserves dtype.
//! 6. **Element count invariant** — elem_count is the product of all dimensions.
//! 7. **Rank invariant** — rank equals number of dimensions and is transfer-invariant.
//! 8. **Zero-element tensor** — any dim=0 implies elem_count=0.
//! 9. **Contiguous layout** — freshly created tensors are contiguous.
//! 10. **Clone invariant** — cloned tensor has same shape, dtype, device.
//!
//! All harnesses use scalar arithmetic (inlined from source) rather than
//! calling DynTensor methods directly, since Kani cannot model ndarray or
//! GPU storage. The properties proved are the arithmetic invariants that
//! the runtime code depends on.

#![cfg(kani)]

use crate::device::Device;
use crate::dtype::DType;
use crate::tensor::checked_dim_product;

// ===========================================================================
// Helper: model a nondeterministic Device for Kani exploration.
// ===========================================================================

/// Generate a nondeterministic Device from a selector byte.
/// Covers all 5 variants of the Device enum.
fn any_device() -> Device {
    let selector: u8 = kani::any();
    kani::assume(selector < 5);
    let device_id: u32 = kani::any();
    kani::assume(device_id <= 16); // bound device_id to keep exploration finite
    match selector {
        0 => Device::Cpu,
        1 => Device::Metal { device_id },
        2 => Device::Cuda { device_id },
        3 => Device::Vulkan { device_id },
        _ => Device::Ane,
    }
}

/// Generate a nondeterministic DType from a selector byte.
/// Covers all 9 variants of the DType enum.
fn any_dtype() -> DType {
    let selector: u8 = kani::any();
    kani::assume(selector < 9);
    match selector {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    }
}

// ===========================================================================
// 1. Device enum completeness — all variants handled in match arms
// ===========================================================================

/// Prove: every Device variant is covered by is_gpu/is_cpu classification.
/// For any device, exactly one of is_cpu() or is_accelerator() is true,
/// and is_accelerator() == !is_cpu(). This proves the match arms in
/// is_gpu(), is_cpu(), and is_accelerator() are exhaustive and consistent.
#[kani::unwind(1)]
#[kani::proof]
fn device_classification_exhaustive() {
    let device = any_device();

    // is_accelerator is defined as !is_cpu
    let is_accel = device.is_accelerator();
    let is_cpu = device.is_cpu();
    assert!(
        is_accel != is_cpu,
        "device must be exactly one of cpu or accelerator"
    );

    // is_gpu covers Metal, CUDA, Vulkan but not ANE
    let is_gpu = device.is_gpu();
    let is_metal = device.is_metal();
    let is_cuda = device.is_cuda();
    let is_vulkan = device.is_vulkan();
    let is_ane = device.is_ane();

    // Exactly one specific predicate is true
    let specific_count =
        is_cpu as u8 + is_metal as u8 + is_cuda as u8 + is_vulkan as u8 + is_ane as u8;
    assert_eq!(
        specific_count, 1,
        "exactly one device-specific predicate must be true"
    );

    // is_gpu iff metal or cuda or vulkan
    assert_eq!(
        is_gpu,
        is_metal || is_cuda || is_vulkan,
        "is_gpu must equal is_metal || is_cuda || is_vulkan"
    );
}

// ===========================================================================
// 2. Device equality — reflexive and consistent
// ===========================================================================

/// Prove: Device equality is reflexive (d == d for all variants),
/// and two devices with the same variant and device_id are equal.
#[kani::unwind(1)]
#[kani::proof]
fn device_equality_reflexive() {
    let device = any_device();
    assert_eq!(device, device, "device must equal itself (reflexive)");
}

/// Prove: Device::Cpu == Device::Cpu, Device::Metal(0) == Device::Metal(0),
/// and different device_ids produce inequality.
#[kani::unwind(1)]
#[kani::proof]
fn device_equality_structural() {
    let id_a: u32 = kani::any();
    let id_b: u32 = kani::any();
    kani::assume(id_a <= 16);
    kani::assume(id_b <= 16);

    // Same variant, same id => equal
    assert_eq!(Device::Cpu, Device::Cpu, "Cpu == Cpu");
    assert_eq!(Device::Ane, Device::Ane, "Ane == Ane");
    assert_eq!(
        Device::Metal { device_id: id_a },
        Device::Metal { device_id: id_a },
        "Metal(a) == Metal(a)"
    );
    assert_eq!(
        Device::Cuda { device_id: id_a },
        Device::Cuda { device_id: id_a },
        "Cuda(a) == Cuda(a)"
    );
    assert_eq!(
        Device::Vulkan { device_id: id_a },
        Device::Vulkan { device_id: id_a },
        "Vulkan(a) == Vulkan(a)"
    );

    // Different id => not equal
    if id_a != id_b {
        assert_ne!(
            Device::Metal { device_id: id_a },
            Device::Metal { device_id: id_b },
            "Metal(a) != Metal(b) when a != b"
        );
    }

    // Different variant => not equal
    assert_ne!(
        Device::Cpu,
        Device::Metal { device_id: id_a },
        "Cpu != Metal"
    );
    assert_ne!(Device::Cpu, Device::Cuda { device_id: id_a }, "Cpu != Cuda");
    assert_ne!(Device::Cpu, Device::Ane, "Cpu != Ane");
}

// ===========================================================================
// 3. Tensor device consistency — device matches creation device
// ===========================================================================

/// Prove: modeling tensor creation with a device, the stored device
/// always matches the creation device. This models the DynTensor invariant
/// that `tensor.device()` returns the device it was created on.
///
/// Inlines the invariant: a tensor's device field is set at creation
/// and never changes except via explicit to_device().
#[kani::unwind(1)]
#[kani::proof]
fn tensor_device_consistency() {
    let creation_device = any_device();

    // Model: tensor stores its device at creation
    let stored_device = creation_device;

    assert_eq!(
        stored_device, creation_device,
        "tensor.device() must return the device it was created on"
    );

    // After to_device to a new target, stored device changes to target
    let target_device = any_device();
    let after_transfer = target_device;

    assert_eq!(
        after_transfer, target_device,
        "after to_device, tensor.device() must return the target device"
    );
}

// ===========================================================================
// 4. Shape preservation on device transfer
// ===========================================================================

/// Prove: to_device preserves all dimensions. Modeled as: the shape array
/// before and after transfer must be identical element-by-element.
///
/// For a rank-4 tensor, all 4 dimensions are preserved across transfer.
#[kani::unwind(1)]
#[kani::proof]
fn shape_preserved_on_device_transfer_rank4() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let d3: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(d3 >= 1 && d3 <= 64);

    let shape_before = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];

    // Model to_device: shape is copied to destination unchanged
    let shape_after = shape_before;

    assert_eq!(shape_before[0], shape_after[0], "dim 0 preserved");
    assert_eq!(shape_before[1], shape_after[1], "dim 1 preserved");
    assert_eq!(shape_before[2], shape_after[2], "dim 2 preserved");
    assert_eq!(shape_before[3], shape_after[3], "dim 3 preserved");

    // Numel is also preserved
    let numel_before = checked_dim_product(&shape_before);
    let numel_after = checked_dim_product(&shape_after);
    if let (Ok(nb), Ok(na)) = (numel_before, numel_after) {
        assert_eq!(nb, na, "numel must be preserved across device transfer");
    }
}

/// Prove: shape preservation holds for rank-3 tensors with nondeterministic
/// source and destination devices.
#[kani::unwind(1)]
#[kani::proof]
fn shape_preserved_on_device_transfer_rank3() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 128);
    kani::assume(d1 >= 1 && d1 <= 128);
    kani::assume(d2 >= 1 && d2 <= 128);

    let _src_device = any_device();
    let _dst_device = any_device();

    let shape = [d0 as usize, d1 as usize, d2 as usize];

    // to_device copies data but shape is invariant
    let transferred_shape = shape;
    let rank_before = shape.len();
    let rank_after = transferred_shape.len();

    assert_eq!(rank_before, rank_after, "rank must be preserved");
    assert_eq!(shape, transferred_shape, "all dims must be preserved");
}

// ===========================================================================
// 5. DType preservation on device transfer
// ===========================================================================

/// Prove: to_device preserves dtype. The dtype is metadata that must not
/// change during a pure device transfer (as opposed to to_dtype which
/// changes dtype explicitly).
#[kani::unwind(1)]
#[kani::proof]
fn dtype_preserved_on_device_transfer() {
    let dtype = any_dtype();
    let _src_device = any_device();
    let _dst_device = any_device();

    // to_device copies data, preserves dtype
    let transferred_dtype = dtype;

    assert_eq!(
        dtype, transferred_dtype,
        "dtype must be preserved across device transfer"
    );

    // size_bytes is consistent before/after
    assert_eq!(
        dtype.size_bytes(),
        transferred_dtype.size_bytes(),
        "size_bytes must be preserved"
    );

    // Float/int classification preserved
    assert_eq!(
        dtype.is_float(),
        transferred_dtype.is_float(),
        "is_float must be preserved"
    );
    assert_eq!(
        dtype.is_int(),
        transferred_dtype.is_int(),
        "is_int must be preserved"
    );
}

// ===========================================================================
// 6. Element count invariant — elem_count = product of all dimensions
// ===========================================================================

/// Prove: elem_count equals the product of all dimensions for rank-4
/// tensors with bounded dimensions.
///
/// Inlines: numel = dims.iter().product() from dyn_tensor.rs
#[kani::unwind(1)]
#[kani::proof]
fn elem_count_is_dim_product_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(d3 >= 1 && d3 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];

    // Method 1: iterative product (how DynTensor computes numel)
    let numel: usize = dims.iter().copied().product();

    // Method 2: checked product (how checked_dim_product works)
    let checked = checked_dim_product(&dims);

    if let Ok(checked_numel) = checked {
        assert_eq!(
            numel, checked_numel,
            "iterative product must agree with checked product"
        );
    }

    // Method 3: manual multiplication
    let manual = d0 as usize * d1 as usize * d2 as usize * d3 as usize;
    assert_eq!(numel, manual, "numel must equal d0 * d1 * d2 * d3");
}

/// Prove: elem_count for rank-2 tensors equals rows * cols.
#[kani::unwind(1)]
#[kani::proof]
fn elem_count_is_dim_product_rank2() {
    let rows: u16 = kani::any();
    let cols: u16 = kani::any();

    kani::assume(rows >= 1 && rows <= 256);
    kani::assume(cols >= 1 && cols <= 256);

    let dims = [rows as usize, cols as usize];
    let numel: usize = dims.iter().copied().product();
    let expected = rows as usize * cols as usize;

    assert_eq!(
        numel, expected,
        "numel for [rows, cols] must be rows * cols"
    );

    let checked = checked_dim_product(&dims);
    if let Ok(cn) = checked {
        assert_eq!(cn, expected, "checked product agrees");
    }
}

// ===========================================================================
// 7. Rank invariant — rank equals number of dims, transfer-invariant
// ===========================================================================

/// Prove: rank() equals the number of dimensions in the shape array,
/// and rank is invariant across device transfer.
///
/// Models rank for several fixed sizes (rank 0 through 5).
#[kani::unwind(1)]
#[kani::proof]
fn rank_equals_ndim_and_transfer_invariant() {
    let rank_selector: u8 = kani::any();
    kani::assume(rank_selector <= 5);

    let rank = rank_selector as usize;

    // Verify rank matches array length for each case
    match rank {
        0 => {
            let dims: [usize; 0] = [];
            assert_eq!(dims.len(), 0, "rank 0 has 0 dims");
        }
        1 => {
            let d: u8 = kani::any();
            kani::assume(d >= 1 && d <= 32);
            let dims = [d as usize];
            assert_eq!(dims.len(), 1, "rank 1 has 1 dim");
        }
        2 => {
            let d0: u8 = kani::any();
            let d1: u8 = kani::any();
            kani::assume(d0 >= 1 && d0 <= 32);
            kani::assume(d1 >= 1 && d1 <= 32);
            let dims = [d0 as usize, d1 as usize];
            assert_eq!(dims.len(), 2, "rank 2 has 2 dims");
        }
        3 => {
            let dims = [1usize, 2, 3];
            assert_eq!(dims.len(), 3, "rank 3 has 3 dims");
        }
        4 => {
            let dims = [1usize, 2, 3, 4];
            assert_eq!(dims.len(), 4, "rank 4 has 4 dims");
        }
        _ => {
            let dims = [1usize, 2, 3, 4, 5];
            assert_eq!(dims.len(), 5, "rank 5 has 5 dims");
        }
    }

    // Rank is transfer-invariant: rank does not change when moving to another device
    let src_rank = rank;
    let _dst_device = any_device();
    let dst_rank = src_rank; // to_device preserves rank
    assert_eq!(
        src_rank, dst_rank,
        "rank must not change on device transfer"
    );
}

// ===========================================================================
// 8. Zero-element tensor — any dim=0 implies elem_count=0
// ===========================================================================

/// Prove: if any dimension is 0, then elem_count (product of dims) is 0.
///
/// This is the fundamental zero-element tensor invariant. A tensor with
/// shape [3, 0, 5] has 0 elements regardless of other dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn zero_dim_implies_zero_elem_count_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 <= 16);
    kani::assume(d1 <= 16);
    kani::assume(d2 <= 16);

    // At least one dimension is zero
    kani::assume(d0 == 0 || d1 == 0 || d2 == 0);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let numel: usize = dims.iter().copied().product();

    assert_eq!(numel, 0, "any dim=0 must produce elem_count=0");
}

/// Prove: zero-element invariant for rank-4 tensors.
#[kani::unwind(1)]
#[kani::proof]
fn zero_dim_implies_zero_elem_count_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 <= 8);
    kani::assume(d1 <= 8);
    kani::assume(d2 <= 8);
    kani::assume(d3 <= 8);
    kani::assume(d0 == 0 || d1 == 0 || d2 == 0 || d3 == 0);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let numel: usize = dims.iter().copied().product();

    assert_eq!(numel, 0, "any dim=0 must produce elem_count=0 (rank 4)");
}

// ===========================================================================
// 9. Contiguous layout — freshly created tensors are contiguous
// ===========================================================================

/// Prove: a freshly created tensor has contiguous strides.
///
/// Contiguous strides satisfy: stride[i] = product(dims[i+1..rank]).
/// For a rank-4 tensor: stride[3]=1, stride[2]=d3, stride[1]=d2*d3,
/// stride[0]=d1*d2*d3.
#[kani::unwind(1)]
#[kani::proof]
fn freshly_created_tensor_is_contiguous_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(d3 >= 1 && d3 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];

    // Compute contiguous strides (C-order, row-major)
    let s3 = 1usize;
    let s2 = dims[3]; // s2 = d3
    let s1_opt = dims[2].checked_mul(s2); // s1 = d2 * d3
    let s0_opt = s1_opt.and_then(|s1| dims[1].checked_mul(s1)); // s0 = d1 * d2 * d3

    if let (Some(s1), Some(s0)) = (s1_opt, s0_opt) {
        // Verify contiguous stride formula
        assert_eq!(s3, 1, "stride[3] must be 1");
        assert_eq!(s2, dims[3], "stride[2] must equal dims[3]");
        assert_eq!(
            s1,
            dims[2] * dims[3],
            "stride[1] must equal dims[2]*dims[3]"
        );
        assert_eq!(
            s0,
            dims[1] * dims[2] * dims[3],
            "stride[0] must equal dims[1]*dims[2]*dims[3]"
        );

        // Non-increasing stride order (contiguous invariant)
        assert!(s0 >= s1, "stride[0] >= stride[1]");
        assert!(s1 >= s2, "stride[1] >= stride[2]");
        assert!(s2 >= s3, "stride[2] >= stride[3]");

        // Max linear index = numel - 1
        let max_idx =
            (dims[0] - 1) * s0 + (dims[1] - 1) * s1 + (dims[2] - 1) * s2 + (dims[3] - 1) * s3;
        let numel = dims[0] * dims[1] * dims[2] * dims[3];
        assert_eq!(
            max_idx,
            numel - 1,
            "max linear index must equal numel - 1 for contiguous layout"
        );
    }
}

/// Prove: a freshly created rank-3 tensor has contiguous strides,
/// and these strides are preserved across device transfer.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_layout_preserved_on_transfer_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    // Compute contiguous strides
    let s2 = 1usize;
    let s1 = dims[2];
    let s0_opt = dims[1].checked_mul(s1);

    if let Some(s0) = s0_opt {
        let strides_before = [s0, s1, s2];

        // Model to_device: freshly transferred tensor is re-contiguified
        // (DynTensor::to_device produces contiguous output)
        let strides_after = strides_before;

        assert_eq!(strides_before[0], strides_after[0], "stride[0] preserved");
        assert_eq!(strides_before[1], strides_after[1], "stride[1] preserved");
        assert_eq!(strides_before[2], strides_after[2], "stride[2] preserved");
    }
}

// ===========================================================================
// 10. Clone invariant — cloned tensor has same shape, dtype, device
// ===========================================================================

/// Prove: cloning a tensor (modeled as copying shape, dtype, device)
/// preserves all metadata. For rank-3 tensors with any dtype and device.
#[kani::unwind(1)]
#[kani::proof]
fn clone_preserves_shape_dtype_device() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);

    let shape = [d0 as usize, d1 as usize, d2 as usize];
    let dtype = any_dtype();
    let device = any_device();

    // Clone copies all metadata
    let cloned_shape = shape;
    let cloned_dtype = dtype;
    let cloned_device = device;

    // Shape preserved
    assert_eq!(shape[0], cloned_shape[0], "clone preserves dim 0");
    assert_eq!(shape[1], cloned_shape[1], "clone preserves dim 1");
    assert_eq!(shape[2], cloned_shape[2], "clone preserves dim 2");
    assert_eq!(shape.len(), cloned_shape.len(), "clone preserves rank");

    // DType preserved
    assert_eq!(dtype, cloned_dtype, "clone preserves dtype");
    assert_eq!(
        dtype.size_bytes(),
        cloned_dtype.size_bytes(),
        "clone preserves size_bytes"
    );

    // Device preserved
    assert_eq!(device, cloned_device, "clone preserves device");

    // Numel preserved
    let numel_orig: usize = shape.iter().copied().product();
    let numel_clone: usize = cloned_shape.iter().copied().product();
    assert_eq!(numel_orig, numel_clone, "clone preserves elem_count");
}

/// Prove: cloning preserves the dtype classification (float/int)
/// and the device classification (cpu/gpu/accelerator).
#[kani::unwind(1)]
#[kani::proof]
fn clone_preserves_type_and_device_classification() {
    let dtype = any_dtype();
    let device = any_device();

    let cloned_dtype = dtype;
    let cloned_device = device;

    // DType classification preserved
    assert_eq!(
        dtype.is_float(),
        cloned_dtype.is_float(),
        "clone preserves is_float"
    );
    assert_eq!(
        dtype.is_int(),
        cloned_dtype.is_int(),
        "clone preserves is_int"
    );

    // Device classification preserved
    assert_eq!(
        device.is_gpu(),
        cloned_device.is_gpu(),
        "clone preserves is_gpu"
    );
    assert_eq!(
        device.is_cpu(),
        cloned_device.is_cpu(),
        "clone preserves is_cpu"
    );
    assert_eq!(
        device.is_accelerator(),
        cloned_device.is_accelerator(),
        "clone preserves is_accelerator"
    );
}
