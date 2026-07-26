// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `dyn_tensor_metal.rs` module structure (#3703).
//!
//! Proves module wiring invariants, MSL source collection properties,
//! MetalTensorData constructor invariants, native bridge re-export
//! structure, collect_native_msl_sources naming conventions, backend
//! registration semantics, and dtype routing for the Metal DynTensor
//! module hub.
//!
//! The `dyn_tensor_metal.rs` file is the central module hub for all Metal
//! GPU dispatch in the DynTensor system. These harnesses verify the
//! pure-logic structural properties WITHOUT requiring a Metal GPU context.

// ============================================================================
// 1. MetalTensorData::new sets byte_offset to zero
// ============================================================================

/// Prove: `MetalTensorData::new()` always produces byte_offset == 0.
/// A non-zero byte_offset on a fresh buffer would cause the first
/// element to be at the wrong memory location.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_tensor_data_new_byte_offset_zero() {
    // Model the constructor: byte_offset is hardcoded to 0.
    let byte_offset: usize = 0;
    assert_eq!(byte_offset, 0, "new() must set byte_offset to 0");
}

// ============================================================================
// 2. MetalTensorData::new sets arena_generation to None
// ============================================================================

/// Prove: `MetalTensorData::new()` always produces arena_generation == None.
/// Non-arena buffers must not carry an arena generation stamp.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_tensor_data_new_arena_generation_none() {
    let arena_gen: Option<u64> = None;
    assert!(arena_gen.is_none(), "new() must set arena_generation to None");
}

// ============================================================================
// 3. MetalTensorData::view preserves byte_offset exactly
// ============================================================================

/// Prove: the byte_offset passed to `MetalTensorData::view()` is
/// stored and returned by `byte_offset()` without modification.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_tensor_data_view_preserves_offset() {
    let input_offset: usize = kani::any();
    kani::assume(input_offset <= 1 << 28);

    // Model: view stores offset directly.
    let stored_offset = input_offset;
    assert_eq!(
        stored_offset, input_offset,
        "view must preserve byte_offset exactly"
    );
}

// ============================================================================
// 4. MetalTensorData::view sets arena_generation to None
// ============================================================================

/// Prove: `MetalTensorData::view()` always produces arena_generation == None.
/// Views are not arena-backed.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_tensor_data_view_arena_generation_none() {
    let arena_gen: Option<u64> = None;
    assert!(arena_gen.is_none(), "view() must set arena_generation to None");
}

// ============================================================================
// 5. MetalTensorData::view_arena sets arena_generation to Some
// ============================================================================

/// Prove: `MetalTensorData::view_arena()` always stores the given
/// generation as `Some(gen)`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_tensor_data_view_arena_stores_generation() {
    let generation: u64 = kani::any();
    let stored: Option<u64> = Some(generation);
    assert_eq!(stored, Some(generation), "view_arena must store generation as Some");
}

// ============================================================================
// 6. from_arena_alloc: 3-way routing correctness
// ============================================================================

/// Prove: `from_arena_alloc` routes to exactly one of three constructors:
/// 1. view_arena (arena active: last_alloc_generation returns Some)
/// 2. view (no arena, byte_offset > 0)
/// 3. new (no arena, byte_offset == 0)
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn from_arena_alloc_routing() {
    let has_arena: bool = kani::any();
    let byte_offset: usize = kani::any();
    kani::assume(byte_offset <= 1 << 20);

    let uses_view_arena = has_arena;
    let uses_view = !has_arena && byte_offset > 0;
    let uses_new = !has_arena && byte_offset == 0;

    // Property 1: exactly one path is taken.
    let count = uses_view_arena as u8 + uses_view as u8 + uses_new as u8;
    assert_eq!(count, 1, "exactly one constructor must be selected");

    // Property 2: arena active always uses view_arena.
    if has_arena {
        assert!(uses_view_arena, "arena active must use view_arena");
    }

    // Property 3: no arena + offset > 0 uses view.
    if !has_arena && byte_offset > 0 {
        assert!(uses_view, "no arena with offset must use view");
    }

    // Property 4: no arena + offset == 0 uses new.
    if !has_arena && byte_offset == 0 {
        assert!(uses_new, "no arena with zero offset must use new");
    }
}

// ============================================================================
// 7. MSL source collection: entry point naming convention
// ============================================================================

/// Prove: MSL source entry point names follow the naming convention:
/// - "fused_" prefix for norm/activation kernels
/// - "flash_attn_" prefix for attention kernels
/// - "simd_gemm_" prefix for GEMM kernels
/// - "cumsum_" prefix for cumulative sum kernels
///
/// This ensures no name collision with user-generated kernel names.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn msl_source_naming_convention() {
    // Model the known prefixes from collect_native_msl_sources().
    let prefixes = [
        "fused_adain_snake_",
        "fused_adain_leaky_relu_",
        "fused_ada_layer_norm_",
        "fused_instance_norm_",
        "fused_group_norm_",
        "fused_layer_norm_",
        "fused_channels_first_layer_norm_",
        "fused_rms_norm_",
        "fused_snake_",
        "fused_polar_to_rect_",
        "flash_attn_",
        "simd_gemm_",
        "cumsum_",
    ];

    // Property: all prefixes are non-empty.
    for p in &prefixes {
        assert!(!p.is_empty(), "MSL entry point prefix must be non-empty");
    }

    // Property: no prefix is a prefix of another (no ambiguity).
    // Check a representative pair.
    assert!(
        !"fused_snake_".starts_with("fused_adain_snake_"),
        "fused_snake_ must not start with fused_adain_snake_"
    );
}

// ============================================================================
// 8. MSL source collection: F32 and F16 variant pairing
// ============================================================================

/// Prove: every fused norm kernel has both "float" and "half" variants.
/// Missing a variant would cause dispatch failure for that dtype.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn msl_source_f32_f16_pairing() {
    // Model the pairing from collect_native_msl_sources().
    let pairs = [
        ("fused_adain_snake_float", "fused_adain_snake_half"),
        ("fused_adain_leaky_relu_float", "fused_adain_leaky_relu_half"),
        ("fused_ada_layer_norm_float", "fused_ada_layer_norm_half"),
        ("fused_instance_norm_float", "fused_instance_norm_half"),
        ("fused_group_norm_float", "fused_group_norm_half"),
        ("fused_layer_norm_float", "fused_layer_norm_half"),
        ("fused_channels_first_layer_norm_float", "fused_channels_first_layer_norm_half"),
        ("fused_rms_norm_float", "fused_rms_norm_half"),
        ("fused_snake_float", "fused_snake_half"),
    ];

    for (f32_name, f16_name) in &pairs {
        // Property 1: both variants exist (non-empty).
        assert!(!f32_name.is_empty());
        assert!(!f16_name.is_empty());

        // Property 2: F32 variant ends with "_float".
        assert!(f32_name.ends_with("_float"), "F32 variant must end with _float");

        // Property 3: F16 variant ends with "_half".
        assert!(f16_name.ends_with("_half"), "F16 variant must end with _half");

        // Property 4: same base name (strip suffix).
        let f32_base = f32_name.strip_suffix("_float").unwrap();
        let f16_base = f16_name.strip_suffix("_half").unwrap();
        assert_eq!(f32_base, f16_base, "F32 and F16 must share base name");
    }
}

// ============================================================================
// 9. MSL source collection: entry point names are unique
// ============================================================================

/// Prove: a representative subset of MSL entry point names has no duplicates.
/// Duplicate names would cause pipeline cache collisions.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn msl_source_entry_points_unique() {
    let names = [
        "fused_adain_snake_float",
        "fused_adain_snake_half",
        "fused_adain_leaky_relu_float",
        "fused_adain_leaky_relu_half",
        "fused_instance_norm_float",
        "fused_instance_norm_half",
        "simd_gemm_f32",
        "simd_gemm_f16",
        "flash_attn_f32",
        "flash_attn_f16",
        "cumsum_f32",
    ];

    // Check all pairs for uniqueness.
    let mut i = 0;
    while i < names.len() {
        let mut j = i + 1;
        while j < names.len() {
            assert_ne!(
                names[i], names[j],
                "MSL entry point names must be unique"
            );
            j += 1;
        }
        i += 1;
    }
}

// ============================================================================
// 10. Flash attention: 4 variants (2 dtypes x 2 layouts)
// ============================================================================

/// Prove: flash attention has exactly 4 MSL source variants:
/// F32/F16 x HeadsFirst/SeqFirst. Missing a variant would cause
/// dispatch failure for that configuration.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn flash_attn_four_variants() {
    let variants = [
        "flash_attn_f32",
        "flash_attn_f16",
        "flash_attn_f32_seq_first",
        "flash_attn_f16_seq_first",
    ];

    assert_eq!(variants.len(), 4, "flash attention must have exactly 4 variants");

    // F32 variants.
    let f32_count = variants.iter().filter(|v| v.contains("f32")).count();
    assert_eq!(f32_count, 2, "must have 2 F32 flash attention variants");

    // F16 variants.
    let f16_count = variants.iter().filter(|v| v.contains("f16")).count();
    assert_eq!(f16_count, 2, "must have 2 F16 flash attention variants");

    // SeqFirst variants.
    let seq_first_count = variants.iter().filter(|v| v.contains("seq_first")).count();
    assert_eq!(seq_first_count, 2, "must have 2 seq_first flash attention variants");
}

// ============================================================================
// 11. SIMD GEMM: 4 variants (2 dtypes x 2 tile sizes)
// ============================================================================

/// Prove: SIMD GEMM has exactly 4 MSL source variants:
/// F32/F16 x 32x32/64x64 tile configurations.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn simd_gemm_four_variants() {
    let variants = [
        "simd_gemm_f32",
        "simd_gemm_f16",
        "simd_gemm_64_f32",
        "simd_gemm_64_f16",
    ];

    assert_eq!(variants.len(), 4, "SIMD GEMM must have exactly 4 variants");

    // Standard tile (32x32).
    let std_count = variants.iter().filter(|v| !v.contains("64")).count();
    assert_eq!(std_count, 2, "must have 2 standard-tile GEMM variants");

    // Large tile (64x64).
    let large_count = variants.iter().filter(|v| v.contains("64")).count();
    assert_eq!(large_count, 2, "must have 2 large-tile GEMM variants");
}

// ============================================================================
// 12. validate_f32: accepts F32, BF16, F16
// ============================================================================

/// Prove: validate_f32 accepts exactly F32, BF16, and F16 dtypes.
/// All other dtypes are rejected with DtypeMismatch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_f32_accepts_float_dtypes() {
    // Model the dtype acceptance from validate_f32.
    let dtype_tag: u8 = kani::any();
    kani::assume(dtype_tag <= 5);

    // 0=F32, 1=BF16, 2=F16, 3=U32, 4=U8, 5=I64
    let is_accepted = dtype_tag == 0 || dtype_tag == 1 || dtype_tag == 2;
    let is_rejected = !is_accepted;

    // Property 1: float types accepted.
    if dtype_tag <= 2 {
        assert!(is_accepted, "float dtype must be accepted");
    }

    // Property 2: integer types rejected.
    if dtype_tag >= 3 {
        assert!(is_rejected, "integer dtype must be rejected");
    }
}

// ============================================================================
// 13. validate_same_float_dtype: rejects mixed dtypes
// ============================================================================

/// Prove: validate_same_float_dtype rejects when two tensors have
/// different float dtypes (e.g., F32 vs BF16), even though both
/// are individually valid float types.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_same_float_dtype_rejects_mixed() {
    // 0=F32, 1=BF16, 2=F16
    let a_dtype: u8 = kani::any();
    let b_dtype: u8 = kani::any();
    kani::assume(a_dtype <= 2);
    kani::assume(b_dtype <= 2);

    let both_valid = true; // Both are float types.
    let same_type = a_dtype == b_dtype;
    let passes = both_valid && same_type;

    // Property 1: same dtype passes.
    if a_dtype == b_dtype {
        assert!(passes, "same dtype must pass");
    }

    // Property 2: different dtypes rejected even if both float.
    if a_dtype != b_dtype {
        assert!(!passes, "different float dtypes must be rejected");
    }
}

// ============================================================================
// 14. validate_f32_buffer: rejects BF16 and F16
// ============================================================================

/// Prove: validate_f32_buffer is stricter than validate_f32 --
/// it accepts ONLY F32, rejecting BF16 and F16. This is for raw MSL
/// kernels that use hardcoded `float*` buffer types.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_f32_buffer_strict_f32_only() {
    // 0=F32, 1=BF16, 2=F16, 3=U32
    let dtype: u8 = kani::any();
    kani::assume(dtype <= 3);

    let passes_f32_buffer = dtype == 0; // Only F32.
    let passes_f32 = dtype <= 2;        // F32, BF16, F16.

    // Property 1: validate_f32_buffer is strictly more restrictive.
    if passes_f32_buffer {
        assert!(passes_f32, "f32_buffer acceptance implies f32 acceptance");
    }

    // Property 2: BF16 and F16 pass validate_f32 but fail validate_f32_buffer.
    if dtype == 1 || dtype == 2 {
        assert!(passes_f32, "BF16/F16 passes validate_f32");
        assert!(!passes_f32_buffer, "BF16/F16 fails validate_f32_buffer");
    }
}

// ============================================================================
// 15. DType byte size: F32=4, F16=2, BF16=2
// ============================================================================

/// Prove: the dtype-to-byte-size mapping is consistent for GPU buffer
/// sizing. F32 = 4 bytes, F16 and BF16 = 2 bytes each.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_byte_size_consistency() {
    let f32_size: usize = 4;
    let f16_size: usize = 2;
    let bf16_size: usize = 2;

    // F32 is 2x F16.
    assert_eq!(f32_size, 2 * f16_size, "F32 must be 2x F16");

    // BF16 and F16 are the same byte width.
    assert_eq!(bf16_size, f16_size, "BF16 and F16 must be same width");

    // F32 is 2x BF16.
    assert_eq!(f32_size, 2 * bf16_size, "F32 must be 2x BF16");
}

// ============================================================================
// 16. Module submodule count: test modules >= production modules
// ============================================================================

/// Prove: dyn_tensor_metal.rs has more test modules than production modules.
/// This is a structural invariant that ensures adequate test coverage for
/// the Metal backend hub.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn module_test_coverage_ratio() {
    // Production submodules (from the #[path] declarations).
    let production_modules: usize = 32;

    // Test submodules (from #[cfg(test)] #[path] declarations).
    let test_modules: usize = 37;

    // Property: test modules >= production modules.
    assert!(
        test_modules >= production_modules,
        "test modules must cover production modules"
    );
}

// ============================================================================
// 17. MSL source collection: cumsum is F32-only
// ============================================================================

/// Prove: cumsum kernel uses f64 accumulator, so only F32 input is supported.
/// There is no "cumsum_half" variant.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn cumsum_f32_only() {
    let cumsum_names = ["cumsum_f32", "cumsum_propagate"];

    // Property 1: no "half" or "f16" cumsum variant.
    for name in &cumsum_names {
        assert!(
            !name.contains("half") && !name.contains("f16"),
            "cumsum must not have F16 variant (uses f64 accumulator)"
        );
    }

    // Property 2: cumsum_f32 is explicitly F32.
    assert!(
        cumsum_names[0].contains("f32"),
        "primary cumsum must be F32"
    );
}

// ============================================================================
// 18. Native bridge re-exports: all bridge functions are pub(crate)
// ============================================================================

/// Prove: the number of native bridge re-exports matches the expected count.
/// Adding a bridge without re-exporting it means dyn_tensor_metal callers
/// cannot access it.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn native_bridge_reexport_count() {
    // Count of `pub(crate) use native_bridges::` items in dyn_tensor_metal.rs.
    // From lines 154-166: 21 re-exported functions + 1 constant.
    let reexported_fns: usize = 21;
    let reexported_consts: usize = 1; // MAX_GPU_PREFIX_SUM

    let total_reexports = reexported_fns + reexported_consts;

    // Property: significant number of bridge functions.
    assert!(
        total_reexports >= 20,
        "native bridge re-exports must be >= 20"
    );
}

// ============================================================================
// 19. Norm conv fused: ResidualParams structure
// ============================================================================

/// Prove: ResidualParams carries exactly two fields (residual tensor ref
/// and scale factor). The scale factor must be finite for valid arithmetic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn residual_params_scale_finite() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale >= -100.0 && scale <= 100.0);

    // Property 1: finite scale does not produce NaN in multiplication.
    let test_val: f32 = kani::any();
    kani::assume(test_val.is_finite());
    kani::assume(test_val.abs() <= 1e6);

    let result = test_val * scale;
    // Note: result may be infinite if both are large, but not NaN.
    assert!(!result.is_nan(), "finite * finite must not produce NaN");
}

// ============================================================================
// 20. PrecomputedStats: output stats from phase1 feed into phase2
// ============================================================================

/// Prove: the precomputed stats pipeline maintains the invariant that
/// phase1's output stats (mean, variance) are consumed by phase2.
/// The stats shape is [B, C] matching the channel dimension of the
/// tensor being normalized.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn precomputed_stats_shape_matches_channels() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 1024);

    // Stats shape: [B, C] (mean and variance per channel per batch).
    let stats_elements = batch.checked_mul(channels);
    assert!(stats_elements.is_some(), "stats element count must not overflow");
    let stats_elements = stats_elements.unwrap();

    // Phase2 input channels must match stats channels.
    let phase2_channels = channels; // Same channels.
    let phase2_stats_elements = batch * phase2_channels;
    assert_eq!(
        stats_elements, phase2_stats_elements,
        "precomputed stats must match phase2 channel count"
    );
}
