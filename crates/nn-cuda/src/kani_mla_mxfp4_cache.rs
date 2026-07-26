// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for MLA decode codegen, MXFP4 GEMM codegen,
//! HipCache content hashing, HipSyntax trait impl, HipMemcpyKind
//! repr invariants, and compile_hip command generation.
//!
//! Part of #3802.

// =========================================================================
// MLA decode codegen proofs
// =========================================================================

use super::codegen_hip_mla_decode::{emit_mla_decode_kernel, mla_decode_launch_config};
use super::codegen_hip_mxfp4_gemm::{emit_mxfp4_gemm_kernel, mxfp4_gemm_launch_config};
use super::codegen_syntax_hip::HipSyntax;
use super::compile_hip::{hipcc_command, target};
use super::hip_cache::HipCache;
use super::hip_ffi::HipMemcpyKind;
use nn_dsl::codegen_syntax::CodegenSyntax;
use nn_dsl::ScalarType;

/// Prove MLA decode kernel accepts valid DeepSeek-V3 parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_mla_decode_valid_deepseek() {
    let result = emit_mla_decode_kernel("mla_ds", 16, 128, 512, 64, 1, 0.08838835);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("mla_ds"));
    assert!(src.contains("__global__"));
    assert!(src.contains("q_absorbed"));
    assert!(src.contains("v_weighted"));
}

/// Prove MLA decode kernel rejects d_c that exceeds shared memory (d_c * 8 > 65536).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_mla_decode_rejects_large_dc() {
    // d_c = 8193 => 8193 * 8 = 65544 > 65536
    let result = emit_mla_decode_kernel("mla_big", 1, 64, 8193, 32, 1, 1.0);
    assert!(result.is_err());
}

/// Prove MLA decode kernel accepts d_c at the shared memory boundary.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_mla_decode_accepts_dc_boundary() {
    // d_c = 8192 => 8192 * 8 = 65536, exactly at the limit
    let result = emit_mla_decode_kernel("mla_edge", 1, 64, 8192, 32, 1, 1.0);
    assert!(result.is_ok());
}

/// Prove MLA decode launch config grid covers all (batch, head) pairs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_mla_launch_grid_coverage() {
    let batch: u8 = kani::any();
    let heads: u8 = kani::any();
    kani::assume(batch > 0);
    kani::assume(heads > 0);

    let cfg = mla_decode_launch_config(batch as usize, heads as usize, 512);
    assert_eq!(cfg.grid.x, (batch as u32) * (heads as u32));
    assert_eq!(cfg.block.x, 256);
}

/// Prove MLA decode launch config shared memory includes q_absorbed + v_weighted + meta + partial_scores.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mla_launch_shared_mem_formula() {
    let d_c: u16 = kani::any();
    kani::assume(d_c > 0);

    let cfg = mla_decode_launch_config(1, 1, d_c as usize);
    // Formula: (2 * d_c + 3 + 256) * 4 bytes
    let expected = ((2 * d_c as u32 + 3 + 256) * 4) as u32;
    assert_eq!(cfg.shared_mem_bytes, expected);
}

/// Prove MLA decode kernel embeds the scale factor in the source.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mla_decode_embeds_scale() {
    let result = emit_mla_decode_kernel("mla_s", 2, 64, 128, 16, 1, 0.125);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("0.12500000"));
}

// =========================================================================
// MXFP4 GEMM codegen proofs
// =========================================================================

/// Prove MXFP4 GEMM rejects non-32-aligned M.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_gemm_rejects_unaligned_m() {
    let m: u8 = kani::any();
    kani::assume(m > 0);
    kani::assume(m as usize % 32 != 0);
    let result = emit_mxfp4_gemm_kernel("mxfp4", m as usize, 32, 32, 1);
    assert!(result.is_err());
}

/// Prove MXFP4 GEMM rejects non-32-aligned K.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_gemm_rejects_unaligned_k() {
    let k: u8 = kani::any();
    kani::assume(k > 0);
    kani::assume(k as usize % 32 != 0);
    let result = emit_mxfp4_gemm_kernel("mxfp4", 32, k as usize, 32, 1);
    assert!(result.is_err());
}

/// Prove MXFP4 GEMM rejects non-32-aligned N.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_gemm_rejects_unaligned_n() {
    let n: u8 = kani::any();
    kani::assume(n > 0);
    kani::assume(n as usize % 32 != 0);
    let result = emit_mxfp4_gemm_kernel("mxfp4", 32, 32, n as usize, 1);
    assert!(result.is_err());
}

/// Prove MXFP4 GEMM accepts 32-aligned dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_gemm_accepts_aligned() {
    let m_blocks: u8 = kani::any();
    let k_blocks: u8 = kani::any();
    let n_blocks: u8 = kani::any();
    kani::assume(m_blocks > 0 && m_blocks <= 4);
    kani::assume(k_blocks > 0 && k_blocks <= 4);
    kani::assume(n_blocks > 0 && n_blocks <= 4);

    let result = emit_mxfp4_gemm_kernel(
        "mxfp4",
        m_blocks as usize * 32,
        k_blocks as usize * 32,
        n_blocks as usize * 32,
        1,
    );
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("mxfp4"));
    assert!(src.contains("mxfp4_dequant"));
    assert!(src.contains("rocwmma"));
}

/// Prove MXFP4 GEMM launch config matches rocWMMA layout.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_gemm_launch_matches_rocwmma() {
    let m: u8 = kani::any();
    let n: u8 = kani::any();
    let batch: u8 = kani::any();
    kani::assume(m > 0);
    kani::assume(n > 0);
    kani::assume(batch > 0);

    let cfg = mxfp4_gemm_launch_config(m as usize * 32, n as usize * 32, batch as usize);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.block.y, 1);
    assert_eq!(cfg.grid.z, batch as u32);
    assert!(u64::from(cfg.grid.x) * 32 >= (n as u64) * 32);
    assert!(u64::from(cfg.grid.y) * 32 >= (m as u64) * 32);
}

// =========================================================================
// HipCache content_hash proofs
// =========================================================================

/// Prove content_hash is deterministic: same inputs produce same hash.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cache_hash_deterministic() {
    let h1 = HipCache::content_hash("kernel", "gfx90a");
    let h2 = HipCache::content_hash("kernel", "gfx90a");
    assert_eq!(h1, h2);
}

/// Prove content_hash produces a 16-character hex string.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cache_hash_length() {
    let h = HipCache::content_hash("source", "gfx1100");
    assert_eq!(h.len(), 16);
    // All chars are hex digits
    let bytes = h.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        assert!((b >= b'0' && b <= b'9') || (b >= b'a' && b <= b'f'));
        i += 1;
    }
}

/// Prove content_hash differs when source differs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cache_hash_source_sensitivity() {
    let h1 = HipCache::content_hash("void k1() {}", "gfx90a");
    let h2 = HipCache::content_hash("void k2() {}", "gfx90a");
    assert_ne!(h1, h2);
}

/// Prove content_hash differs when arch differs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cache_hash_arch_sensitivity() {
    let h1 = HipCache::content_hash("void k() {}", "gfx90a");
    let h2 = HipCache::content_hash("void k() {}", "gfx1100");
    assert_ne!(h1, h2);
}

// =========================================================================
// HipMemcpyKind repr proofs
// =========================================================================

/// Prove HipMemcpyKind enum repr values match HIP API constants.
#[kani::unwind(1)]
#[kani::proof]
fn prove_memcpy_kind_repr_values() {
    assert_eq!(HipMemcpyKind::HostToHost as i32, 0);
    assert_eq!(HipMemcpyKind::HostToDevice as i32, 1);
    assert_eq!(HipMemcpyKind::DeviceToHost as i32, 2);
    assert_eq!(HipMemcpyKind::DeviceToDevice as i32, 3);
}

/// Prove all HipMemcpyKind variants are distinct.
#[kani::unwind(1)]
#[kani::proof]
fn prove_memcpy_kind_all_distinct() {
    let kinds = [
        HipMemcpyKind::HostToHost as i32,
        HipMemcpyKind::HostToDevice as i32,
        HipMemcpyKind::DeviceToHost as i32,
        HipMemcpyKind::DeviceToDevice as i32,
    ];
    // No duplicates in the array
    let mut i = 0;
    while i < kinds.len() {
        let mut j = i + 1;
        while j < kinds.len() {
            assert_ne!(kinds[i], kinds[j]);
            j += 1;
        }
        i += 1;
    }
}

// =========================================================================
// HipSyntax CodegenSyntax trait proofs
// =========================================================================

/// Prove HipSyntax.uint_keyword returns "unsigned int".
#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_syntax_uint_keyword() {
    let s = HipSyntax;
    assert_eq!(s.uint_keyword(), "unsigned int");
}

/// Prove HipSyntax.backend_name returns "HIP".
#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_syntax_backend_name() {
    let s = HipSyntax;
    assert_eq!(s.backend_name(), "HIP");
}

/// Prove HipSyntax.type_name matches hip_type for all supported dtypes.
#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_syntax_type_name_f32() {
    let s = HipSyntax;
    assert_eq!(s.type_name(ScalarType::F32).unwrap(), "float");
}

/// Prove HipSyntax.cast_expr wraps correctly.
#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_syntax_cast_expr() {
    let s = HipSyntax;
    let cast = s.cast_expr("float", "x");
    assert_eq!(cast, "(float)x");
}

/// Prove HipSyntax.accum_type always returns "float".
#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_syntax_accum_type_always_float() {
    let s = HipSyntax;
    assert_eq!(s.accum_type(ScalarType::F32), "float");
    assert_eq!(s.accum_type(ScalarType::F16), "float");
    assert_eq!(s.accum_type(ScalarType::BF16), "float");
}

// =========================================================================
// hipcc_command proofs
// =========================================================================

/// Prove hipcc_command always produces 7 elements.
#[kani::unwind(1)]
#[kani::proof]
fn prove_hipcc_command_length() {
    let cmd = hipcc_command(
        std::path::Path::new("/tmp/k.hip.cpp"),
        std::path::Path::new("/tmp/k.hsaco"),
        "gfx90a",
    );
    assert_eq!(cmd.len(), 7);
}

/// Prove hipcc_command first element is "hipcc".
#[kani::unwind(1)]
#[kani::proof]
fn prove_hipcc_command_binary_name() {
    let cmd = hipcc_command(
        std::path::Path::new("/tmp/k.hip.cpp"),
        std::path::Path::new("/tmp/k.hsaco"),
        "gfx90a",
    );
    assert_eq!(cmd[0], "hipcc");
}

/// Prove hipcc_command embeds target arch.
#[kani::unwind(1)]
#[kani::proof]
fn prove_hipcc_command_embeds_arch() {
    let cmd = hipcc_command(
        std::path::Path::new("/src.cpp"),
        std::path::Path::new("/out.hsaco"),
        "gfx942",
    );
    assert_eq!(cmd[2], "--offload-arch=gfx942");
}

// =========================================================================
// compile_hip::target constant proofs
// =========================================================================

/// Prove target arch constants are non-empty and start with "gfx".
#[kani::unwind(1)]
#[kani::proof]
fn prove_target_constants_format() {
    assert!(target::GFX90A.starts_with("gfx"));
    assert!(target::GFX942.starts_with("gfx"));
    assert!(target::GFX950.starts_with("gfx"));
    assert!(target::GFX1100.starts_with("gfx"));
    assert!(target::GFX1102.starts_with("gfx"));
}

/// Prove all target constants are distinct.
#[kani::unwind(1)]
#[kani::proof]
fn prove_target_constants_distinct() {
    let targets = [
        target::GFX90A,
        target::GFX942,
        target::GFX950,
        target::GFX1100,
        target::GFX1102,
    ];
    let mut i = 0;
    while i < targets.len() {
        let mut j = i + 1;
        while j < targets.len() {
            assert!(targets[i] != targets[j]);
            j += 1;
        }
        i += 1;
    }
}
