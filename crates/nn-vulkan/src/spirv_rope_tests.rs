// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::generate_rope_spirv`] and [`super::generate_rope_neox_spirv`].

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;

// ---- Helpers ----

fn words_to_u32(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

fn assert_valid_header(spirv: &[u32], label: &str) {
    assert!(spirv.len() >= 5, "{label}: module too short");
    assert_eq!(spirv[0], SPIRV_MAGIC, "{label}: wrong magic number");
    assert_eq!(spirv[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
    assert_eq!(spirv[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
    assert!(spirv[3] > 0, "{label}: bound must be > 0");
    assert_eq!(spirv[4], 0, "{label}: schema must be 0");
}

fn has_opcode(spirv: &[u32], target_opcode: u16) -> bool {
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            return true;
        }
        pos += word_count;
    }
    false
}

fn count_opcode(spirv: &[u32], target_opcode: u16) -> usize {
    let mut pos = 5;
    let mut count = 0;
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            count += 1;
        }
        pos += word_count;
    }
    count
}

// ---- SPIR-V header validation ----

#[test]
fn test_rope_spirv_valid_header() {
    let spirv = generate_rope_spirv(128, 64);
    assert_valid_header(&spirv, "rope(128,64)");
}

#[test]
fn test_rope_neox_spirv_valid_header() {
    let spirv = generate_rope_neox_spirv(128, 64);
    assert_valid_header(&spirv, "rope_neox(128,64)");
}

// ---- SPIR-V magic from raw bytes ----

#[test]
fn test_rope_spirv_magic_from_bytes() {
    let spirv = generate_rope_spirv(128, 64);
    let bytes = words_to_u32(&spirv);
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, 0x07230203, "first 4 bytes must be SPIR-V magic");
}

// ---- Entry point ----

#[test]
fn test_rope_spirv_entry_point_main() {
    let spirv = generate_rope_spirv(128, 64);
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("rope: no entry point found"));
    assert_eq!(name, "main", "rope: entry point must be 'main'");
}

#[test]
fn test_rope_neox_spirv_entry_point_main() {
    let spirv = generate_rope_neox_spirv(128, 64);
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("rope_neox: no entry point found"));
    assert_eq!(name, "main", "rope_neox: entry point must be 'main'");
}

// ---- Workgroup size ----

#[test]
fn test_rope_spirv_workgroup_size() {
    let spirv = generate_rope_spirv(128, 64);
    let wg = find_workgroup_size(&spirv).unwrap_or_else(|| panic!("rope: no workgroup size found"));
    assert_eq!(
        wg,
        [ROPE_WORKGROUP_SIZE, 1, 1],
        "rope: workgroup size must be [{ROPE_WORKGROUP_SIZE}, 1, 1]",
    );
}

// ---- Reference correctness: dimension pairing ----

#[test]
fn test_rope_reference_dimension_pairing() {
    // For head_dim=4, dimensions (0,1) and (2,3) form pairs.
    // At position 0, theta=0 for all dims, so cos(0)=1, sin(0)=0.
    // RoPE at pos=0 should be identity.
    let head_dim = 4;
    let seq_len = 2;
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let out = rope_reference(&x, seq_len, head_dim, 10000.0);

    // Position 0: cos(0)=1, sin(0)=0 -> identity
    assert!((out[0] - 1.0).abs() < 1e-6, "pos=0, dim=0");
    assert!((out[1] - 2.0).abs() < 1e-6, "pos=0, dim=1");
    assert!((out[2] - 3.0).abs() < 1e-6, "pos=0, dim=2");
    assert!((out[3] - 4.0).abs() < 1e-6, "pos=0, dim=3");
}

// ---- Reference correctness: zero positions ----

#[test]
fn test_rope_reference_zero_position() {
    // At position 0, theta = 0 for all dimensions, so rotation is identity.
    let head_dim = 8;
    let seq_len = 1;
    let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let out = rope_reference(&x, seq_len, head_dim, 10000.0);

    for i in 0..head_dim {
        assert!(
            (out[i] - x[i]).abs() < 1e-6,
            "position 0 should be identity: dim={i}, expected={}, got={}",
            x[i],
            out[i]
        );
    }
}

// ---- Reference correctness: norm preservation ----

#[test]
fn test_rope_reference_norm_preservation() {
    // RoPE is a rotation, so it preserves the L2 norm of each dimension pair.
    let head_dim = 8;
    let seq_len = 4;
    let batch_heads = 2;
    let total = batch_heads * seq_len * head_dim;
    let x: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let out = rope_reference(&x, seq_len, head_dim, 10000.0);

    let half_dim = head_dim / 2;
    for bh in 0..batch_heads {
        for pos in 0..seq_len {
            for i in 0..half_dim {
                let base_idx = bh * seq_len * head_dim + pos * head_dim;
                let idx0 = base_idx + 2 * i;
                let idx1 = base_idx + 2 * i + 1;

                let norm_in = x[idx0].hypot(x[idx1]);
                let norm_out = out[idx0].hypot(out[idx1]);
                assert!(
                    (norm_in - norm_out).abs() < 1e-5,
                    "norm not preserved: bh={bh}, pos={pos}, pair={i}, in={norm_in}, out={norm_out}"
                );
            }
        }
    }
}

// ---- Various parameter sizes produce valid SPIR-V ----

#[test]
fn test_rope_spirv_various_sizes() {
    let configs: &[(u32, u32)] = &[(1, 2), (16, 32), (128, 64), (512, 128), (2048, 256)];
    for &(seq, hd) in configs {
        let spirv = generate_rope_spirv(seq, hd);
        let label = format!("rope({seq},{hd})");
        assert_valid_header(&spirv, &label);
        let name =
            find_entry_point_name(&spirv).unwrap_or_else(|| panic!("{label}: no entry point"));
        assert_eq!(name, "main", "{label}: wrong entry point");
    }
}

// ---- Standard and NeoX produce different SPIR-V ----

#[test]
fn test_rope_standard_vs_neox_differ() {
    let standard = generate_rope_spirv(128, 64);
    let neox = generate_rope_neox_spirv(128, 64);
    // Both should be valid
    assert_valid_header(&standard, "standard");
    assert_valid_header(&neox, "neox");
    // But they should differ (different index calculation)
    assert_ne!(
        standard, neox,
        "standard and NeoX RoPE should produce different SPIR-V"
    );
}

// ---- Structural: has trig ext instructions ----

#[test]
fn test_rope_spirv_has_trig_operations() {
    let spirv = generate_rope_spirv(128, 64);
    // Must have ExtInst for sin, cos, exp, log (GLSL.std.450)
    let ext_inst_count = count_opcode(&spirv, OP_EXT_INST);
    // At minimum: log(base), exp(exponent), cos(theta), sin(theta) = 4
    assert!(
        ext_inst_count >= 4,
        "rope: must have at least 4 ExtInst (sin, cos, exp, log), got {ext_inst_count}"
    );
}

// ---- Has float multiply for rotation ----

#[test]
fn test_rope_spirv_has_float_arithmetic() {
    let spirv = generate_rope_spirv(128, 64);
    assert!(
        has_opcode(&spirv, OP_FMUL),
        "rope: must have FMul for rotation"
    );
    assert!(
        has_opcode(&spirv, OP_FSUB),
        "rope: must have FSub for rotation"
    );
    assert!(
        has_opcode(&spirv, OP_FADD),
        "rope: must have FAdd for rotation"
    );
    assert!(
        has_opcode(&spirv, OP_FDIV),
        "rope: must have FDiv for freq calculation"
    );
}

// ---- Workgroup size constant value ----

#[test]
fn test_rope_workgroup_size_constant() {
    assert_eq!(ROPE_WORKGROUP_SIZE, 64);
}

// ---- Deterministic output ----

#[test]
fn test_rope_spirv_deterministic() {
    let a = generate_rope_spirv(128, 64);
    let b = generate_rope_spirv(128, 64);
    assert_eq!(a, b, "same params must produce identical SPIR-V");
}

// ---- Has ConvertUToF for position to float conversion ----

#[test]
fn test_rope_spirv_has_uint_to_float_conversion() {
    let spirv = generate_rope_spirv(128, 64);
    assert!(
        has_opcode(&spirv, OP_CONVERT_U_TO_F),
        "rope: must have OpConvertUToF for converting position/dim to float"
    );
}
