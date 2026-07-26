// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the softmax SPIR-V kernel with separate I/O buffers.
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Separate input/output buffer layout (2 StorageBuffer bindings)
//! - Reference softmax correctness (sum-to-one, non-negativity, monotonicity)
//! - Numerical stability with large values
//! - Edge cases (single element, single row, wide rows beyond workgroup size)

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes for structural assertions.
const TEST_OP_CAPABILITY: u16 = 17;
const TEST_OP_MEMORY_MODEL: u16 = 14;
const TEST_OP_FUNCTION: u16 = 54;
const TEST_OP_FUNCTION_END: u16 = 56;
const TEST_OP_LABEL: u16 = 248;
const TEST_OP_RETURN: u16 = 253;
const TEST_OP_LOOP_MERGE: u16 = 246;
const TEST_OP_PHI: u16 = 245;
const TEST_OP_FADD: u16 = 129;
const TEST_OP_FSUB: u16 = 131;
const TEST_OP_FDIV: u16 = 136;
const TEST_OP_EXT_INST: u16 = 12;
const TEST_OP_CONTROL_BARRIER: u16 = 224;
const TEST_OP_VARIABLE: u16 = 59;
const TEST_OP_DECORATE: u16 = 71;
const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const TEST_STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const TEST_STORAGE_CLASS_WORKGROUP: u32 = 4;
const TEST_DECORATION_BINDING: u32 = 33;
const TEST_DECORATION_NON_WRITABLE: u32 = 24;

// ---- Helpers ----

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V byte length must be multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_valid_header(words: &[u32], label: &str) {
    assert!(words.len() >= 5, "{label}: module too short");
    assert_eq!(
        words[0], SPIRV_MAGIC,
        "{label}: wrong magic (expected 0x07230203)"
    );
    assert_eq!(
        words[1], TEST_SPIRV_VERSION_1_0,
        "{label}: wrong version (expected 1.0)"
    );
    assert_eq!(
        words[2], TEST_GENERATOR_MAGIC,
        "{label}: wrong generator magic"
    );
    assert!(words[3] > 0, "{label}: bound must be > 0");
    assert_eq!(words[4], 0, "{label}: schema must be 0");
}

fn has_opcode(words: &[u32], target_opcode: u16) -> bool {
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == target_opcode {
            return true;
        }
        pos += word_count;
    }
    false
}

fn count_opcode(words: &[u32], target_opcode: u16) -> usize {
    let mut pos = 5;
    let mut count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == target_opcode {
            count += 1;
        }
        pos += word_count;
    }
    count
}

fn find_instructions(words: &[u32], target_opcode: u16) -> Vec<Vec<u32>> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == target_opcode {
            results.push(words[pos..pos + word_count].to_vec());
        }
        pos += word_count;
    }
    results
}

// ====================================================================
// SPIR-V structural validity tests
// ====================================================================

#[test]
fn test_softmax_spirv_valid_header_32x128() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "softmax_32x128");
}

#[test]
fn test_softmax_spirv_valid_header_1x16() {
    let bytes = generate_softmax_separate_io_spirv(1, 16);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "softmax_1x16");
}

#[test]
fn test_softmax_spirv_valid_header_64x4096() {
    let bytes = generate_softmax_separate_io_spirv(64, 4096);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "softmax_64x4096");
}

#[test]
fn test_softmax_spirv_valid_header_non_power_of_2() {
    let bytes = generate_softmax_separate_io_spirv(7, 33);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "softmax_7x33");
}

#[test]
fn test_softmax_spirv_entry_point_is_main() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_softmax_spirv_workgroup_size() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [SOFTMAX_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_softmax_spirv_has_capability() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "must have OpCapability"
    );
}

#[test]
fn test_softmax_spirv_has_memory_model() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "must have OpMemoryModel"
    );
}

#[test]
fn test_softmax_spirv_has_function_structure() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_softmax_spirv_has_ext_inst_for_exp_and_fmax() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "softmax must use GLSL.std.450 for Exp and FMax"
    );
}

#[test]
fn test_softmax_spirv_has_fsub_for_stability() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FSUB),
        "softmax must have OpFSub for (x - max)"
    );
}

#[test]
fn test_softmax_spirv_has_fdiv_for_normalization() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FDIV),
        "softmax must have OpFDiv for exp / sum"
    );
}

#[test]
fn test_softmax_spirv_has_fadd_for_sum() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "softmax must have OpFAdd for sum accumulation"
    );
}

#[test]
fn test_softmax_spirv_has_loops_and_phi() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_LOOP_MERGE),
        "softmax must have loops for strided element access"
    );
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "softmax must have OpPhi for loop variable evolution"
    );
}

#[test]
fn test_softmax_spirv_has_barriers() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let barrier_count = count_opcode(&words, TEST_OP_CONTROL_BARRIER);
    // At minimum: after max store, after max tree, after sum store, after sum tree, before phase 3
    assert!(
        barrier_count >= 4,
        "softmax must have at least 4 barriers for shared memory synchronization, found {barrier_count}"
    );
}

// ====================================================================
// Separate I/O buffer layout tests
// ====================================================================

#[test]
fn test_softmax_spirv_has_two_storage_buffer_variables() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 2,
        "softmax with separate I/O must have 2 StorageBuffer variables (input + output), got {sb_count}"
    );
}

#[test]
fn test_softmax_spirv_has_workgroup_variable() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let wg_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_WORKGROUP)
        .count();
    assert!(
        wg_count >= 1,
        "softmax must have at least 1 workgroup variable for shared memory, found {wg_count}"
    );
}

#[test]
fn test_softmax_spirv_binding_numbers() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    assert!(bindings.contains(&0), "must have binding 0 (input buffer)");
    assert!(bindings.contains(&1), "must have binding 1 (output buffer)");
}

#[test]
fn test_softmax_spirv_input_is_nonwritable() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let has_nonwritable = decorations
        .iter()
        .any(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE);
    assert!(
        has_nonwritable,
        "input buffer should be decorated with NonWritable"
    );
}

// ====================================================================
// Byte alignment and size tests
// ====================================================================

#[test]
fn test_softmax_spirv_byte_alignment() {
    for (r, c) in [(1, 1), (1, 16), (7, 33), (32, 128), (64, 4096)] {
        let bytes = generate_softmax_separate_io_spirv(r, c);
        assert_eq!(
            bytes.len() % 4,
            0,
            "softmax {r}x{c}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_softmax_spirv_reasonable_size() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    assert!(
        words.len() > 100,
        "softmax module too small ({} words)",
        words.len()
    );
    assert!(
        words.len() < 5000,
        "softmax module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_softmax_spirv_deterministic() {
    let bytes1 = generate_softmax_separate_io_spirv(32, 128);
    let bytes2 = generate_softmax_separate_io_spirv(32, 128);
    assert_eq!(
        bytes1, bytes2,
        "SPIR-V output must be deterministic across calls"
    );
}

#[test]
fn test_softmax_spirv_various_dimensions() {
    for (r, c) in [
        (1, 1),
        (1, 16),
        (4, 64),
        (7, 33),
        (32, 128),
        (16, 256),
        (8, 4096),
    ] {
        let bytes = generate_softmax_separate_io_spirv(r, c);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("softmax_{r}x{c}"));
    }
}

#[test]
fn test_softmax_spirv_word_counts_consistent() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let mut pos = 5;
    let mut instruction_count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        assert!(
            word_count > 0,
            "instruction at pos {pos} has word_count 0 (opcode {opcode})"
        );
        assert!(
            pos + word_count <= words.len(),
            "instruction at pos {pos} (opcode {opcode}, wc {word_count}) exceeds module length {}",
            words.len()
        );
        pos += word_count;
        instruction_count += 1;
    }
    assert_eq!(
        pos,
        words.len(),
        "instructions did not consume exactly the full module"
    );
    assert!(
        instruction_count > 20,
        "expected at least 20 instructions for softmax, got {instruction_count}"
    );
}

// ====================================================================
// Reference softmax correctness tests (CPU implementation)
// ====================================================================

#[test]
fn test_reference_softmax_basic_1d() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = reference_softmax(&input, 1, 4);

    // All values must be positive.
    for (i, &v) in output.iter().enumerate() {
        assert!(v > 0.0, "softmax output[{i}] = {v} must be > 0");
    }

    // Sum must be 1.0.
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax sum must be 1.0, got {sum}"
    );

    // Monotonicity: larger input => larger output.
    for i in 1..output.len() {
        assert!(
            output[i] > output[i - 1],
            "softmax must be monotonic: output[{i}]={} <= output[{}]={}",
            output[i],
            i - 1,
            output[i - 1]
        );
    }
}

#[test]
fn test_reference_softmax_known_values() {
    // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
    let input = vec![0.0, 0.0, 0.0];
    let output = reference_softmax(&input, 1, 3);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 1.0 / 3.0).abs() < 1e-6,
            "softmax of equal values: output[{i}] = {v}, expected ~0.333"
        );
    }
}

#[test]
fn test_reference_softmax_2d_batch() {
    // Two rows processed independently.
    let input = vec![
        1.0, 2.0, 3.0, // row 0
        10.0, 20.0, 30.0, // row 1
    ];
    let output = reference_softmax(&input, 2, 3);

    // Each row sums to 1.0.
    let sum0: f32 = output[0..3].iter().sum();
    let sum1: f32 = output[3..6].iter().sum();
    assert!(
        (sum0 - 1.0).abs() < 1e-6,
        "row 0 sum must be 1.0, got {sum0}"
    );
    assert!(
        (sum1 - 1.0).abs() < 1e-6,
        "row 1 sum must be 1.0, got {sum1}"
    );

    // Row 1 should have a more peaked distribution (larger spread).
    // The max element's softmax probability should be higher for row 1.
    let max_prob_0 = output[2]; // softmax of 3.0
    let max_prob_1 = output[5]; // softmax of 30.0
    assert!(
        max_prob_1 > max_prob_0,
        "row with larger spread should have higher peak: {max_prob_1} > {max_prob_0}"
    );
}

#[test]
fn test_reference_softmax_numerical_stability_large_values() {
    // Large values that would overflow exp() without the max subtraction trick.
    let input = vec![1000.0, 1001.0, 1002.0];
    let output = reference_softmax(&input, 1, 3);

    // Must not contain NaN or Inf.
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "softmax output[{i}] = {v} must be finite for large inputs"
        );
        assert!(v > 0.0, "softmax output[{i}] = {v} must be > 0");
    }

    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum must be 1.0 for large inputs, got {sum}"
    );

    // The relative ordering should be preserved.
    assert!(output[2] > output[1]);
    assert!(output[1] > output[0]);
}

#[test]
fn test_reference_softmax_numerical_stability_negative_large() {
    // Very negative values.
    let input = vec![-1000.0, -999.0, -998.0];
    let output = reference_softmax(&input, 1, 3);

    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "softmax output[{i}] = {v} must be finite for large negative inputs"
        );
        assert!(v > 0.0, "softmax output[{i}] = {v} must be > 0");
    }

    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum must be 1.0 for large negative inputs, got {sum}"
    );
}

#[test]
fn test_reference_softmax_single_element() {
    let input = vec![42.0];
    let output = reference_softmax(&input, 1, 1);
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "softmax of single element must be 1.0, got {}",
        output[0]
    );
}

#[test]
fn test_reference_softmax_sum_to_one_property() {
    // Test with various input patterns.
    let test_cases: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        vec![100.0, 100.0, 100.0],
        vec![0.0; 10],
        (0..256).map(|i| i as f32 * 0.01).collect(),
    ];

    for (idx, input) in test_cases.iter().enumerate() {
        let cols = input.len();
        let output = reference_softmax(input, 1, cols);
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "test case {idx}: softmax sum must be 1.0, got {sum}"
        );
        for (j, &v) in output.iter().enumerate() {
            assert!(
                v >= 0.0,
                "test case {idx}: softmax output[{j}] = {v} must be >= 0"
            );
        }
    }
}

#[test]
fn test_reference_softmax_comparison_with_manual() {
    // Manually computed: softmax([1, 2]) = [e^1/(e^1+e^2), e^2/(e^1+e^2)]
    let e1 = 1.0f32.exp();
    let e2 = 2.0f32.exp();
    let denom = e1 + e2;
    let expected = [e1 / denom, e2 / denom];

    let input = vec![1.0, 2.0];
    let output = reference_softmax(&input, 1, 2);
    for i in 0..2 {
        assert!(
            (output[i] - expected[i]).abs() < 1e-6,
            "output[{i}] = {}, expected {}",
            output[i],
            expected[i]
        );
    }
}

#[test]
fn test_reference_softmax_wide_row_beyond_workgroup() {
    // cols > SOFTMAX_WORKGROUP_SIZE (256), ensuring strided access works.
    let cols = 1024;
    let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let output = reference_softmax(&input, 1, cols);

    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "wide row softmax sum must be 1.0, got {sum}"
    );

    // All non-negative.
    for (j, &v) in output.iter().enumerate() {
        assert!(v >= 0.0, "wide row output[{j}] = {v} must be >= 0");
    }

    // Monotonicity (inputs are strictly increasing).
    for j in 1..cols {
        assert!(
            output[j] >= output[j - 1],
            "wide row: output[{j}]={} < output[{}]={} (inputs are increasing)",
            output[j],
            j - 1,
            output[j - 1]
        );
    }
}

#[test]
fn test_reference_softmax_multi_row_independence() {
    // Each row should be independent.
    let input_single = vec![1.0, 2.0, 3.0];
    let single_output = reference_softmax(&input_single, 1, 3);

    let input_multi = vec![
        1.0, 2.0, 3.0, // row 0 (same as single)
        100.0, 200.0, 300.0, // row 1 (different)
    ];
    let multi_output = reference_softmax(&input_multi, 2, 3);

    // Row 0 of multi must match single-row result.
    for i in 0..3 {
        assert!(
            (multi_output[i] - single_output[i]).abs() < 1e-6,
            "row 0 of multi-row must match single-row: multi[{i}]={}, single[{i}]={}",
            multi_output[i],
            single_output[i]
        );
    }
}

#[test]
fn test_reference_softmax_invariant_to_constant_shift() {
    // softmax(x + c) == softmax(x) for any constant c.
    let input1 = vec![1.0, 2.0, 3.0, 4.0];
    let input2: Vec<f32> = input1.iter().map(|&x| x + 1000.0).collect();

    let output1 = reference_softmax(&input1, 1, 4);
    let output2 = reference_softmax(&input2, 1, 4);

    for i in 0..4 {
        assert!(
            (output1[i] - output2[i]).abs() < 1e-5,
            "softmax must be shift-invariant: output1[{i}]={}, output2[{i}]={}",
            output1[i],
            output2[i]
        );
    }
}

// ====================================================================
// Cross-validation: GLSL vs SPIR-V binary structure
// ====================================================================

#[test]
fn test_softmax_spirv_loop_count_matches_three_phase_design() {
    // Softmax has 3 phases + 2 tree reductions = at least 5 loop constructs.
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    assert!(
        loop_count >= 5,
        "softmax should have at least 5 loops (3 phases + 2 tree reductions), found {loop_count}"
    );
}

#[test]
fn test_softmax_spirv_ext_inst_count() {
    // Must have GLSL.std.450 calls for: FMax (in phase 1 loop + tree), Exp (in phase 2 + phase 3).
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let ext_count = count_opcode(&words, TEST_OP_EXT_INST);
    // At minimum: 1 FMax in loop, 1 FMax in tree, 1 Exp in phase 2, 1 Exp in phase 3 = 4.
    assert!(
        ext_count >= 4,
        "softmax must have at least 4 ExtInst calls (FMax + Exp), found {ext_count}"
    );
}
