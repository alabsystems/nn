// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary validation tests for the nn-vulkan spirv_binary module.
//!
//! Validates structural correctness of generated SPIR-V binaries at the
//! word level: header fields, opcode presence, decoration values, entry
//! point naming, workgroup sizes, and push constant layout differences
//! across operations. No live Vulkan GPU required.

use nn_vulkan::spirv_binary::{
    emit_add_spirv, emit_mul_spirv, emit_relu_spirv, emit_scalar_mul_spirv, emit_transpose_spirv,
    find_entry_point_name, find_workgroup_size, BINARY_WORKGROUP_SIZE,
};
use nn_vulkan::spirv_emit::SPIRV_MAGIC;

// SPIR-V constants for validation (mirrored from the spec / spirv_binary.rs).
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_VARIABLE: u16 = 59;

// Capability values.
const CAPABILITY_SHADER: u32 = 1;

// Memory model values.
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Decoration constants.
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_OFFSET: u32 = 35;

// Storage class constants.
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;

// Execution mode.
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

/// Helper: iterate SPIR-V instructions from the first post-header word.
/// Calls `f(pos, word_count, opcode, &spirv)` for each instruction.
fn for_each_instruction(spirv: &[u32], mut f: impl FnMut(usize, usize, u16, &[u32])) {
    let mut pos = 5; // skip 5-word header
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }
        f(pos, word_count, opcode, spirv);
        pos += word_count;
    }
}

/// Helper: check if a specific opcode is present.
fn has_opcode(spirv: &[u32], target: u16) -> bool {
    let mut found = false;
    for_each_instruction(spirv, |_, _, opcode, _| {
        if opcode == target {
            found = true;
        }
    });
    found
}

/// Helper: collect all instances of an opcode.
fn find_instructions(spirv: &[u32], target: u16) -> Vec<Vec<u32>> {
    let mut results = Vec::new();
    for_each_instruction(spirv, |pos, wc, opcode, words| {
        if opcode == target {
            results.push(words[pos..pos + wc].to_vec());
        }
    });
    results
}

/// Helper: collect all OpDecorate instructions.
fn find_decorations(spirv: &[u32]) -> Vec<Vec<u32>> {
    find_instructions(spirv, OP_DECORATE)
}

/// Helper: collect all OpVariable instructions.
fn find_variables(spirv: &[u32]) -> Vec<Vec<u32>> {
    find_instructions(spirv, OP_VARIABLE)
}

/// All five emitters for parametric tests.
fn all_spirv_modules() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ]
}

// ============================================================================
// 1. Magic number validation
// ============================================================================

#[test]
fn test_magic_number_is_spirv() {
    for (name, spirv) in all_spirv_modules() {
        assert!(
            spirv.len() >= 5,
            "{name}: SPIR-V module must have at least 5 header words"
        );
        assert_eq!(
            spirv[0], 0x07230203,
            "{name}: first word must be SPIR-V magic 0x07230203"
        );
        assert_eq!(
            spirv[0], SPIRV_MAGIC,
            "{name}: magic must match SPIRV_MAGIC constant"
        );
    }
}

// ============================================================================
// 2. Version field is SPIR-V 1.0
// ============================================================================

#[test]
fn test_version_is_spirv_1_0() {
    for (name, spirv) in all_spirv_modules() {
        assert_eq!(
            spirv[1], SPIRV_VERSION_1_0,
            "{name}: version word must be SPIR-V 1.0 (0x00010000), got {:#010x}",
            spirv[1]
        );
    }
}

// ============================================================================
// 3. Bound field is reasonable (< 1000)
// ============================================================================

#[test]
fn test_bound_field_is_reasonable() {
    for (name, spirv) in all_spirv_modules() {
        let bound = spirv[3];
        assert!(
            bound > 0,
            "{name}: bound (word[3]) must be > 0, got {bound}"
        );
        assert!(
            bound < 1000,
            "{name}: bound (word[3]) must be < 1000, got {bound}"
        );
    }
}

#[test]
fn test_schema_is_zero() {
    for (name, spirv) in all_spirv_modules() {
        assert_eq!(spirv[4], 0, "{name}: schema (word[4]) must be 0");
    }
}

// ============================================================================
// 4. OpEntryPoint present with "main" name
// ============================================================================

#[test]
fn test_entry_point_name_is_main() {
    for (name, spirv) in all_spirv_modules() {
        let ep_name = find_entry_point_name(&spirv)
            .unwrap_or_else(|| panic!("{name}: OpEntryPoint not found in module"));
        assert_eq!(
            ep_name, "main",
            "{name}: entry point name must be \"main\", got \"{ep_name}\""
        );
    }
}

#[test]
fn test_entry_point_opcode_present() {
    for (name, spirv) in all_spirv_modules() {
        assert!(
            has_opcode(&spirv, OP_ENTRY_POINT),
            "{name}: module must contain OpEntryPoint (opcode 15)"
        );
    }
}

#[test]
fn test_entry_point_execution_model_is_glcompute() {
    for (name, spirv) in all_spirv_modules() {
        let entries = find_instructions(&spirv, OP_ENTRY_POINT);
        assert!(
            !entries.is_empty(),
            "{name}: must have at least one OpEntryPoint"
        );
        for entry in &entries {
            // OpEntryPoint layout: [wc|op, execution_model, func_id, name...]
            // execution_model is at index 1
            assert_eq!(
                entry[1],
                5, // GLCompute = 5
                "{name}: OpEntryPoint execution model must be GLCompute (5), got {}",
                entry[1]
            );
        }
    }
}

// ============================================================================
// 5. OpDecorate / OpExecutionMode for workgroup size (256)
// ============================================================================

#[test]
fn test_workgroup_size_is_256() {
    for (name, spirv) in all_spirv_modules() {
        let wg = find_workgroup_size(&spirv)
            .unwrap_or_else(|| panic!("{name}: OpExecutionMode LocalSize not found"));
        assert_eq!(
            wg,
            [BINARY_WORKGROUP_SIZE, 1, 1],
            "{name}: workgroup size must be [{BINARY_WORKGROUP_SIZE}, 1, 1], got {wg:?}"
        );
    }
}

#[test]
fn test_execution_mode_present() {
    for (name, spirv) in all_spirv_modules() {
        let modes = find_instructions(&spirv, OP_EXECUTION_MODE);
        assert!(
            !modes.is_empty(),
            "{name}: must have at least one OpExecutionMode"
        );
        // Find the LocalSize mode specifically.
        let has_local_size = modes
            .iter()
            .any(|inst| inst.len() >= 3 && inst[2] == EXECUTION_MODE_LOCAL_SIZE);
        assert!(
            has_local_size,
            "{name}: must have OpExecutionMode LocalSize"
        );
    }
}

// ============================================================================
// 6. Storage buffer decorations present
// ============================================================================

#[test]
fn test_storage_buffer_binding_decorations_present() {
    for (name, spirv) in all_spirv_modules() {
        let decorations = find_decorations(&spirv);

        // Check that at least one Binding decoration exists.
        let has_binding = decorations
            .iter()
            .any(|d| d.len() >= 3 && d[2] == DECORATION_BINDING);
        assert!(
            has_binding,
            "{name}: must have at least one OpDecorate with Binding decoration"
        );

        // Check that at least one DescriptorSet decoration exists.
        let has_desc_set = decorations
            .iter()
            .any(|d| d.len() >= 3 && d[2] == DECORATION_DESCRIPTOR_SET);
        assert!(
            has_desc_set,
            "{name}: must have at least one OpDecorate with DescriptorSet decoration"
        );
    }
}

#[test]
fn test_block_decoration_present() {
    for (name, spirv) in all_spirv_modules() {
        let decorations = find_decorations(&spirv);
        let has_block = decorations
            .iter()
            .any(|d| d.len() >= 3 && d[2] == DECORATION_BLOCK);
        assert!(
            has_block,
            "{name}: must have at least one OpDecorate Block (for buffer structs)"
        );
    }
}

#[test]
fn test_member_offset_decorations_present() {
    for (name, spirv) in all_spirv_modules() {
        let member_decorations = find_instructions(&spirv, OP_MEMBER_DECORATE);
        let has_offset = member_decorations
            .iter()
            .any(|d| d.len() >= 4 && d[3] == DECORATION_OFFSET);
        assert!(
            has_offset,
            "{name}: must have at least one OpMemberDecorate Offset (for struct layout)"
        );
    }
}

#[test]
fn test_storage_buffer_variables_present() {
    for (name, spirv) in all_spirv_modules() {
        let variables = find_variables(&spirv);
        // OpVariable layout: [wc|op, result_type, result_id, storage_class]
        let sb_count = variables
            .iter()
            .filter(|v| v.len() >= 4 && v[3] == STORAGE_CLASS_STORAGE_BUFFER)
            .count();
        assert!(
            sb_count >= 2,
            "{name}: must have at least 2 StorageBuffer variables (in + out), got {sb_count}"
        );
    }
}

#[test]
fn test_push_constant_variable_present() {
    for (name, spirv) in all_spirv_modules() {
        let variables = find_variables(&spirv);
        let pc_count = variables
            .iter()
            .filter(|v| v.len() >= 4 && v[3] == STORAGE_CLASS_PUSH_CONSTANT)
            .count();
        assert_eq!(
            pc_count, 1,
            "{name}: must have exactly 1 PushConstant variable, got {pc_count}"
        );
    }
}

// ============================================================================
// 7. Each op produces valid SPIR-V (individual tests)
// ============================================================================

#[test]
fn test_add_produces_valid_spirv() {
    let spirv = emit_add_spirv().unwrap();
    assert_eq!(spirv[0], SPIRV_MAGIC);
    assert_eq!(spirv[1], SPIRV_VERSION_1_0);
    assert!(spirv[3] < 1000);
    assert_eq!(find_entry_point_name(&spirv).as_deref(), Some("main"));
    assert_eq!(find_workgroup_size(&spirv), Some([256, 1, 1]));
}

#[test]
fn test_mul_produces_valid_spirv() {
    let spirv = emit_mul_spirv().unwrap();
    assert_eq!(spirv[0], SPIRV_MAGIC);
    assert_eq!(spirv[1], SPIRV_VERSION_1_0);
    assert!(spirv[3] < 1000);
    assert_eq!(find_entry_point_name(&spirv).as_deref(), Some("main"));
    assert_eq!(find_workgroup_size(&spirv), Some([256, 1, 1]));
}

#[test]
fn test_relu_produces_valid_spirv() {
    let spirv = emit_relu_spirv().unwrap();
    assert_eq!(spirv[0], SPIRV_MAGIC);
    assert_eq!(spirv[1], SPIRV_VERSION_1_0);
    assert!(spirv[3] < 1000);
    assert_eq!(find_entry_point_name(&spirv).as_deref(), Some("main"));
    assert_eq!(find_workgroup_size(&spirv), Some([256, 1, 1]));
}

#[test]
fn test_scalar_mul_produces_valid_spirv() {
    let spirv = emit_scalar_mul_spirv().unwrap();
    assert_eq!(spirv[0], SPIRV_MAGIC);
    assert_eq!(spirv[1], SPIRV_VERSION_1_0);
    assert!(spirv[3] < 1000);
    assert_eq!(find_entry_point_name(&spirv).as_deref(), Some("main"));
    assert_eq!(find_workgroup_size(&spirv), Some([256, 1, 1]));
}

#[test]
fn test_transpose_produces_valid_spirv() {
    let spirv = emit_transpose_spirv().unwrap();
    assert_eq!(spirv[0], SPIRV_MAGIC);
    assert_eq!(spirv[1], SPIRV_VERSION_1_0);
    assert!(spirv[3] < 1000);
    assert_eq!(find_entry_point_name(&spirv).as_deref(), Some("main"));
    assert_eq!(find_workgroup_size(&spirv), Some([256, 1, 1]));
}

// ============================================================================
// 8. Different ops produce different push constant layouts
// ============================================================================

#[test]
fn test_push_constant_layouts_differ_by_op() {
    // Elementwise ops (add, mul, relu) use push constant struct: { uint total_elements }.
    // scalar_mul uses: { uint total_elements; float alpha }.
    // transpose uses: { uint total_elements; uint rows; uint cols }.
    //
    // We verify this by collecting the maximum member index seen in
    // OpMemberDecorate Offset decorations. The push constant struct for
    // scalar_mul decorates member 1 (alpha at offset 4), while add only
    // decorates member 0 for its push constant struct. Transpose decorates
    // member 2 (cols at offset 8).

    let add_spirv = emit_add_spirv().unwrap();
    let scalar_mul_spirv = emit_scalar_mul_spirv().unwrap();
    let transpose_spirv = emit_transpose_spirv().unwrap();

    // Collect the maximum member Offset value across all OpMemberDecorate
    // instructions. Ops with more push constant fields have higher max offsets.
    let max_member_offset = |spirv: &[u32]| -> u32 {
        find_instructions(spirv, OP_MEMBER_DECORATE)
            .iter()
            .filter(|d| d.len() >= 5 && d[3] == DECORATION_OFFSET)
            .map(|d| d[4]) // the offset value
            .max()
            .unwrap_or(0)
    };

    let add_max = max_member_offset(&add_spirv);
    let scalar_mul_max = max_member_offset(&scalar_mul_spirv);
    let transpose_max = max_member_offset(&transpose_spirv);

    // add's push constant struct has only total_elements at offset 0.
    // All buffer structs also have member 0 at offset 0. So max is 0.
    assert_eq!(
        add_max, 0,
        "add: max member offset should be 0 (only offset-0 members)"
    );

    // scalar_mul's push constant struct has alpha at offset 4.
    assert!(
        scalar_mul_max >= 4,
        "scalar_mul: max member offset should be >= 4 (alpha at offset 4), got {scalar_mul_max}"
    );

    // transpose's push constant struct has cols at offset 8.
    assert!(
        transpose_max >= 8,
        "transpose: max member offset should be >= 8 (cols at offset 8), got {transpose_max}"
    );

    // Transpose has a strictly larger max offset than scalar_mul.
    assert!(
        transpose_max > scalar_mul_max,
        "transpose max offset ({transpose_max}) should exceed scalar_mul ({scalar_mul_max})"
    );
}

#[test]
fn test_binary_ops_have_three_storage_buffers() {
    // add and mul: input A (binding 0) + input B (binding 1) + output C (binding 2).
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
    ] {
        let variables = find_variables(&spirv);
        let sb_count = variables
            .iter()
            .filter(|v| v.len() >= 4 && v[3] == STORAGE_CLASS_STORAGE_BUFFER)
            .count();
        assert_eq!(
            sb_count, 3,
            "{name}: binary elementwise op must have 3 StorageBuffer variables, got {sb_count}"
        );
    }
}

#[test]
fn test_unary_ops_have_two_storage_buffers() {
    // relu and scalar_mul: input (binding 0) + output (binding 1).
    for (name, spirv) in [
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
    ] {
        let variables = find_variables(&spirv);
        let sb_count = variables
            .iter()
            .filter(|v| v.len() >= 4 && v[3] == STORAGE_CLASS_STORAGE_BUFFER)
            .count();
        assert_eq!(
            sb_count, 2,
            "{name}: unary op must have 2 StorageBuffer variables, got {sb_count}"
        );
    }
}

#[test]
fn test_transpose_has_two_storage_buffers() {
    let spirv = emit_transpose_spirv().unwrap();
    let variables = find_variables(&spirv);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 2,
        "transpose must have 2 StorageBuffer variables (in + out), got {sb_count}"
    );
}

// ============================================================================
// 9. OpCapability (Shader) is correct
// ============================================================================

#[test]
fn test_capability_is_shader() {
    for (name, spirv) in all_spirv_modules() {
        let capabilities = find_instructions(&spirv, OP_CAPABILITY);
        assert!(
            !capabilities.is_empty(),
            "{name}: must have at least one OpCapability"
        );
        // OpCapability layout: [wc|op, capability]
        let has_shader = capabilities
            .iter()
            .any(|c| c.len() >= 2 && c[1] == CAPABILITY_SHADER);
        assert!(
            has_shader,
            "{name}: must have OpCapability Shader (value 1)"
        );
    }
}

// ============================================================================
// 10. OpMemoryModel (Logical, GLSL450) is correct
// ============================================================================

#[test]
fn test_memory_model_is_logical_glsl450() {
    for (name, spirv) in all_spirv_modules() {
        let models = find_instructions(&spirv, OP_MEMORY_MODEL);
        assert_eq!(
            models.len(),
            1,
            "{name}: must have exactly one OpMemoryModel"
        );
        let model = &models[0];
        // OpMemoryModel layout: [wc|op, addressing_model, memory_model]
        assert_eq!(
            model[1], ADDRESSING_LOGICAL,
            "{name}: addressing model must be Logical (0), got {}",
            model[1]
        );
        assert_eq!(
            model[2], MEMORY_MODEL_GLSL450,
            "{name}: memory model must be GLSL450 (1), got {}",
            model[2]
        );
    }
}

// ============================================================================
// Additional structural validation
// ============================================================================

#[test]
fn test_all_modules_have_descriptor_set_zero() {
    // All buffer bindings use descriptor set 0.
    for (name, spirv) in all_spirv_modules() {
        let decorations = find_decorations(&spirv);
        let desc_sets: Vec<u32> = decorations
            .iter()
            .filter(|d| d.len() >= 4 && d[2] == DECORATION_DESCRIPTOR_SET)
            .map(|d| d[3])
            .collect();
        assert!(
            !desc_sets.is_empty(),
            "{name}: must have DescriptorSet decorations"
        );
        for &ds in &desc_sets {
            assert_eq!(ds, 0, "{name}: all descriptor sets must be 0, got {ds}");
        }
    }
}

#[test]
fn test_binding_numbers_are_contiguous_from_zero() {
    for (name, spirv) in all_spirv_modules() {
        let decorations = find_decorations(&spirv);
        let mut bindings: Vec<u32> = decorations
            .iter()
            .filter(|d| d.len() >= 4 && d[2] == DECORATION_BINDING)
            .map(|d| d[3])
            .collect();
        bindings.sort_unstable();
        bindings.dedup();
        assert!(
            !bindings.is_empty(),
            "{name}: must have at least one Binding decoration"
        );
        assert_eq!(bindings[0], 0, "{name}: first binding must be 0");
        for (i, &b) in bindings.iter().enumerate() {
            assert_eq!(
                b, i as u32,
                "{name}: bindings must be contiguous from 0, expected {i} got {b}"
            );
        }
    }
}

#[test]
fn test_modules_are_deterministic() {
    // Calling the same emitter twice produces identical output.
    for (name, first) in all_spirv_modules() {
        let second = match name {
            "add" => emit_add_spirv().unwrap(),
            "mul" => emit_mul_spirv().unwrap(),
            "relu" => emit_relu_spirv().unwrap(),
            "scalar_mul" => emit_scalar_mul_spirv().unwrap(),
            "transpose" => emit_transpose_spirv().unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(
            first, second,
            "{name}: SPIR-V output must be deterministic across calls"
        );
    }
}

#[test]
fn test_module_word_counts_are_consistent() {
    // Walk every instruction and verify word counts don't exceed module bounds.
    for (name, spirv) in all_spirv_modules() {
        let mut pos = 5;
        let mut instruction_count = 0;
        while pos < spirv.len() {
            let word = spirv[pos];
            let word_count = (word >> 16) as usize;
            let opcode = word & 0xFFFF;
            assert!(
                word_count > 0,
                "{name}: instruction at pos {pos} has word_count 0 (opcode {opcode})"
            );
            assert!(
                pos + word_count <= spirv.len(),
                "{name}: instruction at pos {pos} (opcode {opcode}, wc {word_count}) \
                 exceeds module length {}",
                spirv.len()
            );
            pos += word_count;
            instruction_count += 1;
        }
        assert_eq!(
            pos,
            spirv.len(),
            "{name}: instructions did not consume exactly the full module"
        );
        assert!(
            instruction_count > 10,
            "{name}: expected at least 10 instructions, got {instruction_count}"
        );
    }
}
