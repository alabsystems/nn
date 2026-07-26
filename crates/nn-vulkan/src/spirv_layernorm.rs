// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for layer normalization compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for normalization operations:
//!
//! - [`generate_layernorm_spirv`]: Full LayerNorm (mean, variance, normalize, affine).
//! - [`generate_rmsnorm_spirv`]: RMSNorm variant (no mean subtraction, no bias).
//!
//! Both kernels are row-parallel: each workgroup handles one row of size
//! `normalized_shape`. Within the workgroup, threads cooperate on reduction
//! passes using shared memory and barriers.
//!
//! **LayerNorm** algorithm:
//! 1. Compute mean = sum(x) / N
//! 2. Compute variance = sum((x - mean)^2) / N
//! 3. Normalize: (x - mean) / sqrt(variance + eps)
//! 4. Affine: weight * normalized + bias
//!
//! **RMSNorm** algorithm:
//! 1. Compute rms = sqrt(sum(x^2) / N + eps)
//! 2. Normalize: (x / rms) * weight
//!
//! All shaders use SPIR-V 1.0 for maximum Vulkan compatibility, `StorageBuffer`
//! storage class with `std430` layout, and push constants for dimensions.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for layer normalization kernels (1D dispatch).
pub const LAYERNORM_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants (duplicated to keep module independent) ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// Opcodes.
const OP_CAPABILITY: u16 = 17;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_U_LESS_THAN: u16 = 176;
const OP_IADD: u16 = 128;
const OP_IMUL: u16 = 132;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_EXT_INST: u16 = 12;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-ins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_WORKGROUP: u32 = 4;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Execution model / mode.
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Capability.
const CAPABILITY_SHADER: u32 = 1;

// Memory model.
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Function control.
const FUNCTION_CONTROL_NONE: u32 = 0;

// Memory semantics for barriers.
const SCOPE_WORKGROUP: u32 = 2;
const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100;
const MEMORY_SEMANTICS_ACQUIRE_RELEASE: u32 = 0x8;

// GLSL.std.450 extended instruction set opcodes.
const GLSL_STD_450_SQRT: u32 = 31;

/// Encode a string as SPIR-V literal words (null-terminated, padded to 4-byte boundary).
fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let word_count = (bytes.len() + 1).div_ceil(4);
    let mut words = vec![0u32; word_count];
    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i / 4;
        let byte_idx = i % 4;
        words[word_idx] |= u32::from(b) << (byte_idx * 8);
    }
    words
}

/// Convert a `Vec<u32>` SPIR-V module to `Vec<u8>` (little-endian).
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// SPIR-V module builder (local to this module, mirrors spirv_reduction.rs).
struct SpirVBuilder {
    bound: u32,
    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    type_declarations: Vec<u32>,
    functions: Vec<u32>,
}

impl SpirVBuilder {
    fn new() -> Self {
        Self {
            bound: 1,
            capabilities: Vec::new(),
            extensions: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            annotations: Vec::new(),
            type_declarations: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.bound;
        self.bound += 1;
        id
    }

    fn capability(&mut self, cap: u32) {
        self.capabilities.push(op(2, OP_CAPABILITY));
        self.capabilities.push(cap);
    }

    fn ext_inst_import(&mut self, name: &str) -> u32 {
        let result = self.id();
        let name_words = encode_string(name);
        let wc = 2 + name_words.len() as u16;
        self.extensions.push(op(wc, OP_EXT_INST_IMPORT));
        self.extensions.push(result);
        self.extensions.extend_from_slice(&name_words);
        result
    }

    fn memory_model(&mut self, addressing: u32, model: u32) {
        self.memory_model.push(op(3, OP_MEMORY_MODEL));
        self.memory_model.push(addressing);
        self.memory_model.push(model);
    }

    fn entry_point_compute(&mut self, func_id: u32, name: &str, interface_ids: &[u32]) {
        let name_words = encode_string(name);
        let wc = 3 + name_words.len() as u16 + interface_ids.len() as u16;
        self.entry_points.push(op(wc, OP_ENTRY_POINT));
        self.entry_points.push(EXECUTION_MODEL_GL_COMPUTE);
        self.entry_points.push(func_id);
        self.entry_points.extend_from_slice(&name_words);
        self.entry_points.extend_from_slice(interface_ids);
    }

    fn execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.execution_modes.push(op(6, OP_EXECUTION_MODE));
        self.execution_modes.push(func_id);
        self.execution_modes.push(EXECUTION_MODE_LOCAL_SIZE);
        self.execution_modes.push(x);
        self.execution_modes.push(y);
        self.execution_modes.push(z);
    }

    fn decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let wc = 3 + operands.len() as u16;
        self.annotations.push(op(wc, OP_DECORATE));
        self.annotations.push(target);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    fn member_decorate(
        &mut self,
        struct_type: u32,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let wc = 4 + operands.len() as u16;
        self.annotations.push(op(wc, OP_MEMBER_DECORATE));
        self.annotations.push(struct_type);
        self.annotations.push(member);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    // ---- Type declarations ----

    fn type_void(&mut self) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(2, OP_TYPE_VOID));
        self.type_declarations.push(result);
        result
    }

    fn type_bool(&mut self) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(2, OP_TYPE_BOOL));
        self.type_declarations.push(result);
        result
    }

    fn type_int(&mut self, width: u32, signedness: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_INT));
        self.type_declarations.push(result);
        self.type_declarations.push(width);
        self.type_declarations.push(signedness);
        result
    }

    fn type_float(&mut self, width: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(3, OP_TYPE_FLOAT));
        self.type_declarations.push(result);
        self.type_declarations.push(width);
        result
    }

    fn type_vector(&mut self, component_type: u32, count: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_VECTOR));
        self.type_declarations.push(result);
        self.type_declarations.push(component_type);
        self.type_declarations.push(count);
        result
    }

    fn type_array(&mut self, element_type: u32, length_id: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_ARRAY));
        self.type_declarations.push(result);
        self.type_declarations.push(element_type);
        self.type_declarations.push(length_id);
        result
    }

    fn type_runtime_array(&mut self, element_type: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(3, OP_TYPE_RUNTIME_ARRAY));
        self.type_declarations.push(result);
        self.type_declarations.push(element_type);
        result
    }

    fn type_struct(&mut self, member_types: &[u32]) -> u32 {
        let result = self.id();
        let wc = 2 + member_types.len() as u16;
        self.type_declarations.push(op(wc, OP_TYPE_STRUCT));
        self.type_declarations.push(result);
        self.type_declarations.extend_from_slice(member_types);
        result
    }

    fn type_pointer(&mut self, storage_class: u32, pointee_type: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_POINTER));
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        self.type_declarations.push(pointee_type);
        result
    }

    fn type_function(&mut self, return_type: u32, param_types: &[u32]) -> u32 {
        let result = self.id();
        let wc = 3 + param_types.len() as u16;
        self.type_declarations.push(op(wc, OP_TYPE_FUNCTION));
        self.type_declarations.push(result);
        self.type_declarations.push(return_type);
        self.type_declarations.extend_from_slice(param_types);
        result
    }

    fn constant_u32(&mut self, type_id: u32, value: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value);
        result
    }

    fn constant_f32(&mut self, type_id: u32, value: f32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value.to_bits());
        result
    }

    fn variable_global(&mut self, ptr_type: u32, storage_class: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_VARIABLE));
        self.type_declarations.push(ptr_type);
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        result
    }

    // ---- Function body instructions ----

    fn func_begin(&mut self, result_type: u32, func_id: u32, control: u32, func_type: u32) {
        self.functions.push(op(5, OP_FUNCTION));
        self.functions.push(result_type);
        self.functions.push(func_id);
        self.functions.push(control);
        self.functions.push(func_type);
    }

    fn func_end(&mut self) {
        self.functions.push(op(1, OP_FUNCTION_END));
    }

    fn label(&mut self) -> u32 {
        let result = self.id();
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(result);
        result
    }

    fn label_with_id(&mut self, id: u32) {
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(id);
    }

    fn op_return(&mut self) {
        self.functions.push(op(1, OP_RETURN));
    }

    fn branch(&mut self, target_label: u32) {
        self.functions.push(op(2, OP_BRANCH));
        self.functions.push(target_label);
    }

    fn branch_conditional(&mut self, condition: u32, true_label: u32, false_label: u32) {
        self.functions.push(op(4, OP_BRANCH_CONDITIONAL));
        self.functions.push(condition);
        self.functions.push(true_label);
        self.functions.push(false_label);
    }

    fn selection_merge(&mut self, merge_label: u32) {
        self.functions.push(op(3, OP_SELECTION_MERGE));
        self.functions.push(merge_label);
        self.functions.push(0);
    }

    fn loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.functions.push(op(4, OP_LOOP_MERGE));
        self.functions.push(merge_label);
        self.functions.push(continue_label);
        self.functions.push(0);
    }

    fn phi(&mut self, result_type: u32, operands: &[(u32, u32)]) -> u32 {
        let result = self.id();
        let wc = 3 + (operands.len() as u16) * 2;
        self.functions.push(op(wc, OP_PHI));
        self.functions.push(result_type);
        self.functions.push(result);
        for &(val, parent) in operands {
            self.functions.push(val);
            self.functions.push(parent);
        }
        result
    }

    fn load(&mut self, result_type: u32, pointer: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_LOAD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(pointer);
        result
    }

    fn store(&mut self, pointer: u32, object: u32) {
        self.functions.push(op(3, OP_STORE));
        self.functions.push(pointer);
        self.functions.push(object);
    }

    fn access_chain(&mut self, result_type: u32, base: u32, indices: &[u32]) -> u32 {
        let result = self.id();
        let wc = 4 + indices.len() as u16;
        self.functions.push(op(wc, OP_ACCESS_CHAIN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(base);
        self.functions.extend_from_slice(indices);
        result
    }

    fn composite_extract(&mut self, result_type: u32, composite: u32, index: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_COMPOSITE_EXTRACT));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(composite);
        self.functions.push(index);
        result
    }

    fn fadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn fsub(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FSUB));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn fmul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn fdiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn convert_u_to_f(&mut self, result_type: u32, value: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_CONVERT_U_TO_F));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(value);
        result
    }

    fn u_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn iadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn imul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn shift_right_logical(&mut self, result_type: u32, base: u32, shift: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_SHIFT_RIGHT_LOGICAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(base);
        self.functions.push(shift);
        result
    }

    fn control_barrier(&mut self, execution: u32, memory: u32, semantics: u32) {
        self.functions.push(op(4, OP_CONTROL_BARRIER));
        self.functions.push(execution);
        self.functions.push(memory);
        self.functions.push(semantics);
    }

    fn ext_inst(
        &mut self,
        result_type: u32,
        ext_set: u32,
        instruction: u32,
        operands: &[u32],
    ) -> u32 {
        let result = self.id();
        let wc = 5 + operands.len() as u16;
        self.functions.push(op(wc, OP_EXT_INST));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(ext_set);
        self.functions.push(instruction);
        self.functions.extend_from_slice(operands);
        result
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(512);
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0);
        module.extend_from_slice(&self.capabilities);
        module.extend_from_slice(&self.extensions);
        module.extend_from_slice(&self.memory_model);
        module.extend_from_slice(&self.entry_points);
        module.extend_from_slice(&self.execution_modes);
        module.extend_from_slice(&self.annotations);
        module.extend_from_slice(&self.type_declarations);
        module.extend_from_slice(&self.functions);
        module
    }
}

/// Fixup a phi instruction to add an additional (value, parent) operand.
fn fixup_phi(functions: &mut Vec<u32>, phi_id: u32, value: u32, parent: u32) {
    let mut pos = 0;
    while pos < functions.len() {
        let word = functions[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if opcode == OP_PHI && pos + 2 < functions.len() && functions[pos + 2] == phi_id {
            let insert_pos = pos + word_count;
            functions.insert(insert_pos, parent);
            functions.insert(insert_pos, value);
            let new_wc = word_count + 2;
            functions[pos] = op(new_wc as u16, OP_PHI);
            return;
        }
        pos += word_count;
    }
}

/// Emit ISub (opcode 130) inline — not in the SpirVBuilder because reduction
/// modules use it only for row-index derivation.
fn emit_isub(b: &mut SpirVBuilder, result_type: u32, a: u32, subtrahend: u32) -> u32 {
    let result = b.id();
    const OP_ISUB: u16 = 130;
    b.functions.push(op(5, OP_ISUB));
    b.functions.push(result_type);
    b.functions.push(result);
    b.functions.push(a);
    b.functions.push(subtrahend);
    result
}

/// Helper: emit a tree reduction loop over shared memory using `fadd`.
///
/// Expects shared memory to already contain per-thread partial results.
/// After return, shared\[0\] holds the reduction result. A barrier is issued
/// after the loop so the caller can safely read shared\[0\].
fn emit_sum_tree_reduction(
    b: &mut SpirVBuilder,
    ty_uint: u32,
    ty_float: u32,
    ty_bool: u32,
    ptr_wg_float: u32,
    var_shared: u32,
    lid_x: u32,
    const_0u: u32,
    const_1u: u32,
    const_half_wg: u32,
    const_scope_wg: u32,
    const_mem_sem: u32,
    predecessor_label: u32,
) {
    let tree_header = b.id();
    let tree_body = b.id();
    let tree_continue = b.id();
    let tree_merge = b.id();

    b.branch(tree_header);
    b.label_with_id(tree_header);
    b.loop_merge(tree_merge, tree_continue);

    let phi_stride = b.phi(ty_uint, &[(const_half_wg, predecessor_label)]);
    let cmp_stride = b.u_less_than(ty_bool, const_0u, phi_stride);
    b.branch_conditional(cmp_stride, tree_body, tree_merge);

    b.label_with_id(tree_body);
    let cmp_lid = b.u_less_than(ty_bool, lid_x, phi_stride);
    let reduce_label = b.id();
    let skip_label = b.id();
    b.selection_merge(skip_label);
    b.branch_conditional(cmp_lid, reduce_label, skip_label);

    b.label_with_id(reduce_label);
    let ptr_a = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    let val_a = b.load(ty_float, ptr_a);
    let lid_plus_stride = b.iadd(ty_uint, lid_x, phi_stride);
    let ptr_b = b.access_chain(ptr_wg_float, var_shared, &[lid_plus_stride]);
    let val_b = b.load(ty_float, ptr_b);
    let reduced = b.fadd(ty_float, val_a, val_b);
    b.store(ptr_a, reduced);
    b.branch(skip_label);

    b.label_with_id(skip_label);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);
    b.branch(tree_continue);

    b.label_with_id(tree_continue);
    let stride_next = b.shift_right_logical(ty_uint, phi_stride, const_1u);
    b.branch(tree_header);

    fixup_phi(&mut b.functions, phi_stride, stride_next, tree_continue);

    b.label_with_id(tree_merge);
}

/// Generate a SPIR-V binary for LayerNorm.
///
/// Each workgroup handles one row of `normalized_shape` elements. The algorithm:
/// 1. Compute mean = sum(x) / N
/// 2. Compute variance = sum((x - mean)^2) / N
/// 3. Output = weight * (x - mean) / sqrt(variance + eps) + bias
///
/// # Arguments
///
/// * `normalized_shape` - Number of elements per row (compile-time hint; actual from push constants).
/// * `eps` - Epsilon for numerical stability (baked into the shader as a constant).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[total_elements\])
/// - Binding 1: Weight/gamma buffer (float\[normalized_shape\])
/// - Binding 2: Bias/beta buffer (float\[normalized_shape\])
/// - Binding 3: Output buffer (float\[total_elements\])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
/// - `uint normalized_shape` at offset 4
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_layernorm::generate_layernorm_spirv;
/// let spirv = generate_layernorm_spirv(768, 1e-5);
/// assert_eq!(spirv.len() % 4, 0); // 4-byte aligned
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_layernorm_spirv(normalized_shape: usize, eps: f32) -> Vec<u8> {
    let _ = normalized_shape; // hint; actual dim from push constants.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (binding 0).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Weight buffer struct (binding 1).
    let ty_struct_weight = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_weight, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_weight, 0, DECORATION_OFFSET, &[0]);

    // Bias buffer struct (binding 2).
    let ty_struct_bias = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_bias, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_bias, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct (binding 3).
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint total_elements; uint normalized_shape; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Shared memory: float[LAYERNORM_WORKGROUP_SIZE]
    let const_wg_size = b.constant_u32(ty_uint, LAYERNORM_WORKGROUP_SIZE);
    let ty_shared_arr = b.type_array(ty_float, const_wg_size);
    b.decorate(ty_shared_arr, DECORATION_ARRAY_STRIDE, &[4]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_weight = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_weight);
    let ptr_sb_bias = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_bias);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_wg_shared = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_shared_arr);
    let ptr_wg_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_f0 = b.constant_f32(ty_float, 0.0);
    let const_eps = b.constant_f32(ty_float, eps);
    let const_half_wg = b.constant_u32(ty_uint, LAYERNORM_WORKGROUP_SIZE / 2);
    let const_scope_wg = b.constant_u32(ty_uint, SCOPE_WORKGROUP);
    let const_mem_sem = b.constant_u32(
        ty_uint,
        MEMORY_SEMANTICS_WORKGROUP | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );
    let const_8u = b.constant_u32(ty_uint, 8); // log2(256)

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);

    let var_weight = b.variable_global(ptr_sb_weight, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_weight, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_weight, DECORATION_BINDING, &[1]);

    let var_bias = b.variable_global(ptr_sb_bias, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_bias, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_bias, DECORATION_BINDING, &[2]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[3]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    let var_lid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);

    let var_shared = b.variable_global(ptr_wg_shared, STORAGE_CLASS_WORKGROUP);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid, var_lid]);
    b.execution_mode_local_size(func_id, LAYERNORM_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let entry_label = b.label();

    // Load invocation IDs.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);
    let loaded_lid = b.load(ty_uvec3, var_lid);
    let lid_x = b.composite_extract(ty_uint, loaded_lid, 0);

    // Load push constants.
    let pc_total_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let _dim_total = b.load(ty_uint, pc_total_ptr);
    let pc_norm_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let dim_norm = b.load(ty_uint, pc_norm_ptr);

    // Row index: row = (gid_x - lid_x) >> 8 (since WG_SIZE = 256 = 2^8).
    let gid_minus_lid = emit_isub(&mut b, ty_uint, gid_x, lid_x);
    let row = b.shift_right_logical(ty_uint, gid_minus_lid, const_8u);

    // Base offset for this row: row_base = row * normalized_shape.
    let row_base = b.imul(ty_uint, row, dim_norm);

    // ================================================================
    // Phase 1: Compute mean = sum(x) / N
    // ================================================================
    // Serial accumulation: thread lid accumulates elements lid, lid+WG_SIZE, etc.
    let p1_loop_header = b.id();
    let p1_loop_body = b.id();
    let p1_loop_continue = b.id();
    let p1_loop_merge = b.id();

    b.branch(p1_loop_header);
    b.label_with_id(p1_loop_header);
    b.loop_merge(p1_loop_merge, p1_loop_continue);

    let phi_col1 = b.phi(ty_uint, &[(lid_x, entry_label)]);
    let phi_sum1 = b.phi(ty_float, &[(const_f0, entry_label)]);

    let cmp_col1 = b.u_less_than(ty_bool, phi_col1, dim_norm);
    b.branch_conditional(cmp_col1, p1_loop_body, p1_loop_merge);

    b.label_with_id(p1_loop_body);
    let elem_idx1 = b.iadd(ty_uint, row_base, phi_col1);
    let ptr_elem1 = b.access_chain(ptr_sb_float, var_input, &[const_0u, elem_idx1]);
    let val_elem1 = b.load(ty_float, ptr_elem1);
    let new_sum1 = b.fadd(ty_float, phi_sum1, val_elem1);
    let next_col1 = b.iadd(ty_uint, phi_col1, const_wg_size);
    b.branch(p1_loop_continue);

    b.label_with_id(p1_loop_continue);
    b.branch(p1_loop_header);

    fixup_phi(&mut b.functions, phi_col1, next_col1, p1_loop_continue);
    fixup_phi(&mut b.functions, phi_sum1, new_sum1, p1_loop_continue);

    b.label_with_id(p1_loop_merge);

    // Store partial sum to shared memory and reduce.
    let ptr_s_lid1 = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    b.store(ptr_s_lid1, phi_sum1);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // Tree reduction for sum -> mean.
    emit_sum_tree_reduction(
        &mut b,
        ty_uint,
        ty_float,
        ty_bool,
        ptr_wg_float,
        var_shared,
        lid_x,
        const_0u,
        const_1u,
        const_half_wg,
        const_scope_wg,
        const_mem_sem,
        p1_loop_merge,
    );
    // After emit_sum_tree_reduction, we are at the tree_merge label.

    // Load sum from shared[0] and compute mean.
    let ptr_s0_sum = b.access_chain(ptr_wg_float, var_shared, &[const_0u]);
    let total_sum = b.load(ty_float, ptr_s0_sum);
    let n_float = b.convert_u_to_f(ty_float, dim_norm);
    let mean = b.fdiv(ty_float, total_sum, n_float);

    // Barrier before reusing shared memory.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // ================================================================
    // Phase 2: Compute variance = sum((x - mean)^2) / N
    // ================================================================
    let phase2_predecessor = b.id(); // We need this label ID for phi references.
                                     // Actually, we're already inside a label from the tree merge. Record current position
                                     // by noting that emit_sum_tree_reduction ended with label_with_id(tree_merge).
                                     // We issued instructions after it (load, convert, fdiv, barrier) but we are still
                                     // in the same basic block. We need a label for the phi predecessor.
                                     // Since we are still in the tree_merge label, we can just branch to a new label
                                     // for the phase 2 loop.

    let p2_loop_header = b.id();
    let p2_loop_body = b.id();
    let p2_loop_continue = b.id();
    let p2_loop_merge = b.id();

    // We need the predecessor label for phi. The current block is the tree_merge
    // from the sum reduction. We can't easily get that label ID back from emit_sum_tree_reduction
    // without refactoring. Instead, create a new label to serve as predecessor.
    b.branch(phase2_predecessor);
    b.label_with_id(phase2_predecessor);

    b.branch(p2_loop_header);
    b.label_with_id(p2_loop_header);
    b.loop_merge(p2_loop_merge, p2_loop_continue);

    let phi_col2 = b.phi(ty_uint, &[(lid_x, phase2_predecessor)]);
    let phi_var_sum = b.phi(ty_float, &[(const_f0, phase2_predecessor)]);

    let cmp_col2 = b.u_less_than(ty_bool, phi_col2, dim_norm);
    b.branch_conditional(cmp_col2, p2_loop_body, p2_loop_merge);

    b.label_with_id(p2_loop_body);
    let elem_idx2 = b.iadd(ty_uint, row_base, phi_col2);
    let ptr_elem2 = b.access_chain(ptr_sb_float, var_input, &[const_0u, elem_idx2]);
    let val_elem2 = b.load(ty_float, ptr_elem2);
    let diff = b.fsub(ty_float, val_elem2, mean);
    let diff_sq = b.fmul(ty_float, diff, diff);
    let new_var_sum = b.fadd(ty_float, phi_var_sum, diff_sq);
    let next_col2 = b.iadd(ty_uint, phi_col2, const_wg_size);
    b.branch(p2_loop_continue);

    b.label_with_id(p2_loop_continue);
    b.branch(p2_loop_header);

    fixup_phi(&mut b.functions, phi_col2, next_col2, p2_loop_continue);
    fixup_phi(&mut b.functions, phi_var_sum, new_var_sum, p2_loop_continue);

    b.label_with_id(p2_loop_merge);

    // Store partial variance sum to shared and reduce.
    let ptr_s_lid2 = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    b.store(ptr_s_lid2, phi_var_sum);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    emit_sum_tree_reduction(
        &mut b,
        ty_uint,
        ty_float,
        ty_bool,
        ptr_wg_float,
        var_shared,
        lid_x,
        const_0u,
        const_1u,
        const_half_wg,
        const_scope_wg,
        const_mem_sem,
        p2_loop_merge,
    );

    // Load variance sum from shared[0] and compute variance.
    let ptr_s0_var = b.access_chain(ptr_wg_float, var_shared, &[const_0u]);
    let var_total = b.load(ty_float, ptr_s0_var);
    let variance = b.fdiv(ty_float, var_total, n_float);

    // inv_std = 1.0 / sqrt(variance + eps)
    let var_plus_eps = b.fadd(ty_float, variance, const_eps);
    let sqrt_val = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_SQRT, &[var_plus_eps]);
    let const_f1 = b.constant_f32(ty_float, 1.0);
    let inv_std = b.fdiv(ty_float, const_f1, sqrt_val);

    // Barrier before reusing shared memory.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // ================================================================
    // Phase 3: Normalize and apply affine: out = weight * (x - mean) * inv_std + bias
    // ================================================================
    let phase3_predecessor = b.id();
    b.branch(phase3_predecessor);
    b.label_with_id(phase3_predecessor);

    let p3_loop_header = b.id();
    let p3_loop_body = b.id();
    let p3_loop_continue = b.id();
    let p3_loop_merge = b.id();

    b.branch(p3_loop_header);
    b.label_with_id(p3_loop_header);
    b.loop_merge(p3_loop_merge, p3_loop_continue);

    let phi_col3 = b.phi(ty_uint, &[(lid_x, phase3_predecessor)]);
    let cmp_col3 = b.u_less_than(ty_bool, phi_col3, dim_norm);
    b.branch_conditional(cmp_col3, p3_loop_body, p3_loop_merge);

    b.label_with_id(p3_loop_body);
    // Load input[row_base + col].
    let elem_idx3 = b.iadd(ty_uint, row_base, phi_col3);
    let ptr_elem3 = b.access_chain(ptr_sb_float, var_input, &[const_0u, elem_idx3]);
    let val_elem3 = b.load(ty_float, ptr_elem3);

    // normalized = (x - mean) * inv_std
    let diff3 = b.fsub(ty_float, val_elem3, mean);
    let normalized = b.fmul(ty_float, diff3, inv_std);

    // Load weight[col] and bias[col].
    let ptr_w = b.access_chain(ptr_sb_float, var_weight, &[const_0u, phi_col3]);
    let w_val = b.load(ty_float, ptr_w);
    let ptr_b_bias = b.access_chain(ptr_sb_float, var_bias, &[const_0u, phi_col3]);
    let b_val = b.load(ty_float, ptr_b_bias);

    // out = weight * normalized + bias
    let scaled = b.fmul(ty_float, w_val, normalized);
    let result = b.fadd(ty_float, scaled, b_val);

    // Store to output[row_base + col].
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, elem_idx3]);
    b.store(ptr_out, result);

    let next_col3 = b.iadd(ty_uint, phi_col3, const_wg_size);
    b.branch(p3_loop_continue);

    b.label_with_id(p3_loop_continue);
    b.branch(p3_loop_header);

    fixup_phi(&mut b.functions, phi_col3, next_col3, p3_loop_continue);

    b.label_with_id(p3_loop_merge);

    // Return.
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

/// Generate a SPIR-V binary for RMSNorm (Root Mean Square Layer Normalization).
///
/// Each workgroup handles one row of `normalized_shape` elements. The algorithm:
/// 1. Compute rms = sqrt(sum(x^2) / N + eps)
/// 2. Output = weight * (x / rms)
///
/// RMSNorm does not subtract the mean and has no bias parameter, making it
/// faster than full LayerNorm while achieving similar results in practice.
///
/// # Arguments
///
/// * `normalized_shape` - Number of elements per row (compile-time hint; actual from push constants).
/// * `eps` - Epsilon for numerical stability (baked into the shader as a constant).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[total_elements\])
/// - Binding 1: Weight buffer (float\[normalized_shape\])
/// - Binding 2: Output buffer (float\[total_elements\])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
/// - `uint normalized_shape` at offset 4
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_layernorm::generate_rmsnorm_spirv;
/// let spirv = generate_rmsnorm_spirv(768, 1e-5);
/// assert_eq!(spirv.len() % 4, 0); // 4-byte aligned
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_rmsnorm_spirv(normalized_shape: usize, eps: f32) -> Vec<u8> {
    let _ = normalized_shape; // hint; actual dim from push constants.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (binding 0).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Weight buffer struct (binding 1).
    let ty_struct_weight = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_weight, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_weight, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct (binding 2).
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint total_elements; uint normalized_shape; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Shared memory: float[LAYERNORM_WORKGROUP_SIZE]
    let const_wg_size = b.constant_u32(ty_uint, LAYERNORM_WORKGROUP_SIZE);
    let ty_shared_arr = b.type_array(ty_float, const_wg_size);
    b.decorate(ty_shared_arr, DECORATION_ARRAY_STRIDE, &[4]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_weight = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_weight);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_wg_shared = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_shared_arr);
    let ptr_wg_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_f0 = b.constant_f32(ty_float, 0.0);
    let const_eps = b.constant_f32(ty_float, eps);
    let const_half_wg = b.constant_u32(ty_uint, LAYERNORM_WORKGROUP_SIZE / 2);
    let const_scope_wg = b.constant_u32(ty_uint, SCOPE_WORKGROUP);
    let const_mem_sem = b.constant_u32(
        ty_uint,
        MEMORY_SEMANTICS_WORKGROUP | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );
    let const_8u = b.constant_u32(ty_uint, 8); // log2(256)

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);

    let var_weight = b.variable_global(ptr_sb_weight, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_weight, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_weight, DECORATION_BINDING, &[1]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    let var_lid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);

    let var_shared = b.variable_global(ptr_wg_shared, STORAGE_CLASS_WORKGROUP);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid, var_lid]);
    b.execution_mode_local_size(func_id, LAYERNORM_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let entry_label = b.label();

    // Load invocation IDs.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);
    let loaded_lid = b.load(ty_uvec3, var_lid);
    let lid_x = b.composite_extract(ty_uint, loaded_lid, 0);

    // Load push constants.
    let pc_total_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let _dim_total = b.load(ty_uint, pc_total_ptr);
    let pc_norm_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let dim_norm = b.load(ty_uint, pc_norm_ptr);

    // Row index: row = (gid_x - lid_x) >> 8.
    let gid_minus_lid = emit_isub(&mut b, ty_uint, gid_x, lid_x);
    let row = b.shift_right_logical(ty_uint, gid_minus_lid, const_8u);

    // Base offset for this row.
    let row_base = b.imul(ty_uint, row, dim_norm);

    // ================================================================
    // Phase 1: Compute sum(x^2)
    // ================================================================
    let p1_loop_header = b.id();
    let p1_loop_body = b.id();
    let p1_loop_continue = b.id();
    let p1_loop_merge = b.id();

    b.branch(p1_loop_header);
    b.label_with_id(p1_loop_header);
    b.loop_merge(p1_loop_merge, p1_loop_continue);

    let phi_col1 = b.phi(ty_uint, &[(lid_x, entry_label)]);
    let phi_sq_sum = b.phi(ty_float, &[(const_f0, entry_label)]);

    let cmp_col1 = b.u_less_than(ty_bool, phi_col1, dim_norm);
    b.branch_conditional(cmp_col1, p1_loop_body, p1_loop_merge);

    b.label_with_id(p1_loop_body);
    let elem_idx1 = b.iadd(ty_uint, row_base, phi_col1);
    let ptr_elem1 = b.access_chain(ptr_sb_float, var_input, &[const_0u, elem_idx1]);
    let val_elem1 = b.load(ty_float, ptr_elem1);
    let val_sq = b.fmul(ty_float, val_elem1, val_elem1);
    let new_sq_sum = b.fadd(ty_float, phi_sq_sum, val_sq);
    let next_col1 = b.iadd(ty_uint, phi_col1, const_wg_size);
    b.branch(p1_loop_continue);

    b.label_with_id(p1_loop_continue);
    b.branch(p1_loop_header);

    fixup_phi(&mut b.functions, phi_col1, next_col1, p1_loop_continue);
    fixup_phi(&mut b.functions, phi_sq_sum, new_sq_sum, p1_loop_continue);

    b.label_with_id(p1_loop_merge);

    // Store partial sq sum to shared and reduce.
    let ptr_s_lid1 = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    b.store(ptr_s_lid1, phi_sq_sum);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    emit_sum_tree_reduction(
        &mut b,
        ty_uint,
        ty_float,
        ty_bool,
        ptr_wg_float,
        var_shared,
        lid_x,
        const_0u,
        const_1u,
        const_half_wg,
        const_scope_wg,
        const_mem_sem,
        p1_loop_merge,
    );

    // Load sq sum from shared[0] and compute rms.
    let ptr_s0 = b.access_chain(ptr_wg_float, var_shared, &[const_0u]);
    let sq_sum_total = b.load(ty_float, ptr_s0);
    let n_float = b.convert_u_to_f(ty_float, dim_norm);
    let mean_sq = b.fdiv(ty_float, sq_sum_total, n_float);
    let mean_sq_eps = b.fadd(ty_float, mean_sq, const_eps);
    let rms_val = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_SQRT, &[mean_sq_eps]);
    let const_f1 = b.constant_f32(ty_float, 1.0);
    let inv_rms = b.fdiv(ty_float, const_f1, rms_val);

    // Barrier before reusing shared memory.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // ================================================================
    // Phase 2: Normalize and apply weight: out = weight * (x * inv_rms)
    // ================================================================
    let phase2_predecessor = b.id();
    b.branch(phase2_predecessor);
    b.label_with_id(phase2_predecessor);

    let p2_loop_header = b.id();
    let p2_loop_body = b.id();
    let p2_loop_continue = b.id();
    let p2_loop_merge = b.id();

    b.branch(p2_loop_header);
    b.label_with_id(p2_loop_header);
    b.loop_merge(p2_loop_merge, p2_loop_continue);

    let phi_col2 = b.phi(ty_uint, &[(lid_x, phase2_predecessor)]);
    let cmp_col2 = b.u_less_than(ty_bool, phi_col2, dim_norm);
    b.branch_conditional(cmp_col2, p2_loop_body, p2_loop_merge);

    b.label_with_id(p2_loop_body);
    // Load input[row_base + col].
    let elem_idx2 = b.iadd(ty_uint, row_base, phi_col2);
    let ptr_elem2 = b.access_chain(ptr_sb_float, var_input, &[const_0u, elem_idx2]);
    let val_elem2 = b.load(ty_float, ptr_elem2);

    // normalized = x * inv_rms
    let normalized = b.fmul(ty_float, val_elem2, inv_rms);

    // Load weight[col].
    let ptr_w = b.access_chain(ptr_sb_float, var_weight, &[const_0u, phi_col2]);
    let w_val = b.load(ty_float, ptr_w);

    // out = weight * normalized
    let result = b.fmul(ty_float, w_val, normalized);

    // Store to output[row_base + col].
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, elem_idx2]);
    b.store(ptr_out, result);

    let next_col2 = b.iadd(ty_uint, phi_col2, const_wg_size);
    b.branch(p2_loop_continue);

    b.label_with_id(p2_loop_continue);
    b.branch(p2_loop_header);

    fixup_phi(&mut b.functions, phi_col2, next_col2, p2_loop_continue);

    b.label_with_id(p2_loop_merge);

    // Return.
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

#[cfg(test)]
#[path = "spirv_layernorm_tests.rs"]
mod spirv_layernorm_tests;
