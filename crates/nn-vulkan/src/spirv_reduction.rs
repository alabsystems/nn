// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for reduction and softmax compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for common reduction operations:
//!
//! - [`generate_sum_spirv`]: Sum reduction over a flat buffer of n floats.
//! - [`generate_max_spirv`]: Max reduction over a flat buffer of n floats.
//! - [`generate_mean_spirv`]: Mean reduction (sum / n).
//! - [`generate_softmax_spirv`]: Per-row softmax over a [rows, cols] matrix.
//!
//! All reduction kernels use a two-phase approach:
//! 1. **Workgroup-level parallel reduction** using shared memory and barriers.
//! 2. **Final atomic/serial reduction** in the first workgroup thread.
//!
//! Softmax is row-parallel: each workgroup handles one row and performs
//! max, subtract, exp, sum, and divide in sequence.
//!
//! All shaders use SPIR-V 1.0 for maximum Vulkan compatibility, `StorageBuffer`
//! storage class with `std430` layout, and push constants for dimensions.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for reduction kernels (1D dispatch).
pub const REDUCTION_WORKGROUP_SIZE: u32 = 256;

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
const OP_FDIV: u16 = 136;
const OP_FSUB: u16 = 131;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_U_LESS_THAN: u16 = 176;
const OP_IADD: u16 = 128;
const OP_IMUL: u16 = 132;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_I_EQUAL: u16 = 170;
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
const GLSL_STD_450_FMAX: u32 = 40;
const GLSL_STD_450_EXP: u32 = 27;

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

/// SPIR-V module builder (local to this module, mirrors spirv_binary.rs).
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

    fn fdiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FDIV));
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

    fn i_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_I_EQUAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
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

/// Common type IDs and variable IDs for reduction shaders.
struct ReductionSetup {
    ty_void: u32,
    ty_float: u32,
    ty_uint: u32,
    ty_bool: u32,
    ty_fn_void: u32,
    ptr_sb_float: u32,
    ptr_pc_uint: u32,
    ptr_wg_float: u32,
    const_0u: u32,
    const_1u: u32,
    var_buf_in: u32,
    var_buf_out: u32,
    var_pc: u32,
    var_gid: u32,
    var_lid: u32,
    var_shared: u32,
    glsl_ext: u32,
    /// Scope workgroup constant ID (for barriers).
    const_scope_wg: u32,
    /// Memory semantics constant ID (for barriers).
    const_mem_sem: u32,
}

/// Set up types, decorations, and global variables for a single-output reduction shader.
///
/// Layout:
/// - Binding 0: Input buffer (float[])
/// - Binding 1: Output buffer (float[], single element for sum/max/mean)
/// - Push constants: { uint n; }
/// - Shared memory: float[REDUCTION_WORKGROUP_SIZE]
fn setup_reduction_types(b: &mut SpirVBuilder) -> ReductionSetup {
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays of float for storage buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct.
    let ty_struct_in = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct.
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Shared memory: float[REDUCTION_WORKGROUP_SIZE]
    let const_wg_size = b.constant_u32(ty_uint, REDUCTION_WORKGROUP_SIZE);
    let ty_shared_arr = b.type_array(ty_float, const_wg_size);
    b.decorate(ty_shared_arr, DECORATION_ARRAY_STRIDE, &[4]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_wg_shared = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_shared_arr);
    let ptr_wg_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);

    // Scope and memory semantics constants (for OpControlBarrier).
    let const_scope_wg = b.constant_u32(ty_uint, SCOPE_WORKGROUP);
    let const_mem_sem = b.constant_u32(
        ty_uint,
        MEMORY_SEMANTICS_WORKGROUP | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );

    // Global variables.
    let var_buf_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_in, DECORATION_BINDING, &[0]);

    let var_buf_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    let var_lid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);

    // Shared memory variable.
    let var_shared = b.variable_global(ptr_wg_shared, STORAGE_CLASS_WORKGROUP);

    // GLSL.std.450 extension.
    let glsl_ext = b.ext_inst_import("GLSL.std.450");

    ReductionSetup {
        ty_void,
        ty_float,
        ty_uint,
        ty_bool,
        ty_fn_void,
        ptr_sb_float,
        ptr_pc_uint,
        ptr_wg_float,
        const_0u,
        const_1u,
        var_buf_in,
        var_buf_out,
        var_pc,
        var_gid,
        var_lid,
        var_shared,
        glsl_ext,
        const_scope_wg,
        const_mem_sem,
    }
}

/// The kind of reduction operation to perform.
enum ReductionKind {
    Sum,
    Max,
    Mean,
}

/// Generate a SPIR-V reduction kernel (sum, max, or mean).
///
/// Algorithm (single-workgroup for n <= WORKGROUP_SIZE, multi-workgroup partial):
///
/// ```text
/// lid = gl_LocalInvocationID.x
/// gid = gl_GlobalInvocationID.x
/// shared[lid] = (gid < n) ? input[gid] : identity
/// barrier()
/// // Tree reduction in shared memory
/// for stride = WORKGROUP_SIZE/2; stride > 0; stride >>= 1:
///     if lid < stride:
///         shared[lid] = op(shared[lid], shared[lid + stride])
///     barrier()
/// if lid == 0:
///     output[0] = shared[0]  // (or shared[0] / n for mean)
/// ```
fn generate_reduction_spirv(n: u32, kind: ReductionKind) -> Vec<u8> {
    let _ = n; // n is a hint; actual size comes from push constants at runtime.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let s = setup_reduction_types(&mut b);

    // Entry point — list all Input interface variables.
    b.entry_point_compute(func_id, "main", &[s.var_gid, s.var_lid]);
    b.execution_mode_local_size(func_id, REDUCTION_WORKGROUP_SIZE, 1, 1);

    // Identity element for the reduction.
    let identity = match kind {
        ReductionKind::Sum | ReductionKind::Mean => b.constant_f32(s.ty_float, 0.0),
        ReductionKind::Max => b.constant_f32(s.ty_float, f32::NEG_INFINITY),
    };

    // Function body.
    b.func_begin(s.ty_void, func_id, FUNCTION_CONTROL_NONE, s.ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x and gl_LocalInvocationID.x.
    let ty_uvec3 = b.type_vector(s.ty_uint, 3);
    let loaded_gid = b.load(ty_uvec3, s.var_gid);
    let gid_x = b.composite_extract(s.ty_uint, loaded_gid, 0);
    let loaded_lid = b.load(ty_uvec3, s.var_lid);
    let lid_x = b.composite_extract(s.ty_uint, loaded_lid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_0u]);
    let dim_n = b.load(s.ty_uint, pc_n_ptr);

    // Bounds check: val = (gid < n) ? input[gid] : identity
    let cmp_bounds = b.u_less_than(s.ty_bool, gid_x, dim_n);
    let load_label = b.id();
    let identity_label = b.id();
    let merge_init = b.id();
    b.selection_merge(merge_init);
    b.branch_conditional(cmp_bounds, load_label, identity_label);

    // Load from input.
    b.label_with_id(load_label);
    let ptr_in = b.access_chain(s.ptr_sb_float, s.var_buf_in, &[s.const_0u, gid_x]);
    let val_loaded = b.load(s.ty_float, ptr_in);
    b.branch(merge_init);

    // Identity path.
    b.label_with_id(identity_label);
    b.branch(merge_init);

    // Merge: phi to select loaded value or identity.
    b.label_with_id(merge_init);
    let init_val = b.phi(
        s.ty_float,
        &[(val_loaded, load_label), (identity, identity_label)],
    );

    // Store initial value to shared memory: shared[lid] = init_val
    let ptr_shared_lid = b.access_chain(s.ptr_wg_float, s.var_shared, &[lid_x]);
    b.store(ptr_shared_lid, init_val);

    // Barrier: wait for all threads to finish writing shared memory.
    b.control_barrier(s.const_scope_wg, s.const_scope_wg, s.const_mem_sem);

    // Tree reduction loop:
    //   stride starts at WORKGROUP_SIZE/2, halves each iteration until 0.
    //
    // Loop structure:
    //   loop_header: phi(stride) + condition check (stride > 0 <==> stride != 0 for uint)
    //   loop_body: if lid < stride { shared[lid] = op(shared[lid], shared[lid+stride]) }; barrier
    //   loop_continue: stride >>= 1 -> back to header
    //   loop_merge: done with reduction
    let const_half_wg = b.constant_u32(s.ty_uint, REDUCTION_WORKGROUP_SIZE / 2);

    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge_label = b.id();

    b.branch(loop_header);

    // Loop header.
    b.label_with_id(loop_header);
    b.loop_merge(loop_merge_label, loop_continue);

    // Phi for stride.
    let phi_stride = b.phi(s.ty_uint, &[(const_half_wg, merge_init)]);

    // Condition: stride > 0 (for unsigned, this is stride != 0, i.e., !(stride == 0)).
    // We use u_less_than(0, stride) which is equivalent to stride > 0 for unsigned.
    let cmp_stride = b.u_less_than(s.ty_bool, s.const_0u, phi_stride);
    b.branch_conditional(cmp_stride, loop_body, loop_merge_label);

    // Loop body.
    b.label_with_id(loop_body);

    // if lid < stride: reduce
    let cmp_lid = b.u_less_than(s.ty_bool, lid_x, phi_stride);
    let reduce_label = b.id();
    let skip_label = b.id();
    b.selection_merge(skip_label);
    b.branch_conditional(cmp_lid, reduce_label, skip_label);

    // Reduce block: shared[lid] = op(shared[lid], shared[lid + stride])
    b.label_with_id(reduce_label);
    let ptr_a = b.access_chain(s.ptr_wg_float, s.var_shared, &[lid_x]);
    let val_a = b.load(s.ty_float, ptr_a);
    let lid_plus_stride = b.iadd(s.ty_uint, lid_x, phi_stride);
    let ptr_b = b.access_chain(s.ptr_wg_float, s.var_shared, &[lid_plus_stride]);
    let val_b = b.load(s.ty_float, ptr_b);

    let reduced = match kind {
        ReductionKind::Sum | ReductionKind::Mean => b.fadd(s.ty_float, val_a, val_b),
        ReductionKind::Max => {
            b.ext_inst(s.ty_float, s.glsl_ext, GLSL_STD_450_FMAX, &[val_a, val_b])
        }
    };

    b.store(ptr_a, reduced);
    b.branch(skip_label);

    // Skip / merge after conditional reduce.
    b.label_with_id(skip_label);

    // Barrier after each reduction step.
    b.control_barrier(s.const_scope_wg, s.const_scope_wg, s.const_mem_sem);

    b.branch(loop_continue);

    // Loop continue: stride >>= 1.
    b.label_with_id(loop_continue);
    let stride_next = b.shift_right_logical(s.ty_uint, phi_stride, s.const_1u);
    b.branch(loop_header);

    // Fixup phi for stride back-edge.
    fixup_phi(&mut b.functions, phi_stride, stride_next, loop_continue);

    // Loop merge: reduction is done. shared[0] holds the result.
    b.label_with_id(loop_merge_label);

    // Only thread 0 writes the output.
    let cmp_zero = b.i_equal(s.ty_bool, lid_x, s.const_0u);
    let write_label = b.id();
    let end_label = b.id();
    b.selection_merge(end_label);
    b.branch_conditional(cmp_zero, write_label, end_label);

    b.label_with_id(write_label);

    // Load final result from shared[0].
    let ptr_shared_0 = b.access_chain(s.ptr_wg_float, s.var_shared, &[s.const_0u]);
    let final_val = b.load(s.ty_float, ptr_shared_0);

    // For mean: divide by n.
    let output_val = match kind {
        ReductionKind::Mean => {
            let n_float = b.convert_u_to_f(s.ty_float, dim_n);
            b.fdiv(s.ty_float, final_val, n_float)
        }
        _ => final_val,
    };

    // Store to output[0].
    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_out, &[s.const_0u, s.const_0u]);
    b.store(ptr_out, output_val);
    b.branch(end_label);

    // End.
    b.label_with_id(end_label);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

/// Generate a SPIR-V binary for sum reduction: output\[0\] = sum(input\[0..n\]).
///
/// Uses workgroup shared memory with parallel tree reduction.
///
/// # Arguments
///
/// * `n` - Number of input elements (compile-time hint; actual value from push constants).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[n\])
/// - Binding 1: Output buffer (float\[1\])
///
/// # Push constants
///
/// - `uint n` at offset 0
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_reduction::generate_sum_spirv;
/// let spirv = generate_sum_spirv(1024);
/// assert_eq!(spirv.len() % 4, 0); // 4-byte aligned
/// // Check SPIR-V magic (little-endian).
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_sum_spirv(n: u32) -> Vec<u8> {
    generate_reduction_spirv(n, ReductionKind::Sum)
}

/// Generate a SPIR-V binary for max reduction: output\[0\] = max(input\[0..n\]).
///
/// Uses workgroup shared memory with parallel tree reduction.
/// Identity element is `f32::NEG_INFINITY`.
///
/// # Arguments
///
/// * `n` - Number of input elements (compile-time hint; actual value from push constants).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[n\])
/// - Binding 1: Output buffer (float\[1\])
///
/// # Push constants
///
/// - `uint n` at offset 0
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_reduction::generate_max_spirv;
/// let spirv = generate_max_spirv(1024);
/// assert_eq!(spirv.len() % 4, 0);
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_max_spirv(n: u32) -> Vec<u8> {
    generate_reduction_spirv(n, ReductionKind::Max)
}

/// Generate a SPIR-V binary for mean reduction: output\[0\] = sum(input\[0..n\]) / n.
///
/// Uses workgroup shared memory with parallel tree reduction for the sum,
/// then divides by n (converted to float via `OpConvertUToF`).
///
/// # Arguments
///
/// * `n` - Number of input elements (compile-time hint; actual value from push constants).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[n\])
/// - Binding 1: Output buffer (float\[1\])
///
/// # Push constants
///
/// - `uint n` at offset 0
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_reduction::generate_mean_spirv;
/// let spirv = generate_mean_spirv(1024);
/// assert_eq!(spirv.len() % 4, 0);
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_mean_spirv(n: u32) -> Vec<u8> {
    generate_reduction_spirv(n, ReductionKind::Mean)
}

/// Generate a SPIR-V binary for per-row softmax over a \[rows, cols\] matrix.
///
/// Each workgroup handles one row. Within the workgroup, threads cooperate
/// on the three reduction passes using shared memory:
///
/// 1. **Max**: row_max = max(row\[0..cols\])
/// 2. **Subtract + Exp**: row\[i\] = exp(row\[i\] - row_max)
/// 3. **Sum**: row_sum = sum(exp_values)
/// 4. **Divide**: row\[i\] = row\[i\] / row_sum
///
/// # Arguments
///
/// * `rows` - Number of rows (compile-time hint; actual from push constants).
/// * `cols` - Number of columns per row (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: Input/output buffer (float\[rows * cols\], in-place softmax)
///
/// # Push constants
///
/// - `uint rows` at offset 0
/// - `uint cols` at offset 4
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_reduction::generate_softmax_spirv;
/// let spirv = generate_softmax_spirv(32, 128);
/// assert_eq!(spirv.len() % 4, 0);
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_softmax_spirv(rows: u32, cols: u32) -> Vec<u8> {
    let _ = (rows, cols); // hints; actual dims from push constants.

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

    // Runtime array for in-place buffer.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer struct (in-place: input = output).
    let ty_struct_buf = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_buf, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_buf, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint rows; uint cols; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Shared memory: float[REDUCTION_WORKGROUP_SIZE]
    let const_wg_size = b.constant_u32(ty_uint, REDUCTION_WORKGROUP_SIZE);
    let ty_shared_arr = b.type_array(ty_float, const_wg_size);
    b.decorate(ty_shared_arr, DECORATION_ARRAY_STRIDE, &[4]);

    // Pointer types.
    let ptr_sb_buf = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_buf);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_wg_shared = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_shared_arr);
    let ptr_wg_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_neg_inf = b.constant_f32(ty_float, f32::NEG_INFINITY);
    let const_f0 = b.constant_f32(ty_float, 0.0);
    let const_half_wg = b.constant_u32(ty_uint, REDUCTION_WORKGROUP_SIZE / 2);
    let const_scope_wg = b.constant_u32(ty_uint, SCOPE_WORKGROUP);
    let const_mem_sem = b.constant_u32(
        ty_uint,
        MEMORY_SEMANTICS_WORKGROUP | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );

    // Global variables.
    let var_buf = b.variable_global(ptr_sb_buf, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf, DECORATION_BINDING, &[0]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    let var_lid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);

    let var_shared = b.variable_global(ptr_wg_shared, STORAGE_CLASS_WORKGROUP);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid, var_lid]);
    b.execution_mode_local_size(func_id, REDUCTION_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let entry_label = b.label();

    // Load invocation IDs.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);
    let loaded_lid = b.load(ty_uvec3, var_lid);
    let lid_x = b.composite_extract(ty_uint, loaded_lid, 0);

    // Load rows, cols from push constants.
    let pc_rows_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let _dim_rows = b.load(ty_uint, pc_rows_ptr);
    let pc_cols_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let dim_cols = b.load(ty_uint, pc_cols_ptr);

    // Row index: row = gid_x / WORKGROUP_SIZE (since we dispatch 1 workgroup per row,
    // the workgroup_id.x IS the row). But for SPIR-V 1.0 without WorkgroupId,
    // we compute row = (gid_x - lid_x) / WORKGROUP_SIZE. However, it is simpler to
    // note that gid_x = row * WORKGROUP_SIZE + lid_x for 1 workgroup per row dispatch.
    // But if cols > WORKGROUP_SIZE, we need multiple passes per row within the workgroup.
    //
    // For simplicity and generality: row = gid_x / WORKGROUP_SIZE.
    // This works because we dispatch ceil(rows) workgroups in x, 1 in y.
    // Actually, for softmax each workgroup handles one row, dispatched as (rows, 1, 1).
    // So gid_x ranges across threads within a row, and workgroup_id.x = row.
    // We need WorkgroupId for that. Let's compute: row = gid_x / WORKGROUP_SIZE.
    // This is valid because dispatch is (rows, 1, 1) with local size (WG_SIZE, 1, 1).
    // gid_x = workgroup_id.x * WG_SIZE + lid_x, so gid_x / WG_SIZE = workgroup_id.x = row.
    // Integer division works because gid_x and WG_SIZE are both uint.
    //
    // However, SPIR-V doesn't have OpUDiv, but we defined IADD and IMUL... Actually we do
    // need division. Let's add it. Actually, we can avoid division by subtracting lid
    // and then dividing. Or better: we can derive row from gl_GlobalInvocationID.x:
    //   row = (gid_x - lid_x) / WG_SIZE
    // But actually for clean dispatch with 1 workgroup per row:
    //   gid_x = row * WG_SIZE + lid_x
    //   row = (gid_x - lid_x) / WG_SIZE
    // Since gid_x - lid_x is exactly row * WG_SIZE, the division is exact.
    //
    // We can use UDiv (opcode 134 — not imported yet, but let's just use the subtraction
    // and shift approach since WG_SIZE is a power of 2).
    // WG_SIZE = 256 = 2^8, so row = (gid_x - lid_x) >> 8.
    let gid_minus_lid = {
        // gid_x - lid_x as unsigned subtraction. SPIR-V: use ISub (opcode 130).
        let result = b.id();
        let op_isub: u16 = 130;
        b.functions.push(op(5, op_isub));
        b.functions.push(ty_uint);
        b.functions.push(result);
        b.functions.push(gid_x);
        b.functions.push(lid_x);
        result
    };
    let const_8u = b.constant_u32(ty_uint, 8); // log2(256) = 8
    let row = b.shift_right_logical(ty_uint, gid_minus_lid, const_8u);

    // Base offset for this row: row_base = row * cols.
    let row_base = b.imul(ty_uint, row, dim_cols);

    // ========================================================
    // Phase 1: Find max of this row using shared memory reduction.
    // ========================================================
    // Each thread loads one element (or neg_inf if lid >= cols).
    // We handle cols > WG_SIZE via a serial accumulation loop first.

    // Serial accumulation loop: thread lid accumulates elements
    // lid, lid+WG_SIZE, lid+2*WG_SIZE, ... that fall within [0, cols).
    // This handles cols > WORKGROUP_SIZE.
    let phase1_loop_header = b.id();
    let phase1_loop_body = b.id();
    let phase1_loop_continue = b.id();
    let phase1_loop_merge = b.id();

    b.branch(phase1_loop_header);
    b.label_with_id(phase1_loop_header);
    b.loop_merge(phase1_loop_merge, phase1_loop_continue);

    let phi_col_idx = b.phi(ty_uint, &[(lid_x, entry_label)]);
    let phi_max_accum = b.phi(ty_float, &[(const_neg_inf, entry_label)]);

    let cmp_col = b.u_less_than(ty_bool, phi_col_idx, dim_cols);
    b.branch_conditional(cmp_col, phase1_loop_body, phase1_loop_merge);

    b.label_with_id(phase1_loop_body);
    let elem_idx = b.iadd(ty_uint, row_base, phi_col_idx);
    let ptr_elem = b.access_chain(ptr_sb_float, var_buf, &[const_0u, elem_idx]);
    let val_elem = b.load(ty_float, ptr_elem);
    let new_max = b.ext_inst(
        ty_float,
        glsl_ext,
        GLSL_STD_450_FMAX,
        &[phi_max_accum, val_elem],
    );
    let next_col_idx = b.iadd(ty_uint, phi_col_idx, const_wg_size);
    b.branch(phase1_loop_continue);

    b.label_with_id(phase1_loop_continue);
    b.branch(phase1_loop_header);

    fixup_phi(
        &mut b.functions,
        phi_col_idx,
        next_col_idx,
        phase1_loop_continue,
    );
    fixup_phi(
        &mut b.functions,
        phi_max_accum,
        new_max,
        phase1_loop_continue,
    );

    b.label_with_id(phase1_loop_merge);

    // Store thread's partial max into shared memory.
    let ptr_s_lid = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    b.store(ptr_s_lid, phi_max_accum);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // Tree reduction for max.
    let max_tree_header = b.id();
    let max_tree_body = b.id();
    let max_tree_continue = b.id();
    let max_tree_merge = b.id();

    b.branch(max_tree_header);
    b.label_with_id(max_tree_header);
    b.loop_merge(max_tree_merge, max_tree_continue);

    let phi_max_stride = b.phi(ty_uint, &[(const_half_wg, phase1_loop_merge)]);
    let cmp_max_stride = b.u_less_than(ty_bool, const_0u, phi_max_stride);
    b.branch_conditional(cmp_max_stride, max_tree_body, max_tree_merge);

    b.label_with_id(max_tree_body);
    let cmp_max_lid = b.u_less_than(ty_bool, lid_x, phi_max_stride);
    let max_reduce_label = b.id();
    let max_skip_label = b.id();
    b.selection_merge(max_skip_label);
    b.branch_conditional(cmp_max_lid, max_reduce_label, max_skip_label);

    b.label_with_id(max_reduce_label);
    let ptr_max_a = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    let val_max_a = b.load(ty_float, ptr_max_a);
    let lid_plus_max_stride = b.iadd(ty_uint, lid_x, phi_max_stride);
    let ptr_max_b = b.access_chain(ptr_wg_float, var_shared, &[lid_plus_max_stride]);
    let val_max_b = b.load(ty_float, ptr_max_b);
    let reduced_max = b.ext_inst(
        ty_float,
        glsl_ext,
        GLSL_STD_450_FMAX,
        &[val_max_a, val_max_b],
    );
    b.store(ptr_max_a, reduced_max);
    b.branch(max_skip_label);

    b.label_with_id(max_skip_label);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);
    b.branch(max_tree_continue);

    b.label_with_id(max_tree_continue);
    let max_stride_next = b.shift_right_logical(ty_uint, phi_max_stride, const_1u);
    b.branch(max_tree_header);

    fixup_phi(
        &mut b.functions,
        phi_max_stride,
        max_stride_next,
        max_tree_continue,
    );

    b.label_with_id(max_tree_merge);

    // Load row_max from shared[0].
    let ptr_s0 = b.access_chain(ptr_wg_float, var_shared, &[const_0u]);
    let row_max = b.load(ty_float, ptr_s0);

    // Barrier before reusing shared memory.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // ========================================================
    // Phase 2 & 3: Subtract max, exp, and sum.
    // ========================================================
    // Serial loop: subtract max and exp in-place, accumulate partial sum.
    let phase2_loop_header = b.id();
    let phase2_loop_body = b.id();
    let phase2_loop_continue = b.id();
    let phase2_loop_merge = b.id();

    b.branch(phase2_loop_header);
    b.label_with_id(phase2_loop_header);
    b.loop_merge(phase2_loop_merge, phase2_loop_continue);

    let phi_col2 = b.phi(ty_uint, &[(lid_x, max_tree_merge)]);
    let phi_sum_accum = b.phi(ty_float, &[(const_f0, max_tree_merge)]);

    let cmp_col2 = b.u_less_than(ty_bool, phi_col2, dim_cols);
    b.branch_conditional(cmp_col2, phase2_loop_body, phase2_loop_merge);

    b.label_with_id(phase2_loop_body);
    let elem_idx2 = b.iadd(ty_uint, row_base, phi_col2);
    let ptr_elem2 = b.access_chain(ptr_sb_float, var_buf, &[const_0u, elem_idx2]);
    let val_elem2 = b.load(ty_float, ptr_elem2);
    let shifted = b.fsub(ty_float, val_elem2, row_max);
    let exp_val = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_EXP, &[shifted]);
    // Write exp back in-place.
    b.store(ptr_elem2, exp_val);
    let new_sum = b.fadd(ty_float, phi_sum_accum, exp_val);
    let next_col2 = b.iadd(ty_uint, phi_col2, const_wg_size);
    b.branch(phase2_loop_continue);

    b.label_with_id(phase2_loop_continue);
    b.branch(phase2_loop_header);

    fixup_phi(&mut b.functions, phi_col2, next_col2, phase2_loop_continue);
    fixup_phi(
        &mut b.functions,
        phi_sum_accum,
        new_sum,
        phase2_loop_continue,
    );

    b.label_with_id(phase2_loop_merge);

    // Store partial sum to shared memory.
    let ptr_s_lid2 = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    b.store(ptr_s_lid2, phi_sum_accum);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // Tree reduction for sum.
    let sum_tree_header = b.id();
    let sum_tree_body = b.id();
    let sum_tree_continue = b.id();
    let sum_tree_merge = b.id();

    b.branch(sum_tree_header);
    b.label_with_id(sum_tree_header);
    b.loop_merge(sum_tree_merge, sum_tree_continue);

    let phi_sum_stride = b.phi(ty_uint, &[(const_half_wg, phase2_loop_merge)]);
    let cmp_sum_stride = b.u_less_than(ty_bool, const_0u, phi_sum_stride);
    b.branch_conditional(cmp_sum_stride, sum_tree_body, sum_tree_merge);

    b.label_with_id(sum_tree_body);
    let cmp_sum_lid = b.u_less_than(ty_bool, lid_x, phi_sum_stride);
    let sum_reduce_label = b.id();
    let sum_skip_label = b.id();
    b.selection_merge(sum_skip_label);
    b.branch_conditional(cmp_sum_lid, sum_reduce_label, sum_skip_label);

    b.label_with_id(sum_reduce_label);
    let ptr_sum_a = b.access_chain(ptr_wg_float, var_shared, &[lid_x]);
    let val_sum_a = b.load(ty_float, ptr_sum_a);
    let lid_plus_sum_stride = b.iadd(ty_uint, lid_x, phi_sum_stride);
    let ptr_sum_b = b.access_chain(ptr_wg_float, var_shared, &[lid_plus_sum_stride]);
    let val_sum_b = b.load(ty_float, ptr_sum_b);
    let reduced_sum = b.fadd(ty_float, val_sum_a, val_sum_b);
    b.store(ptr_sum_a, reduced_sum);
    b.branch(sum_skip_label);

    b.label_with_id(sum_skip_label);
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);
    b.branch(sum_tree_continue);

    b.label_with_id(sum_tree_continue);
    let sum_stride_next = b.shift_right_logical(ty_uint, phi_sum_stride, const_1u);
    b.branch(sum_tree_header);

    fixup_phi(
        &mut b.functions,
        phi_sum_stride,
        sum_stride_next,
        sum_tree_continue,
    );

    b.label_with_id(sum_tree_merge);

    // Load row_sum from shared[0].
    let ptr_s0_sum = b.access_chain(ptr_wg_float, var_shared, &[const_0u]);
    let row_sum = b.load(ty_float, ptr_s0_sum);

    // Barrier before reusing shared memory.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // ========================================================
    // Phase 4: Divide each element by row_sum.
    // ========================================================
    let phase4_loop_header = b.id();
    let phase4_loop_body = b.id();
    let phase4_loop_continue = b.id();
    let phase4_loop_merge = b.id();

    b.branch(phase4_loop_header);
    b.label_with_id(phase4_loop_header);
    b.loop_merge(phase4_loop_merge, phase4_loop_continue);

    let phi_col4 = b.phi(ty_uint, &[(lid_x, sum_tree_merge)]);
    let cmp_col4 = b.u_less_than(ty_bool, phi_col4, dim_cols);
    b.branch_conditional(cmp_col4, phase4_loop_body, phase4_loop_merge);

    b.label_with_id(phase4_loop_body);
    let elem_idx4 = b.iadd(ty_uint, row_base, phi_col4);
    let ptr_elem4 = b.access_chain(ptr_sb_float, var_buf, &[const_0u, elem_idx4]);
    let val_elem4 = b.load(ty_float, ptr_elem4);
    let normed = b.fdiv(ty_float, val_elem4, row_sum);
    b.store(ptr_elem4, normed);
    let next_col4 = b.iadd(ty_uint, phi_col4, const_wg_size);
    b.branch(phase4_loop_continue);

    b.label_with_id(phase4_loop_continue);
    b.branch(phase4_loop_header);

    fixup_phi(&mut b.functions, phi_col4, next_col4, phase4_loop_continue);

    b.label_with_id(phase4_loop_merge);

    // Return.
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

    // ---- Helpers ----

    fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
        assert_eq!(bytes.len() % 4, 0, "SPIR-V binary must be 4-byte aligned");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn assert_valid_header(words: &[u32], label: &str) {
        assert!(words.len() >= 5, "{label}: module too short");
        assert_eq!(words[0], SPIRV_MAGIC, "{label}: wrong magic number");
        assert_eq!(words[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
        assert_eq!(words[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
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

    // ---- generate_sum_spirv ----

    #[test]
    fn test_sum_spirv_header_256() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "sum_256");
    }

    #[test]
    fn test_sum_spirv_header_16() {
        let bytes = generate_sum_spirv(16);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "sum_16");
    }

    #[test]
    fn test_sum_spirv_header_4096() {
        let bytes = generate_sum_spirv(4096);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "sum_4096");
    }

    #[test]
    fn test_sum_spirv_header_non_power_of_2() {
        let bytes = generate_sum_spirv(100);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "sum_100");
    }

    #[test]
    fn test_sum_spirv_entry_point() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_sum_spirv_workgroup_size() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
    }

    #[test]
    fn test_sum_spirv_has_capability() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_CAPABILITY),
            "sum must have OpCapability"
        );
    }

    #[test]
    fn test_sum_spirv_has_memory_model() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_MEMORY_MODEL),
            "sum must have OpMemoryModel"
        );
    }

    #[test]
    fn test_sum_spirv_has_loop() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_LOOP_MERGE),
            "sum must have loop (OpLoopMerge)"
        );
        assert!(
            has_opcode(&words, OP_PHI),
            "sum must have OpPhi for loop variables"
        );
    }

    #[test]
    fn test_sum_spirv_has_barrier() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_CONTROL_BARRIER),
            "sum must have OpControlBarrier for shared memory sync"
        );
    }

    #[test]
    fn test_sum_spirv_has_fadd() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_FADD),
            "sum must have OpFAdd for accumulation"
        );
    }

    #[test]
    fn test_sum_spirv_has_function_structure() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(has_opcode(&words, OP_FUNCTION), "must have OpFunction");
        assert!(
            has_opcode(&words, OP_FUNCTION_END),
            "must have OpFunctionEnd"
        );
        assert!(has_opcode(&words, OP_LABEL), "must have OpLabel");
        assert!(has_opcode(&words, OP_RETURN), "must have OpReturn");
    }

    #[test]
    fn test_sum_spirv_byte_alignment() {
        for n in [16, 100, 256, 1024, 4096] {
            let bytes = generate_sum_spirv(n);
            assert_eq!(
                bytes.len() % 4,
                0,
                "sum n={n}: SPIR-V binary must be 4-byte aligned"
            );
        }
    }

    #[test]
    fn test_sum_spirv_reasonable_size() {
        let bytes = generate_sum_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            words.len() > 50,
            "sum module too small ({} words)",
            words.len()
        );
        assert!(
            words.len() < 2000,
            "sum module too large ({} words)",
            words.len()
        );
    }

    // ---- generate_max_spirv ----

    #[test]
    fn test_max_spirv_header_256() {
        let bytes = generate_max_spirv(256);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "max_256");
    }

    #[test]
    fn test_max_spirv_header_16() {
        let bytes = generate_max_spirv(16);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "max_16");
    }

    #[test]
    fn test_max_spirv_header_4096() {
        let bytes = generate_max_spirv(4096);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "max_4096");
    }

    #[test]
    fn test_max_spirv_header_non_power_of_2() {
        let bytes = generate_max_spirv(77);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "max_77");
    }

    #[test]
    fn test_max_spirv_entry_point() {
        let bytes = generate_max_spirv(256);
        let words = bytes_to_words(&bytes);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_max_spirv_workgroup_size() {
        let bytes = generate_max_spirv(256);
        let words = bytes_to_words(&bytes);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
    }

    #[test]
    fn test_max_spirv_has_ext_inst() {
        let bytes = generate_max_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_EXT_INST),
            "max must use GLSL.std.450 FMax (OpExtInst)"
        );
    }

    #[test]
    fn test_max_spirv_has_barrier() {
        let bytes = generate_max_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_CONTROL_BARRIER),
            "max must have OpControlBarrier"
        );
    }

    #[test]
    fn test_max_spirv_has_loop() {
        let bytes = generate_max_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(has_opcode(&words, OP_LOOP_MERGE), "max must have loop");
        assert!(has_opcode(&words, OP_PHI), "max must have OpPhi");
    }

    #[test]
    fn test_max_spirv_byte_alignment() {
        for n in [16, 77, 256, 1024, 4096] {
            let bytes = generate_max_spirv(n);
            assert_eq!(
                bytes.len() % 4,
                0,
                "max n={n}: SPIR-V binary must be 4-byte aligned"
            );
        }
    }

    // ---- generate_mean_spirv ----

    #[test]
    fn test_mean_spirv_header_256() {
        let bytes = generate_mean_spirv(256);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "mean_256");
    }

    #[test]
    fn test_mean_spirv_header_16() {
        let bytes = generate_mean_spirv(16);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "mean_16");
    }

    #[test]
    fn test_mean_spirv_header_4096() {
        let bytes = generate_mean_spirv(4096);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "mean_4096");
    }

    #[test]
    fn test_mean_spirv_header_non_power_of_2() {
        let bytes = generate_mean_spirv(33);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "mean_33");
    }

    #[test]
    fn test_mean_spirv_entry_point() {
        let bytes = generate_mean_spirv(256);
        let words = bytes_to_words(&bytes);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_mean_spirv_workgroup_size() {
        let bytes = generate_mean_spirv(256);
        let words = bytes_to_words(&bytes);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
    }

    #[test]
    fn test_mean_spirv_has_fdiv() {
        let bytes = generate_mean_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_FDIV),
            "mean must have OpFDiv for division by n"
        );
    }

    #[test]
    fn test_mean_spirv_has_convert_u_to_f() {
        let bytes = generate_mean_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_CONVERT_U_TO_F),
            "mean must have OpConvertUToF to convert n to float"
        );
    }

    #[test]
    fn test_mean_spirv_has_fadd() {
        let bytes = generate_mean_spirv(256);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_FADD),
            "mean must have OpFAdd for sum accumulation"
        );
    }

    #[test]
    fn test_mean_spirv_byte_alignment() {
        for n in [16, 33, 256, 1024, 4096] {
            let bytes = generate_mean_spirv(n);
            assert_eq!(
                bytes.len() % 4,
                0,
                "mean n={n}: SPIR-V binary must be 4-byte aligned"
            );
        }
    }

    // ---- generate_softmax_spirv ----

    #[test]
    fn test_softmax_spirv_header_32x128() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "softmax_32x128");
    }

    #[test]
    fn test_softmax_spirv_header_1x16() {
        let bytes = generate_softmax_spirv(1, 16);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "softmax_1x16");
    }

    #[test]
    fn test_softmax_spirv_header_64x4096() {
        let bytes = generate_softmax_spirv(64, 4096);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "softmax_64x4096");
    }

    #[test]
    fn test_softmax_spirv_header_non_power_of_2() {
        let bytes = generate_softmax_spirv(7, 33);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, "softmax_7x33");
    }

    #[test]
    fn test_softmax_spirv_entry_point() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_softmax_spirv_workgroup_size() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
    }

    #[test]
    fn test_softmax_spirv_has_ext_inst() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_EXT_INST),
            "softmax must use GLSL.std.450 for FMax and Exp"
        );
    }

    #[test]
    fn test_softmax_spirv_has_fsub() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_FSUB),
            "softmax must have OpFSub for (x - max)"
        );
    }

    #[test]
    fn test_softmax_spirv_has_fdiv() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_FDIV),
            "softmax must have OpFDiv for normalization"
        );
    }

    #[test]
    fn test_softmax_spirv_has_fadd() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_FADD),
            "softmax must have OpFAdd for sum"
        );
    }

    #[test]
    fn test_softmax_spirv_has_loops() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_LOOP_MERGE),
            "softmax must have loops for reduction"
        );
        assert!(
            has_opcode(&words, OP_PHI),
            "softmax must have OpPhi for loop variables"
        );
    }

    #[test]
    fn test_softmax_spirv_has_barrier() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(
            has_opcode(&words, OP_CONTROL_BARRIER),
            "softmax must have barriers for shared memory sync"
        );
    }

    #[test]
    fn test_softmax_spirv_has_function_structure() {
        let bytes = generate_softmax_spirv(32, 128);
        let words = bytes_to_words(&bytes);
        assert!(has_opcode(&words, OP_FUNCTION), "must have OpFunction");
        assert!(
            has_opcode(&words, OP_FUNCTION_END),
            "must have OpFunctionEnd"
        );
        assert!(has_opcode(&words, OP_LABEL), "must have OpLabel");
        assert!(has_opcode(&words, OP_RETURN), "must have OpReturn");
    }

    #[test]
    fn test_softmax_spirv_byte_alignment() {
        for (r, c) in [(1, 16), (7, 33), (32, 128), (64, 4096)] {
            let bytes = generate_softmax_spirv(r, c);
            assert_eq!(
                bytes.len() % 4,
                0,
                "softmax {r}x{c}: SPIR-V binary must be 4-byte aligned"
            );
        }
    }

    #[test]
    fn test_softmax_spirv_reasonable_size() {
        let bytes = generate_softmax_spirv(32, 128);
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

    // ---- Cross-cutting tests ----

    #[test]
    fn test_all_reduction_kernels_have_capability() {
        for (name, bytes) in [
            ("sum", generate_sum_spirv(256)),
            ("max", generate_max_spirv(256)),
            ("mean", generate_mean_spirv(256)),
            ("softmax", generate_softmax_spirv(32, 128)),
        ] {
            let words = bytes_to_words(&bytes);
            assert!(
                has_opcode(&words, OP_CAPABILITY),
                "{name}: must have OpCapability"
            );
        }
    }

    #[test]
    fn test_all_reduction_kernels_have_memory_model() {
        for (name, bytes) in [
            ("sum", generate_sum_spirv(256)),
            ("max", generate_max_spirv(256)),
            ("mean", generate_mean_spirv(256)),
            ("softmax", generate_softmax_spirv(32, 128)),
        ] {
            let words = bytes_to_words(&bytes);
            assert!(
                has_opcode(&words, OP_MEMORY_MODEL),
                "{name}: must have OpMemoryModel"
            );
        }
    }

    #[test]
    fn test_all_reduction_kernels_have_bounds_check() {
        for (name, bytes) in [
            ("sum", generate_sum_spirv(256)),
            ("max", generate_max_spirv(256)),
            ("mean", generate_mean_spirv(256)),
        ] {
            let words = bytes_to_words(&bytes);
            assert!(
                has_opcode(&words, OP_U_LESS_THAN),
                "{name}: must have bounds check (OpULessThan)"
            );
            assert!(
                has_opcode(&words, OP_BRANCH_CONDITIONAL),
                "{name}: must have conditional branch"
            );
        }
    }

    #[test]
    fn test_all_reduction_kernels_various_sizes() {
        // Power-of-2, non-power-of-2, small, large.
        for n in [16, 33, 64, 100, 256, 512, 1000, 4096] {
            let sum = generate_sum_spirv(n);
            let max = generate_max_spirv(n);
            let mean = generate_mean_spirv(n);
            let sum_words = bytes_to_words(&sum);
            let max_words = bytes_to_words(&max);
            let mean_words = bytes_to_words(&mean);
            assert_valid_header(&sum_words, &format!("sum_{n}"));
            assert_valid_header(&max_words, &format!("max_{n}"));
            assert_valid_header(&mean_words, &format!("mean_{n}"));
        }
    }

    #[test]
    fn test_softmax_various_dimensions() {
        for (r, c) in [(1, 16), (4, 64), (7, 33), (32, 128), (16, 256), (8, 4096)] {
            let bytes = generate_softmax_spirv(r, c);
            let words = bytes_to_words(&bytes);
            assert_valid_header(&words, &format!("softmax_{r}x{c}"));
        }
    }

    #[test]
    fn test_sum_and_mean_share_structure() {
        // Sum and mean should have similar sizes (mean has extra FDiv + ConvertUToF).
        let sum_bytes = generate_sum_spirv(256);
        let mean_bytes = generate_mean_spirv(256);
        let sum_words = bytes_to_words(&sum_bytes);
        let mean_words = bytes_to_words(&mean_bytes);
        // Mean should be slightly larger than sum (extra division).
        assert!(
            mean_words.len() > sum_words.len(),
            "mean ({} words) should be larger than sum ({} words)",
            mean_words.len(),
            sum_words.len(),
        );
        // But not dramatically larger.
        assert!(
            mean_words.len() < sum_words.len() + 20,
            "mean ({} words) should not be much larger than sum ({} words)",
            mean_words.len(),
            sum_words.len(),
        );
    }

    #[test]
    fn test_reduction_has_workgroup_variables() {
        // All reduction kernels must have Workgroup storage class variables for shared memory.
        for (name, bytes) in [
            ("sum", generate_sum_spirv(256)),
            ("max", generate_max_spirv(256)),
            ("mean", generate_mean_spirv(256)),
            ("softmax", generate_softmax_spirv(32, 128)),
        ] {
            let words = bytes_to_words(&bytes);
            let mut wg_count = 0;
            let mut pos = 5;
            while pos < words.len() {
                let word = words[pos];
                let word_count = (word >> 16) as usize;
                let opcode = (word & 0xFFFF) as u16;
                if word_count == 0 || pos + word_count > words.len() {
                    break;
                }
                if opcode == OP_VARIABLE
                    && word_count >= 4
                    && words[pos + 3] == STORAGE_CLASS_WORKGROUP
                {
                    wg_count += 1;
                }
                pos += word_count;
            }
            assert!(
                wg_count >= 1,
                "{name}: must have at least 1 workgroup variable for shared memory (found {wg_count})"
            );
        }
    }
}
