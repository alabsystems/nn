// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for convolution and pooling compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for 1D convolution and pooling operations
//! needed by audio model pipelines (Kokoro, HTDemucs, etc.):
//!
//! - [`generate_conv1d_spirv`]: 1D convolution with stride, padding, and dilation.
//! - [`generate_max_pool1d_spirv`]: 1D max pooling with stride and padding.
//! - [`generate_avg_pool1d_spirv`]: 1D average pooling with stride and padding.
//!
//! All shaders use:
//! - Workgroup size of 256 threads (1D dispatch)
//! - Push constants for tensor dimensions and conv/pool parameters
//! - `StorageBuffer` storage class with `std430` layout
//! - SPIR-V 1.0 for maximum Vulkan compatibility
//! - Bounds checking for non-aligned dimensions
//!
//! # Buffer layouts
//!
//! **Conv1d:**
//! - Binding 0: input \[batch, in\_channels, length\] (row-major float\[\])
//! - Binding 1: weight \[out\_channels, in\_channels, kernel\_size\] (row-major float\[\])
//! - Binding 2: output \[batch, out\_channels, out\_length\] (row-major float\[\])
//!
//! **MaxPool1d / AvgPool1d:**
//! - Binding 0: input \[batch, channels, length\] (row-major float\[\])
//! - Binding 1: output \[batch, channels, out\_length\] (row-major float\[\])

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for conv/pool kernels (1D dispatch).
pub const CONV_WORKGROUP_SIZE: u32 = 256;

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
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_U_LESS_THAN: u16 = 176;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_EXT_INST: u16 = 12;
const OP_CONVERT_U_TO_F: u16 = 112;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-ins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Execution model / mode.
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Capability.
const CAPABILITY_SHADER: u32 = 1;

// Memory model.
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// GLSL.std.450 extended instruction set opcodes.
const GLSL_STD_450_FMAX: u32 = 40;

// Function control.
const FUNCTION_CONTROL_NONE: u32 = 0;

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
        self.functions.push(0); // SelectionControl: None
    }

    fn loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.functions.push(op(4, OP_LOOP_MERGE));
        self.functions.push(merge_label);
        self.functions.push(continue_label);
        self.functions.push(0); // LoopControl: None
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

    fn u_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_greater_than_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN_EQUAL));
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

    fn iadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn convert_u_to_f(&mut self, result_type: u32, operand: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_CONVERT_U_TO_F));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(operand);
        result
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
        module.push(0); // Reserved schema.
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
///
/// Finds the phi instruction that produces `phi_id` and rewrites it in place
/// to include the new back-edge operand.
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

// ---- Output length helpers ----

/// Compute the output length of a 1D convolution.
///
/// `out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1`
///
/// When `dilation == 1`, this simplifies to:
/// `(length + 2 * padding - kernel_size) / stride + 1`
pub fn conv1d_output_length(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_ks = dilation * (kernel_size - 1) + 1;
    (length + 2 * padding - effective_ks) / stride + 1
}

/// Compute the output length of a 1D pooling operation.
///
/// `out_length = (length + 2 * padding - kernel_size) / stride + 1`
///
/// Same formula as conv1d but provided separately for clarity.
pub fn pool1d_output_length(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    (length + 2 * padding - kernel_size) / stride + 1
}

// ---- Type setup helpers ----

/// Common types and variables for a 3-buffer conv1d shader.
struct Conv1dSetup {
    ty_void: u32,
    ty_float: u32,
    ty_uint: u32,
    ty_bool: u32,
    ty_fn_void: u32,
    ptr_sb_float: u32,
    ptr_pc_uint: u32,
    const_0u: u32,
    const_1u: u32,
    const_2u: u32,
    const_3u: u32,
    const_4u: u32,
    const_5u: u32,
    const_6u: u32,
    const_7u: u32,
    var_buf_input: u32,
    var_buf_weight: u32,
    var_buf_output: u32,
    var_pc: u32,
    var_gid: u32,
}

/// Set up types, decorations, and global variables for a conv1d compute shader.
///
/// Push constant layout: { uint batch, uint in_channels, uint out_channels,
///                          uint length, uint kernel_size, uint stride, uint padding,
///                          uint dilation }
fn setup_conv1d_types(b: &mut SpirVBuilder) -> Conv1dSetup {
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays of float for storage buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer input struct.
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Buffer weight struct.
    let ty_struct_weight = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_weight, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_weight, 0, DECORATION_OFFSET, &[0]);

    // Buffer output struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint batch, uint in_channels, uint out_channels,
    //                          uint length, uint kernel_size, uint stride, uint padding,
    //                          uint dilation }
    let ty_struct_pc = b.type_struct(&[
        ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint,
    ]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    for i in 0..8u32 {
        b.member_decorate(ty_struct_pc, i, DECORATION_OFFSET, &[i * 4]);
    }

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_weight = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_weight);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_3u = b.constant_u32(ty_uint, 3);
    let const_4u = b.constant_u32(ty_uint, 4);
    let const_5u = b.constant_u32(ty_uint, 5);
    let const_6u = b.constant_u32(ty_uint, 6);
    let const_7u = b.constant_u32(ty_uint, 7);

    // Global variables.
    let var_buf_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_input, DECORATION_BINDING, &[0]);

    let var_buf_weight = b.variable_global(ptr_sb_weight, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_weight, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_weight, DECORATION_BINDING, &[1]);

    let var_buf_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_output, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    Conv1dSetup {
        ty_void,
        ty_float,
        ty_uint,
        ty_bool,
        ty_fn_void,
        ptr_sb_float,
        ptr_pc_uint,
        const_0u,
        const_1u,
        const_2u,
        const_3u,
        const_4u,
        const_5u,
        const_6u,
        const_7u,
        var_buf_input,
        var_buf_weight,
        var_buf_output,
        var_pc,
        var_gid,
    }
}

/// Common types and variables for a 2-buffer pool1d shader.
struct Pool1dSetup {
    ty_void: u32,
    ty_float: u32,
    ty_uint: u32,
    ty_bool: u32,
    ty_fn_void: u32,
    ptr_sb_float: u32,
    ptr_pc_uint: u32,
    glsl_ext: u32,
    const_0u: u32,
    const_1u: u32,
    const_2u: u32,
    const_3u: u32,
    const_4u: u32,
    const_5u: u32,
    var_buf_input: u32,
    var_buf_output: u32,
    var_pc: u32,
    var_gid: u32,
}

/// Set up types, decorations, and global variables for a pool1d compute shader.
///
/// Push constant layout: { uint batch, uint channels, uint length,
///                          uint kernel_size, uint stride, uint padding }
fn setup_pool1d_types(b: &mut SpirVBuilder) -> Pool1dSetup {
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    let glsl_ext = b.ext_inst_import("GLSL.std.450");

    // Runtime arrays of float for storage buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer input struct.
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Buffer output struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint batch, uint channels, uint length,
    //                          uint kernel_size, uint stride, uint padding }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    for i in 0..6u32 {
        b.member_decorate(ty_struct_pc, i, DECORATION_OFFSET, &[i * 4]);
    }

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_3u = b.constant_u32(ty_uint, 3);
    let const_4u = b.constant_u32(ty_uint, 4);
    let const_5u = b.constant_u32(ty_uint, 5);

    // Global variables.
    let var_buf_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_input, DECORATION_BINDING, &[0]);

    let var_buf_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_output, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    Pool1dSetup {
        ty_void,
        ty_float,
        ty_uint,
        ty_bool,
        ty_fn_void,
        ptr_sb_float,
        ptr_pc_uint,
        glsl_ext,
        const_0u,
        const_1u,
        const_2u,
        const_3u,
        const_4u,
        const_5u,
        var_buf_input,
        var_buf_output,
        var_pc,
        var_gid,
    }
}

/// Generate a SPIR-V 1.0 binary for 1D convolution.
///
/// Each thread computes one element of the output tensor by iterating over
/// `in_channels * kernel_size` weights, accumulating the dot product.
/// Dilation inserts gaps between kernel elements, increasing the effective
/// receptive field without increasing the number of parameters.
///
/// # Arguments
///
/// * `_in_channels` - Number of input channels (compile-time hint; runtime from push constants).
/// * `_out_channels` - Number of output channels.
/// * `_kernel_size` - Convolution kernel width.
/// * `_stride` - Stride of the convolution.
/// * `_padding` - Zero-padding added to both sides of the input.
/// * `_dilation` - Spacing between kernel elements (1 = standard convolution).
///
/// # Buffers
///
/// - Binding 0: input \[batch, in\_channels, length\] (row-major float\[\])
/// - Binding 1: weight \[out\_channels, in\_channels, kernel\_size\] (row-major float\[\])
/// - Binding 2: output \[batch, out\_channels, out\_length\] (row-major float\[\])
///
/// # Push constants
///
/// - `uint batch` at offset 0
/// - `uint in_channels` at offset 4
/// - `uint out_channels` at offset 8
/// - `uint length` at offset 12
/// - `uint kernel_size` at offset 16
/// - `uint stride` at offset 20
/// - `uint padding` at offset 24
/// - `uint dilation` at offset 28
///
/// # Algorithm
///
/// ```text
/// gid = gl_GlobalInvocationID.x
/// // Decode 3D output index from flat gid:
/// //   output is [batch, out_channels, out_length]
/// //   effective_ks = dilation * (kernel_size - 1) + 1
/// //   out_length = (length + 2*padding - effective_ks) / stride + 1
/// // gid -> (n, oc, ox) where:
/// //   ox = gid % out_length
/// //   oc = (gid / out_length) % out_channels
/// //   n  = gid / (out_channels * out_length)
///
/// sum = 0.0
/// for ic in 0..in_channels {
///     for k in 0..kernel_size {
///         ix = ox * stride + k * dilation - padding
///         if ix < length {  // unsigned comparison handles ix < 0 case
///             input_idx = n * (in_channels * length) + ic * length + ix
///             weight_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k
///             sum += input[input_idx] * weight[weight_idx]
///         }
///     }
/// }
/// output[gid] = sum
/// ```
pub fn generate_conv1d_spirv(
    _in_channels: u32,
    _out_channels: u32,
    _kernel_size: u32,
    _stride: u32,
    _padding: u32,
    _dilation: u32,
) -> Vec<u8> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let s = setup_conv1d_types(&mut b);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[s.var_gid]);
    b.execution_mode_local_size(func_id, CONV_WORKGROUP_SIZE, 1, 1);

    // Additional constants.
    let const_f0 = b.constant_f32(s.ty_float, 0.0);

    // Function body.
    b.func_begin(s.ty_void, func_id, FUNCTION_CONTROL_NONE, s.ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let ty_uvec3 = b.type_vector(s.ty_uint, 3);
    let loaded_gid = b.load(ty_uvec3, s.var_gid);
    let gid = b.composite_extract(s.ty_uint, loaded_gid, 0);

    // Load push constants.
    let pc_batch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_0u]);
    let dim_batch = b.load(s.ty_uint, pc_batch_ptr);
    let pc_in_ch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_1u]);
    let dim_in_ch = b.load(s.ty_uint, pc_in_ch_ptr);
    let pc_out_ch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_2u]);
    let dim_out_ch = b.load(s.ty_uint, pc_out_ch_ptr);
    let pc_length_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_3u]);
    let dim_length = b.load(s.ty_uint, pc_length_ptr);
    let pc_ks_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_4u]);
    let dim_ks = b.load(s.ty_uint, pc_ks_ptr);
    let pc_stride_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_5u]);
    let dim_stride = b.load(s.ty_uint, pc_stride_ptr);
    let pc_pad_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_6u]);
    let dim_pad = b.load(s.ty_uint, pc_pad_ptr);
    let pc_dil_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_7u]);
    let dim_dilation = b.load(s.ty_uint, pc_dil_ptr);

    // Compute effective_ks = dilation * (kernel_size - 1) + 1
    let ks_minus_1 = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(dim_ks);
        b.functions.push(s.const_1u);
        result
    };
    let dil_times_ks_m1 = b.imul(s.ty_uint, dim_dilation, ks_minus_1);
    let effective_ks = b.iadd(s.ty_uint, dil_times_ks_m1, s.const_1u);

    // Compute out_length = (length + 2*padding - effective_ks) / stride + 1
    // All unsigned: length + 2*padding >= effective_ks is assumed.
    let two = b.constant_u32(s.ty_uint, 2);
    let pad2 = b.imul(s.ty_uint, two, dim_pad);
    let len_plus_pad2 = b.iadd(s.ty_uint, dim_length, pad2);
    // Subtraction: SPIR-V ISub opcode = 130
    // We emit ISub inline since the builder does not have a method for it.
    let len_padded_minus_ks = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(len_plus_pad2);
        b.functions.push(effective_ks);
        result
    };
    // Division: OpUDiv = 134
    let out_len_minus1 = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(len_padded_minus_ks);
        b.functions.push(dim_stride);
        result
    };
    let out_len = b.iadd(s.ty_uint, out_len_minus1, s.const_1u);

    // Compute total output elements = batch * out_channels * out_length.
    let oc_times_ol = b.imul(s.ty_uint, dim_out_ch, out_len);
    let total_out = b.imul(s.ty_uint, dim_batch, oc_times_ol);

    // Bounds check: gid >= total_out -> return.
    let cmp_oob = b.u_greater_than_equal(s.ty_bool, gid, total_out);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    // Body: gid < total_out.
    b.label_with_id(body_label);

    // Decode (n, oc, ox) from flat gid.
    // ox = gid % out_len
    let ox = {
        let result = b.id();
        b.functions.push(op(5, 137)); // OpUMod
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(out_len);
        result
    };
    // tmp = gid / out_len
    let tmp = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(out_len);
        result
    };
    // oc = tmp % out_channels
    let oc = {
        let result = b.id();
        b.functions.push(op(5, 137)); // OpUMod
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(tmp);
        b.functions.push(dim_out_ch);
        result
    };
    // n = tmp / out_channels
    let n = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(tmp);
        b.functions.push(dim_out_ch);
        result
    };

    // Precompute base offsets.
    // input_base = n * (in_channels * length)
    let in_ch_times_len = b.imul(s.ty_uint, dim_in_ch, dim_length);
    let input_base = b.imul(s.ty_uint, n, in_ch_times_len);

    // weight_base = oc * (in_channels * kernel_size)
    let in_ch_times_ks = b.imul(s.ty_uint, dim_in_ch, dim_ks);
    let weight_base = b.imul(s.ty_uint, oc, in_ch_times_ks);

    // ox_times_stride = ox * stride
    let ox_times_stride = b.imul(s.ty_uint, ox, dim_stride);

    // --- Outer loop: ic in 0..in_channels ---
    let ic_loop_header = b.id();
    let ic_loop_body = b.id();
    let ic_loop_continue = b.id();
    let ic_loop_merge = b.id();

    b.branch(ic_loop_header);

    // IC loop header.
    b.label_with_id(ic_loop_header);
    b.loop_merge(ic_loop_merge, ic_loop_continue);
    let phi_ic = b.phi(s.ty_uint, &[(s.const_0u, body_label)]);
    let phi_sum_ic = b.phi(s.ty_float, &[(const_f0, body_label)]);
    let cmp_ic = b.u_less_than(s.ty_bool, phi_ic, dim_in_ch);
    b.branch_conditional(cmp_ic, ic_loop_body, ic_loop_merge);

    // IC loop body: inner loop over kernel_size.
    b.label_with_id(ic_loop_body);

    // Precompute for inner loop.
    // ic_input_offset = input_base + ic * length
    let ic_times_len = b.imul(s.ty_uint, phi_ic, dim_length);
    let ic_input_offset = b.iadd(s.ty_uint, input_base, ic_times_len);

    // ic_weight_offset = weight_base + ic * kernel_size
    let ic_times_ks = b.imul(s.ty_uint, phi_ic, dim_ks);
    let ic_weight_offset = b.iadd(s.ty_uint, weight_base, ic_times_ks);

    // --- Inner loop: k in 0..kernel_size ---
    let k_loop_header = b.id();
    let k_loop_body = b.id();
    let k_loop_continue = b.id();
    let k_loop_merge = b.id();

    b.branch(k_loop_header);

    // K loop header.
    b.label_with_id(k_loop_header);
    b.loop_merge(k_loop_merge, k_loop_continue);
    let phi_k = b.phi(s.ty_uint, &[(s.const_0u, ic_loop_body)]);
    let phi_sum_k = b.phi(s.ty_float, &[(phi_sum_ic, ic_loop_body)]);
    let cmp_k = b.u_less_than(s.ty_bool, phi_k, dim_ks);
    b.branch_conditional(cmp_k, k_loop_body, k_loop_merge);

    // K loop body.
    b.label_with_id(k_loop_body);

    // ix = ox * stride + k * dilation - padding
    // Using unsigned arithmetic: if the result underflows (ix < 0), the unsigned
    // result wraps to a large value, which will fail the ix < length bounds check.
    let k_times_dil = b.imul(s.ty_uint, phi_k, dim_dilation);
    let ox_stride_plus_k = b.iadd(s.ty_uint, ox_times_stride, k_times_dil);
    let ix = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(ox_stride_plus_k);
        b.functions.push(dim_pad);
        result
    };

    // Bounds check: if (ix < length) — unsigned, so wrapping underflow > length.
    let cmp_ix = b.u_less_than(s.ty_bool, ix, dim_length);
    let load_label = b.id();
    let skip_label = b.id();
    b.selection_merge(skip_label);
    b.branch_conditional(cmp_ix, load_label, skip_label);

    // Load and accumulate.
    b.label_with_id(load_label);

    // input_idx = ic_input_offset + ix
    let input_idx = b.iadd(s.ty_uint, ic_input_offset, ix);
    let ptr_in = b.access_chain(s.ptr_sb_float, s.var_buf_input, &[s.const_0u, input_idx]);
    let val_in = b.load(s.ty_float, ptr_in);

    // weight_idx = ic_weight_offset + k
    let weight_idx = b.iadd(s.ty_uint, ic_weight_offset, phi_k);
    let ptr_w = b.access_chain(s.ptr_sb_float, s.var_buf_weight, &[s.const_0u, weight_idx]);
    let val_w = b.load(s.ty_float, ptr_w);

    let product = b.fmul(s.ty_float, val_in, val_w);
    let new_sum_loaded = b.fadd(s.ty_float, phi_sum_k, product);

    b.branch(skip_label);

    // Skip label: phi to select the sum (loaded or unchanged).
    b.label_with_id(skip_label);
    let phi_sum_after_k = b.phi(
        s.ty_float,
        &[(new_sum_loaded, load_label), (phi_sum_k, k_loop_body)],
    );

    // k_next = k + 1
    let k_next = b.iadd(s.ty_uint, phi_k, s.const_1u);
    b.branch(k_loop_continue);

    // K loop continue.
    b.label_with_id(k_loop_continue);
    b.branch(k_loop_header);

    // Fixup K loop phis.
    fixup_phi(&mut b.functions, phi_k, k_next, k_loop_continue);
    fixup_phi(
        &mut b.functions,
        phi_sum_k,
        phi_sum_after_k,
        k_loop_continue,
    );

    // K loop merge: result of inner loop in phi_sum_k.
    b.label_with_id(k_loop_merge);

    // ic_next = ic + 1
    let ic_next = b.iadd(s.ty_uint, phi_ic, s.const_1u);
    b.branch(ic_loop_continue);

    // IC loop continue.
    b.label_with_id(ic_loop_continue);
    b.branch(ic_loop_header);

    // Fixup IC loop phis.
    fixup_phi(&mut b.functions, phi_ic, ic_next, ic_loop_continue);
    fixup_phi(&mut b.functions, phi_sum_ic, phi_sum_k, ic_loop_continue);

    // IC loop merge: store result.
    b.label_with_id(ic_loop_merge);

    // output[gid] = sum
    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_output, &[s.const_0u, gid]);
    b.store(ptr_out, phi_sum_ic);

    b.branch(return_label);

    // Return block.
    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

/// Generate a SPIR-V 1.0 binary for 1D max pooling.
///
/// Each thread computes one element of the output by finding the maximum
/// value in a sliding window over the input.
///
/// # Arguments
///
/// * `_kernel_size` - Pooling window width.
/// * `_stride` - Stride of the pooling window.
/// * `_padding` - Zero-padding added to both sides of the input.
///
/// # Buffers
///
/// - Binding 0: input \[batch, channels, length\] (row-major float\[\])
/// - Binding 1: output \[batch, channels, out\_length\] (row-major float\[\])
///
/// # Push constants
///
/// - `uint batch` at offset 0
/// - `uint channels` at offset 4
/// - `uint length` at offset 8
/// - `uint kernel_size` at offset 12
/// - `uint stride` at offset 16
/// - `uint padding` at offset 20
///
/// # Algorithm
///
/// ```text
/// gid = gl_GlobalInvocationID.x
/// out_length = (length + 2*padding - kernel_size) / stride + 1
/// ox = gid % out_length
/// ch = (gid / out_length) % channels
/// n  = gid / (channels * out_length)
///
/// max_val = -3.4028235e+38 (f32::MIN)
/// for k in 0..kernel_size {
///     ix = ox * stride + k - padding
///     if ix < length {  // unsigned comparison
///         val = input[n * channels * length + ch * length + ix]
///         max_val = fmax(max_val, val)
///     }
/// }
/// output[gid] = max_val
/// ```
pub fn generate_max_pool1d_spirv(_kernel_size: u32, _stride: u32, _padding: u32) -> Vec<u8> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let s = setup_pool1d_types(&mut b);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[s.var_gid]);
    b.execution_mode_local_size(func_id, CONV_WORKGROUP_SIZE, 1, 1);

    // Additional constants.
    let const_f_neg_max = b.constant_f32(s.ty_float, f32::MIN);

    // Function body.
    b.func_begin(s.ty_void, func_id, FUNCTION_CONTROL_NONE, s.ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let ty_uvec3 = b.type_vector(s.ty_uint, 3);
    let loaded_gid = b.load(ty_uvec3, s.var_gid);
    let gid = b.composite_extract(s.ty_uint, loaded_gid, 0);

    // Load push constants.
    let pc_batch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_0u]);
    let dim_batch = b.load(s.ty_uint, pc_batch_ptr);
    let pc_ch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_1u]);
    let dim_ch = b.load(s.ty_uint, pc_ch_ptr);
    let pc_length_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_2u]);
    let dim_length = b.load(s.ty_uint, pc_length_ptr);
    let pc_ks_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_3u]);
    let dim_ks = b.load(s.ty_uint, pc_ks_ptr);
    let pc_stride_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_4u]);
    let dim_stride = b.load(s.ty_uint, pc_stride_ptr);
    let pc_pad_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_5u]);
    let dim_pad = b.load(s.ty_uint, pc_pad_ptr);

    // Compute out_length.
    let two = b.constant_u32(s.ty_uint, 2);
    let pad2 = b.imul(s.ty_uint, two, dim_pad);
    let len_plus_pad2 = b.iadd(s.ty_uint, dim_length, pad2);
    let len_padded_minus_ks = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(len_plus_pad2);
        b.functions.push(dim_ks);
        result
    };
    let out_len_minus1 = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(len_padded_minus_ks);
        b.functions.push(dim_stride);
        result
    };
    let out_len = b.iadd(s.ty_uint, out_len_minus1, s.const_1u);

    // Total output elements = batch * channels * out_length.
    let ch_times_ol = b.imul(s.ty_uint, dim_ch, out_len);
    let total_out = b.imul(s.ty_uint, dim_batch, ch_times_ol);

    // Bounds check.
    let cmp_oob = b.u_greater_than_equal(s.ty_bool, gid, total_out);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    // Body.
    b.label_with_id(body_label);

    // Decode (n, ch, ox).
    let ox = {
        let result = b.id();
        b.functions.push(op(5, 137)); // OpUMod
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(out_len);
        result
    };
    let tmp = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(out_len);
        result
    };
    let ch = {
        let result = b.id();
        b.functions.push(op(5, 137)); // OpUMod
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(tmp);
        b.functions.push(dim_ch);
        result
    };
    let n = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(tmp);
        b.functions.push(dim_ch);
        result
    };

    // input_base = n * (channels * length) + ch * length
    let ch_times_len = b.imul(s.ty_uint, dim_ch, dim_length);
    let n_times_chl = b.imul(s.ty_uint, n, ch_times_len);
    let ch_offset = b.imul(s.ty_uint, ch, dim_length);
    let input_base = b.iadd(s.ty_uint, n_times_chl, ch_offset);

    let ox_times_stride = b.imul(s.ty_uint, ox, dim_stride);

    // --- Loop: k in 0..kernel_size ---
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    b.branch(loop_header);

    b.label_with_id(loop_header);
    b.loop_merge(loop_merge, loop_continue);
    let phi_k = b.phi(s.ty_uint, &[(s.const_0u, body_label)]);
    let phi_max = b.phi(s.ty_float, &[(const_f_neg_max, body_label)]);
    let cmp_k = b.u_less_than(s.ty_bool, phi_k, dim_ks);
    b.branch_conditional(cmp_k, loop_body, loop_merge);

    // Loop body.
    b.label_with_id(loop_body);

    // ix = ox * stride + k - padding (unsigned).
    let ox_stride_plus_k = b.iadd(s.ty_uint, ox_times_stride, phi_k);
    let ix = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(ox_stride_plus_k);
        b.functions.push(dim_pad);
        result
    };

    // Bounds check.
    let cmp_ix = b.u_less_than(s.ty_bool, ix, dim_length);
    let load_label = b.id();
    let skip_label = b.id();
    b.selection_merge(skip_label);
    b.branch_conditional(cmp_ix, load_label, skip_label);

    // Load and max.
    b.label_with_id(load_label);
    let input_idx = b.iadd(s.ty_uint, input_base, ix);
    let ptr_in = b.access_chain(s.ptr_sb_float, s.var_buf_input, &[s.const_0u, input_idx]);
    let val_in = b.load(s.ty_float, ptr_in);
    let new_max = b.ext_inst(
        s.ty_float,
        s.glsl_ext,
        GLSL_STD_450_FMAX,
        &[phi_max, val_in],
    );
    b.branch(skip_label);

    // Merge: select max.
    b.label_with_id(skip_label);
    let phi_max_after = b.phi(s.ty_float, &[(new_max, load_label), (phi_max, loop_body)]);

    let k_next = b.iadd(s.ty_uint, phi_k, s.const_1u);
    b.branch(loop_continue);

    b.label_with_id(loop_continue);
    b.branch(loop_header);

    fixup_phi(&mut b.functions, phi_k, k_next, loop_continue);
    fixup_phi(&mut b.functions, phi_max, phi_max_after, loop_continue);

    // Loop merge: store result.
    b.label_with_id(loop_merge);
    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_output, &[s.const_0u, gid]);
    b.store(ptr_out, phi_max);

    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

/// Generate a SPIR-V 1.0 binary for 1D average pooling.
///
/// Each thread computes one element of the output by averaging the values
/// in a sliding window over the input. Uses "count_include_pad" semantics:
/// the divisor is always `kernel_size`, matching PyTorch's default.
///
/// # Arguments
///
/// * `_kernel_size` - Pooling window width.
/// * `_stride` - Stride of the pooling window.
/// * `_padding` - Zero-padding added to both sides of the input.
///
/// # Buffers
///
/// - Binding 0: input \[batch, channels, length\] (row-major float\[\])
/// - Binding 1: output \[batch, channels, out\_length\] (row-major float\[\])
///
/// # Push constants
///
/// - `uint batch` at offset 0
/// - `uint channels` at offset 4
/// - `uint length` at offset 8
/// - `uint kernel_size` at offset 12
/// - `uint stride` at offset 16
/// - `uint padding` at offset 20
///
/// # Algorithm
///
/// ```text
/// gid = gl_GlobalInvocationID.x
/// out_length = (length + 2*padding - kernel_size) / stride + 1
/// ox = gid % out_length
/// ch = (gid / out_length) % channels
/// n  = gid / (channels * out_length)
///
/// sum = 0.0
/// for k in 0..kernel_size {
///     ix = ox * stride + k - padding
///     if ix < length {
///         sum += input[n * channels * length + ch * length + ix]
///     }
/// }
/// output[gid] = sum / float(kernel_size)
/// ```
pub fn generate_avg_pool1d_spirv(_kernel_size: u32, _stride: u32, _padding: u32) -> Vec<u8> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let s = setup_pool1d_types(&mut b);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[s.var_gid]);
    b.execution_mode_local_size(func_id, CONV_WORKGROUP_SIZE, 1, 1);

    // Additional constants.
    let const_f0 = b.constant_f32(s.ty_float, 0.0);

    // Function body.
    b.func_begin(s.ty_void, func_id, FUNCTION_CONTROL_NONE, s.ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let ty_uvec3 = b.type_vector(s.ty_uint, 3);
    let loaded_gid = b.load(ty_uvec3, s.var_gid);
    let gid = b.composite_extract(s.ty_uint, loaded_gid, 0);

    // Load push constants.
    let pc_batch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_0u]);
    let dim_batch = b.load(s.ty_uint, pc_batch_ptr);
    let pc_ch_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_1u]);
    let dim_ch = b.load(s.ty_uint, pc_ch_ptr);
    let pc_length_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_2u]);
    let dim_length = b.load(s.ty_uint, pc_length_ptr);
    let pc_ks_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_3u]);
    let dim_ks = b.load(s.ty_uint, pc_ks_ptr);
    let pc_stride_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_4u]);
    let dim_stride = b.load(s.ty_uint, pc_stride_ptr);
    let pc_pad_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_5u]);
    let dim_pad = b.load(s.ty_uint, pc_pad_ptr);

    // Compute out_length.
    let two = b.constant_u32(s.ty_uint, 2);
    let pad2 = b.imul(s.ty_uint, two, dim_pad);
    let len_plus_pad2 = b.iadd(s.ty_uint, dim_length, pad2);
    let len_padded_minus_ks = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(len_plus_pad2);
        b.functions.push(dim_ks);
        result
    };
    let out_len_minus1 = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(len_padded_minus_ks);
        b.functions.push(dim_stride);
        result
    };
    let out_len = b.iadd(s.ty_uint, out_len_minus1, s.const_1u);

    // Total output elements.
    let ch_times_ol = b.imul(s.ty_uint, dim_ch, out_len);
    let total_out = b.imul(s.ty_uint, dim_batch, ch_times_ol);

    // Bounds check.
    let cmp_oob = b.u_greater_than_equal(s.ty_bool, gid, total_out);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    // Body.
    b.label_with_id(body_label);

    // Decode (n, ch, ox).
    let ox = {
        let result = b.id();
        b.functions.push(op(5, 137)); // OpUMod
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(out_len);
        result
    };
    let tmp = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(out_len);
        result
    };
    let ch = {
        let result = b.id();
        b.functions.push(op(5, 137)); // OpUMod
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(tmp);
        b.functions.push(dim_ch);
        result
    };
    let n = {
        let result = b.id();
        b.functions.push(op(5, 134)); // OpUDiv
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(tmp);
        b.functions.push(dim_ch);
        result
    };

    // input_base = n * (channels * length) + ch * length
    let ch_times_len = b.imul(s.ty_uint, dim_ch, dim_length);
    let n_times_chl = b.imul(s.ty_uint, n, ch_times_len);
    let ch_offset = b.imul(s.ty_uint, ch, dim_length);
    let input_base = b.iadd(s.ty_uint, n_times_chl, ch_offset);

    let ox_times_stride = b.imul(s.ty_uint, ox, dim_stride);

    // --- Loop: k in 0..kernel_size ---
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    b.branch(loop_header);

    b.label_with_id(loop_header);
    b.loop_merge(loop_merge, loop_continue);
    let phi_k = b.phi(s.ty_uint, &[(s.const_0u, body_label)]);
    let phi_sum = b.phi(s.ty_float, &[(const_f0, body_label)]);
    let cmp_k = b.u_less_than(s.ty_bool, phi_k, dim_ks);
    b.branch_conditional(cmp_k, loop_body, loop_merge);

    // Loop body.
    b.label_with_id(loop_body);

    // ix = ox * stride + k - padding.
    let ox_stride_plus_k = b.iadd(s.ty_uint, ox_times_stride, phi_k);
    let ix = {
        let result = b.id();
        b.functions.push(op(5, 130)); // OpISub
        b.functions.push(s.ty_uint);
        b.functions.push(result);
        b.functions.push(ox_stride_plus_k);
        b.functions.push(dim_pad);
        result
    };

    // Bounds check.
    let cmp_ix = b.u_less_than(s.ty_bool, ix, dim_length);
    let load_label = b.id();
    let skip_label = b.id();
    b.selection_merge(skip_label);
    b.branch_conditional(cmp_ix, load_label, skip_label);

    // Load and accumulate.
    b.label_with_id(load_label);
    let input_idx = b.iadd(s.ty_uint, input_base, ix);
    let ptr_in = b.access_chain(s.ptr_sb_float, s.var_buf_input, &[s.const_0u, input_idx]);
    let val_in = b.load(s.ty_float, ptr_in);
    let new_sum = b.fadd(s.ty_float, phi_sum, val_in);
    b.branch(skip_label);

    // Merge.
    b.label_with_id(skip_label);
    let phi_sum_after = b.phi(s.ty_float, &[(new_sum, load_label), (phi_sum, loop_body)]);

    let k_next = b.iadd(s.ty_uint, phi_k, s.const_1u);
    b.branch(loop_continue);

    b.label_with_id(loop_continue);
    b.branch(loop_header);

    fixup_phi(&mut b.functions, phi_k, k_next, loop_continue);
    fixup_phi(&mut b.functions, phi_sum, phi_sum_after, loop_continue);

    // Loop merge: divide by kernel_size and store.
    b.label_with_id(loop_merge);

    // Convert kernel_size to float for division.
    let ks_float = b.convert_u_to_f(s.ty_float, dim_ks);
    let avg = b.fdiv(s.ty_float, phi_sum, ks_float);

    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_output, &[s.const_0u, gid]);
    b.store(ptr_out, avg);

    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

#[cfg(test)]
#[path = "spirv_conv_tests.rs"]
mod spirv_conv_tests;
