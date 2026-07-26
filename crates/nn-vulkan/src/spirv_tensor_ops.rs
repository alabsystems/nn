// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for tensor manipulation compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for structural tensor operations:
//!
//! - [`generate_concat_spirv`]: Concatenate two tensors along the first axis
//! - [`generate_slice_spirv`]: Slice a contiguous range from a tensor
//! - [`generate_repeat_spirv`]: Repeat each element a fixed number of times
//! - [`generate_fill_spirv`]: Fill a buffer with a constant value
//!
//! All shaders use:
//! - Workgroup size of [`TENSOR_OPS_WORKGROUP_SIZE`] threads (1D dispatch)
//! - Push constants for tensor dimensions and parameters
//! - `StorageBuffer` storage class with `std430` layout
//! - SPIR-V 1.0 for maximum Vulkan compatibility
//!
//! # Buffer layouts
//!
//! **Concat:**
//! - Binding 0 (set 0): Input A buffer `float[]` (readonly)
//! - Binding 1 (set 0): Input B buffer `float[]` (readonly)
//! - Binding 2 (set 0): Output buffer `float[]`
//! - Push constants: `{ uint n_a; uint n_b; }` — lengths of A and B
//!
//! **Slice:**
//! - Binding 0 (set 0): Input buffer `float[]` (readonly)
//! - Binding 1 (set 0): Output buffer `float[]`
//! - Push constants: `{ uint n; uint start; uint len; }` — input length, slice start, slice length
//!
//! **Repeat:**
//! - Binding 0 (set 0): Input buffer `float[]` (readonly)
//! - Binding 1 (set 0): Output buffer `float[]`
//! - Push constants: `{ uint n; uint repeats; }` — input length, repeat count
//!
//! **Fill:**
//! - Binding 0 (set 0): Output buffer `float[]`
//! - Push constants: `{ uint n; float value; }` — element count and fill value

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for tensor manipulation kernels (1D dispatch).
pub const TENSOR_OPS_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants ----

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
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_U_LESS_THAN: u16 = 176;
const OP_IADD: u16 = 128;
const OP_ISUB: u16 = 130;
const OP_UDIV: u16 = 134;

// Decoration constants.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;
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

/// SPIR-V module builder.
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

    fn memory_model_decl(&mut self, addressing: u32, model: u32) {
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

    fn variable_global(&mut self, ptr_type: u32, storage_class: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_VARIABLE));
        self.type_declarations.push(ptr_type);
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        result
    }

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

    fn u_gte(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN_EQUAL));
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

    fn iadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn isub(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_ISUB));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn udiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(256);
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

// ============================================================================
// Concat: output = [A..., B...]
// ============================================================================

/// Generate a SPIR-V 1.0 binary module for concatenation of two tensors.
///
/// The kernel writes `output[gid] = A[gid]` if `gid < n_a`, else
/// `output[gid] = B[gid - n_a]`. Total output length is `n_a + n_b`.
///
/// # Buffers
///
/// - Binding 0 (set 0): Input A buffer `float[]` (readonly)
/// - Binding 1 (set 0): Input B buffer `float[]` (readonly)
/// - Binding 2 (set 0): Output buffer `float[]`
///
/// # Push constants
///
/// - `uint n_a` at offset 0 — length of A
/// - `uint n_b` at offset 4 — length of B
#[must_use]
pub fn generate_concat_spirv(n_a: u32, n_b: u32) -> Vec<u32> {
    let _ = (n_a, n_b); // Dimensions reserved for future specialization.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input A struct.
    let ty_struct_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_a, 0, DECORATION_OFFSET, &[0]);

    // Input B struct.
    let ty_struct_b = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_b, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_b, 0, DECORATION_OFFSET, &[0]);

    // Output struct.
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n_a; uint n_b; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_a);
    let ptr_sb_b = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_b);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);

    // Global variables.
    let var_a = b.variable_global(ptr_sb_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_a, DECORATION_BINDING, &[0]);
    b.decorate(var_a, DECORATION_NON_WRITABLE, &[]);

    let var_b = b.variable_global(ptr_sb_b, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_b, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_b, DECORATION_BINDING, &[1]);
    b.decorate(var_b, DECORATION_NON_WRITABLE, &[]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, TENSOR_OPS_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n_a and n_b from push constants.
    let pc_na_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let na_val = b.load(ty_uint, pc_na_ptr);
    let pc_nb_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let nb_val = b.load(ty_uint, pc_nb_ptr);

    // total = n_a + n_b
    let total = b.iadd(ty_uint, na_val, nb_val);

    // Bounds check: if (gid_x >= total) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, total);
    let label_in_range = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_in_range);

    // In-range body.
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_in_range);

    // Branch: if (gid_x < n_a) read from A else read from B.
    let cmp_in_a = b.u_less_than(ty_bool, gid_x, na_val);
    let label_from_a = b.id();
    let label_from_b = b.id();
    let label_merge = b.id();
    b.selection_merge(label_merge);
    b.branch_conditional(cmp_in_a, label_from_a, label_from_b);

    // --- From A ---
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_from_a);
    let ptr_a_val = b.access_chain(ptr_sb_float, var_a, &[const_0u, gid_x]);
    let a_val = b.load(ty_float, ptr_a_val);
    let ptr_out_a = b.access_chain(ptr_sb_float, var_out, &[const_0u, gid_x]);
    b.store(ptr_out_a, a_val);
    b.branch(label_merge);

    // --- From B ---
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_from_b);
    let b_idx = b.isub(ty_uint, gid_x, na_val);
    let ptr_b_val = b.access_chain(ptr_sb_float, var_b, &[const_0u, b_idx]);
    let b_val = b.load(ty_float, ptr_b_val);
    let ptr_out_b = b.access_chain(ptr_sb_float, var_out, &[const_0u, gid_x]);
    b.store(ptr_out_b, b_val);
    b.branch(label_merge);

    // Merge.
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_merge);
    b.branch(label_exit);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);
    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation for concatenation of two tensors.
#[must_use]
pub fn concat_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

// ============================================================================
// Slice: output[i] = input[start + i] for i in 0..len
// ============================================================================

/// Generate a SPIR-V 1.0 binary module for slicing a tensor.
///
/// Each invocation copies one element: `output[gid] = input[start + gid]`
/// for `gid < len`.
///
/// # Buffers
///
/// - Binding 0 (set 0): Input buffer `float[]` (readonly)
/// - Binding 1 (set 0): Output buffer `float[]`
///
/// # Push constants
///
/// - `uint n` at offset 0 — total input length (reserved for future bounds checks)
/// - `uint start` at offset 4 — start index of the slice
/// - `uint len` at offset 8 — number of elements to copy
#[must_use]
pub fn generate_slice_spirv(n: u32, start: u32, len: u32) -> Vec<u32> {
    let _ = (n, start, len); // Dimensions reserved for future specialization.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input struct.
    let ty_struct_in = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Output struct.
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; uint start; uint len; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);

    // Global variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);
    b.decorate(var_in, DECORATION_NON_WRITABLE, &[]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, TENSOR_OPS_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants: start, len.
    let pc_start_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let start_val = b.load(ty_uint, pc_start_ptr);
    let pc_len_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let len_val = b.load(ty_uint, pc_len_ptr);

    // Bounds check: if (gid_x >= len) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, len_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // src_idx = start + gid_x
    let src_idx = b.iadd(ty_uint, start_val, gid_x);

    // val = input[src_idx]
    let ptr_val = b.access_chain(ptr_sb_float, var_in, &[const_0u, src_idx]);
    let val = b.load(ty_float, ptr_val);

    // output[gid_x] = val
    let ptr_out = b.access_chain(ptr_sb_float, var_out, &[const_0u, gid_x]);
    b.store(ptr_out, val);

    b.branch(label_exit);
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);
    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation for tensor slice.
///
/// Returns `input[start..start+len]`.
///
/// # Panics
///
/// Panics if `start + len > input.len()`.
#[must_use]
pub fn slice_reference(input: &[f32], start: usize, len: usize) -> Vec<f32> {
    assert!(
        start + len <= input.len(),
        "slice out of bounds: start={start}, len={len}, input.len()={}",
        input.len()
    );
    input[start..start + len].to_vec()
}

// ============================================================================
// Repeat: output[i * repeats + j] = input[i] for j in 0..repeats
// ============================================================================

/// Generate a SPIR-V 1.0 binary module for repeating each element.
///
/// Output has `n * repeats` elements. Thread `gid` computes:
/// - `src_idx = gid / repeats`
/// - `output[gid] = input[src_idx]`
///
/// # Buffers
///
/// - Binding 0 (set 0): Input buffer `float[]` (readonly)
/// - Binding 1 (set 0): Output buffer `float[]`
///
/// # Push constants
///
/// - `uint n` at offset 0 — input length
/// - `uint repeats` at offset 4 — repeat count per element
#[must_use]
pub fn generate_repeat_spirv(n: u32, repeats: u32) -> Vec<u32> {
    let _ = (n, repeats); // Dimensions reserved for future specialization.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input struct.
    let ty_struct_in = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Output struct.
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; uint repeats; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);

    // Global variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);
    b.decorate(var_in, DECORATION_NON_WRITABLE, &[]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, TENSOR_OPS_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants: n, repeats.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);
    let pc_r_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let r_val = b.load(ty_uint, pc_r_ptr);

    // total = n * repeats (computed via iadd chain would be complex; use
    // the output dispatch size for bounds instead).
    // Simpler: compute total_out on CPU side. We use n * repeats for bounds.
    // Actually we just need: output_len. Dispatch already controls invocation count.
    // We use the standard pattern: total = n * repeats, check gid_x >= total.
    // But we don't have IMUL in our opcode set — we can compute n * repeats
    // by repeated add... or we just add IMUL. Let's use a different approach:
    // the dispatch is sized to n*repeats, so just use a simple in-bounds check
    // with the total computed by the host. We pass n and repeats to compute src_idx.

    // For bounds, we trust dispatch sizing (standard SPIR-V pattern). The host
    // dispatches ceil(n*repeats / WORKGROUP_SIZE) workgroups. We do a soft check
    // using n and repeats.
    //
    // We need OpIMul — add it as a raw instruction since the builder doesn't have it.
    // Actually, let's just add IMUL to the builder. We have the opcode constant.
    // OpIMul: opcode 132
    let total_out = imul_raw(&mut b, ty_uint, n_val, r_val);

    // Bounds check: if (gid_x >= total_out) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, total_out);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // src_idx = gid_x / repeats
    let src_idx = b.udiv(ty_uint, gid_x, r_val);

    // val = input[src_idx]
    let ptr_val = b.access_chain(ptr_sb_float, var_in, &[const_0u, src_idx]);
    let val = b.load(ty_float, ptr_val);

    // output[gid_x] = val
    let ptr_out = b.access_chain(ptr_sb_float, var_out, &[const_0u, gid_x]);
    b.store(ptr_out, val);

    b.branch(label_exit);
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);
    b.op_return();
    b.func_end();

    b.build()
}

/// Emit OpIMul (opcode 132) on the SpirVBuilder's function section.
fn imul_raw(b: &mut SpirVBuilder, result_type: u32, a: u32, b_val: u32) -> u32 {
    const OP_IMUL: u16 = 132;
    let result = b.id();
    b.functions.push(op(5, OP_IMUL));
    b.functions.push(result_type);
    b.functions.push(result);
    b.functions.push(a);
    b.functions.push(b_val);
    result
}

/// CPU reference implementation for repeat.
///
/// Each element in `input` is repeated `repeats` times consecutively.
/// Output length is `input.len() * repeats`.
#[must_use]
pub fn repeat_reference(input: &[f32], repeats: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() * repeats);
    for &val in input {
        for _ in 0..repeats {
            out.push(val);
        }
    }
    out
}

// ============================================================================
// Fill: output[i] = value for i in 0..n
// ============================================================================

/// Generate a SPIR-V 1.0 binary module for filling a buffer with a constant.
///
/// Each invocation writes: `output[gid] = value`.
///
/// # Buffers
///
/// - Binding 0 (set 0): Output buffer `float[]`
///
/// # Push constants
///
/// - `uint n` at offset 0 — number of elements
/// - `float value` at offset 4 — fill value
#[must_use]
pub fn generate_fill_spirv(n: u32, value: f32) -> Vec<u32> {
    let _ = (n, value); // Dimensions reserved for future specialization.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Output struct.
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; float value; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_float]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_pc_float = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);

    // Global variables.
    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[0]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, TENSOR_OPS_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);

    // Load value from push constants.
    let pc_val_ptr = b.access_chain(ptr_pc_float, var_pc, &[const_1u]);
    let fill_val = b.load(ty_float, pc_val_ptr);

    // Bounds check: if (gid_x >= n) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, n_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // output[gid_x] = value
    let ptr_out_elem = b.access_chain(ptr_sb_float, var_out, &[const_0u, gid_x]);
    b.store(ptr_out_elem, fill_val);

    b.branch(label_exit);
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);
    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation for fill.
///
/// Returns a vector of `n` elements, all set to `value`.
#[must_use]
pub fn fill_reference(n: usize, value: f32) -> Vec<f32> {
    vec![value; n]
}
