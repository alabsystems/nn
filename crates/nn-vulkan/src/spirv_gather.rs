// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for gather and scatter compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for index-based memory operations:
//!
//! - [`generate_gather_spirv`]: Gather: `output[i] = input[indices[i]]`
//! - [`generate_scatter_spirv`]: Scatter: `output[indices[i]] = values[i]`
//!
//! All shaders use:
//! - Configurable workgroup size (1D dispatch)
//! - Push constants for element count
//! - `StorageBuffer` storage class with `std430` layout
//! - SPIR-V 1.0 for maximum Vulkan compatibility
//!
//! # Buffer layouts
//!
//! **Gather:**
//! - Binding 0 (set 0): Input buffer `float[]` (readonly)
//! - Binding 1 (set 0): Indices buffer `uint[]` (readonly)
//! - Binding 2 (set 0): Output buffer `float[]`
//! - Push constants: `{ uint n; }` — total elements in input (for bounds)
//!
//! **Scatter:**
//! - Binding 0 (set 0): Values buffer `float[]` (readonly)
//! - Binding 1 (set 0): Indices buffer `uint[]` (readonly)
//! - Binding 2 (set 0): Output buffer `float[]`
//! - Push constants: `{ uint n; }` — number of scatter operations (= index_count)

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for gather/scatter kernels (1D dispatch).
pub const GATHER_WORKGROUP_SIZE: u32 = 256;

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

/// Generate a SPIR-V 1.0 binary module for gather: `output[i] = input[indices[i]]`.
///
/// Each invocation reads one index from the indices buffer and gathers the
/// corresponding value from the input buffer into the output buffer.
///
/// # Buffers
///
/// - Binding 0 (set 0): Input buffer `float[]` (readonly)
/// - Binding 1 (set 0): Indices buffer `uint[]` (readonly)
/// - Binding 2 (set 0): Output buffer `float[]`
///
/// # Push constants
///
/// - `uint n` at offset 0 — total elements in input (for documentation; bounds
///   check uses `index_count` via dispatch size)
///
/// # Arguments
///
/// * `_n` — Total number of elements in the input buffer (reserved for future
///   specialization; currently used only for documentation).
/// * `_index_count` — Number of indices to gather (reserved for future use;
///   dispatch size controls invocation count).
#[must_use]
pub fn generate_gather_spirv(_n: u32, _index_count: u32) -> Vec<u32> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    // Capability + extensions.
    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    let _ = glsl_ext;

    // Memory model.
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    let ty_rtarr_uint = b.type_runtime_array(ty_uint);
    b.decorate(ty_rtarr_uint, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (float[], readonly).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Indices buffer struct (uint[], readonly).
    let ty_struct_indices = b.type_struct(&[ty_rtarr_uint]);
    b.decorate(ty_struct_indices, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_indices, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct (float[]).
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_indices = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_indices);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_sb_uint = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_indices = b.variable_global(ptr_sb_indices, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_indices, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_indices, DECORATION_BINDING, &[1]);
    b.decorate(var_indices, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, GATHER_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);

    // Bounds check: if (gid_x >= n) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, n_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    // Body label.
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // idx = indices[gid_x]
    let ptr_idx = b.access_chain(ptr_sb_uint, var_indices, &[const_0u, gid_x]);
    let idx = b.load(ty_uint, ptr_idx);

    // val = input[idx]
    let ptr_in = b.access_chain(ptr_sb_float, var_input, &[const_0u, idx]);
    let val = b.load(ty_float, ptr_in);

    // output[gid_x] = val
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(ptr_out, val);

    // Branch to exit.
    b.branch(label_exit);
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);
    b.op_return();
    b.func_end();

    b.build()
}

/// Generate a SPIR-V 1.0 binary module for scatter: `output[indices[i]] = values[i]`.
///
/// Each invocation writes one value into the output buffer at the location
/// specified by the corresponding index.
///
/// # Buffers
///
/// - Binding 0 (set 0): Values buffer `float[]` (readonly)
/// - Binding 1 (set 0): Indices buffer `uint[]` (readonly)
/// - Binding 2 (set 0): Output buffer `float[]`
///
/// # Push constants
///
/// - `uint n` at offset 0 — number of scatter operations (= index_count)
///
/// # Arguments
///
/// * `_n` — Total number of elements in the output buffer (reserved for future use).
/// * `_index_count` — Number of indices to scatter (reserved for future use;
///   dispatch size controls invocation count).
#[must_use]
pub fn generate_scatter_spirv(_n: u32, _index_count: u32) -> Vec<u32> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    let _ = glsl_ext;

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

    let ty_rtarr_uint = b.type_runtime_array(ty_uint);
    b.decorate(ty_rtarr_uint, DECORATION_ARRAY_STRIDE, &[4]);

    // Values buffer struct (float[], readonly).
    let ty_struct_values = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_values, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_values, 0, DECORATION_OFFSET, &[0]);

    // Indices buffer struct (uint[], readonly).
    let ty_struct_indices = b.type_struct(&[ty_rtarr_uint]);
    b.decorate(ty_struct_indices, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_indices, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct (float[]).
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Pointer types.
    let ptr_sb_values = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_values);
    let ptr_sb_indices = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_indices);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_sb_uint = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    let const_0u = b.constant_u32(ty_uint, 0);

    // Global variables.
    let var_values = b.variable_global(ptr_sb_values, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_values, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_values, DECORATION_BINDING, &[0]);
    b.decorate(var_values, DECORATION_NON_WRITABLE, &[]);

    let var_indices = b.variable_global(ptr_sb_indices, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_indices, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_indices, DECORATION_BINDING, &[1]);
    b.decorate(var_indices, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, GATHER_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);

    // Bounds check: if (gid_x >= n) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, n_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // val = values[gid_x]
    let ptr_val = b.access_chain(ptr_sb_float, var_values, &[const_0u, gid_x]);
    let val = b.load(ty_float, ptr_val);

    // idx = indices[gid_x]
    let ptr_idx = b.access_chain(ptr_sb_uint, var_indices, &[const_0u, gid_x]);
    let idx = b.load(ty_uint, ptr_idx);

    // output[idx] = val
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, idx]);
    b.store(ptr_out, val);

    b.branch(label_exit);
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);
    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation for gather: `output[i] = input[indices[i]]`.
///
/// # Panics
///
/// Panics if any index in `indices` is out of bounds for `input`.
#[must_use]
pub fn gather_reference(input: &[f32], indices: &[u32]) -> Vec<f32> {
    indices
        .iter()
        .map(|&idx| {
            let i = idx as usize;
            assert!(
                i < input.len(),
                "gather index {i} out of bounds for input of length {}",
                input.len()
            );
            input[i]
        })
        .collect()
}

/// CPU reference implementation for scatter: `output[indices[i]] = values[i]`.
///
/// Creates a zero-initialized output buffer of `output_size` elements and writes
/// each value at the position given by the corresponding index. If multiple
/// values map to the same index, the last one wins (same as GPU scatter with
/// no atomics).
///
/// # Panics
///
/// Panics if any index in `indices` is out of bounds for the output.
#[must_use]
pub fn scatter_reference(values: &[f32], indices: &[u32], output_size: usize) -> Vec<f32> {
    assert_eq!(
        values.len(),
        indices.len(),
        "scatter: values and indices must have the same length"
    );
    let mut output = vec![0.0f32; output_size];
    for (&val, &idx) in values.iter().zip(indices.iter()) {
        let i = idx as usize;
        assert!(
            i < output_size,
            "scatter index {i} out of bounds for output of size {output_size}"
        );
        output[i] = val;
    }
    output
}
