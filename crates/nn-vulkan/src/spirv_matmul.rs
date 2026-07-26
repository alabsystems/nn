// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for matrix multiplication compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for C = A * B where:
//! - A is [M, K] (row-major)
//! - B is [K, N] (row-major)
//! - C is [M, N] (row-major)
//!
//! Two variants are provided:
//!
//! - [`generate_matmul_spirv`]: Tiled matmul using workgroup shared memory (16x16 tiles)
//!   for better memory access patterns and cache reuse.
//! - [`generate_matmul_spirv_naive`]: Simple per-element matmul without tiling, useful
//!   for correctness verification and small matrices.
//!
//! Both variants use:
//! - 3 storage buffers (A at binding 0, B at binding 1, C at binding 2)
//! - Push constants for M, N, K dimensions
//! - Bounds checking for non-power-of-2 dimensions
//! - `StorageBuffer` storage class with `std430` layout
//! - SPIR-V 1.0 for maximum Vulkan compatibility

use crate::spirv_emit::SPIRV_MAGIC;

/// Default tile size for the tiled matmul kernel (16x16 workgroup).
pub const MATMUL_TILE_SIZE: u32 = 16;

// ---- SPIR-V constants (duplicated from spirv_binary.rs to keep modules independent) ----

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
const OP_FMUL: u16 = 133;
const OP_U_LESS_THAN: u16 = 176;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_CONTROL_BARRIER: u16 = 224;

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
const BUILTIN_WORKGROUP_ID: u32 = 26;

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
const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100; // WorkgroupMemory
const MEMORY_SEMANTICS_ACQUIRE_RELEASE: u32 = 0x8; // AcquireRelease

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

    fn control_barrier(&mut self, execution: u32, memory: u32, semantics: u32) {
        self.functions.push(op(4, OP_CONTROL_BARRIER));
        self.functions.push(execution);
        self.functions.push(memory);
        self.functions.push(semantics);
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

/// Declare the standard buffer/push-constant types for a 3-buffer matmul shader.
///
/// Returns a struct with all type and variable IDs needed by both naive and tiled
/// matmul implementations.
struct MatmulSetup {
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
    var_buf_a: u32,
    var_buf_b: u32,
    var_buf_c: u32,
    var_pc: u32,
    var_gid: u32,
}

/// Set up types, decorations, and global variables for a matmul compute shader.
///
/// Push constant layout: { uint M; uint N; uint K; }
fn setup_matmul_types(b: &mut SpirVBuilder) -> MatmulSetup {
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays of float for storage buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer A struct.
    let ty_struct_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_a, 0, DECORATION_OFFSET, &[0]);

    // Buffer B struct.
    let ty_struct_b = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_b, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_b, 0, DECORATION_OFFSET, &[0]);

    // Buffer C struct (output).
    let ty_struct_c = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_c, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_c, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint M; uint N; uint K; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // Pointer types.
    let ptr_sb_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_a);
    let ptr_sb_b = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_b);
    let ptr_sb_c = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_c);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);

    // Global variables.
    let var_buf_a = b.variable_global(ptr_sb_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_a, DECORATION_BINDING, &[0]);

    let var_buf_b = b.variable_global(ptr_sb_b, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_b, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_b, DECORATION_BINDING, &[1]);

    let var_buf_c = b.variable_global(ptr_sb_c, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_c, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_c, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    MatmulSetup {
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
        var_buf_a,
        var_buf_b,
        var_buf_c,
        var_pc,
        var_gid,
    }
}

/// Generate a SPIR-V 1.0 binary for naive matrix multiplication: C = A * B.
///
/// Each thread computes one element of C by iterating over the K dimension.
/// No shared memory or tiling. Workgroup size is 16x16x1 (dispatched as 2D).
///
/// # Arguments
///
/// * `m` - Number of rows of A and C (compile-time specialization hint; actual
///   value comes from push constants at runtime).
/// * `n` - Number of columns of B and C.
/// * `k` - Shared inner dimension (columns of A, rows of B).
///
/// The `m`, `n`, `k` parameters are embedded in the SPIR-V module as documentation
/// but the actual dimensions are read from push constants at runtime. This allows
/// the same SPIR-V binary to work with any dimensions (with bounds checking).
///
/// # Buffers
///
/// - Binding 0: A \[M, K\] (row-major float\[\])
/// - Binding 1: B \[K, N\] (row-major float\[\])
/// - Binding 2: C \[M, N\] (row-major float\[\], output)
///
/// # Push constants
///
/// - `uint M` at offset 0
/// - `uint N` at offset 4
/// - `uint K` at offset 8
///
/// # Algorithm
///
/// ```text
/// row = gl_GlobalInvocationID.y
/// col = gl_GlobalInvocationID.x
/// if (row < M && col < N) {
///     sum = 0.0
///     for t in 0..K {
///         sum += A[row * K + t] * B[t * N + col]
///     }
///     C[row * N + col] = sum
/// }
/// ```
pub fn generate_matmul_spirv_naive(_m: u32, _n: u32, _k: u32) -> Vec<u8> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let s = setup_matmul_types(&mut b);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[s.var_gid]);
    b.execution_mode_local_size(func_id, MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);

    // Additional constants.
    let const_f0 = b.constant_f32(s.ty_float, 0.0);

    // Function body.
    b.func_begin(s.ty_void, func_id, FUNCTION_CONTROL_NONE, s.ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.
    let ty_uvec3 = b.type_vector(s.ty_uint, 3);
    let loaded_gid = b.load(ty_uvec3, s.var_gid);
    let col = b.composite_extract(s.ty_uint, loaded_gid, 0); // .x = column
    let row = b.composite_extract(s.ty_uint, loaded_gid, 1); // .y = row

    // Load M, N, K from push constants.
    let pc_m_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_0u]);
    let dim_m = b.load(s.ty_uint, pc_m_ptr);
    let pc_n_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_1u]);
    let dim_n = b.load(s.ty_uint, pc_n_ptr);
    let pc_k_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_2u]);
    let dim_k = b.load(s.ty_uint, pc_k_ptr);

    // Bounds check: if (row >= M || col >= N) return.
    let cmp_row = b.u_greater_than_equal(s.ty_bool, row, dim_m);
    let cmp_col = b.u_greater_than_equal(s.ty_bool, col, dim_n);
    // Combine: out_of_bounds = row >= M || col >= N
    // We use two nested branches for simplicity (SPIR-V has no OpLogicalOr
    // that we need for this; we could add it but nested branches are clearer).
    let return_label = b.id();
    let check_col_label = b.id();
    b.selection_merge(check_col_label);
    b.branch_conditional(cmp_row, return_label, check_col_label);

    // check_col block.
    b.label_with_id(check_col_label);
    let body_label = b.id();
    let return_label2 = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_col, return_label2, body_label);

    // body: row < M && col < N.
    b.label_with_id(body_label);

    // --- K-loop: accumulate sum = sum_t A[row,t] * B[t,col] ---
    // Loop structure:
    //   loop_header: phi(t, sum) + condition check
    //   loop_body: load, multiply, accumulate
    //   loop_continue: t++ -> back to header
    //   loop_merge: store result
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    b.branch(loop_header);

    // Loop header.
    b.label_with_id(loop_header);
    b.loop_merge(loop_merge, loop_continue);

    // Phi for t (loop index) and sum (accumulator).
    let phi_t = b.phi(s.ty_uint, &[(s.const_0u, body_label)]);
    let phi_sum = b.phi(s.ty_float, &[(const_f0, body_label)]);

    // Condition: t < K.
    let cmp_t = b.u_less_than(s.ty_bool, phi_t, dim_k);
    b.branch_conditional(cmp_t, loop_body, loop_merge);

    // Loop body.
    b.label_with_id(loop_body);

    // a_idx = row * K + t
    let row_times_k = b.imul(s.ty_uint, row, dim_k);
    let a_idx = b.iadd(s.ty_uint, row_times_k, phi_t);

    // b_idx = t * N + col
    let t_times_n = b.imul(s.ty_uint, phi_t, dim_n);
    let b_idx = b.iadd(s.ty_uint, t_times_n, col);

    // Load A[a_idx] and B[b_idx].
    let ptr_a = b.access_chain(s.ptr_sb_float, s.var_buf_a, &[s.const_0u, a_idx]);
    let val_a = b.load(s.ty_float, ptr_a);
    let ptr_b = b.access_chain(s.ptr_sb_float, s.var_buf_b, &[s.const_0u, b_idx]);
    let val_b = b.load(s.ty_float, ptr_b);

    // sum += A[a_idx] * B[b_idx]
    let product = b.fmul(s.ty_float, val_a, val_b);
    let new_sum = b.fadd(s.ty_float, phi_sum, product);

    // t_next = t + 1
    let t_next = b.iadd(s.ty_uint, phi_t, s.const_1u);

    b.branch(loop_continue);

    // Loop continue target.
    b.label_with_id(loop_continue);
    b.branch(loop_header);

    // Now patch the phi nodes: we need to add the back-edge operands.
    // SPIR-V requires all phi operands at the point of the phi instruction.
    // Since we built the phi before emitting the loop body, we need to
    // fixup the phi instructions to include the back-edge values.
    //
    // The phi for t was: phi(const_0u from body_label) -- needs (t_next from loop_continue)
    // The phi for sum was: phi(const_f0 from body_label) -- needs (new_sum from loop_continue)
    //
    // We fix this by rewriting the phi instructions in the functions vec.
    fixup_phi(&mut b.functions, phi_t, t_next, loop_continue);
    fixup_phi(&mut b.functions, phi_sum, new_sum, loop_continue);

    // Loop merge: store result.
    b.label_with_id(loop_merge);

    // c_idx = row * N + col
    let row_times_n = b.imul(s.ty_uint, row, dim_n);
    let c_idx = b.iadd(s.ty_uint, row_times_n, col);
    let ptr_c = b.access_chain(s.ptr_sb_float, s.var_buf_c, &[s.const_0u, c_idx]);

    // After loop, phi_sum holds the final accumulated value.
    // But we need a phi at the merge that selects phi_sum from loop_header.
    // Actually, the loop_merge is the merge of the loop, so phi_sum at
    // the point where we branched to loop_merge (from loop_header when t >= K)
    // will contain the final value. We can use phi_sum directly since
    // it dominates loop_merge through the loop structure.
    b.store(ptr_c, phi_sum);

    b.branch(return_label);

    // Return blocks.
    b.label_with_id(return_label2);
    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

/// Fixup a phi instruction to add an additional (value, parent) operand.
///
/// Finds the phi instruction that produces `phi_id` and rewrites it in place
/// to include the new back-edge operand.
fn fixup_phi(functions: &mut Vec<u32>, phi_id: u32, value: u32, parent: u32) {
    // Scan for the OpPhi that produces phi_id.
    let mut pos = 0;
    while pos < functions.len() {
        let word = functions[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;

        if word_count == 0 {
            break;
        }

        if opcode == OP_PHI && pos + 2 < functions.len() && functions[pos + 2] == phi_id {
            // Found it. Insert the new operand pair at the end of this instruction.
            let insert_pos = pos + word_count;
            functions.insert(insert_pos, parent);
            functions.insert(insert_pos, value);
            // Update word count.
            let new_wc = word_count + 2;
            functions[pos] = op(new_wc as u16, OP_PHI);
            return;
        }

        pos += word_count;
    }
}

/// Generate a SPIR-V 1.0 binary for tiled matrix multiplication: C = A * B.
///
/// Uses workgroup shared memory with tiles of size [`MATMUL_TILE_SIZE`] x
/// [`MATMUL_TILE_SIZE`] (16x16) for improved memory access patterns.
///
/// # Arguments
///
/// * `m` - Number of rows of A and C.
/// * `n` - Number of columns of B and C.
/// * `k` - Shared inner dimension.
///
/// Like the naive version, actual dimensions come from push constants at runtime.
///
/// # Buffers
///
/// - Binding 0: A \[M, K\] (row-major float\[\])
/// - Binding 1: B \[K, N\] (row-major float\[\])
/// - Binding 2: C \[M, N\] (row-major float\[\], output)
///
/// # Push constants
///
/// - `uint M` at offset 0
/// - `uint N` at offset 4
/// - `uint K` at offset 8
///
/// # Algorithm
///
/// ```text
/// shared float tile_a[TILE][TILE];
/// shared float tile_b[TILE][TILE];
///
/// row = workgroup_id.y * TILE + local_id.y
/// col = workgroup_id.x * TILE + local_id.x
/// sum = 0.0
///
/// for t in (0..K).step_by(TILE) {
///     // Cooperative load into shared memory with bounds check
///     if (row < M && t + local_id.x < K)
///         tile_a[local_id.y][local_id.x] = A[row * K + t + local_id.x]
///     else
///         tile_a[local_id.y][local_id.x] = 0.0
///
///     if (t + local_id.y < K && col < N)
///         tile_b[local_id.y][local_id.x] = B[(t + local_id.y) * N + col]
///     else
///         tile_b[local_id.y][local_id.x] = 0.0
///
///     barrier()
///
///     for kk in 0..TILE {
///         sum += tile_a[local_id.y][kk] * tile_b[kk][local_id.x]
///     }
///
///     barrier()
/// }
///
/// if (row < M && col < N)
///     C[row * N + col] = sum
/// ```
pub fn generate_matmul_spirv(_m: u32, _n: u32, _k: u32) -> Vec<u8> {
    let tile = MATMUL_TILE_SIZE;
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types (base).
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays for storage buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer structs (A, B, C).
    let ty_struct_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_a, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_b = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_b, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_b, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_c = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_c, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_c, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint M; uint N; uint K; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // Shared memory: tile_a[TILE*TILE] and tile_b[TILE*TILE] as flat arrays.
    let const_tile_sq = b.constant_u32(ty_uint, tile * tile);
    let ty_arr_float_tile = b.type_array(ty_float, const_tile_sq);
    b.decorate(ty_arr_float_tile, DECORATION_ARRAY_STRIDE, &[4]);

    // Pointer types.
    let ptr_sb_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_a);
    let ptr_sb_b = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_b);
    let ptr_sb_c = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_c);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_wg_arr = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_arr_float_tile);
    let ptr_wg_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_tile = b.constant_u32(ty_uint, tile);
    let const_f0 = b.constant_f32(ty_float, 0.0);

    // Scope/semantics constants for barriers.
    let const_scope_wg = b.constant_u32(ty_uint, SCOPE_WORKGROUP);
    let const_mem_sem = b.constant_u32(
        ty_uint,
        MEMORY_SEMANTICS_WORKGROUP | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );

    // Global variables: storage buffers.
    let var_buf_a = b.variable_global(ptr_sb_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_a, DECORATION_BINDING, &[0]);

    let var_buf_b = b.variable_global(ptr_sb_b, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_b, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_b, DECORATION_BINDING, &[1]);

    let var_buf_c = b.variable_global(ptr_sb_c, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_c, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_c, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    // Input built-ins: GlobalInvocationID, LocalInvocationID, WorkgroupID.
    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    let var_lid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);

    let var_wgid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_wgid, DECORATION_BUILTIN, &[BUILTIN_WORKGROUP_ID]);

    // Shared memory variables.
    let var_tile_a = b.variable_global(ptr_wg_arr, STORAGE_CLASS_WORKGROUP);
    let var_tile_b = b.variable_global(ptr_wg_arr, STORAGE_CLASS_WORKGROUP);

    // Entry point — interface includes all Input variables.
    b.entry_point_compute(func_id, "main", &[var_gid, var_lid, var_wgid]);
    b.execution_mode_local_size(func_id, tile, tile, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let entry_label = b.label();

    // Load built-ins.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let global_col = b.composite_extract(ty_uint, loaded_gid, 0);
    let global_row = b.composite_extract(ty_uint, loaded_gid, 1);

    let loaded_lid = b.load(ty_uvec3, var_lid);
    let local_x = b.composite_extract(ty_uint, loaded_lid, 0);
    let local_y = b.composite_extract(ty_uint, loaded_lid, 1);

    // Load dimensions from push constants.
    let pc_m_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let dim_m = b.load(ty_uint, pc_m_ptr);
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let dim_n = b.load(ty_uint, pc_n_ptr);
    let pc_k_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let dim_k = b.load(ty_uint, pc_k_ptr);

    // ---- Outer tile loop over K dimension ----
    // for (t = 0; t < K; t += TILE)
    let outer_header = b.id();
    let outer_body = b.id();
    let outer_continue = b.id();
    let outer_merge = b.id();

    b.branch(outer_header);

    // Outer loop header.
    b.label_with_id(outer_header);
    b.loop_merge(outer_merge, outer_continue);
    let phi_t = b.phi(ty_uint, &[(const_0u, entry_label)]);
    let phi_sum = b.phi(ty_float, &[(const_f0, entry_label)]);

    let cmp_outer = b.u_less_than(ty_bool, phi_t, dim_k);
    b.branch_conditional(cmp_outer, outer_body, outer_merge);

    // Outer loop body: load tiles into shared memory.
    b.label_with_id(outer_body);

    // --- Load tile_a[local_y][local_x] ---
    // a_col = t + local_x; a_row = global_row
    // if (a_row < M && a_col < K) tile_a[...] = A[a_row * K + a_col] else 0.0
    let a_col = b.iadd(ty_uint, phi_t, local_x);
    let cmp_a_row = b.u_less_than(ty_bool, global_row, dim_m);
    let cmp_a_col = b.u_less_than(ty_bool, a_col, dim_k);

    // Combined check using nested selection (simpler than logical AND).
    let tile_a_valid = b.id();
    let tile_a_invalid = b.id();
    let tile_a_merge = b.id();
    b.selection_merge(tile_a_merge);
    b.branch_conditional(cmp_a_row, tile_a_valid, tile_a_invalid);

    // tile_a valid row check.
    b.label_with_id(tile_a_valid);
    let tile_a_col_valid = b.id();
    let tile_a_col_invalid = b.id();
    let tile_a_col_merge = b.id();
    b.selection_merge(tile_a_col_merge);
    b.branch_conditional(cmp_a_col, tile_a_col_valid, tile_a_col_invalid);

    // Both checks pass: load from A.
    b.label_with_id(tile_a_col_valid);
    let a_idx = {
        let row_k = b.imul(ty_uint, global_row, dim_k);
        b.iadd(ty_uint, row_k, a_col)
    };
    let ptr_a_elem = b.access_chain(ptr_sb_float, var_buf_a, &[const_0u, a_idx]);
    let val_a = b.load(ty_float, ptr_a_elem);
    // Store to shared tile_a[local_y * TILE + local_x].
    let tile_a_idx = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, local_x)
    };
    let ptr_tile_a = b.access_chain(ptr_wg_float, var_tile_a, &[tile_a_idx]);
    b.store(ptr_tile_a, val_a);
    b.branch(tile_a_col_merge);

    // Col invalid: store 0.
    b.label_with_id(tile_a_col_invalid);
    let tile_a_idx2 = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, local_x)
    };
    let ptr_tile_a2 = b.access_chain(ptr_wg_float, var_tile_a, &[tile_a_idx2]);
    b.store(ptr_tile_a2, const_f0);
    b.branch(tile_a_col_merge);

    b.label_with_id(tile_a_col_merge);
    b.branch(tile_a_merge);

    // Row invalid: store 0.
    b.label_with_id(tile_a_invalid);
    let tile_a_idx3 = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, local_x)
    };
    let ptr_tile_a3 = b.access_chain(ptr_wg_float, var_tile_a, &[tile_a_idx3]);
    b.store(ptr_tile_a3, const_f0);
    b.branch(tile_a_merge);

    b.label_with_id(tile_a_merge);

    // --- Load tile_b[local_y][local_x] ---
    // b_row = t + local_y; b_col = global_col
    // if (b_row < K && b_col < N) tile_b[...] = B[b_row * N + b_col] else 0.0
    let b_row = b.iadd(ty_uint, phi_t, local_y);
    let cmp_b_row = b.u_less_than(ty_bool, b_row, dim_k);
    let cmp_b_col = b.u_less_than(ty_bool, global_col, dim_n);

    let tile_b_valid = b.id();
    let tile_b_invalid = b.id();
    let tile_b_merge = b.id();
    b.selection_merge(tile_b_merge);
    b.branch_conditional(cmp_b_row, tile_b_valid, tile_b_invalid);

    b.label_with_id(tile_b_valid);
    let tile_b_col_valid = b.id();
    let tile_b_col_invalid = b.id();
    let tile_b_col_merge = b.id();
    b.selection_merge(tile_b_col_merge);
    b.branch_conditional(cmp_b_col, tile_b_col_valid, tile_b_col_invalid);

    b.label_with_id(tile_b_col_valid);
    let b_idx = {
        let row_n = b.imul(ty_uint, b_row, dim_n);
        b.iadd(ty_uint, row_n, global_col)
    };
    let ptr_b_elem = b.access_chain(ptr_sb_float, var_buf_b, &[const_0u, b_idx]);
    let val_b = b.load(ty_float, ptr_b_elem);
    let tile_b_idx = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, local_x)
    };
    let ptr_tile_b = b.access_chain(ptr_wg_float, var_tile_b, &[tile_b_idx]);
    b.store(ptr_tile_b, val_b);
    b.branch(tile_b_col_merge);

    b.label_with_id(tile_b_col_invalid);
    let tile_b_idx2 = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, local_x)
    };
    let ptr_tile_b2 = b.access_chain(ptr_wg_float, var_tile_b, &[tile_b_idx2]);
    b.store(ptr_tile_b2, const_f0);
    b.branch(tile_b_col_merge);

    b.label_with_id(tile_b_col_merge);
    b.branch(tile_b_merge);

    b.label_with_id(tile_b_invalid);
    let tile_b_idx3 = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, local_x)
    };
    let ptr_tile_b3 = b.access_chain(ptr_wg_float, var_tile_b, &[tile_b_idx3]);
    b.store(ptr_tile_b3, const_f0);
    b.branch(tile_b_merge);

    b.label_with_id(tile_b_merge);

    // --- Workgroup barrier: wait for all threads to finish loading tiles ---
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // --- Inner loop: accumulate from tiles ---
    // for (kk = 0; kk < TILE; kk++) sum += tile_a[local_y][kk] * tile_b[kk][local_x]
    let inner_header = b.id();
    let inner_body = b.id();
    let inner_continue = b.id();
    let inner_merge = b.id();

    let pre_inner_label = b.id();
    b.branch(pre_inner_label);
    b.label_with_id(pre_inner_label);
    b.branch(inner_header);

    b.label_with_id(inner_header);
    b.loop_merge(inner_merge, inner_continue);
    let phi_kk = b.phi(ty_uint, &[(const_0u, pre_inner_label)]);
    let phi_inner_sum = b.phi(ty_float, &[(phi_sum, pre_inner_label)]);

    let cmp_inner = b.u_less_than(ty_bool, phi_kk, const_tile);
    b.branch_conditional(cmp_inner, inner_body, inner_merge);

    b.label_with_id(inner_body);

    // tile_a[local_y * TILE + kk]
    let ta_idx = {
        let y_tile = b.imul(ty_uint, local_y, const_tile);
        b.iadd(ty_uint, y_tile, phi_kk)
    };
    let ptr_ta = b.access_chain(ptr_wg_float, var_tile_a, &[ta_idx]);
    let va = b.load(ty_float, ptr_ta);

    // tile_b[kk * TILE + local_x]
    let tb_idx = {
        let kk_tile = b.imul(ty_uint, phi_kk, const_tile);
        b.iadd(ty_uint, kk_tile, local_x)
    };
    let ptr_tb = b.access_chain(ptr_wg_float, var_tile_b, &[tb_idx]);
    let vb = b.load(ty_float, ptr_tb);

    let prod = b.fmul(ty_float, va, vb);
    let new_inner_sum = b.fadd(ty_float, phi_inner_sum, prod);
    let kk_next = b.iadd(ty_uint, phi_kk, const_1u);

    b.branch(inner_continue);

    b.label_with_id(inner_continue);
    b.branch(inner_header);

    // Fixup inner loop phis.
    fixup_phi(&mut b.functions, phi_kk, kk_next, inner_continue);
    fixup_phi(
        &mut b.functions,
        phi_inner_sum,
        new_inner_sum,
        inner_continue,
    );

    // Inner loop merge.
    b.label_with_id(inner_merge);

    // --- Second workgroup barrier before next tile iteration ---
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // t_next = t + TILE
    let t_next = b.iadd(ty_uint, phi_t, const_tile);

    b.branch(outer_continue);

    // Outer continue.
    b.label_with_id(outer_continue);
    b.branch(outer_header);

    // Fixup outer loop phis.
    fixup_phi(&mut b.functions, phi_t, t_next, outer_continue);
    // After inner loop, phi_inner_sum holds the new accumulated sum.
    fixup_phi(&mut b.functions, phi_sum, phi_inner_sum, outer_continue);

    // Outer loop merge: store result to C if in bounds.
    b.label_with_id(outer_merge);

    let store_valid = b.id();
    let store_merge = b.id();
    let cmp_store_row = b.u_less_than(ty_bool, global_row, dim_m);
    let cmp_check_col = b.id();
    b.selection_merge(store_merge);
    b.branch_conditional(cmp_store_row, cmp_check_col, store_merge);

    b.label_with_id(cmp_check_col);
    let cmp_store_col = b.u_less_than(ty_bool, global_col, dim_n);
    b.selection_merge(store_valid);
    b.branch_conditional(cmp_store_col, store_valid, store_merge);

    b.label_with_id(store_valid);
    let c_idx = {
        let row_n = b.imul(ty_uint, global_row, dim_n);
        b.iadd(ty_uint, row_n, global_col)
    };
    let ptr_c = b.access_chain(ptr_sb_float, var_buf_c, &[const_0u, c_idx]);
    b.store(ptr_c, phi_sum);
    b.branch(store_merge);

    b.label_with_id(store_merge);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

#[cfg(test)]
#[path = "spirv_matmul_tests.rs"]
mod tests;
