// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct SPIR-V binary generation for Vulkan compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules (`Vec<u32>`) directly without any
//! external compiler (no glslangValidator, no shaderc, no naga). Each
//! function produces a complete, valid SPIR-V module with:
//!
//! - Header (magic, version, generator, bound, schema)
//! - Capability and memory model declarations
//! - Entry point and execution mode
//! - Type declarations (void, float, uint, pointers, structs, runtime arrays)
//! - Decorations (bindings, descriptor sets, offsets, array strides, block)
//! - Global variables (storage buffers, push constants, built-in variables)
//! - Function body implementing the compute kernel
//!
//! # Supported operations
//!
//! - [`emit_add_spirv`]: Element-wise addition (A + B -> C)
//! - [`emit_mul_spirv`]: Element-wise multiplication (A * B -> C)
//! - [`emit_relu_spirv`]: ReLU activation (max(0, x))
//! - [`emit_scalar_mul_spirv`]: Broadcast scalar multiply (alpha * x -> y)
//! - [`emit_transpose_spirv`]: 2D matrix transpose
//!
//! All shaders use:
//! - Workgroup size of 256 threads (1D dispatch)
//! - Push constants for tensor dimensions
//! - `StorageBuffer` storage class with `std430` layout
//! - SPIR-V 1.0 for maximum Vulkan compatibility
//!
//! # SPIR-V spec references
//!
//! - Magic: `0x07230203`
//! - Version 1.0: `0x00010000`
//! - Capabilities: `Shader` (cap 1)
//! - Memory model: `Logical` / `GLSL450`
//! - Storage class: `StorageBuffer` (12) for buffers, `PushConstant` (9)
//! - Built-in: `GlobalInvocationId` (28)

use crate::error::VulkanError;

/// SPIR-V 1.0 version word.
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;

/// Generator magic (nn-vulkan = 0x4E4E0000, "NN\0").
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

/// Workgroup size used by all generated shaders.
pub const BINARY_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V opcode helpers ----

/// Encode a SPIR-V instruction word: (word_count << 16) | opcode.
const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// SPIR-V opcodes used in this module.
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
const OP_FADD: u16 = 129;
const OP_FMUL: u16 = 133;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_EXT_INST: u16 = 12;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;

// Decoration constants.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-in constants.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage class constants.
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
    // +1 for null terminator, round up to multiple of 4.
    let word_count = (bytes.len() + 1).div_ceil(4);
    let mut words = vec![0u32; word_count];
    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i / 4;
        let byte_idx = i % 4;
        words[word_idx] |= u32::from(b) << (byte_idx * 8);
    }
    // Null terminator: the remaining bytes in the last word are already 0.
    words
}

/// SPIR-V module builder. Accumulates instructions in the correct section order.
struct SpirVBuilder {
    /// Current ID bound (next free ID).
    bound: u32,
    /// Capability declarations.
    capabilities: Vec<u32>,
    /// Extension imports (GLSL.std.450).
    extensions: Vec<u32>,
    /// Memory model.
    memory_model: Vec<u32>,
    /// Entry points.
    entry_points: Vec<u32>,
    /// Execution modes.
    execution_modes: Vec<u32>,
    /// Debug names (OpName, OpMemberName) — omitted for compactness.
    /// Annotations (decorations).
    annotations: Vec<u32>,
    /// Type declarations, constants, global variables.
    type_declarations: Vec<u32>,
    /// Function definitions.
    functions: Vec<u32>,
}

impl SpirVBuilder {
    fn new() -> Self {
        Self {
            bound: 1, // ID 0 is reserved.
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

    /// Allocate the next result ID.
    fn id(&mut self) -> u32 {
        let id = self.bound;
        self.bound += 1;
        id
    }

    /// Emit OpCapability.
    fn capability(&mut self, cap: u32) {
        self.capabilities.push(op(2, OP_CAPABILITY));
        self.capabilities.push(cap);
    }

    /// Emit OpExtInstImport and return the result ID.
    fn ext_inst_import(&mut self, name: &str) -> u32 {
        let result = self.id();
        let name_words = encode_string(name);
        let wc = 2 + name_words.len() as u16;
        self.extensions.push(op(wc, OP_EXT_INST_IMPORT));
        self.extensions.push(result);
        self.extensions.extend_from_slice(&name_words);
        result
    }

    /// Emit OpMemoryModel.
    fn memory_model(&mut self, addressing: u32, model: u32) {
        self.memory_model.push(op(3, OP_MEMORY_MODEL));
        self.memory_model.push(addressing);
        self.memory_model.push(model);
    }

    /// Emit OpEntryPoint for GLCompute.
    fn entry_point_compute(&mut self, func_id: u32, name: &str, interface_ids: &[u32]) {
        let name_words = encode_string(name);
        let wc = 3 + name_words.len() as u16 + interface_ids.len() as u16;
        self.entry_points.push(op(wc, OP_ENTRY_POINT));
        self.entry_points.push(EXECUTION_MODEL_GL_COMPUTE);
        self.entry_points.push(func_id);
        self.entry_points.extend_from_slice(&name_words);
        self.entry_points.extend_from_slice(interface_ids);
    }

    /// Emit OpExecutionMode LocalSize.
    fn execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.execution_modes.push(op(6, OP_EXECUTION_MODE));
        self.execution_modes.push(func_id);
        self.execution_modes.push(EXECUTION_MODE_LOCAL_SIZE);
        self.execution_modes.push(x);
        self.execution_modes.push(y);
        self.execution_modes.push(z);
    }

    /// Emit OpDecorate.
    fn decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let wc = 3 + operands.len() as u16;
        self.annotations.push(op(wc, OP_DECORATE));
        self.annotations.push(target);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    /// Emit OpMemberDecorate.
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

    /// Emit OpConstant for a 32-bit value.
    fn constant_u32(&mut self, type_id: u32, value: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value);
        result
    }

    /// Emit OpConstant for a 32-bit float.
    fn constant_f32(&mut self, type_id: u32, value: f32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value.to_bits());
        result
    }

    /// Emit OpVariable in the type_declarations section (global).
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

    fn u_greater_than_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN_EQUAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
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

    fn udiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn umod(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UMOD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    /// Assemble the complete SPIR-V module.
    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(256);

        // Header.
        module.push(crate::spirv_emit::SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0); // Reserved schema.

        // Sections in required order.
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

/// Shared setup for an elementwise compute shader with 2 input buffers + 1 output buffer.
///
/// Returns (builder, func_id, glsl_ext, idx, types) where:
/// - `func_id`: the main function ID (pre-allocated)
/// - `glsl_ext`: GLSL.std.450 import ID
/// - `idx`: the `gl_GlobalInvocationID.x` value as uint
/// - types hold type IDs for later use
///
/// The function body starts after the bounds-check branch (in the "then" block).
/// Caller must emit the operation, store the result, then call `finish_elementwise`.
struct ElementwiseSetup {
    builder: SpirVBuilder,
    #[allow(dead_code)]
    func_id: u32,
    glsl_ext: u32,
    /// gl_GlobalInvocationID.x as uint
    idx: u32,
    /// Type IDs.
    ty_float: u32,
    #[allow(dead_code)]
    ty_uint: u32,
    #[allow(dead_code)]
    ty_bool: u32,
    /// Pointer types for AccessChain into storage buffer float arrays.
    ptr_sb_float: u32,
    /// Pointer to push constant uint.
    #[allow(dead_code)]
    ptr_pc_uint: u32,
    /// Constant uint 0.
    const_0u: u32,
    /// Input buffer A variable.
    var_buf_a: u32,
    /// Input buffer B variable (may be unused for unary ops).
    var_buf_b: u32,
    /// Output buffer variable.
    var_buf_out: u32,
    /// Push constant variable.
    #[allow(dead_code)]
    var_pc: u32,
    /// Merge label (for the bounds-check branch).
    merge_label: u32,
    /// gl_GlobalInvocationID variable.
    #[allow(dead_code)]
    var_invocation_id: u32,
}

/// Set up a 2-input elementwise shader.
///
/// `num_input_bufs`: 1 for unary ops (ReLU), 2 for binary ops (add, mul).
fn setup_elementwise(num_input_bufs: u32) -> ElementwiseSetup {
    let mut b = SpirVBuilder::new();

    // Pre-allocate the main function ID so we can reference it in OpEntryPoint.
    let func_id = b.id();

    // Capability + extensions.
    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");

    // Memory model.
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0); // unsigned 32-bit
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays of float.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Struct for each buffer: { float data[]; }
    let ty_struct_buf_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_buf_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_buf_a, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_buf_b = if num_input_bufs >= 2 {
        let s = b.type_struct(&[ty_rtarr_float]);
        b.decorate(s, DECORATION_BLOCK, &[]);
        b.member_decorate(s, 0, DECORATION_OFFSET, &[0]);
        s
    } else {
        0 // unused
    };

    let ty_struct_buf_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_buf_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_buf_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint total_elements; }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Pointer types.
    let ptr_sb_buf_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_buf_a);
    let ptr_sb_buf_b = if num_input_bufs >= 2 {
        b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_buf_b)
    } else {
        0
    };
    let ptr_sb_buf_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_buf_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);

    // Global variables.
    let var_buf_a = b.variable_global(ptr_sb_buf_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_a, DECORATION_BINDING, &[0]);

    let var_buf_b = if num_input_bufs >= 2 {
        let v = b.variable_global(ptr_sb_buf_b, STORAGE_CLASS_STORAGE_BUFFER);
        b.decorate(v, DECORATION_DESCRIPTOR_SET, &[0]);
        b.decorate(v, DECORATION_BINDING, &[1]);
        v
    } else {
        0
    };

    let out_binding = if num_input_bufs >= 2 { 2 } else { 1 };
    let var_buf_out = b.variable_global(ptr_sb_buf_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_out, DECORATION_BINDING, &[out_binding]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_invocation_id = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(
        var_invocation_id,
        DECORATION_BUILTIN,
        &[BUILTIN_GLOBAL_INVOCATION_ID],
    );

    // Entry point — must list all Input/Output interface variables.
    let interface = vec![var_invocation_id];
    // Vulkan 1.0 with StorageBuffer requires listing storage buffer variables in the interface
    // only for SPIR-V 1.4+. For SPIR-V 1.0, only Input/Output variables are listed.
    b.entry_point_compute(func_id, "main", &interface);
    b.execution_mode_local_size(func_id, BINARY_WORKGROUP_SIZE, 1, 1);

    // --- Function body ---
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    // Load gl_GlobalInvocationID and extract .x
    let loaded_gid = b.load(ty_uvec3, var_invocation_id);
    let idx = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load total_elements from push constants.
    let pc_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let total = b.load(ty_uint, pc_ptr);

    // Bounds check: if (idx >= total_elements) return.
    let cmp = b.u_greater_than_equal(ty_bool, idx, total);
    let merge_label = b.id();
    let then_label = b.id();
    b.selection_merge(merge_label);
    b.branch_conditional(cmp, merge_label, then_label);

    // Then block (idx < total_elements — we branched here on !cmp, i.e., false path).
    b.label_with_id(then_label);

    ElementwiseSetup {
        builder: b,
        func_id,
        glsl_ext,
        idx,
        ty_float,
        ty_uint,
        ty_bool,
        ptr_sb_float,
        ptr_pc_uint,
        const_0u,
        var_buf_a,
        var_buf_b,
        var_buf_out,
        var_pc,
        merge_label,
        var_invocation_id,
    }
}

/// Finish an elementwise shader after the operation has been stored.
fn finish_elementwise(mut b: SpirVBuilder, merge_label: u32) -> Vec<u32> {
    // Branch to merge block.
    b.branch(merge_label);

    // Merge block — just return.
    b.label_with_id(merge_label);
    b.op_return();
    b.func_end();

    b.build()
}

/// Generate a SPIR-V binary for element-wise addition: `C[i] = A[i] + B[i]`.
///
/// # Buffers
///
/// - Binding 0: Input buffer A (float[])
/// - Binding 1: Input buffer B (float[])
/// - Binding 2: Output buffer C (float[])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_binary::emit_add_spirv;
/// let spirv = emit_add_spirv().unwrap();
/// assert_eq!(spirv[0], 0x07230203); // SPIR-V magic
/// ```
pub fn emit_add_spirv() -> Result<Vec<u32>, VulkanError> {
    let s = setup_elementwise(2);
    let mut b = s.builder;

    // Load A[idx].
    let ptr_a = b.access_chain(s.ptr_sb_float, s.var_buf_a, &[s.const_0u, s.idx]);
    let val_a = b.load(s.ty_float, ptr_a);

    // Load B[idx].
    let ptr_b = b.access_chain(s.ptr_sb_float, s.var_buf_b, &[s.const_0u, s.idx]);
    let val_b = b.load(s.ty_float, ptr_b);

    // C[idx] = A[idx] + B[idx].
    let result = b.fadd(s.ty_float, val_a, val_b);

    // Store to output.
    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_out, &[s.const_0u, s.idx]);
    b.store(ptr_out, result);

    Ok(finish_elementwise(b, s.merge_label))
}

/// Generate a SPIR-V binary for element-wise multiplication: `C[i] = A[i] * B[i]`.
///
/// # Buffers
///
/// - Binding 0: Input buffer A (float[])
/// - Binding 1: Input buffer B (float[])
/// - Binding 2: Output buffer C (float[])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_binary::emit_mul_spirv;
/// let spirv = emit_mul_spirv().unwrap();
/// assert_eq!(spirv[0], 0x07230203); // SPIR-V magic
/// ```
pub fn emit_mul_spirv() -> Result<Vec<u32>, VulkanError> {
    let s = setup_elementwise(2);
    let mut b = s.builder;

    // Load A[idx].
    let ptr_a = b.access_chain(s.ptr_sb_float, s.var_buf_a, &[s.const_0u, s.idx]);
    let val_a = b.load(s.ty_float, ptr_a);

    // Load B[idx].
    let ptr_b = b.access_chain(s.ptr_sb_float, s.var_buf_b, &[s.const_0u, s.idx]);
    let val_b = b.load(s.ty_float, ptr_b);

    // C[idx] = A[idx] * B[idx].
    let result = b.fmul(s.ty_float, val_a, val_b);

    // Store to output.
    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_out, &[s.const_0u, s.idx]);
    b.store(ptr_out, result);

    Ok(finish_elementwise(b, s.merge_label))
}

/// Generate a SPIR-V binary for ReLU activation: `Y[i] = max(0.0, X[i])`.
///
/// # Buffers
///
/// - Binding 0: Input buffer X (float[])
/// - Binding 1: Output buffer Y (float[])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_binary::emit_relu_spirv;
/// let spirv = emit_relu_spirv().unwrap();
/// assert_eq!(spirv[0], 0x07230203); // SPIR-V magic
/// ```
pub fn emit_relu_spirv() -> Result<Vec<u32>, VulkanError> {
    let s = setup_elementwise(1);
    let mut b = s.builder;

    // Load X[idx].
    let ptr_x = b.access_chain(s.ptr_sb_float, s.var_buf_a, &[s.const_0u, s.idx]);
    let val_x = b.load(s.ty_float, ptr_x);

    // max(0.0, x) via GLSL.std.450 FMax.
    let const_zero = b.constant_f32(s.ty_float, 0.0);
    let result = b.ext_inst(
        s.ty_float,
        s.glsl_ext,
        GLSL_STD_450_FMAX,
        &[const_zero, val_x],
    );

    // Store to output.
    let ptr_out = b.access_chain(s.ptr_sb_float, s.var_buf_out, &[s.const_0u, s.idx]);
    b.store(ptr_out, result);

    Ok(finish_elementwise(b, s.merge_label))
}

/// Generate a SPIR-V binary for broadcast scalar multiply: `Y[i] = alpha * X[i]`.
///
/// The scalar `alpha` is passed as a float in push constants at offset 4
/// (after `total_elements`).
///
/// # Buffers
///
/// - Binding 0: Input buffer X (float[])
/// - Binding 1: Output buffer Y (float[])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
/// - `float alpha` at offset 4
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_binary::emit_scalar_mul_spirv;
/// let spirv = emit_scalar_mul_spirv().unwrap();
/// assert_eq!(spirv[0], 0x07230203); // SPIR-V magic
/// ```
pub fn emit_scalar_mul_spirv() -> Result<Vec<u32>, VulkanError> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

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

    // Push constant struct: { uint total_elements; float alpha; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_float]);
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
    let ptr_pc_float = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);

    // Variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, BINARY_WORKGROUP_SIZE, 1, 1);

    // Function body.
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let idx = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load total_elements.
    let pc_total_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let total = b.load(ty_uint, pc_total_ptr);

    // Bounds check.
    let cmp = b.u_greater_than_equal(ty_bool, idx, total);
    let merge_label = b.id();
    let then_label = b.id();
    b.selection_merge(merge_label);
    b.branch_conditional(cmp, merge_label, then_label);

    b.label_with_id(then_label);

    // Load alpha from push constants.
    let pc_alpha_ptr = b.access_chain(ptr_pc_float, var_pc, &[const_1u]);
    let alpha = b.load(ty_float, pc_alpha_ptr);

    // Load X[idx].
    let ptr_x = b.access_chain(ptr_sb_float, var_in, &[const_0u, idx]);
    let val_x = b.load(ty_float, ptr_x);

    // Y[idx] = alpha * X[idx].
    let result = b.fmul(ty_float, alpha, val_x);

    // Store.
    let ptr_y = b.access_chain(ptr_sb_float, var_out, &[const_0u, idx]);
    b.store(ptr_y, result);

    Ok(finish_elementwise(b, merge_label))
}

/// Generate a SPIR-V binary for 2D matrix transpose: `B[j][i] = A[i][j]`.
///
/// Each thread transposes one element. The thread's global index is mapped
/// to `(row, col)` coordinates using `rows` and `cols` from push constants.
///
/// # Buffers
///
/// - Binding 0: Input buffer A (float[rows * cols], row-major)
/// - Binding 1: Output buffer B (float[cols * rows], row-major)
///
/// # Push constants
///
/// - `uint total_elements` at offset 0 (= rows * cols)
/// - `uint rows` at offset 4
/// - `uint cols` at offset 8
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_binary::emit_transpose_spirv;
/// let spirv = emit_transpose_spirv().unwrap();
/// assert_eq!(spirv[0], 0x07230203); // SPIR-V magic
/// ```
pub fn emit_transpose_spirv() -> Result<Vec<u32>, VulkanError> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

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

    // Push constant struct: { uint total_elements; uint rows; uint cols; }
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

    // Variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, BINARY_WORKGROUP_SIZE, 1, 1);

    // Function body.
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let idx = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load total_elements from push constants.
    let pc_total_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let total = b.load(ty_uint, pc_total_ptr);

    // Bounds check.
    let cmp = b.u_greater_than_equal(ty_bool, idx, total);
    let merge_label = b.id();
    let then_label = b.id();
    b.selection_merge(merge_label);
    b.branch_conditional(cmp, merge_label, then_label);

    b.label_with_id(then_label);

    // Load rows and cols from push constants.
    let pc_cols_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let cols = b.load(ty_uint, pc_cols_ptr);

    // Compute row = idx / cols, col = idx % cols.
    let row = b.udiv(ty_uint, idx, cols);
    let col = b.umod(ty_uint, idx, cols);

    // Load A[row * cols + col] = A[idx].
    let ptr_a = b.access_chain(ptr_sb_float, var_in, &[const_0u, idx]);
    let val = b.load(ty_float, ptr_a);

    // Load rows for computing destination index.
    let pc_rows_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let rows = b.load(ty_uint, pc_rows_ptr);

    // Store to B[col * rows + row].
    let col_times_rows = b.imul(ty_uint, col, rows);
    let dst_idx = b.iadd(ty_uint, col_times_rows, row);
    let ptr_b = b.access_chain(ptr_sb_float, var_out, &[const_0u, dst_idx]);
    b.store(ptr_b, val);

    Ok(finish_elementwise(b, merge_label))
}

/// Find the entry point name in a SPIR-V module by scanning for OpEntryPoint.
///
/// Returns the name string if found.
pub fn find_entry_point_name(spirv: &[u32]) -> Option<String> {
    if spirv.len() < 5 || spirv[0] != crate::spirv_emit::SPIRV_MAGIC {
        return None;
    }

    let mut pos = 5; // Skip header.
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;

        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }

        if opcode == OP_ENTRY_POINT {
            // OpEntryPoint: [word_count|op] execution_model func_id name... interface...
            // Name starts at word index 3.
            if word_count > 3 {
                let name_words = &spirv[pos + 3..pos + word_count];
                // Decode the null-terminated string from SPIR-V words.
                let mut bytes = Vec::new();
                'outer: for &w in name_words {
                    for byte_idx in 0..4 {
                        let b = ((w >> (byte_idx * 8)) & 0xFF) as u8;
                        if b == 0 {
                            break 'outer;
                        }
                        bytes.push(b);
                    }
                }
                return String::from_utf8(bytes).ok();
            }
        }

        pos += word_count;
    }

    None
}

/// Extract the workgroup size from OpExecutionMode LocalSize in a SPIR-V module.
///
/// Returns `[x, y, z]` if found.
pub fn find_workgroup_size(spirv: &[u32]) -> Option<[u32; 3]> {
    if spirv.len() < 5 || spirv[0] != crate::spirv_emit::SPIRV_MAGIC {
        return None;
    }

    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;

        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }

        if opcode == OP_EXECUTION_MODE && word_count >= 6 {
            let mode = spirv[pos + 2];
            if mode == EXECUTION_MODE_LOCAL_SIZE {
                return Some([spirv[pos + 3], spirv[pos + 4], spirv[pos + 5]]);
            }
        }

        pos += word_count;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_emit::SPIRV_MAGIC;

    // ---- Header validation ----

    fn assert_valid_header(spirv: &[u32], label: &str) {
        assert!(spirv.len() >= 5, "{label}: module too short");
        assert_eq!(spirv[0], SPIRV_MAGIC, "{label}: wrong magic number");
        assert_eq!(spirv[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
        assert_eq!(spirv[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
        assert!(spirv[3] > 0, "{label}: bound must be > 0");
        assert_eq!(spirv[4], 0, "{label}: schema must be 0");
    }

    // ---- Entry point ----

    fn assert_entry_point_main(spirv: &[u32], label: &str) {
        let name =
            find_entry_point_name(spirv).unwrap_or_else(|| panic!("{label}: no entry point found"));
        assert_eq!(name, "main", "{label}: entry point name must be 'main'");
    }

    // ---- Workgroup size ----

    fn assert_workgroup_size_256(spirv: &[u32], label: &str) {
        let wg = find_workgroup_size(spirv)
            .unwrap_or_else(|| panic!("{label}: no workgroup size found"));
        assert_eq!(
            wg,
            [BINARY_WORKGROUP_SIZE, 1, 1],
            "{label}: wrong workgroup size"
        );
    }

    // ---- Structural: check for key opcodes ----

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

    // ---- emit_add_spirv ----

    #[test]
    fn test_add_spirv_valid_header() {
        let spirv = emit_add_spirv().unwrap();
        assert_valid_header(&spirv, "add");
    }

    #[test]
    fn test_add_spirv_entry_point() {
        let spirv = emit_add_spirv().unwrap();
        assert_entry_point_main(&spirv, "add");
    }

    #[test]
    fn test_add_spirv_workgroup_size() {
        let spirv = emit_add_spirv().unwrap();
        assert_workgroup_size_256(&spirv, "add");
    }

    #[test]
    fn test_add_spirv_contains_fadd() {
        let spirv = emit_add_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_FADD),
            "add shader must contain OpFAdd"
        );
    }

    #[test]
    fn test_add_spirv_contains_access_chain() {
        let spirv = emit_add_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_ACCESS_CHAIN),
            "add shader must contain OpAccessChain"
        );
    }

    // ---- emit_mul_spirv ----

    #[test]
    fn test_mul_spirv_valid_header() {
        let spirv = emit_mul_spirv().unwrap();
        assert_valid_header(&spirv, "mul");
    }

    #[test]
    fn test_mul_spirv_entry_point() {
        let spirv = emit_mul_spirv().unwrap();
        assert_entry_point_main(&spirv, "mul");
    }

    #[test]
    fn test_mul_spirv_workgroup_size() {
        let spirv = emit_mul_spirv().unwrap();
        assert_workgroup_size_256(&spirv, "mul");
    }

    #[test]
    fn test_mul_spirv_contains_fmul() {
        let spirv = emit_mul_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_FMUL),
            "mul shader must contain OpFMul"
        );
    }

    // ---- emit_relu_spirv ----

    #[test]
    fn test_relu_spirv_valid_header() {
        let spirv = emit_relu_spirv().unwrap();
        assert_valid_header(&spirv, "relu");
    }

    #[test]
    fn test_relu_spirv_entry_point() {
        let spirv = emit_relu_spirv().unwrap();
        assert_entry_point_main(&spirv, "relu");
    }

    #[test]
    fn test_relu_spirv_workgroup_size() {
        let spirv = emit_relu_spirv().unwrap();
        assert_workgroup_size_256(&spirv, "relu");
    }

    #[test]
    fn test_relu_spirv_contains_ext_inst() {
        let spirv = emit_relu_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_EXT_INST),
            "relu shader must use GLSL.std.450 FMax"
        );
    }

    // ---- emit_scalar_mul_spirv ----

    #[test]
    fn test_scalar_mul_spirv_valid_header() {
        let spirv = emit_scalar_mul_spirv().unwrap();
        assert_valid_header(&spirv, "scalar_mul");
    }

    #[test]
    fn test_scalar_mul_spirv_entry_point() {
        let spirv = emit_scalar_mul_spirv().unwrap();
        assert_entry_point_main(&spirv, "scalar_mul");
    }

    #[test]
    fn test_scalar_mul_spirv_workgroup_size() {
        let spirv = emit_scalar_mul_spirv().unwrap();
        assert_workgroup_size_256(&spirv, "scalar_mul");
    }

    #[test]
    fn test_scalar_mul_spirv_contains_fmul() {
        let spirv = emit_scalar_mul_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_FMUL),
            "scalar_mul shader must contain OpFMul"
        );
    }

    // ---- emit_transpose_spirv ----

    #[test]
    fn test_transpose_spirv_valid_header() {
        let spirv = emit_transpose_spirv().unwrap();
        assert_valid_header(&spirv, "transpose");
    }

    #[test]
    fn test_transpose_spirv_entry_point() {
        let spirv = emit_transpose_spirv().unwrap();
        assert_entry_point_main(&spirv, "transpose");
    }

    #[test]
    fn test_transpose_spirv_workgroup_size() {
        let spirv = emit_transpose_spirv().unwrap();
        assert_workgroup_size_256(&spirv, "transpose");
    }

    #[test]
    fn test_transpose_spirv_contains_udiv_umod() {
        let spirv = emit_transpose_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_UDIV),
            "transpose must contain OpUDiv for row computation"
        );
        assert!(
            has_opcode(&spirv, OP_UMOD),
            "transpose must contain OpUMod for col computation"
        );
    }

    #[test]
    fn test_transpose_spirv_contains_imul() {
        let spirv = emit_transpose_spirv().unwrap();
        assert!(
            has_opcode(&spirv, OP_IMUL),
            "transpose must contain OpIMul for index calculation"
        );
    }

    // ---- Cross-cutting structural tests ----

    #[test]
    fn test_all_shaders_have_capability_shader() {
        for (name, spirv) in [
            ("add", emit_add_spirv().unwrap()),
            ("mul", emit_mul_spirv().unwrap()),
            ("relu", emit_relu_spirv().unwrap()),
            ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
            ("transpose", emit_transpose_spirv().unwrap()),
        ] {
            assert!(
                has_opcode(&spirv, OP_CAPABILITY),
                "{name}: must have OpCapability"
            );
        }
    }

    #[test]
    fn test_all_shaders_have_memory_model() {
        for (name, spirv) in [
            ("add", emit_add_spirv().unwrap()),
            ("mul", emit_mul_spirv().unwrap()),
            ("relu", emit_relu_spirv().unwrap()),
            ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
            ("transpose", emit_transpose_spirv().unwrap()),
        ] {
            assert!(
                has_opcode(&spirv, OP_MEMORY_MODEL),
                "{name}: must have OpMemoryModel"
            );
        }
    }

    #[test]
    fn test_all_shaders_have_function_structure() {
        for (name, spirv) in [
            ("add", emit_add_spirv().unwrap()),
            ("mul", emit_mul_spirv().unwrap()),
            ("relu", emit_relu_spirv().unwrap()),
            ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
            ("transpose", emit_transpose_spirv().unwrap()),
        ] {
            assert!(
                has_opcode(&spirv, OP_FUNCTION),
                "{name}: must have OpFunction"
            );
            assert!(
                has_opcode(&spirv, OP_FUNCTION_END),
                "{name}: must have OpFunctionEnd"
            );
            assert!(has_opcode(&spirv, OP_LABEL), "{name}: must have OpLabel");
            assert!(has_opcode(&spirv, OP_RETURN), "{name}: must have OpReturn");
        }
    }

    #[test]
    fn test_all_shaders_have_bounds_check() {
        for (name, spirv) in [
            ("add", emit_add_spirv().unwrap()),
            ("mul", emit_mul_spirv().unwrap()),
            ("relu", emit_relu_spirv().unwrap()),
            ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
            ("transpose", emit_transpose_spirv().unwrap()),
        ] {
            assert!(
                has_opcode(&spirv, OP_U_GREATER_THAN_EQUAL),
                "{name}: must have bounds check (OpUGreaterThanEqual)"
            );
            assert!(
                has_opcode(&spirv, OP_BRANCH_CONDITIONAL),
                "{name}: must have conditional branch for bounds check"
            );
        }
    }

    #[test]
    fn test_module_sizes_are_reasonable() {
        // Sanity check: modules should be a reasonable size (not empty, not huge).
        for (name, spirv) in [
            ("add", emit_add_spirv().unwrap()),
            ("mul", emit_mul_spirv().unwrap()),
            ("relu", emit_relu_spirv().unwrap()),
            ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
            ("transpose", emit_transpose_spirv().unwrap()),
        ] {
            assert!(
                spirv.len() > 50,
                "{name}: module too small ({} words)",
                spirv.len()
            );
            assert!(
                spirv.len() < 1000,
                "{name}: module too large ({} words)",
                spirv.len()
            );
        }
    }

    #[test]
    fn test_string_encoding() {
        let words = encode_string("main");
        // "main" = 4 bytes + 1 null = 5 bytes -> 2 words.
        assert_eq!(words.len(), 2);
        // First word: 'm' | 'a'<<8 | 'i'<<16 | 'n'<<24
        let expected0 =
            u32::from(b'm') | u32::from(b'a') << 8 | u32::from(b'i') << 16 | u32::from(b'n') << 24;
        assert_eq!(words[0], expected0);
        // Second word: null terminator padded.
        assert_eq!(words[1], 0);
    }

    #[test]
    fn test_string_encoding_empty() {
        let words = encode_string("");
        assert_eq!(words.len(), 1);
        assert_eq!(words[0], 0); // Just null terminator.
    }

    #[test]
    fn test_find_entry_point_name_invalid_module() {
        assert_eq!(find_entry_point_name(&[]), None);
        assert_eq!(find_entry_point_name(&[0, 0, 0, 0, 0]), None);
    }

    #[test]
    fn test_find_workgroup_size_invalid_module() {
        assert_eq!(find_workgroup_size(&[]), None);
        assert_eq!(find_workgroup_size(&[0, 0, 0, 0, 0]), None);
    }
}
