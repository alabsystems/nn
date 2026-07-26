// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for transpose compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for 2D and batched 3D transpose operations:
//!
//! - [`generate_transpose_spirv`]: 2D matrix transpose (rows x cols -> cols x rows).
//! - [`generate_batch_transpose_spirv`]: Batched 2D transpose over leading dimension.
//! - [`transpose_reference`]: CPU reference implementation for differential testing.
//!
//! All shaders use SPIR-V 1.0 for maximum Vulkan compatibility, `StorageBuffer`
//! storage class with `std430` layout, and push constants for dimensions.
//!
//! The transpose kernel uses a 2D workgroup (TRANSPOSE_WORKGROUP_SIZE x TRANSPOSE_WORKGROUP_SIZE)
//! for coalesced memory access patterns.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for transpose kernels (2D: 16x16 = 256 threads).
pub const TRANSPOSE_WORKGROUP_SIZE: u32 = 16;

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
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;

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

    fn u_greater_than_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
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

/// Generate a SPIR-V binary for 2D matrix transpose.
///
/// Transposes a `rows x cols` matrix to `cols x rows`.
/// Each thread handles one element: reads `A[row * cols + col]`, writes to
/// `B[col * rows + row]`.
///
/// # Arguments
///
/// * `rows` - Number of rows in the input matrix.
/// * `cols` - Number of columns in the input matrix.
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[rows * cols\])
/// - Binding 1: Output buffer (float\[cols * rows\])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0
/// - `uint rows` at offset 4
/// - `uint cols` at offset 8
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>` (word array).
pub fn generate_transpose_spirv(rows: u32, cols: u32) -> Vec<u32> {
    let _ = (rows, cols); // Dimensions are runtime via push constants.
    build_transpose_kernel(false)
}

/// Generate a SPIR-V binary for batched 2D matrix transpose.
///
/// Transposes `batch` matrices each of size `rows x cols` to `cols x rows`.
/// Each thread handles one element across the full batch. The batch index is
/// computed from the global invocation ID.
///
/// # Arguments
///
/// * `batch` - Number of matrices in the batch.
/// * `rows` - Number of rows per matrix.
/// * `cols` - Number of columns per matrix.
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[batch * rows * cols\])
/// - Binding 1: Output buffer (float\[batch * cols * rows\])
///
/// # Push constants
///
/// - `uint total_elements` at offset 0 (batch * rows * cols)
/// - `uint rows` at offset 4
/// - `uint cols` at offset 8
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>` (word array).
pub fn generate_batch_transpose_spirv(batch: u32, rows: u32, cols: u32) -> Vec<u32> {
    let _ = (batch, rows, cols); // Dimensions are runtime via push constants.
    build_transpose_kernel(true)
}

/// CPU reference implementation for 2D matrix transpose.
///
/// Transposes a `rows x cols` matrix stored in row-major order.
///
/// # Panics
///
/// Panics if `data.len() != rows * cols`.
pub fn transpose_reference(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        data.len(),
        rows * cols,
        "transpose_reference: data length {} != rows * cols = {}",
        data.len(),
        rows * cols
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Build the transpose SPIR-V kernel.
///
/// When `batched` is true, the kernel computes a batch index from the global
/// invocation ID and offsets input/output pointers per batch slice.
fn build_transpose_kernel(batched: bool) -> Vec<u32> {
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
    b.execution_mode_local_size(
        func_id,
        TRANSPOSE_WORKGROUP_SIZE,
        TRANSPOSE_WORKGROUP_SIZE,
        1,
    );

    // Function body.
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);
    let gid_y = b.composite_extract(ty_uint, loaded_gid, 1);

    // Load total_elements from push constants.
    let pc_total_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let total = b.load(ty_uint, pc_total_ptr);

    // Load rows and cols from push constants.
    let pc_rows_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let rows_val = b.load(ty_uint, pc_rows_ptr);
    let pc_cols_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let cols_val = b.load(ty_uint, pc_cols_ptr);

    if batched {
        // Batched: gid_x = col within matrix, gid_y encodes (batch * rows + row).
        // matrix_size = rows * cols
        let matrix_size = b.imul(ty_uint, rows_val, cols_val);
        // batch_idx = gid_y / rows
        let batch_idx = b.udiv(ty_uint, gid_y, rows_val);
        // row = gid_y % rows
        let row = b.umod(ty_uint, gid_y, rows_val);
        let col = gid_x;

        // Compute linear index = batch_idx * matrix_size + row * cols + col
        let batch_offset = b.imul(ty_uint, batch_idx, matrix_size);
        let row_offset = b.imul(ty_uint, row, cols_val);
        let src_inner = b.iadd(ty_uint, row_offset, col);
        let src_idx = b.iadd(ty_uint, batch_offset, src_inner);

        // Bounds check: src_idx >= total -> skip
        let cmp = b.u_greater_than_equal(ty_bool, src_idx, total);
        let merge_label = b.id();
        let then_label = b.id();
        b.selection_merge(merge_label);
        b.branch_conditional(cmp, merge_label, then_label);

        b.label_with_id(then_label);

        // Load from input.
        let ptr_a = b.access_chain(ptr_sb_float, var_in, &[const_0u, src_idx]);
        let val = b.load(ty_float, ptr_a);

        // Destination: batch_idx * matrix_size + col * rows + row
        let col_times_rows = b.imul(ty_uint, col, rows_val);
        let dst_inner = b.iadd(ty_uint, col_times_rows, row);
        let dst_idx = b.iadd(ty_uint, batch_offset, dst_inner);
        let ptr_b = b.access_chain(ptr_sb_float, var_out, &[const_0u, dst_idx]);
        b.store(ptr_b, val);

        b.branch(merge_label);
        b.label_with_id(merge_label);
    } else {
        // Non-batched: gid_x = col, gid_y = row.
        let row = gid_y;
        let col = gid_x;

        // Compute linear index = row * cols + col
        let row_offset = b.imul(ty_uint, row, cols_val);
        let src_idx = b.iadd(ty_uint, row_offset, col);

        // Bounds check: src_idx >= total -> skip
        let cmp = b.u_greater_than_equal(ty_bool, src_idx, total);
        let merge_label = b.id();
        let then_label = b.id();
        b.selection_merge(merge_label);
        b.branch_conditional(cmp, merge_label, then_label);

        b.label_with_id(then_label);

        // Load from input[row * cols + col].
        let ptr_a = b.access_chain(ptr_sb_float, var_in, &[const_0u, src_idx]);
        let val = b.load(ty_float, ptr_a);

        // Store to output[col * rows + row].
        let col_times_rows = b.imul(ty_uint, col, rows_val);
        let dst_idx = b.iadd(ty_uint, col_times_rows, row);
        let ptr_b = b.access_chain(ptr_sb_float, var_out, &[const_0u, dst_idx]);
        b.store(ptr_b, val);

        b.branch(merge_label);
        b.label_with_id(merge_label);
    }

    b.op_return();
    b.func_end();

    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

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

    // ---- transpose_reference ----

    #[test]
    fn test_transpose_reference_square_2x2() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = transpose_reference(&data, 2, 2);
        assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_reference_non_square_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_reference(&data, 2, 3);
        // Input:  [[1,2,3],[4,5,6]]  (2x3)
        // Output: [[1,4],[2,5],[3,6]] (3x2) stored row-major
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_reference_non_square_3x2() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_reference(&data, 3, 2);
        // Input:  [[1,2],[3,4],[5,6]]  (3x2)
        // Output: [[1,3,5],[2,4,6]]    (2x3) stored row-major
        assert_eq!(result, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_reference_single_element() {
        let data = vec![42.0];
        let result = transpose_reference(&data, 1, 1);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_transpose_reference_single_row() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = transpose_reference(&data, 1, 4);
        // 1x4 -> 4x1
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_reference_single_column() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = transpose_reference(&data, 4, 1);
        // 4x1 -> 1x4
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_reference_double_transpose_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let once = transpose_reference(&data, 2, 3);
        let twice = transpose_reference(&once, 3, 2);
        assert_eq!(twice, data, "double transpose must be identity");
    }

    // ---- generate_transpose_spirv ----

    #[test]
    fn test_transpose_spirv_header() {
        let words = generate_transpose_spirv(4, 4);
        assert_valid_header(&words, "transpose_4x4");
    }

    #[test]
    fn test_transpose_spirv_non_square_header() {
        let words = generate_transpose_spirv(3, 5);
        assert_valid_header(&words, "transpose_3x5");
    }

    #[test]
    fn test_transpose_spirv_entry_point() {
        let words = generate_transpose_spirv(8, 8);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_transpose_spirv_workgroup_size() {
        let words = generate_transpose_spirv(16, 16);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [TRANSPOSE_WORKGROUP_SIZE, TRANSPOSE_WORKGROUP_SIZE, 1]);
    }

    #[test]
    fn test_transpose_spirv_has_capability() {
        let words = generate_transpose_spirv(4, 4);
        assert!(
            has_opcode(&words, OP_CAPABILITY),
            "transpose must have OpCapability"
        );
    }

    #[test]
    fn test_transpose_spirv_has_memory_model() {
        let words = generate_transpose_spirv(4, 4);
        assert!(
            has_opcode(&words, OP_MEMORY_MODEL),
            "transpose must have OpMemoryModel"
        );
    }

    #[test]
    fn test_transpose_spirv_has_store() {
        let words = generate_transpose_spirv(4, 4);
        assert!(has_opcode(&words, OP_STORE), "transpose must have OpStore");
    }

    #[test]
    fn test_transpose_spirv_single_element() {
        let words = generate_transpose_spirv(1, 1);
        assert_valid_header(&words, "transpose_1x1");
    }

    // ---- generate_batch_transpose_spirv ----

    #[test]
    fn test_batch_transpose_spirv_header() {
        let words = generate_batch_transpose_spirv(2, 4, 4);
        assert_valid_header(&words, "batch_transpose_2x4x4");
    }

    #[test]
    fn test_batch_transpose_spirv_entry_point() {
        let words = generate_batch_transpose_spirv(3, 8, 8);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_batch_transpose_spirv_workgroup_size() {
        let words = generate_batch_transpose_spirv(2, 16, 16);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [TRANSPOSE_WORKGROUP_SIZE, TRANSPOSE_WORKGROUP_SIZE, 1]);
    }

    #[test]
    fn test_batch_transpose_spirv_has_store() {
        let words = generate_batch_transpose_spirv(2, 4, 4);
        assert!(
            has_opcode(&words, OP_STORE),
            "batch transpose must have OpStore"
        );
    }

    #[test]
    fn test_batch_transpose_spirv_has_udiv_for_batch() {
        let words = generate_batch_transpose_spirv(2, 4, 4);
        assert!(
            has_opcode(&words, OP_UDIV),
            "batch transpose must have OpUDiv for batch index computation"
        );
    }
}
