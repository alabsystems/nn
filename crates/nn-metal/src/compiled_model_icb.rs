// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal Indirect Command Buffer (ICB) replay for `CompiledModel`.
//!
//! Pre-encodes GPU dispatch commands at build time into a Metal ICB.
//! At execution time, only input buffer bindings are updated; the rest
//! is replayed via a single `executeCommandsInBuffer` call.
//!
//! Based on tinygrad's `MetalGraph` pattern
//! (`~/tinygrad-ref/tinygrad/runtime/graph/metal.py`).
//!
//! Part of #3206, #3259.

use objc::rc::autoreleasepool;
use objc::runtime::{Class, Object, Sel};
use objc::Message;

use crate::buffer::MetalBuffer;
use crate::dispatch::CommandBatch;
use crate::error::MetalError;
use crate::pipeline::ComputePipeline;

/// Send an ObjC message with no arguments to a raw Object pointer.
///
/// # Safety
/// `obj` must be a valid, non-null ObjC object pointer.
unsafe fn msg0<T: 'static>(obj: *mut Object, sel: &str) -> T {
    unsafe {
        (*obj)
            .send_message(Sel::register(sel), ())
            .expect("msg0 failed")
    }
}

/// Send an ObjC message with one argument to a raw Object pointer.
///
/// # Safety
/// `obj` must be a valid, non-null ObjC object pointer.
unsafe fn msg1<T: 'static, A: objc::Encode>(obj: *mut Object, sel: &str, a: A) -> T {
    unsafe {
        (*obj)
            .send_message(Sel::register(sel), (a,))
            .expect("msg1 failed")
    }
}

/// Wrapper around `MTLIndirectCommandBuffer`.
///
/// Pre-encodes a fixed sequence of compute dispatches. At replay time,
/// variable buffer bindings are updated via `setKernelBuffer:offset:atIndex:`
/// on individual commands, then the entire ICB is executed with a single
/// `executeCommandsInBuffer:withRange:` call.
pub(crate) struct IndirectCommandBuffer {
    /// The underlying `MTLIndirectCommandBuffer` ObjC object.
    icb: *mut Object,
    /// Number of commands pre-encoded.
    command_count: usize,
    /// Whether this device needs the M1/M2 pipeline-state fix.
    needs_pipeline_fix: bool,
    /// All unique pipeline states referenced by commands.
    pipelines: Vec<ComputePipeline>,
    /// All Metal buffers retained to prevent deallocation.
    retained_buffers: Vec<MetalBuffer>,
}

// SAFETY: The `icb` field is a `*mut Object` wrapping a Metal
// `MTLIndirectCommandBuffer` — an Objective-C object with thread-safe
// reference counting (ARC). Metal ICBs can be encoded on one thread and
// executed on another. `pipelines` and `retained_buffers` contain
// `ComputePipeline` and `MetalBuffer`, both of which are Send.
// This is the same safety argument as `WeightMap` and `MetalTensorData`.
unsafe impl Send for IndirectCommandBuffer {}

impl IndirectCommandBuffer {
    /// Create a new ICB with capacity for `max_commands` compute dispatches.
    pub(crate) fn new(
        device: &metal::DeviceRef,
        max_commands: usize,
        max_buffer_bindings: usize,
    ) -> Result<Self, MetalError> {
        if max_commands == 0 {
            return Err(MetalError::DispatchFailed("ICB: zero commands".into()));
        }

        // Cap max_commands to prevent unreasonable ICB allocation sizes.
        // Metal ICBs pre-allocate storage for all commands; unbounded values
        // could exhaust GPU memory or cause Metal driver issues.
        const MAX_ICB_COMMANDS: usize = 65536;
        if max_commands > MAX_ICB_COMMANDS {
            return Err(MetalError::DispatchFailed(format!(
                "ICB: max_commands {max_commands} exceeds limit {MAX_ICB_COMMANDS}"
            )));
        }

        // SAFETY: All ObjC message sends target valid Metal API objects:
        // - `desc_cls` is a non-null Class pointer obtained from the ObjC runtime.
        // - `desc` is null-checked after creation.
        // - `dev_ptr` is derived from a valid `&metal::DeviceRef`.
        // - `icb` is null-checked after creation.
        // - All selectors match documented Metal API signatures.
        // Wrapped in autoreleasepool for ObjC temporary object cleanup.
        autoreleasepool(|| unsafe {
            let desc_cls = Class::get("MTLIndirectCommandBufferDescriptor").ok_or_else(|| {
                MetalError::DispatchFailed(
                    "ICB: MTLIndirectCommandBufferDescriptor class not found".into(),
                )
            })?;
            let desc: *mut Object = msg0(std::ptr::from_ref(desc_cls) as *mut Object, "new");
            if desc.is_null() {
                return Err(MetalError::DispatchFailed(
                    "ICB: failed to create descriptor".into(),
                ));
            }

            // MTLIndirectCommandTypeConcurrentDispatch = (1 << 5) = 32
            let _: () = msg1(desc, "setCommandTypes:", 32u64);
            let _: () = msg1(desc, "setInheritBuffers:", false);
            let _: () = msg1(desc, "setInheritPipelineState:", false);
            let _: () = msg1(
                desc,
                "setMaxKernelBufferBindCount:",
                max_buffer_bindings as u64,
            );

            let dev_ptr = std::ptr::from_ref(device) as *mut Object;
            let icb: *mut Object = (*dev_ptr)
                .send_message(
                    Sel::register(
                        "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
                    ),
                    (desc, max_commands as u64, 0u64),
                )
                .expect("invariant: MTLDevice responds to newIndirectCommandBufferWithDescriptor");
            let _: () = msg0(desc, "release");

            if icb.is_null() {
                return Err(MetalError::DispatchFailed(
                    "ICB: device does not support indirect command buffers".into(),
                ));
            }

            let needs_fix = detect_needs_pipeline_fix(device);

            Ok(Self {
                icb,
                command_count: 0,
                needs_pipeline_fix: needs_fix,
                pipelines: Vec::new(),
                retained_buffers: Vec::new(),
            })
        })
    }

    /// Pre-encode a compute dispatch at the given command index.
    pub(crate) fn encode_command(
        &mut self,
        index: usize,
        pipeline: &ComputePipeline,
        buffers: &[(usize, &MetalBuffer, usize)],
        grid_size: [u32; 3],
        threadgroup_size: [u32; 3],
        needs_barrier: bool,
    ) -> Result<(), MetalError> {
        // Validate grid dimensions are non-zero to prevent undefined Metal behavior.
        for (i, &dim) in grid_size.iter().enumerate() {
            if dim == 0 {
                let dimension = match i {
                    0 => "grid width",
                    1 => "grid height",
                    _ => "grid depth",
                };
                return Err(MetalError::InvalidGridDimension {
                    dimension,
                    value: dim,
                });
            }
        }
        for (i, &dim) in threadgroup_size.iter().enumerate() {
            if dim == 0 {
                let dimension = match i {
                    0 => "threadgroup width",
                    1 => "threadgroup height",
                    _ => "threadgroup depth",
                };
                return Err(MetalError::InvalidGridDimension {
                    dimension,
                    value: dim,
                });
            }
        }

        // SAFETY: All ObjC message sends target valid Metal API objects:
        // - `self.icb` is a non-null ICB pointer (validated at construction).
        // - `cmd` is null-checked after retrieval from the ICB.
        // - `pipeline.inner()` returns a valid pipeline state reference.
        // - `buf.inner()` returns a valid Metal buffer reference for each binding.
        // - All selectors match documented Metal ICB API signatures.
        // Wrapped in autoreleasepool for ObjC temporary object cleanup.
        autoreleasepool(|| unsafe {
            let cmd: *mut Object = msg1(self.icb, "indirectComputeCommandAtIndex:", index as u64);
            if cmd.is_null() {
                return Err(MetalError::DispatchFailed(format!(
                    "ICB: null command at index {index}"
                )));
            }

            let pso_ref: &metal::ComputePipelineStateRef = pipeline.inner();
            let pso = std::ptr::from_ref(pso_ref) as *mut Object;
            let _: () = msg1(cmd, "setComputePipelineState:", pso);

            for &(arg_idx, buf, byte_offset) in buffers {
                let buf_ptr = std::ptr::from_ref(buf.inner()) as *mut Object;
                let _: () = (*cmd)
                    .send_message(
                        Sel::register("setKernelBuffer:offset:atIndex:"),
                        (buf_ptr, byte_offset as u64, arg_idx as u64),
                    )
                    .expect("invariant: ICB compute command responds to setKernelBuffer");
            }

            let grid = metal::MTLSize::new(
                u64::from(grid_size[0]),
                u64::from(grid_size[1]),
                u64::from(grid_size[2]),
            );
            let tg = metal::MTLSize::new(
                u64::from(threadgroup_size[0]),
                u64::from(threadgroup_size[1]),
                u64::from(threadgroup_size[2]),
            );
            let _: () = (*cmd)
                .send_message(
                    Sel::register("concurrentDispatchThreadgroups:threadsPerThreadgroup:"),
                    (grid, tg),
                )
                .expect(
                    "invariant: ICB compute command responds to concurrentDispatchThreadgroups",
                );

            if needs_barrier {
                let _: () = msg0(cmd, "setBarrier");
            }

            Ok(())
        })?;

        let pso_ref: &metal::ComputePipelineStateRef = pipeline.inner();
        let pso_ptr = std::ptr::from_ref(pso_ref).cast::<()>();
        if !self.pipelines.iter().any(|p| {
            let r: &metal::ComputePipelineStateRef = p.inner();
            std::ptr::eq(std::ptr::from_ref(r).cast::<()>(), pso_ptr)
        }) {
            self.pipelines.push(pipeline.clone());
        }

        self.command_count = self.command_count.max(index + 1);
        Ok(())
    }

    /// Update a buffer binding on an existing command (for variable inputs).
    pub(crate) fn update_buffer(
        &self,
        command_index: usize,
        arg_index: usize,
        buffer: &MetalBuffer,
        byte_offset: usize,
    ) -> Result<(), MetalError> {
        // Validate command_index is within the pre-encoded range.
        if command_index >= self.command_count {
            return Err(MetalError::DispatchFailed(format!(
                "ICB: update_buffer command_index {command_index} >= command_count {}",
                self.command_count
            )));
        }
        // Validate byte_offset does not exceed the buffer's byte length.
        if byte_offset > buffer.len() {
            return Err(MetalError::BufferBoundsExceeded {
                buffer_len: buffer.len(),
                offset: byte_offset,
                size: 0,
                role: "ICB update_buffer",
            });
        }

        // SAFETY: All ObjC message sends target valid Metal API objects:
        // - `self.icb` is a non-null ICB pointer (validated at construction).
        // - `command_index < self.command_count` validated above.
        // - `cmd` is null-checked after retrieval from the ICB.
        // - `buffer.inner()` returns a valid Metal buffer reference.
        // Wrapped in autoreleasepool for ObjC temporary object cleanup.
        autoreleasepool(|| unsafe {
            let cmd: *mut Object = msg1(
                self.icb,
                "indirectComputeCommandAtIndex:",
                command_index as u64,
            );
            if cmd.is_null() {
                return Err(MetalError::DispatchFailed(format!(
                    "ICB: null command at index {command_index}"
                )));
            }
            let buf_ptr = std::ptr::from_ref(buffer.inner()) as *mut Object;
            let _: () = (*cmd)
                .send_message(
                    Sel::register("setKernelBuffer:offset:atIndex:"),
                    (buf_ptr, byte_offset as u64, arg_index as u64),
                )
                .expect("invariant: ICB compute command responds to setKernelBuffer");
            Ok(())
        })
    }

    /// Retain a buffer to keep it alive for the ICB's lifetime.
    #[allow(dead_code)] // ICB wiring in progress (#3259)
    pub(crate) fn retain_buffer(&mut self, buffer: MetalBuffer) {
        self.retained_buffers.push(buffer);
    }

    /// Execute the pre-encoded ICB via a compute command encoder.
    ///
    /// `resources`: all Metal buffers the ICB may read from or write to.
    pub(crate) fn execute(
        &self,
        batch: &CommandBatch,
        resources: &[&MetalBuffer],
    ) -> Result<(), MetalError> {
        if self.command_count == 0 {
            return Ok(());
        }

        let encoder = batch.new_encoder()?;
        let raw = encoder.raw_encoder();
        let raw_ptr = std::ptr::from_ref(raw) as *mut Object;

        // SAFETY: All ObjC message sends target valid Metal API objects:
        // - `raw_ptr` is derived from a valid `ComputeCommandEncoder` reference.
        // - `res.inner()` returns a valid Metal buffer reference for each resource.
        // - `self.icb` is a non-null ICB pointer (validated at construction).
        // - `self.command_count > 0` checked above, so NSRange is non-empty.
        // - Pipeline states in `self.pipelines` are retained by this struct.
        // Wrapped in autoreleasepool for ObjC temporary object cleanup.
        autoreleasepool(|| unsafe {
            // Declare resource usage for all referenced buffers.
            // MTLResourceUsageRead | MTLResourceUsageWrite = 0x3
            for res in resources {
                let buf_ptr = std::ptr::from_ref(res.inner()) as *mut Object;
                let _: () = (*raw_ptr)
                    .send_message(Sel::register("useResource:usage:"), (buf_ptr, 3u64))
                    .expect("invariant: compute encoder responds to useResource:usage:");
            }

            // M1/M2 fix: zero-size dispatches to "use" each pipeline state.
            if self.needs_pipeline_fix {
                for ps in &self.pipelines {
                    raw.set_compute_pipeline_state(ps.inner());
                    let zero = metal::MTLSize::new(0, 0, 0);
                    raw.dispatch_thread_groups(zero, zero);
                }
            }

            // Execute the ICB.
            let range = NSRange {
                location: 0,
                length: self.command_count as u64,
            };
            let _: () = (*raw_ptr)
                .send_message(
                    Sel::register("executeCommandsInBuffer:withRange:"),
                    (self.icb, range),
                )
                .expect("invariant: compute encoder responds to executeCommandsInBuffer");
        });

        encoder.end_encoding();
        Ok(())
    }

    /// Number of pre-encoded commands.
    #[allow(dead_code)] // ICB wiring in progress (#3259)
    pub(crate) fn command_count(&self) -> usize {
        self.command_count
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for IndirectCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndirectCommandBuffer")
            .field("command_count", &self.command_count)
            .field("needs_pipeline_fix", &self.needs_pipeline_fix)
            .field("pipelines", &self.pipelines.len())
            .field("retained_buffers", &self.retained_buffers.len())
            .finish()
    }
}

impl Drop for IndirectCommandBuffer {
    fn drop(&mut self) {
        if !self.icb.is_null() {
            // SAFETY: `self.icb` is a non-null ObjC object pointer that was
            // allocated via `newIndirectCommandBufferWithDescriptor:` in `new()`.
            // This is the sole release call, matching the +1 retain from `new`.
            // `pipelines` and `retained_buffers` are dropped after this by Rust's
            // drop order (fields drop in declaration order), which is correct:
            // the ICB must be released before its referenced pipelines/buffers.
            unsafe {
                let _: () = msg0(self.icb, "release");
            }
        }
    }
}

/// NSRange for `executeCommandsInBuffer:withRange:`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct NSRange {
    location: u64,
    length: u64,
}

// SAFETY: NSRange is a POD struct matching the ObjC {_NSRange=QQ} layout.
unsafe impl objc::Encode for NSRange {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{_NSRange=QQ}") }
    }
}

/// Detect whether the device needs the M1/M2 ICB pipeline-state fix.
fn detect_needs_pipeline_fix(device: &metal::DeviceRef) -> bool {
    let name = device.name().to_string();
    if name.contains("M3") || name.contains("M4") || name.contains("M5") {
        return false;
    }
    true
}

#[path = "compiled_model_icb_analysis.rs"]
mod analysis;
pub(crate) use analysis::{
    analyze_gpu_dispatch_steps, analyze_icb_eligibility, compute_concurrent_barriers, IcbSegment,
};

#[path = "compiled_model_icb_preencode.rs"]
mod preencode;
pub(crate) use preencode::pre_compile_icb_segments;

#[path = "compiled_model_icb_encode.rs"]
mod encode;
pub(crate) use encode::{encode_icb_from_segment, update_icb_bindings, IcbStepBindings};

#[path = "compiled_model_icb_autocast.rs"]
mod autocast;

#[path = "compiled_model_icb_native.rs"]
mod native;

#[path = "compiled_model_icb_frame_bucket.rs"]
pub(crate) mod frame_bucket;

#[path = "compiled_model_icb_replay.rs"]
pub(crate) mod replay;
#[allow(unused_imports)] // ICB replay wiring in progress (#4264)
pub(crate) use replay::{
    IcbReplayBuffer, IcbReplayBufferStats, IcbReplayConfig, IcbReplayRecorder, IcbReplaySegment,
    ReplayPhase, ReplayStats, ShapeKey,
};
