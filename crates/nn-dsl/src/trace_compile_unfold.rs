// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spatial compile-op helpers: unfold, upsample1d, upsample2d,
//! adaptive_avg_pool2d, pixel_shuffle, pixel_unshuffle.
//!
//! Extracted from trace_compile to keep files under 450 lines.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode};

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorIRError;

use super::{resolve_input_shape, CompiledKernel, CompiledStep};

// -- Unfold (sliding window extraction) ----------------------------------------

/// Compile `unfold(x, dim, size, step)` as N windows of narrow + permute + reshape.
///
/// Output shape: `[d0, ..., n_windows, ..., dN, size]` where dim axis is replaced
/// by `n_windows = (d_dim - size) / step + 1` and `size` is appended at end.
pub(super) fn compile_unfold(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
    size: usize,
    step: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let dim_size = input_shape[dim];
    if step == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "unfold (step == 0)".into(),
        });
    }
    if size > dim_size {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("unfold (size {size} > dim_size {dim_size})"),
        });
    }
    let n_windows = (dim_size - size) / step + 1;
    let rank = input_shape.len();

    let mut b = TensorBlockBuilder::new("unfold");
    let input = b.add_input("input_0", input_shape);

    // After narrow: [..., size, ...] (same rank, dim axis has `size` elements).
    let mut narrow_shape = input_shape.to_vec();
    narrow_shape[dim] = size;

    // Need permute when dim is not the last axis (move dim to end for correct layout).
    let need_permute = dim < rank - 1;

    // Permute axes: move dim to end. [0,..,dim-1, dim+1,..,rank-1, dim]
    let perm_axes: Vec<usize> = if need_permute {
        let mut a: Vec<usize> = (0..rank).filter(|&x| x != dim).collect();
        a.push(dim);
        a
    } else {
        (0..rank).collect()
    };
    let perm_shape: Vec<usize> = perm_axes.iter().map(|&a| narrow_shape[a]).collect();

    // Window shape: insert 1 at position dim (window count axis).
    let mut window_shape = perm_shape.clone();
    let insert_pos = dim;
    window_shape.insert(insert_pos, 1);

    // Concat shape: same as window_shape but insert_pos = n_windows.
    let mut concat_shape = window_shape.clone();
    concat_shape[insert_pos] = n_windows;

    // Build per-window IR: narrow -> permute (if needed) -> reshape.
    let use_dim0_workaround = insert_pos == 0;
    let (cat_dim, cat_window_shape, cat_concat_shape) = if use_dim0_workaround {
        let mut ws = vec![1usize];
        ws.extend_from_slice(&window_shape);
        let mut cs = vec![1usize];
        cs.extend_from_slice(&concat_shape);
        (1, ws, cs)
    } else {
        (insert_pos, window_shape.clone(), concat_shape.clone())
    };

    let windows: Vec<_> = (0..n_windows)
        .map(|w| {
            let start = w * step;
            let narrowed = b.add_narrow(input, dim, start, size, &narrow_shape);
            let permuted = if need_permute {
                b.add_transpose(narrowed, &perm_axes, &perm_shape)
            } else {
                narrowed
            };
            if use_dim0_workaround {
                let mut padded = vec![1usize];
                padded.extend_from_slice(&window_shape);
                b.add_reshape(permuted, &padded)
            } else {
                b.add_reshape(permuted, &cat_window_shape)
            }
        })
        .collect();

    let cat = if n_windows <= 1 {
        windows.first().copied().unwrap_or(input)
    } else {
        b.add_concat(&windows, cat_dim, &cat_concat_shape)
    };

    // Strip the dim-0 workaround wrapper and ensure output matches expected shape.
    let output = if use_dim0_workaround || concat_shape != node.output_shape() {
        b.add_reshape(cat, node.output_shape())
    } else {
        cat
    };

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- Upsample1d (nearest: reshape + broadcast) ---------------------------------

/// Compile `upsample1d(x, factor)` for nearest-neighbor mode.
///
/// Decomposition: reshape (insert 1) -> broadcast -> reshape (merge).
/// `[..., T]` -> `[..., T, 1]` -> `[..., T, factor]` -> `[..., T*factor]`.
pub(super) fn compile_upsample1d(
    node: &TraceNode,
    graph: &ComputationGraph,
    factor: usize,
) -> Result<CompiledStep, TensorIRError> {
    if factor == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "upsample1d (zero factor)".into(),
        });
    }

    let input_shape = resolve_input_shape(node, 0, graph)?;
    let rank = input_shape.len();
    if rank == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "upsample1d (rank 0)".into(),
        });
    }
    let in_t = input_shape[rank - 1];

    let mut b = TensorBlockBuilder::new("upsample1d");
    let input = b.add_input("input_0", input_shape);

    // [..., T] -> [..., T, 1] (insert dim for factor)
    let mut unsq = input_shape.to_vec();
    unsq.push(1);
    let r1 = b.add_reshape(input, &unsq);

    // [..., T, 1] -> [..., T, factor] (broadcast)
    let mut exp = input_shape.to_vec();
    exp.push(factor);
    let r2 = b.add_broadcast(r1, &exp);

    // [..., T, factor] -> [..., T*factor] (merge)
    let mut out_shape = input_shape.to_vec();
    out_shape[rank - 1] =
        in_t.checked_mul(factor)
            .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
                name: format!("upsample1d (overflow: {in_t} * {factor})"),
            })?;
    let output = b.add_reshape(r2, &out_shape);

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- Upsample2d (nearest: reshape + broadcast) ---------------------------------

/// Compile `upsample2d(x, mode, scale_h, scale_w)` for nearest-neighbor mode.
///
/// Decomposition (per dimension): reshape (insert 1) -> broadcast -> reshape (merge).
/// Bilinear mode returns `UnsupportedTraceOp`.
pub(super) fn compile_upsample2d(
    node: &TraceNode,
    graph: &ComputationGraph,
    mode: &str,
    scale_h: f64,
    scale_w: f64,
) -> Result<CompiledStep, TensorIRError> {
    if mode != "nearest" {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("upsample2d (mode={mode})"),
        });
    }
    let sh = scale_h.round() as usize;
    let sw = scale_w.round() as usize;
    if sh == 0 || sw == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "upsample2d (zero scale)".into(),
        });
    }

    let input_shape = resolve_input_shape(node, 0, graph)?;
    let rank = input_shape.len();
    if rank < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "upsample2d (rank < 2)".into(),
        });
    }
    let in_h = input_shape[rank - 2];
    let in_w = input_shape[rank - 1];
    let out_h = in_h
        .checked_mul(sh)
        .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
            name: format!("upsample2d (overflow: {in_h} * {sh})"),
        })?;
    let out_w = in_w
        .checked_mul(sw)
        .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
            name: format!("upsample2d (overflow: {in_w} * {sw})"),
        })?;

    let mut b = TensorBlockBuilder::new("upsample2d");
    let input = b.add_input("input_0", input_shape);

    // Step 1: Repeat along H.
    // [..., H, W] -> [..., H, 1, W] (insert dim for scale_h)
    let mut unsq_h = input_shape.to_vec();
    unsq_h.insert(rank - 1, 1);
    let r1 = b.add_reshape(input, &unsq_h);

    // [..., H, 1, W] -> [..., H, sh, W] (broadcast)
    let mut exp_h = input_shape.to_vec();
    exp_h.insert(rank - 1, sh);
    let r2 = b.add_broadcast(r1, &exp_h);

    // [..., H, sh, W] -> [..., H*sh, W] (merge)
    let mut mid = input_shape.to_vec();
    mid[rank - 2] = out_h;
    let r3 = b.add_reshape(r2, &mid);

    // Step 2: Repeat along W.
    // [..., H*sh, W] -> [..., H*sh, W, 1]
    let mut unsq_w = mid.clone();
    unsq_w.push(1);
    let r4 = b.add_reshape(r3, &unsq_w);

    // [..., H*sh, W, 1] -> [..., H*sh, W, sw]
    let mut exp_w = mid.clone();
    exp_w.push(sw);
    let r5 = b.add_broadcast(r4, &exp_w);

    // [..., H*sh, W, sw] -> [..., H*sh, W*sw]
    let mut out_shape = input_shape.to_vec();
    out_shape[rank - 2] = out_h;
    out_shape[rank - 1] = out_w;
    let output = b.add_reshape(r5, &out_shape);

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- AdaptiveAvgPool2d (decomposed to AvgPool2d) --------------------------------

/// Compile `AdaptiveAvgPool2d { output_size }` by computing equivalent
/// AvgPool2d kernel/stride parameters.
pub(super) fn compile_adaptive_avg_pool2d(
    node: &TraceNode,
    graph: &ComputationGraph,
    output_size: &[usize; 2],
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    if input_shape.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "adaptive_avg_pool2d (rank < 2)".into(),
        });
    }
    let in_h = input_shape[input_shape.len() - 2];
    let in_w = input_shape[input_shape.len() - 1];
    let [out_h, out_w] = *output_size;
    if out_h == 0 || out_w == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "adaptive_avg_pool2d (zero output size)".into(),
        });
    }
    let stride_h = in_h / out_h;
    let stride_w = in_w / out_w;
    let kernel_h = in_h - (out_h - 1) * stride_h;
    let kernel_w = in_w - (out_w - 1) * stride_w;

    let mut b = TensorBlockBuilder::new("adaptive_avg_pool2d");
    let input = b.add_input("input_0", input_shape);
    let output = b.add_avg_pool_2d(
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        0,
        0,
        node.output_shape(),
    );
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- PixelShuffle (reshape + permute + reshape) --------------------------------

/// Compile `PixelShuffle`: `[B, C*r², H, W] -> [B, C, H*r, W*r]`.
///
/// Decomposition: reshape `[B,C,r,r,H,W]` -> permute `[B,C,H,r,W,r]`
/// -> reshape `[B,C,H*r,W*r]`.
pub(super) fn compile_pixel_shuffle(
    node: &TraceNode,
    graph: &ComputationGraph,
    upscale_factor: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    if input_shape.len() != 4 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "pixel_shuffle (rank != 4)".into(),
        });
    }
    let r = upscale_factor;
    if r == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "pixel_shuffle (upscale_factor == 0)".into(),
        });
    }
    let [bd, c_r2, h, w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let r_sq = r
        .checked_mul(r)
        .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
            name: format!("pixel_shuffle (overflow: {r} * {r})"),
        })?;
    if c_r2 % r_sq != 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("pixel_shuffle (channels {c_r2} not divisible by r²={r_sq})"),
        });
    }
    let c = c_r2 / r_sq;

    let mut b = TensorBlockBuilder::new("pixel_shuffle");
    let input = b.add_input("input_0", input_shape);
    let reshaped = b.add_reshape(input, &[bd, c, r, r, h, w]);
    let permuted = b.add_transpose(reshaped, &[0, 1, 4, 2, 5, 3], &[bd, c, h, r, w, r]);
    let output = b.add_reshape(permuted, node.output_shape());

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- PixelUnshuffle (reshape + permute + reshape) ------------------------------

/// Compile `PixelUnshuffle`: `[B, C, H*r, W*r] -> [B, C*r², H, W]`.
///
/// Inverse of PixelShuffle: reshape `[B,C,H,r,W,r]` ->
/// permute `[B,C,r,r,H,W]` -> reshape `[B,C*r²,H,W]`.
pub(super) fn compile_pixel_unshuffle(
    node: &TraceNode,
    graph: &ComputationGraph,
    downscale_factor: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    if input_shape.len() != 4 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "pixel_unshuffle (rank != 4)".into(),
        });
    }
    let r = downscale_factor;
    if r == 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "pixel_unshuffle (downscale_factor == 0)".into(),
        });
    }
    let [bd, c, hr, wr] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    if hr % r != 0 || wr % r != 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("pixel_unshuffle (H={hr} or W={wr} not divisible by r={r})"),
        });
    }
    let (h, w) = (hr / r, wr / r);

    let mut b = TensorBlockBuilder::new("pixel_unshuffle");
    let input = b.add_input("input_0", input_shape);
    let reshaped = b.add_reshape(input, &[bd, c, h, r, w, r]);
    let permuted = b.add_transpose(reshaped, &[0, 1, 3, 5, 2, 4], &[bd, c, r, r, h, w]);
    let output = b.add_reshape(permuted, node.output_shape());

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}
