// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Argument extraction helpers for the aten op mapper.
//!
//! Extracted from `op_map.rs` for file-size compliance (#2575).
//! Used by `op_map_impl.rs`, `op_map_impl_ext.rs`, and `op_map_expand.rs`.

use nn_core::dyn_tensor::trace::WeightRef;

use super::{Argument, ImportError, Node, OpMapContext};

pub(super) fn get_arg<'a>(node: &'a Node, name: &str) -> Result<&'a Argument, ImportError> {
    node.inputs
        .iter()
        .find(|a| a.name == name)
        .map(|a| &a.arg)
        .ok_or_else(|| ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: name.to_string(),
        })
}

pub(super) fn require_int(node: &Node, name: &str) -> Result<i64, ImportError> {
    get_arg(node, name)?
        .as_int()
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: name.to_string(),
            expected: "int",
            actual: "non-int".to_string(),
        })
}

pub(super) fn require_ints(node: &Node, name: &str) -> Result<Vec<i64>, ImportError> {
    get_arg(node, name)?
        .as_ints()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: name.to_string(),
            expected: "int list",
            actual: "non-int-list".to_string(),
        })
}

pub(super) fn require_tensor_name(node: &Node, name: &str) -> Result<String, ImportError> {
    get_arg(node, name)?
        .as_tensor_name()
        .map(String::from)
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: name.to_string(),
            expected: "tensor",
            actual: "non-tensor".to_string(),
        })
}

pub(super) fn optional_int(node: &Node, name: &str) -> Option<i64> {
    get_arg(node, name)
        .ok()
        .and_then(|a| if a.is_none() { None } else { a.as_int() })
}

pub(super) fn optional_float(node: &Node, name: &str) -> Option<f64> {
    get_arg(node, name)
        .ok()
        .and_then(|a| if a.is_none() { None } else { a.as_float() })
}

pub(super) fn optional_bool(node: &Node, name: &str, default: bool) -> bool {
    get_arg(node, name)
        .ok()
        .and_then(Argument::as_bool_val)
        .unwrap_or(default)
}

pub(super) fn first_tensor_name(node: &Node) -> Result<String, ImportError> {
    node.inputs
        .first()
        .and_then(|na| na.arg.as_tensor_name())
        .map(String::from)
        .ok_or_else(|| ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "input".to_string(),
        })
}

pub(super) fn resolve_weight(name: &str, ctx: &OpMapContext<'_>) -> Result<WeightRef, ImportError> {
    ctx.weights
        .get(name)
        .map(|w| WeightRef::new(w.data.clone(), w.shape.clone()))
        .transpose()
        .map_err(ImportError::Tensor)?
        .ok_or_else(|| ImportError::MissingWeight {
            fqn: name.to_string(),
        })
}

pub(super) fn optional_weight(name: Option<&str>, ctx: &OpMapContext<'_>) -> Option<WeightRef> {
    name.and_then(|n| ctx.weights.get(n))
        .and_then(|w| WeightRef::new(w.data.clone(), w.shape.clone()).ok())
}

// --- Safe i64 → usize conversion (#2355) ---

pub(super) fn safe_usize(val: i64, arg_name: &str, op: &str) -> Result<usize, ImportError> {
    usize::try_from(val).map_err(|_| ImportError::NegativeDimension {
        op_target: op.to_string(),
        arg_name: arg_name.to_string(),
        value: val,
    })
}

/// Resolve a potentially negative dimension index using the tensor's rank.
///
/// PyTorch uses -1 for "last dim", -2 for "second to last", etc.
/// If `ndim` is 0 (shape unknown), falls back to `safe_usize`.
pub(super) fn resolve_dim(
    val: i64,
    ndim: usize,
    arg_name: &str,
    op: &str,
) -> Result<usize, ImportError> {
    if val >= 0 {
        safe_usize(val, arg_name, op)
    } else if ndim > 0 {
        let resolved = val + ndim as i64;
        safe_usize(resolved, arg_name, op)
    } else {
        // ndim unknown, can't resolve negative dim.
        Err(ImportError::NegativeDimension {
            op_target: op.to_string(),
            arg_name: arg_name.to_string(),
            value: val,
        })
    }
}

pub(super) fn safe_usize_vec(
    vals: Vec<i64>,
    arg_name: &str,
    op: &str,
) -> Result<Vec<usize>, ImportError> {
    vals.into_iter()
        .map(|v| safe_usize(v, arg_name, op))
        .collect()
}

pub(super) fn safe_usize_allow_neg1(
    val: i64,
    arg_name: &str,
    op: &str,
) -> Result<usize, ImportError> {
    if val == -1 {
        Ok(usize::MAX)
    } else {
        safe_usize(val, arg_name, op)
    }
}

pub(super) fn require_single_dim(
    dims: &[i64],
    _arg_name: &str,
    op_kind: &'static str,
    op_target: &str,
) -> Result<i64, ImportError> {
    if dims.len() > 1 {
        return Err(ImportError::MultiAxisNotSupported {
            op_target: op_target.to_string(),
            op_kind,
            dims: dims.to_vec(),
        });
    }
    Ok(dims.first().copied().unwrap_or(0))
}

pub(super) fn reduce_params(node: &Node) -> Result<(String, usize, bool), ImportError> {
    let input = first_tensor_name(node)?;
    let dims = require_ints(node, "dim")?;
    let keepdim = optional_bool(node, "keepdim", false);
    let raw = require_single_dim(&dims, "dim", "reduction", &node.target)?;
    let dim = safe_usize(raw, "dim", &node.target)?;
    Ok((input, dim, keepdim))
}

pub(super) fn parse_pool1d_params(node: &Node) -> Result<(usize, usize, usize), ImportError> {
    let t = &node.target;
    let ks = require_ints(node, "kernel_size")?;
    let stride = get_arg(node, "stride")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| ks.clone());
    let padding = get_arg(node, "padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0]);
    Ok((
        safe_usize(ks[0], "kernel_size", t)?,
        safe_usize(stride[0], "stride", t)?,
        safe_usize(padding[0], "padding", t)?,
    ))
}

pub(super) fn parse_pool2d_params(
    node: &Node,
) -> Result<([usize; 2], [usize; 2], [usize; 2]), ImportError> {
    let t = &node.target;
    let ks = require_ints(node, "kernel_size")?;
    let stride = get_arg(node, "stride")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| ks.clone());
    let padding = get_arg(node, "padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0, 0]);
    let pair = |v: &[i64], name| -> Result<[usize; 2], ImportError> {
        Ok([
            safe_usize(v[0], name, t)?,
            safe_usize(v.get(1).copied().unwrap_or(v[0]), name, t)?,
        ])
    };
    Ok((
        pair(&ks, "kernel_size")?,
        pair(&stride, "stride")?,
        pair(&padding, "padding")?,
    ))
}

/// Map ScalarType integer to [`nn_core::DType`].
///
/// Returns an error for unrecognized scalar types instead of silently
/// defaulting to f32 (which would corrupt models using int/bool dtypes).
pub(super) fn scalar_type_to_name(
    st: i32,
    op_target: &str,
) -> Result<nn_core::DType, ImportError> {
    match st {
        1 => Ok(nn_core::DType::U8),
        5 => Ok(nn_core::DType::I64),
        6 => Ok(nn_core::DType::F16),
        7 => Ok(nn_core::DType::F32),
        8 => Ok(nn_core::DType::F64),
        13 => Ok(nn_core::DType::BF16),
        _ => Err(ImportError::WrongArgumentType {
            op_target: op_target.to_string(),
            arg_name: "dtype".to_string(),
            expected: "known ScalarType (1=u8, 5=i64, 6=f16, 7=f32, 8=f64, 13=bf16)",
            actual: format!("ScalarType({st})"),
        }),
    }
}
