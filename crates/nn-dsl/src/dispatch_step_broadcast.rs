// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Broadcast types for `DispatchStep` binary ops.

use crate::tensor_ir::BroadcastAlignment;

/// Describes which operand needs broadcast indexing and its shape info.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryBroadcastInfo {
    /// Which operand is the broadcast (smaller) input.
    pub side: BroadcastSide,
    /// Shape of the broadcast input (before replication).
    pub input_shape: Vec<usize>,
    /// Shape of the output (= shape of the flat operand).
    pub output_shape: Vec<usize>,
    /// Broadcast alignment (Left or Right).
    pub alignment: BroadcastAlignment,
}

/// Which side of a binary op is the broadcast (smaller) operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BroadcastSide {
    Left,
    Right,
}
