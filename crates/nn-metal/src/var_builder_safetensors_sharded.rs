// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sharded SafeTensors backend — extracted from `var_builder_safetensors.rs` (#1377).
//!
//! Supports models that split weights across multiple safetensors files
//! (e.g., Qwen3.5 with `model-00001-of-00004.safetensors`).

use nn_core::var_builder::TensorBackend;
use nn_core::{DType, Device, DynTensor, Result as TensorResult, TensorError};

use crate::safetensors::WeightMap;

/// SafeTensors backend supporting multiple (sharded) weight files.
///
/// Each shard is a separate `WeightMap` with its own mmap and Metal buffer.
/// Tensor lookups search all shards in order, returning from the first shard
/// that contains the tensor name. This matches candle's multi-file
/// `from_mmaped_safetensors` behavior for models like Qwen3.5 that split
/// weights across multiple safetensors files.
pub(crate) struct ShardedSafeTensorsBackend {
    pub(crate) shards: Vec<WeightMap>,
}

impl ShardedSafeTensorsBackend {
    fn find_shard(&self, name: &str) -> Option<&WeightMap> {
        self.shards.iter().find(|wm| wm.tensor_info(name).is_ok())
    }
}

impl TensorBackend for ShardedSafeTensorsBackend {
    fn get(
        &self,
        dims: &[usize],
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> TensorResult<DynTensor> {
        let wm = self.find_shard(name).ok_or(TensorError::TensorNotFound {
            name: name.to_string(),
        })?;
        let info = wm
            .tensor_info(name)
            .map_err(|_| TensorError::TensorNotFound {
                name: name.to_string(),
            })?;
        if info.shape.as_slice() != dims {
            return Err(TensorError::shape_mismatch(
                dims.to_vec(),
                info.shape.clone(),
            ));
        }
        super::load_tensor_from_weight_map(wm, name, &info.shape, info.dtype, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> TensorResult<DynTensor> {
        let wm = self.find_shard(name).ok_or(TensorError::TensorNotFound {
            name: name.to_string(),
        })?;
        let info = wm
            .tensor_info(name)
            .map_err(|_| TensorError::TensorNotFound {
                name: name.to_string(),
            })?;
        super::load_tensor_from_weight_map(wm, name, &info.shape, info.dtype, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.find_shard(name).is_some()
    }

    fn tensor_names(&self) -> Vec<String> {
        self.shards
            .iter()
            .flat_map(|wm| wm.tensor_names().map(String::from))
            .collect()
    }
}
