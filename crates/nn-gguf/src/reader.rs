// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF file reader — the main entry point.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use memmap2::Mmap;

use crate::dequant::{self, GgufDType};
use crate::error::GgufError;
use crate::header::GgufHeader;
use crate::metadata::GgufMetadata;
use crate::tensor_info::GgufTensorInfo;

/// Maximum number of tensors we will accept from a GGUF header.
///
/// Even the largest models (Llama-405B at 4-bit) have ~600 tensors.
/// 100_000 is generous enough for any legitimate model while preventing
/// allocation bombs from crafted files.
const MAX_TENSOR_COUNT: u64 = 100_000;

/// Maximum number of metadata key-value pairs we will accept.
///
/// Legitimate GGUF files rarely exceed a few hundred metadata entries.
const MAX_METADATA_KV_COUNT: u64 = 100_000;

/// A parsed GGUF file.
///
/// Created via [`GgufFile::open`] (mmap-backed, zero-copy access to tensor
/// data) or [`GgufFile::read_from`] (stream-based, for in-memory buffers).
///
/// When opened via `open`, the underlying mmap is held for the lifetime
/// of the `GgufFile` and all `tensor_data()` slices borrow from it.
#[derive(Debug)]
pub struct GgufFile {
    /// File header.
    pub header: GgufHeader,
    /// Metadata key-value pairs.
    pub metadata: GgufMetadata,
    /// Tensor info entries, keyed by tensor name.
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Byte offset where the tensor data section starts.
    pub data_offset: u64,
    /// Optional mmap backing. Present when opened via `GgufFile::open`.
    mmap: Option<Mmap>,
}

impl GgufFile {
    /// Open a GGUF file from disk, memory-mapping it for zero-copy access.
    ///
    /// This is the preferred entry point for loading GGUF files. The file
    /// is memory-mapped, so tensor data is accessed on demand without
    /// copying the entire file into RAM. After opening, use
    /// [`tensor_data`](Self::tensor_data) for zero-copy access to raw
    /// tensor bytes, or [`read_tensor_f32`](Self::read_tensor_f32) to
    /// dequantize to f32.
    ///
    /// # Example
    /// ```rust,ignore
    /// let gguf = GgufFile::open("model.gguf")?;
    /// println!("Architecture: {:?}", gguf.architecture());
    /// let raw = gguf.tensor_data("token_embd.weight").unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: We hold the Mmap for the lifetime of GgufFile. The file
        // must not be truncated while mapped; this is the standard contract
        // for mmap-based file readers (same as safetensors, llama.cpp).
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let header = GgufHeader::read_from(&mut cursor)?;

        // Validate declared counts before allocating capacity.
        if header.tensor_count > MAX_TENSOR_COUNT {
            return Err(GgufError::TensorCountExceeded {
                count: header.tensor_count,
                max: MAX_TENSOR_COUNT,
            });
        }
        if header.metadata_kv_count > MAX_METADATA_KV_COUNT {
            return Err(GgufError::MetadataCountExceeded {
                count: header.metadata_kv_count,
                max: MAX_METADATA_KV_COUNT,
            });
        }

        let metadata = GgufMetadata::read_from(&mut cursor, header.metadata_kv_count)?;

        let capped_tensor_count = (header.tensor_count as usize).min(100_000);
        let mut tensors = HashMap::with_capacity(capped_tensor_count);
        for _ in 0..header.tensor_count {
            let info = GgufTensorInfo::read_from(&mut cursor)?;
            tensors.insert(info.name.clone(), info);
        }

        let current_pos = cursor.position();
        let alignment = 32u64;
        let data_offset = current_pos.div_ceil(alignment) * alignment;

        // Validate every tensor's data region fits within the mmap.
        let file_len = mmap.len() as u64;
        for (name, info) in &tensors {
            let byte_size = info.checked_byte_size()?;
            let abs_start = data_offset.checked_add(info.offset).ok_or_else(|| {
                GgufError::DataOffsetOverflow {
                    name: name.clone(),
                    data_offset,
                    tensor_offset: info.offset,
                }
            })?;
            let abs_end =
                abs_start
                    .checked_add(byte_size)
                    .ok_or_else(|| GgufError::DataOffsetOverflow {
                        name: name.clone(),
                        data_offset,
                        tensor_offset: info.offset,
                    })?;
            if abs_end > file_len {
                return Err(GgufError::DataOutOfBounds {
                    name: name.clone(),
                    start: abs_start,
                    end: abs_end,
                    file_len,
                });
            }
        }

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset,
            mmap: Some(mmap),
        })
    }

    /// Parse a GGUF file from a reader.
    ///
    /// Reads the header, metadata, and tensor info. Does NOT read tensor data
    /// (that happens lazily via `read_tensor_f32`).
    ///
    /// For file-based access with zero-copy tensor data, prefer
    /// [`GgufFile::open`] which uses mmap.
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufError> {
        let header = GgufHeader::read_from(reader)?;

        // Validate declared counts before allocating capacity.
        if header.tensor_count > MAX_TENSOR_COUNT {
            return Err(GgufError::TensorCountExceeded {
                count: header.tensor_count,
                max: MAX_TENSOR_COUNT,
            });
        }
        if header.metadata_kv_count > MAX_METADATA_KV_COUNT {
            return Err(GgufError::MetadataCountExceeded {
                count: header.metadata_kv_count,
                max: MAX_METADATA_KV_COUNT,
            });
        }

        let metadata = GgufMetadata::read_from(reader, header.metadata_kv_count)?;

        let capped_tensor_count = (header.tensor_count as usize).min(100_000);
        let mut tensors = HashMap::with_capacity(capped_tensor_count);
        for _ in 0..header.tensor_count {
            let info = GgufTensorInfo::read_from(reader)?;
            tensors.insert(info.name.clone(), info);
        }

        // Data section starts at the next alignment boundary after tensor info.
        // GGUF v3 aligns to 32 bytes.
        let current_pos = reader.stream_position()?;
        let alignment = 32u64;
        let data_offset = current_pos.div_ceil(alignment) * alignment;

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset,
            mmap: None,
        })
    }

    /// Read a tensor's data and dequantize to f32.
    ///
    /// Returns `(data, shape)` where data is the dequantized f32 values
    /// and shape is the tensor dimensions.
    pub fn read_tensor_f32<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), GgufError> {
        let info = self.tensors.get(name).ok_or_else(|| {
            GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor not found: {name}"),
            ))
        })?;

        let byte_size = info.checked_byte_size()?;
        let abs_offset = self.data_offset.checked_add(info.offset).ok_or_else(|| {
            GgufError::DataOffsetOverflow {
                name: name.to_string(),
                data_offset: self.data_offset,
                tensor_offset: info.offset,
            }
        })?;
        let abs_end =
            abs_offset
                .checked_add(byte_size)
                .ok_or_else(|| GgufError::DataOffsetOverflow {
                    name: name.to_string(),
                    data_offset: self.data_offset,
                    tensor_offset: info.offset,
                })?;

        // Validate data region against stream length to give a clear error
        // instead of a generic "unexpected end of file".
        let file_len = reader.seek(SeekFrom::End(0))?;
        if abs_end > file_len {
            return Err(GgufError::DataOutOfBounds {
                name: name.to_string(),
                start: abs_offset,
                end: abs_end,
                file_len,
            });
        }

        reader.seek(SeekFrom::Start(abs_offset))?;
        let byte_size_usize =
            usize::try_from(byte_size).map_err(|_| GgufError::TensorTooLarge {
                name: name.to_string(),
                byte_size,
                max: usize::MAX as u64,
            })?;
        let mut raw_data = vec![0u8; byte_size_usize];
        reader.read_exact(&mut raw_data)?;

        let num_elements_u64 = info.checked_num_elements()?;
        let num_elements =
            usize::try_from(num_elements_u64).map_err(|_| GgufError::ElementCountOverflow {
                name: name.to_string(),
            })?;
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();

        let f32_data = match info.dtype {
            GgufDType::F32 => {
                // Direct reinterpret. Limit output to available data.
                let available = raw_data.len() / 4;
                let count = num_elements.min(available);
                let mut out = vec![0f32; count];
                for (i, chunk) in raw_data.chunks_exact(4).take(count).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                out
            }
            GgufDType::F16 => {
                let available = raw_data.len() / 2;
                let count = num_elements.min(available);
                let mut out = Vec::with_capacity(count);
                for chunk in raw_data.chunks_exact(2).take(count) {
                    out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                }
                out
            }
            GgufDType::Q4_0 => dequant::dequantize_q4_0(&raw_data, num_elements),
            GgufDType::Q4_1 => dequant::dequantize_q4_1(&raw_data, num_elements),
            GgufDType::Q5_0 => dequant::dequantize_q5_0(&raw_data, num_elements),
            GgufDType::Q5_1 => dequant::dequantize_q5_1(&raw_data, num_elements),
            GgufDType::Q8_0 => dequant::dequantize_q8_0(&raw_data, num_elements),
            GgufDType::Q2K => dequant::dequantize_q2_k(&raw_data, num_elements),
            GgufDType::Q3K => dequant::dequantize_q3_k(&raw_data, num_elements),
            GgufDType::Q4K => dequant::dequantize_q4_k(&raw_data, num_elements),
            GgufDType::Q5K => dequant::dequantize_q5_k(&raw_data, num_elements),
            GgufDType::Q6K => dequant::dequantize_q6_k(&raw_data, num_elements),
            dtype => return Err(GgufError::UnsupportedDType { dtype }),
        };

        Ok((f32_data, shape))
    }

    /// Get raw tensor bytes by name (zero-copy when mmap-backed).
    ///
    /// Returns `None` if the tensor is not found. Returns `Some(Err(...))`
    /// if the file was opened via `read_from` (no mmap backing) and therefore
    /// cannot provide a zero-copy slice.
    ///
    /// For mmap-backed files (via [`GgufFile::open`]), this returns a
    /// borrowed slice into the memory-mapped file -- no allocation, no copy.
    pub fn tensor_data(&self, name: &str) -> Option<Result<&[u8], GgufError>> {
        let info = self.tensors.get(name)?;
        let mmap = match &self.mmap {
            Some(m) => m,
            None => {
                return Some(Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "tensor_data() requires mmap backing (use GgufFile::open)",
                ))));
            }
        };

        let start = match self.data_offset.checked_add(info.offset) {
            Some(s) => s,
            None => {
                return Some(Err(GgufError::DataOffsetOverflow {
                    name: name.to_string(),
                    data_offset: self.data_offset,
                    tensor_offset: info.offset,
                }));
            }
        };
        let byte_size = match info.checked_byte_size() {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        let end = match start.checked_add(byte_size) {
            Some(e) => e,
            None => {
                return Some(Err(GgufError::DataOffsetOverflow {
                    name: name.to_string(),
                    data_offset: self.data_offset,
                    tensor_offset: info.offset,
                }));
            }
        };
        let bytes = mmap.as_ref();

        if end as usize > bytes.len() {
            return Some(Err(GgufError::TensorSizeMismatch {
                name: name.to_string(),
                expected: byte_size,
                actual: (bytes.len() as u64).saturating_sub(start),
            }));
        }

        Some(Ok(&bytes[start as usize..end as usize]))
    }

    /// Dequantize a tensor to f32 using the mmap backing (zero-copy read).
    ///
    /// This is a convenience method that combines `tensor_data()` with
    /// dequantization. Only available for mmap-backed files (via `open`).
    pub fn dequantize_tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), GgufError> {
        let info = self.tensors.get(name).ok_or_else(|| {
            GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor not found: {name}"),
            ))
        })?;

        let raw_data = match self.tensor_data(name) {
            Some(result) => result?,
            None => {
                return Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("tensor not found: {name}"),
                )));
            }
        };

        let num_elements_u64 = info.checked_num_elements()?;
        let num_elements =
            usize::try_from(num_elements_u64).map_err(|_| GgufError::ElementCountOverflow {
                name: name.to_string(),
            })?;
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();

        let f32_data = match info.dtype {
            GgufDType::F32 => {
                let available = raw_data.len() / 4;
                let count = num_elements.min(available);
                let mut out = vec![0f32; count];
                for (i, chunk) in raw_data.chunks_exact(4).take(count).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                out
            }
            GgufDType::F16 => {
                let available = raw_data.len() / 2;
                let count = num_elements.min(available);
                let mut out = Vec::with_capacity(count);
                for chunk in raw_data.chunks_exact(2).take(count) {
                    out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                }
                out
            }
            GgufDType::Q4_0 => dequant::dequantize_q4_0(raw_data, num_elements),
            GgufDType::Q4_1 => dequant::dequantize_q4_1(raw_data, num_elements),
            GgufDType::Q5_0 => dequant::dequantize_q5_0(raw_data, num_elements),
            GgufDType::Q5_1 => dequant::dequantize_q5_1(raw_data, num_elements),
            GgufDType::Q8_0 => dequant::dequantize_q8_0(raw_data, num_elements),
            GgufDType::Q2K => dequant::dequantize_q2_k(raw_data, num_elements),
            GgufDType::Q3K => dequant::dequantize_q3_k(raw_data, num_elements),
            GgufDType::Q4K => dequant::dequantize_q4_k(raw_data, num_elements),
            GgufDType::Q5K => dequant::dequantize_q5_k(raw_data, num_elements),
            GgufDType::Q6K => dequant::dequantize_q6_k(raw_data, num_elements),
            dtype => return Err(GgufError::UnsupportedDType { dtype }),
        };

        Ok((f32_data, shape))
    }

    /// Model architecture string from metadata (e.g., "llama").
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get_str("general.architecture")
    }

    /// Model name from metadata.
    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get_str("general.name")
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Build a `ComputationGraph` from the GGUF model metadata (shape-only).
    ///
    /// Auto-detects the architecture from `general.architecture` metadata
    /// and constructs the appropriate graph. Currently supports:
    /// - `"llama"` — Llama 2/3, Code Llama, TinyLlama, Mistral
    ///
    /// Weight data is NOT embedded in the graph — all weights are shape-only
    /// placeholders. Use [`to_computation_graph_with_weights`] for a graph
    /// ready for GPU execution.
    pub fn to_computation_graph(
        &self,
    ) -> Result<nn_core::dyn_tensor::trace::ComputationGraph, GgufError> {
        let arch = self.architecture().unwrap_or("");
        match arch {
            "llama" => {
                let config = crate::arch_llama::LlamaConfig::from_gguf(self)?;
                Ok(crate::arch_llama::build_llama_graph(&config))
            }
            other => Err(GgufError::ArchitectureMismatch {
                expected: "llama".to_string(),
                found: other.to_string(),
            }),
        }
    }

    /// Build a `ComputationGraph` with actual weight data from the GGUF file.
    ///
    /// Same as [`to_computation_graph`] but dequantizes all model weights from
    /// the GGUF file and embeds them in the `WeightRef`s. The resulting graph
    /// is ready for `CompiledModel::builder(graph, cache).build()`.
    ///
    /// This is the hero API for GGUF → compiled model:
    /// ```rust,ignore
    /// let mut f = File::open("model.gguf")?;
    /// let gguf = GgufFile::read_from(&mut f)?;
    /// let graph = gguf.to_computation_graph_with_weights(&mut f)?;
    /// let model = CompiledModel::builder(&graph, &cache).build()?;
    /// ```
    pub fn to_computation_graph_with_weights<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> Result<nn_core::dyn_tensor::trace::ComputationGraph, GgufError> {
        let arch = self.architecture().unwrap_or("");
        match arch {
            "llama" => {
                let config = crate::arch_llama::LlamaConfig::from_gguf(self)?;
                crate::arch_llama::build_llama_graph_with_weights(&config, self, reader)
            }
            other => Err(GgufError::ArchitectureMismatch {
                expected: "llama".to_string(),
                found: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::header::GGUF_MAGIC;

    /// Build a minimal valid GGUF file in memory (header + 0 metadata + 0 tensors).
    fn minimal_gguf() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
                                                     // Pad to 32-byte alignment for data section.
        while data.len() % 32 != 0 {
            data.push(0);
        }
        data
    }

    /// Build a GGUF with one F32 tensor and one metadata entry.
    fn gguf_with_tensor() -> Vec<u8> {
        let mut data = Vec::new();

        // Header.
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count = 1

        // Metadata: "general.architecture" = "llama"
        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&8u32.to_le_bytes()); // STRING type
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        // Tensor info: "test_weight" [4, 3] F32, offset=0.
        let tname = b"test_weight";
        data.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        data.extend_from_slice(tname);
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        data.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&3u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&(GgufDType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // Pad to 32-byte alignment.
        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Tensor data: 12 f32 values.
        for i in 0..12 {
            data.extend_from_slice(&(i as f32 * 0.1).to_le_bytes());
        }

        data
    }

    #[test]
    fn test_read_minimal_gguf() {
        let data = minimal_gguf();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(file.header.version, 3);
        assert_eq!(file.header.tensor_count, 0);
        assert_eq!(file.header.metadata_kv_count, 0);
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn test_read_gguf_with_tensor() {
        let data = gguf_with_tensor();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(file.header.tensor_count, 1);
        assert_eq!(file.header.metadata_kv_count, 1);
        assert_eq!(file.architecture(), Some("llama"));

        let info = file
            .tensors
            .get("test_weight")
            .expect("tensor should exist");
        assert_eq!(info.shape, vec![4, 3]);
        assert_eq!(info.dtype, GgufDType::F32);
        assert_eq!(info.num_elements(), 12);
        assert_eq!(info.byte_size(), 48);
    }

    #[test]
    fn test_read_tensor_f32_roundtrip() {
        let data = gguf_with_tensor();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();

        let (values, shape) = file
            .read_tensor_f32(&mut cursor, "test_weight")
            .expect("should read tensor");
        assert_eq!(shape, vec![4, 3]);
        assert_eq!(values.len(), 12);
        for i in 0..12 {
            let expected = i as f32 * 0.1;
            assert!(
                (values[i] - expected).abs() < 1e-6,
                "index {i}: expected {expected}, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_tensor_data_requires_mmap() {
        let data = gguf_with_tensor();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();

        // tensor_data should return an error for non-mmap files.
        let result = file.tensor_data("test_weight");
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_tensor_data_not_found() {
        let data = minimal_gguf();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();

        assert!(file.tensor_data("nonexistent").is_none());
    }

    #[test]
    fn test_tensor_names() {
        let data = gguf_with_tensor();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();

        let names = file.tensor_names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test_weight"));
    }

    #[test]
    fn test_open_mmap_and_tensor_data() {
        // Write gguf_with_tensor to a temp file, then open via mmap.
        let data = gguf_with_tensor();
        let dir = std::env::temp_dir();
        let path = dir.join("nn_gguf_test_mmap.gguf");
        std::fs::write(&path, &data).expect("write temp file");

        let file = GgufFile::open(&path).expect("should open via mmap");
        assert_eq!(file.header.version, 3);
        assert_eq!(file.header.tensor_count, 1);
        assert_eq!(file.architecture(), Some("llama"));

        // tensor_data should return raw bytes.
        let raw = file
            .tensor_data("test_weight")
            .expect("tensor should exist")
            .expect("should be accessible via mmap");
        assert_eq!(raw.len(), 48); // 12 * 4 bytes

        // Verify raw bytes decode to correct f32 values.
        for i in 0..12 {
            let val =
                f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            let expected = i as f32 * 0.1;
            assert!(
                (val - expected).abs() < 1e-6,
                "index {i}: expected {expected}, got {val}"
            );
        }

        // dequantize_tensor should also work.
        let (values, shape) = file
            .dequantize_tensor("test_weight")
            .expect("should dequantize");
        assert_eq!(shape, vec![4, 3]);
        assert_eq!(values.len(), 12);
        for i in 0..12 {
            let expected = i as f32 * 0.1;
            assert!(
                (values[i] - expected).abs() < 1e-6,
                "dequantize index {i}: expected {expected}, got {}",
                values[i]
            );
        }

        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_open_nonexistent_file() {
        let result = GgufFile::open("/tmp/nonexistent_gguf_file_12345.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_excessive_tensor_count_rejected() {
        // Craft a GGUF header with tensor_count = u64::MAX to trigger OOM.
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor_count = absurd
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0

        let mut cursor = std::io::Cursor::new(data);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        assert!(
            matches!(err, GgufError::TensorCountExceeded { .. }),
            "expected TensorCountExceeded, got: {err}"
        );
    }

    #[test]
    fn test_excessive_metadata_count_rejected() {
        // Craft a GGUF header with metadata_kv_count = u64::MAX.
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // metadata_kv_count = absurd

        let mut cursor = std::io::Cursor::new(data);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        assert!(
            matches!(err, GgufError::MetadataCountExceeded { .. }),
            "expected MetadataCountExceeded, got: {err}"
        );
    }

    #[test]
    fn test_excessive_string_length_rejected() {
        // Craft a GGUF file with a metadata string whose length prefix is absurd.
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count = 1
                                                     // Key with absurd length
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // string length = absurd

        let mut cursor = std::io::Cursor::new(data);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        assert!(
            matches!(err, GgufError::StringLengthExceeded { .. }),
            "expected StringLengthExceeded, got: {err}"
        );
    }

    #[test]
    fn test_metadata_values() {
        let data = gguf_with_tensor();
        let mut cursor = std::io::Cursor::new(data);
        let file = GgufFile::read_from(&mut cursor).unwrap();

        // Test get_str.
        assert_eq!(file.metadata.get_str("general.architecture"), Some("llama"));

        // Test get on missing key.
        assert!(file.metadata.get("nonexistent_key").is_none());
        assert!(file.metadata.get_str("nonexistent_key").is_none());
        assert!(file.metadata.get_u32("nonexistent_key").is_none());
    }
}
