// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD weight tensor definitions and loading (safetensors + SVAD binary).

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::SileroVadError;

/// Weight tensors for Silero VAD 16kHz model.
///
/// All tensors are flattened `Vec<f32>` in row-major order.
/// Load from safetensors via `WeightMap::load_tensor` and flatten.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct SileroVadWeights {
    /// STFT basis filter `[258, 1, 256]` = 66048 floats.
    pub stft_basis: Vec<f32>,
    /// Encoder Conv1d weights per block: `[out_ch, in_ch, 3]`.
    pub enc_weights: [Vec<f32>; 4],
    /// Encoder Conv1d biases per block: `[out_ch]`.
    pub enc_biases: [Vec<f32>; 4],
    /// LSTM input-to-hidden weight `[512, 128]` = 65536 floats.
    pub lstm_weight_ih: Vec<f32>,
    /// LSTM hidden-to-hidden weight `[512, 128]` = 65536 floats.
    pub lstm_weight_hh: Vec<f32>,
    /// LSTM input-to-hidden bias `[512]`.
    pub lstm_bias_ih: Vec<f32>,
    /// LSTM hidden-to-hidden bias `[512]`.
    pub lstm_bias_hh: Vec<f32>,
    /// Output Linear weight `[1, 128]` = 128 floats.
    pub output_weight: Vec<f32>,
    /// Output Linear bias `[1]` = 1 float.
    pub output_bias: Vec<f32>,
}

impl SileroVadWeights {
    /// Create weights from raw tensor data.
    ///
    /// Prefer `from_safetensors_file`, `from_svad_file`, or `from_weight_map`
    /// for production use. This constructor is for testing and custom loaders.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stft_basis: Vec<f32>,
        enc_weights: [Vec<f32>; 4],
        enc_biases: [Vec<f32>; 4],
        lstm_weight_ih: Vec<f32>,
        lstm_weight_hh: Vec<f32>,
        lstm_bias_ih: Vec<f32>,
        lstm_bias_hh: Vec<f32>,
        output_weight: Vec<f32>,
        output_bias: Vec<f32>,
    ) -> Self {
        Self {
            stft_basis,
            enc_weights,
            enc_biases,
            lstm_weight_ih,
            lstm_weight_hh,
            lstm_bias_ih,
            lstm_bias_hh,
            output_weight,
            output_bias,
        }
    }
}

/// SVAD binary weight format magic bytes.
const SVAD_MAGIC: &[u8; 4] = b"SVAD";
/// Supported SVAD format version.
const SVAD_VERSION: u32 = 1;
/// Maximum tensor name length in SVAD files (4 KiB).
const MAX_NAME_LEN: usize = 4096;
/// Maximum tensor dimensionality in SVAD files.
const MAX_NDIM: usize = 16;

impl SileroVadWeights {
    /// Load weights from a dvoice SVAD binary file.
    ///
    /// The SVAD format is a simple binary container used by dvoice:
    /// `"SVAD"` magic (4 bytes), `u32` version (1), `u32` num_tensors,
    /// then per tensor: `u32` name_len, name bytes, `u32` ndim,
    /// `ndim × u32` shape, `u32` data_size, `data_size` bytes of f32 (LE).
    ///
    /// Weight names match exactly between SVAD and safetensors formats.
    ///
    /// # Errors
    ///
    /// Returns `SileroVadError::SvadFormat` on format violations or missing
    /// tensors, `SileroVadError::Io` on read failures.
    pub fn from_svad_file(path: impl AsRef<Path>) -> Result<Self, SileroVadError> {
        let data = std::fs::read(path)?;
        let mut cursor = &data[..];

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != SVAD_MAGIC {
            return Err(SileroVadError::SvadFormat("bad magic".into()));
        }

        let version = read_u32(&mut cursor)?;
        if version != SVAD_VERSION {
            return Err(SileroVadError::SvadFormat(format!(
                "version {version}, expected {SVAD_VERSION}"
            )));
        }

        let num_tensors = read_u32(&mut cursor)? as usize;
        if num_tensors > 1000 {
            return Err(SileroVadError::SvadFormat(format!(
                "num_tensors {num_tensors} exceeds maximum 1000"
            )));
        }

        let mut tensors: HashMap<String, Vec<f32>> = HashMap::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            let name_len = read_u32(&mut cursor)? as usize;
            if name_len > MAX_NAME_LEN {
                return Err(SileroVadError::SvadFormat(format!(
                    "tensor name_len {name_len} exceeds maximum {MAX_NAME_LEN}"
                )));
            }
            let mut name_buf = vec![0u8; name_len];
            cursor.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| SileroVadError::SvadFormat(format!("non-UTF8 tensor name: {e}")))?;

            let ndim = read_u32(&mut cursor)? as usize;
            if ndim > MAX_NDIM {
                return Err(SileroVadError::SvadFormat(format!(
                    "tensor '{name}' ndim {ndim} exceeds maximum {MAX_NDIM}"
                )));
            }
            let mut shape_elems: usize = 1;
            for _ in 0..ndim {
                shape_elems = shape_elems.saturating_mul(read_u32(&mut cursor)? as usize);
            }

            let data_size = read_u32(&mut cursor)? as usize;
            if data_size > cursor.len() {
                return Err(SileroVadError::SvadFormat(format!(
                    "tensor '{name}' data_size {data_size} exceeds remaining file ({} bytes)",
                    cursor.len()
                )));
            }
            if !data_size.is_multiple_of(4) {
                return Err(SileroVadError::SvadFormat(format!(
                    "tensor '{name}' data_size {data_size} not a multiple of 4"
                )));
            }
            let num_floats = data_size / 4;
            if num_floats != shape_elems {
                return Err(SileroVadError::SvadFormat(format!(
                    "tensor '{name}': {num_floats} floats but shape requires {shape_elems}"
                )));
            }

            let mut raw = vec![0u8; data_size];
            cursor.read_exact(&mut raw)?;
            let floats: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(name, floats);
        }

        Self::from_tensor_map(&mut tensors)
    }

    /// Construct from a name→data map (shared by SVAD and future loaders).
    fn from_tensor_map(tensors: &mut HashMap<String, Vec<f32>>) -> Result<Self, SileroVadError> {
        fn take(
            tensors: &mut HashMap<String, Vec<f32>>,
            name: &str,
        ) -> Result<Vec<f32>, SileroVadError> {
            let data = tensors
                .remove(name)
                .ok_or_else(|| SileroVadError::SvadFormat(format!("missing tensor '{name}'")))?;
            let count = data.iter().filter(|v| !v.is_finite()).count();
            if count > 0 {
                return Err(SileroVadError::NonFiniteWeight {
                    name: name.to_string(),
                    count,
                });
            }
            Ok(data)
        }

        Ok(Self {
            stft_basis: take(tensors, "stft_forward_basis_buffer")?,
            enc_weights: [
                take(tensors, "encoder_0_weight")?,
                take(tensors, "encoder_1_weight")?,
                take(tensors, "encoder_2_weight")?,
                take(tensors, "encoder_3_weight")?,
            ],
            enc_biases: [
                take(tensors, "encoder_0_bias")?,
                take(tensors, "encoder_1_bias")?,
                take(tensors, "encoder_2_bias")?,
                take(tensors, "encoder_3_bias")?,
            ],
            lstm_weight_ih: take(tensors, "decoder_rnn_weight_ih")?,
            lstm_weight_hh: take(tensors, "decoder_rnn_weight_hh")?,
            lstm_bias_ih: take(tensors, "decoder_rnn_bias_ih")?,
            lstm_bias_hh: take(tensors, "decoder_rnn_bias_hh")?,
            output_weight: take(tensors, "decoder_output_weight")?,
            output_bias: take(tensors, "decoder_output_bias")?,
        })
    }

    /// Load all Silero VAD weight tensors from an opened `WeightMap`.
    ///
    /// Tensor names match the output of `convert_silero_vad_safetensors.py`:
    /// `stft_forward_basis_buffer`, `encoder_{0..3}_{weight,bias}`,
    /// `decoder_rnn_{weight_ih,weight_hh,bias_ih,bias_hh}`,
    /// `decoder_output_{weight,bias}`.
    ///
    /// # Errors
    ///
    /// Returns `SileroVadError::WeightLoad` if any tensor is missing or has
    /// non-f32-aligned byte length.
    pub fn from_weight_map(wm: &crate::safetensors::WeightMap) -> Result<Self, SileroVadError> {
        fn extract(
            wm: &crate::safetensors::WeightMap,
            name: &str,
        ) -> Result<Vec<f32>, SileroVadError> {
            let bytes = wm.tensor_data(name)?;
            if bytes.len() % 4 != 0 {
                return Err(SileroVadError::SafetensorsFormat(format!(
                    "tensor '{name}' byte length {} not a multiple of 4",
                    bytes.len()
                )));
            }
            // Use f32::from_le_bytes instead of bytemuck::cast_slice to avoid
            // panics from misaligned byte slices. Safetensors data offsets are
            // not guaranteed to be 4-byte aligned.
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let count = floats.iter().filter(|v| !v.is_finite()).count();
            if count > 0 {
                return Err(SileroVadError::NonFiniteWeight {
                    name: name.to_string(),
                    count,
                });
            }
            Ok(floats)
        }

        Ok(Self {
            stft_basis: extract(wm, "stft_forward_basis_buffer")?,
            enc_weights: [
                extract(wm, "encoder_0_weight")?,
                extract(wm, "encoder_1_weight")?,
                extract(wm, "encoder_2_weight")?,
                extract(wm, "encoder_3_weight")?,
            ],
            enc_biases: [
                extract(wm, "encoder_0_bias")?,
                extract(wm, "encoder_1_bias")?,
                extract(wm, "encoder_2_bias")?,
                extract(wm, "encoder_3_bias")?,
            ],
            lstm_weight_ih: extract(wm, "decoder_rnn_weight_ih")?,
            lstm_weight_hh: extract(wm, "decoder_rnn_weight_hh")?,
            lstm_bias_ih: extract(wm, "decoder_rnn_bias_ih")?,
            lstm_bias_hh: extract(wm, "decoder_rnn_bias_hh")?,
            output_weight: extract(wm, "decoder_output_weight")?,
            output_bias: extract(wm, "decoder_output_bias")?,
        })
    }

    /// Load weights from a safetensors file without mmap (fully safe).
    ///
    /// Reads the entire file into memory, parses the safetensors header, and
    /// extracts tensor data as owned `Vec<f32>`. Unlike [`from_weight_map`](Self::from_weight_map),
    /// this does not require mmap, `unsafe`, or a Metal context — suitable for
    /// consumers that want a safe loading path.
    ///
    /// For zero-copy loading with shared Metal buffers, use
    /// [`from_weight_map`](Self::from_weight_map) with a [`WeightMap`](crate::safetensors::WeightMap) instead.
    ///
    /// # Errors
    ///
    /// Returns `SileroVadError::Io` on read failures or
    /// `SileroVadError::SafetensorsFormat` on parse/extraction errors.
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> Result<Self, SileroVadError> {
        let data = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| SileroVadError::SafetensorsFormat(e.to_string()))?;

        fn extract_tensor(
            st: &safetensors::SafeTensors<'_>,
            name: &str,
        ) -> Result<Vec<f32>, SileroVadError> {
            let view = st
                .tensor(name)
                .map_err(|e| SileroVadError::SafetensorsFormat(format!("tensor '{name}': {e}")))?;
            let bytes = view.data();
            if bytes.len() % 4 != 0 {
                return Err(SileroVadError::SafetensorsFormat(format!(
                    "tensor '{name}' byte length {} not a multiple of 4",
                    bytes.len()
                )));
            }
            // Use f32::from_le_bytes instead of bytemuck::cast_slice to avoid
            // panics from misaligned byte slices.
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let count = floats.iter().filter(|v| !v.is_finite()).count();
            if count > 0 {
                return Err(SileroVadError::NonFiniteWeight {
                    name: name.to_string(),
                    count,
                });
            }
            Ok(floats)
        }

        Ok(Self {
            stft_basis: extract_tensor(&st, "stft_forward_basis_buffer")?,
            enc_weights: [
                extract_tensor(&st, "encoder_0_weight")?,
                extract_tensor(&st, "encoder_1_weight")?,
                extract_tensor(&st, "encoder_2_weight")?,
                extract_tensor(&st, "encoder_3_weight")?,
            ],
            enc_biases: [
                extract_tensor(&st, "encoder_0_bias")?,
                extract_tensor(&st, "encoder_1_bias")?,
                extract_tensor(&st, "encoder_2_bias")?,
                extract_tensor(&st, "encoder_3_bias")?,
            ],
            lstm_weight_ih: extract_tensor(&st, "decoder_rnn_weight_ih")?,
            lstm_weight_hh: extract_tensor(&st, "decoder_rnn_weight_hh")?,
            lstm_bias_ih: extract_tensor(&st, "decoder_rnn_bias_ih")?,
            lstm_bias_hh: extract_tensor(&st, "decoder_rnn_bias_hh")?,
            output_weight: extract_tensor(&st, "decoder_output_weight")?,
            output_bias: extract_tensor(&st, "decoder_output_bias")?,
        })
    }
}

/// Read a little-endian u32 from a byte cursor.
fn read_u32(cursor: &mut &[u8]) -> Result<u32, SileroVadError> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
