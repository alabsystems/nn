// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF quantization type definitions and dequantization routines.
//!
//! Supports the quantization formats most commonly used in the llama.cpp
//! ecosystem: Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0,
//! and F32/F16.

/// GGUF tensor data types (quantization formats).
///
/// Type IDs match the GGUF spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgufDType {
    /// Parse from the raw u32 type ID in the GGUF file.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size for quantized types. Returns 1 for non-quantized types.
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => 1,
        }
    }

    /// Bytes per block for this type.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18, // 2 (scale) + 16 (4 bits * 32 values)
            Self::Q4_1 => 20, // 2 (scale) + 2 (min) + 16
            Self::Q5_0 => 22, // 2 (scale) + 4 (high bits) + 16
            Self::Q5_1 => 24, // 2 + 2 + 4 + 16
            Self::Q8_0 => 34, // 2 (scale) + 32 (8 bits * 32 values)
            Self::Q8_1 => 40, // 4 (scale f32) + 4 (min f32) + 32
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            _ => 0, // Unknown/unsupported
        }
    }
}

/// Validate that the input buffer is large enough for the given number of
/// elements and block parameters. Returns the number of complete blocks.
///
/// # Panics
///
/// Panics if `bytes_per_block` is 0.
fn validate_dequant_input(
    data: &[u8],
    num_elements: usize,
    block_size: usize,
    bytes_per_block: usize,
) -> usize {
    assert!(bytes_per_block > 0, "bytes_per_block must be nonzero");
    let num_blocks = num_elements / block_size;
    let required_bytes = num_blocks.saturating_mul(bytes_per_block);
    // Clamp to available data: if the buffer is short, reduce the number of
    // blocks we process rather than panicking on an out-of-bounds index.
    if data.len() < required_bytes {
        data.len() / bytes_per_block
    } else {
        num_blocks
    }
}

/// Dequantize Q4_0 block data to f32.
///
/// Q4_0 format: each block of 32 values is stored as:
/// - 2 bytes: f16 scale factor
/// - 16 bytes: 32 x 4-bit signed quantized values (packed 2 per byte)
///
/// Dequantization: val = scale * (q - 8)
pub fn dequantize_q4_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 18;
    let num_blocks = validate_dequant_input(data, num_elements, block_size, bytes_per_block);
    let mut output = Vec::with_capacity(num_blocks * block_size);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * bytes_per_block;
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        for j in 0..16 {
            let byte = data[block_start + 2 + j];
            let lo = i32::from(byte & 0x0F) - 8;
            let hi = i32::from((byte >> 4) & 0x0F) - 8;
            output.push(scale * lo as f32);
            output.push(scale * hi as f32);
        }
    }

    output
}

/// Dequantize Q8_0 block data to f32.
///
/// Q8_0 format: each block of 32 values is stored as:
/// - 2 bytes: f16 scale factor
/// - 32 bytes: 32 x 8-bit signed quantized values
///
/// Dequantization: val = scale * q
pub fn dequantize_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 34;
    let num_blocks = validate_dequant_input(data, num_elements, block_size, bytes_per_block);
    let mut output = Vec::with_capacity(num_blocks * block_size);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * bytes_per_block;
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        for j in 0..32 {
            let q = data[block_start + 2 + j] as i8;
            output.push(scale * f32::from(q));
        }
    }

    output
}

/// Dequantize Q4_1 block data to f32.
///
/// Q4_1 format: each block of 32 values is stored as:
/// - 2 bytes: f16 scale factor (`d`)
/// - 2 bytes: f16 minimum value (`m`)
/// - 16 bytes: 32 x 4-bit unsigned quantized values (packed 2 per byte)
///
/// Dequantization: val = d * q + m
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q4_1`
pub fn dequantize_q4_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 20; // 2 (d) + 2 (m) + 16 (quants)
    let num_blocks = validate_dequant_input(data, num_elements, block_size, bytes_per_block);
    let mut output = Vec::with_capacity(num_blocks * block_size);

    for block_idx in 0..num_blocks {
        let b = block_idx * bytes_per_block;
        let d = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();
        let m = half::f16::from_le_bytes([data[b + 2], data[b + 3]]).to_f32();

        for j in 0..16 {
            let byte = data[b + 4 + j];
            let lo = f32::from(byte & 0x0F);
            let hi = f32::from((byte >> 4) & 0x0F);
            output.push(d * lo + m);
            output.push(d * hi + m);
        }
    }

    output
}

/// Dequantize Q5_0 block data to f32.
///
/// Q5_0 format: each block of 32 values is stored as:
/// - 2 bytes: f16 scale factor
/// - 4 bytes: 32 high bits (bit 4 of each quantized value)
/// - 16 bytes: 32 x 4-bit low quantized values (packed 2 per byte)
///
/// Dequantization: q = lo4 | (hi1 << 4), val = scale * (q - 16)
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q5_0`
pub fn dequantize_q5_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 22; // 2 (scale) + 4 (high bits) + 16 (low nibbles)
    let num_blocks = validate_dequant_input(data, num_elements, block_size, bytes_per_block);
    let mut output = Vec::with_capacity(num_blocks * block_size);

    for block_idx in 0..num_blocks {
        let b = block_idx * bytes_per_block;
        let scale = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();

        // 4 bytes of high bits packed as a u32 (little-endian).
        let qh = u32::from_le_bytes([data[b + 2], data[b + 3], data[b + 4], data[b + 5]]);

        for j in 0..16 {
            let byte = data[b + 6 + j];
            let lo_nibble = byte & 0x0F;
            let hi_nibble = (byte >> 4) & 0x0F;

            // Extract the 5th bit for the two values at positions 2*j and 2*j+1.
            let x0_h = ((qh >> (2 * j)) & 1) as u8;
            let x1_h = ((qh >> (2 * j + 1)) & 1) as u8;

            let x0 = i32::from(lo_nibble | (x0_h << 4)) - 16;
            let x1 = i32::from(hi_nibble | (x1_h << 4)) - 16;

            output.push(scale * x0 as f32);
            output.push(scale * x1 as f32);
        }
    }

    output
}

/// Dequantize Q5_1 block data to f32.
///
/// Q5_1 format: each block of 32 values is stored as:
/// - 2 bytes: f16 scale factor (`d`)
/// - 2 bytes: f16 minimum value (`m`)
/// - 4 bytes: 32 high bits (bit 4 of each quantized value)
/// - 16 bytes: 32 x 4-bit low quantized values (packed 2 per byte)
///
/// Dequantization: q = lo4 | (hi1 << 4), val = d * q + m
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q5_1`
pub fn dequantize_q5_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 24; // 2 (d) + 2 (m) + 4 (high bits) + 16 (low nibbles)
    let num_blocks = validate_dequant_input(data, num_elements, block_size, bytes_per_block);
    let mut output = Vec::with_capacity(num_blocks * block_size);

    for block_idx in 0..num_blocks {
        let b = block_idx * bytes_per_block;
        let d = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();
        let m = half::f16::from_le_bytes([data[b + 2], data[b + 3]]).to_f32();

        // 4 bytes of high bits packed as a u32 (little-endian).
        let qh = u32::from_le_bytes([data[b + 4], data[b + 5], data[b + 6], data[b + 7]]);

        for j in 0..16 {
            let byte = data[b + 8 + j];
            let lo_nibble = byte & 0x0F;
            let hi_nibble = (byte >> 4) & 0x0F;

            let x0_h = ((qh >> (2 * j)) & 1) as u8;
            let x1_h = ((qh >> (2 * j + 1)) & 1) as u8;

            let x0 = lo_nibble | (x0_h << 4);
            let x1 = hi_nibble | (x1_h << 4);

            output.push(d * f32::from(x0) + m);
            output.push(d * f32::from(x1) + m);
        }
    }

    output
}

/// Dequantize Q4_K block data to f32.
///
/// Q4_K format: 256-element super-blocks, 144 bytes each:
/// - 2 bytes: f16 `d` (super-block scale)
/// - 2 bytes: f16 `dmin` (super-block minimum)
/// - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks
/// - 128 bytes: 256 x 4-bit quantized values (packed 2 per byte)
///
/// The 12-byte scales section encodes 8 scales and 8 mins, each 6 bits:
/// - Bytes 0..4: low 4 bits of scales[0..8] (packed 2 per byte, lo/hi nibble)
/// - Bytes 4..8: low 4 bits of mins[0..8] (packed 2 per byte, lo/hi nibble)
/// - Bytes 8..12: high 2 bits of scales and mins (packed 4 per byte)
///
/// Dequantization for sub-block j, element i:
///   val = d * scale_j * q_ij - dmin * min_j
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q4_K`
pub fn dequantize_q4_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144;
    let num_blocks = validate_dequant_input(data, num_elements, BLOCK_SIZE, BYTES_PER_BLOCK);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let b = block_idx * BYTES_PER_BLOCK;

        // Super-block scale and minimum.
        let d = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();
        let dmin = half::f16::from_le_bytes([data[b + 2], data[b + 3]]).to_f32();

        // Unpack 8 x 6-bit scales and 8 x 6-bit mins from bytes b+4..b+16.
        let scales_data = &data[b + 4..b + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // Low 4 bits of scales: bytes 0..4, two per byte (lo nibble = even
        // index, hi nibble = odd index).
        for j in 0..8 {
            scales[j] = (scales_data[j / 2] >> (4 * (j % 2))) & 0x0F;
        }
        // Low 4 bits of mins: bytes 4..8.
        for j in 0..8 {
            mins[j] = (scales_data[4 + j / 2] >> (4 * (j % 2))) & 0x0F;
        }
        // High 2 bits: bytes 8..12.
        // Byte 8: high bits for scales[0..4]
        // Byte 9: high bits for scales[4..8]
        // Byte 10: high bits for mins[0..4]
        // Byte 11: high bits for mins[4..8]
        for j in 0..4 {
            scales[j] |= ((scales_data[8] >> (2 * j)) & 3) << 4;
        }
        for j in 0..4 {
            scales[4 + j] |= ((scales_data[9] >> (2 * j)) & 3) << 4;
        }
        for j in 0..4 {
            mins[j] |= ((scales_data[10] >> (2 * j)) & 3) << 4;
        }
        for j in 0..4 {
            mins[4 + j] |= ((scales_data[11] >> (2 * j)) & 3) << 4;
        }

        // Dequantize 8 sub-blocks of 32 elements each.
        let qs = &data[b + 16..b + BYTES_PER_BLOCK];
        for j in 0..8 {
            let sc = d * f32::from(scales[j]);
            let m = dmin * f32::from(mins[j]);

            // Each sub-block: 32 elements from 16 bytes (4 bits each).
            for i in 0..16 {
                let byte = qs[j * 16 + i];
                let lo = f32::from(byte & 0x0F);
                let hi = f32::from((byte >> 4) & 0x0F);
                output.push(sc * lo - m);
                output.push(sc * hi - m);
            }
        }
    }

    output
}

/// Dequantize Q6_K block data to f32.
///
/// Q6_K format: 256-element super-blocks, 210 bytes each:
/// - 128 bytes: `ql[128]` — low 4 bits of each 6-bit quantized value (packed 2 per byte)
/// - 64 bytes: `qh[64]` — high 2 bits of each 6-bit quantized value (packed 4 per byte)
/// - 16 bytes: `scales[16]` — int8 per-sub-block scales (16 sub-blocks of 16 values)
/// - 2 bytes: `d` — f16 super-block scale
///
/// The 6-bit quantized value is reconstructed as:
///   q6 = (ql_lo4) | ((qh_2bit) << 4)   (range 0..63)
///
/// Dequantization:
///   val = d * scale_j * (q6 - 32)
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q6_K`
pub fn dequantize_q6_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 210;
    let num_blocks = validate_dequant_input(data, num_elements, BLOCK_SIZE, BYTES_PER_BLOCK);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let b = block_idx * BYTES_PER_BLOCK;

        // Layout: ql[128] | qh[64] | scales[16] | d[2]
        let ql = &data[b..b + 128];
        let qh = &data[b + 128..b + 192];
        let scales = &data[b + 192..b + 208];
        let d = half::f16::from_le_bytes([data[b + 208], data[b + 209]]).to_f32();

        // 256 values = 16 sub-blocks of 16 values each.
        // ql: 128 bytes = 256 nibbles (low 4 bits). Byte i holds values 2*i (lo) and 2*i+1 (hi).
        // qh: 64 bytes = 256 crumbs (high 2 bits). Byte i holds 4 values, 2 bits each.
        for sub in 0..16 {
            let sc = f32::from(scales[sub] as i8);
            let base = sub * 16; // value index within the 256-element block

            for k in 0..16 {
                let idx = base + k; // value index 0..255

                // Low 4 bits from ql: byte idx/2, low or high nibble.
                let ql_byte = ql[idx / 2];
                let lo4 = if idx % 2 == 0 {
                    ql_byte & 0x0F
                } else {
                    (ql_byte >> 4) & 0x0F
                };

                // High 2 bits from qh: byte idx/4, shift by (idx%4)*2.
                let qh_byte = qh[idx / 4];
                let hi2 = (qh_byte >> ((idx % 4) * 2)) & 0x03;

                let q6 = lo4 | (hi2 << 4); // 6-bit value, range 0..63
                output.push(d * sc * (i32::from(q6) - 32) as f32);
            }
        }
    }

    output
}

/// Dequantize Q2_K block data to f32.
///
/// Q2_K format: 256-element super-blocks, 84 bytes each:
/// - 16 bytes: `scales[16]` — uint8 packed scale+min for 16 sub-blocks
///   (low 4 bits = scale, high 4 bits = min)
/// - 64 bytes: `qs[64]` — 256 x 2-bit quantized values (packed 4 per byte)
/// - 2 bytes: `d` — f16 super-block scale
/// - 2 bytes: `dmin` — f16 super-block minimum
///
/// Dequantization for sub-block j, element i:
///   val = d * scale_j * q_ij - dmin * min_j
///
/// where scale_j = low 4 bits of scales[j], min_j = high 4 bits of scales[j].
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q2_K`
pub fn dequantize_q2_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 84;
    let num_blocks = validate_dequant_input(data, num_elements, BLOCK_SIZE, BYTES_PER_BLOCK);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let b = block_idx * BYTES_PER_BLOCK;

        // Layout: scales[16] | qs[64] | d[2] | dmin[2]
        let scales_raw = &data[b..b + 16];
        let qs = &data[b + 16..b + 80];
        let d = half::f16::from_le_bytes([data[b + 80], data[b + 81]]).to_f32();
        let dmin = half::f16::from_le_bytes([data[b + 82], data[b + 83]]).to_f32();

        // 256 values = 16 sub-blocks of 16 values each.
        // qs: 64 bytes = 256 crumbs (2-bit quantized values, 4 per byte).
        for sub in 0..16 {
            let sc = f32::from(scales_raw[sub] & 0x0F);
            let m = f32::from((scales_raw[sub] >> 4) & 0x0F);

            let base = sub * 16; // value index within the 256-element block

            for k in 0..16 {
                let idx = base + k; // value index 0..255

                // 2-bit quant from qs: byte idx/4, shift by (idx%4)*2.
                let qs_byte = qs[idx / 4];
                let q2 = f32::from((qs_byte >> ((idx % 4) * 2)) & 0x03);

                output.push(d * sc * q2 - dmin * m);
            }
        }
    }

    output
}

/// Dequantize Q3_K block data to f32.
///
/// Q3_K format: 256-element super-blocks, 110 bytes each:
/// - 32 bytes: `hmask[32]` — high bit (bit 2) of each 3-bit quantized value.
///   Bit j of byte i corresponds to value (i * 8 + j). When set, the high bit
///   is 1, contributing +4 to the 3-bit value.
/// - 64 bytes: `qs[64]` — low 2 bits of each 3-bit quantized value, packed
///   4 per byte (2 bits each).
/// - 12 bytes: `scales[12]` — packed 6-bit signed scales for 16 sub-blocks
///   of 16 values each.
/// - 2 bytes: `d` — f16 super-block scale.
///
/// Scale packing (12 bytes → 16 × 6-bit signed scales):
///   Byte k (k < 8): low 4 bits → scale for sub-block k via shift = 4*(k%2).
///   Bytes 8..12: high 2 bits packed by pairs.
///   Final 6-bit value is sign-extended to i8 (subtract 32).
///
/// Dequantization for sub-block j, element i:
///   q3 = (qs_2bit) | (hmask_bit << 2)     (range 0..7)
///   val = d * scale_j * (q3 - 4)
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q3_K`
pub fn dequantize_q3_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 110;
    let num_blocks = validate_dequant_input(data, num_elements, BLOCK_SIZE, BYTES_PER_BLOCK);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let b = block_idx * BYTES_PER_BLOCK;

        // Layout: hmask[32] | qs[64] | scales[12] | d[2]
        let hmask = &data[b..b + 32];
        let qs = &data[b + 32..b + 96];
        let scales_raw = &data[b + 96..b + 108];
        let d = half::f16::from_le_bytes([data[b + 108], data[b + 109]]).to_f32();

        // Unpack 16 x 6-bit signed scales from 12 bytes.
        // Bytes 0..8: low 4 bits, two per byte (lo/hi nibble), for sub-blocks 0..15.
        // Bytes 8..12: high 2 bits.
        let mut scales = [0i8; 16];

        // Low 4 bits: bytes 0..8, sub-blocks 0..15.
        for k in 0..16 {
            let lo4 = (scales_raw[k / 2] >> (4 * (k % 2))) & 0x0F;
            scales[k] = lo4 as i8;
        }

        // High 2 bits: bytes 8..12.
        // Byte 8: high 2 bits for sub-blocks 0..3.
        // Byte 9: high 2 bits for sub-blocks 4..7.
        // Byte 10: high 2 bits for sub-blocks 8..11.
        // Byte 11: high 2 bits for sub-blocks 12..15.
        for k in 0..16 {
            let hi_byte = scales_raw[8 + k / 4];
            let hi2 = ((hi_byte >> (2 * (k % 4))) & 0x03) as i8;
            // Combine: 6-bit unsigned = lo4 | (hi2 << 4), then sign-extend
            // by subtracting 32.
            scales[k] = ((scales[k] as u8 | ((hi2 as u8) << 4)) as i8).wrapping_sub(32);
        }

        // Dequantize 16 sub-blocks of 16 elements each.
        for sub in 0..16 {
            let sc = d * f32::from(scales[sub]);
            let base = sub * 16;

            for k in 0..16 {
                let idx = base + k;

                // Low 2 bits from qs: byte idx/4, shift by (idx%4)*2.
                let q2 = (qs[idx / 4] >> ((idx % 4) * 2)) & 0x03;

                // High bit from hmask: byte idx/8, bit idx%8.
                let hb = (hmask[idx / 8] >> (idx % 8)) & 1;

                let q3 = q2 | (hb << 2); // 3-bit value, range 0..7
                output.push(sc * (i32::from(q3) - 4) as f32);
            }
        }
    }

    output
}

/// Dequantize Q5_K block data to f32.
///
/// Q5_K format: 256-element super-blocks, 176 bytes each:
/// - 2 bytes: f16 `d` (super-block scale)
/// - 2 bytes: f16 `dmin` (super-block minimum)
/// - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks (same as Q4_K)
/// - 32 bytes: `qh[32]` — high bit (bit 4) of each 5-bit quantized value.
///   Bit j of byte i corresponds to value (i * 8 + j).
/// - 128 bytes: `qs[128]` — low 4 bits of each 5-bit quantized value, packed
///   2 per byte (lo/hi nibble).
///
/// Dequantization for sub-block j, element i:
///   q5 = qs_lo4 | (qh_bit << 4)     (range 0..31)
///   val = d * scale_j * q5 - dmin * min_j
///
/// Reference: llama.cpp ggml-quants.c `dequantize_row_q5_K`
pub fn dequantize_q5_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 176;
    let num_blocks = validate_dequant_input(data, num_elements, BLOCK_SIZE, BYTES_PER_BLOCK);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let b = block_idx * BYTES_PER_BLOCK;

        // Layout: d[2] | dmin[2] | scales[12] | qh[32] | qs[128]
        let d = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();
        let dmin = half::f16::from_le_bytes([data[b + 2], data[b + 3]]).to_f32();

        // Unpack 8 x 6-bit scales and 8 x 6-bit mins from bytes b+4..b+16.
        // Same packing as Q4_K.
        let scales_data = &data[b + 4..b + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // Low 4 bits of scales: bytes 0..4, two per byte.
        for j in 0..8 {
            scales[j] = (scales_data[j / 2] >> (4 * (j % 2))) & 0x0F;
        }
        // Low 4 bits of mins: bytes 4..8.
        for j in 0..8 {
            mins[j] = (scales_data[4 + j / 2] >> (4 * (j % 2))) & 0x0F;
        }
        // High 2 bits: bytes 8..12.
        for j in 0..4 {
            scales[j] |= ((scales_data[8] >> (2 * j)) & 3) << 4;
        }
        for j in 0..4 {
            scales[4 + j] |= ((scales_data[9] >> (2 * j)) & 3) << 4;
        }
        for j in 0..4 {
            mins[j] |= ((scales_data[10] >> (2 * j)) & 3) << 4;
        }
        for j in 0..4 {
            mins[4 + j] |= ((scales_data[11] >> (2 * j)) & 3) << 4;
        }

        let qh = &data[b + 16..b + 48];
        let qs = &data[b + 48..b + BYTES_PER_BLOCK];

        // Dequantize 8 sub-blocks of 32 elements each.
        for j in 0..8 {
            let sc = d * f32::from(scales[j]);
            let m = dmin * f32::from(mins[j]);

            for i in 0..32 {
                let idx = j * 32 + i; // value index 0..255

                // Low 4 bits from qs: byte idx/2, lo or hi nibble.
                let qs_byte = qs[idx / 2];
                let lo4 = if idx % 2 == 0 {
                    qs_byte & 0x0F
                } else {
                    (qs_byte >> 4) & 0x0F
                };

                // High bit from qh: byte idx/8, bit idx%8.
                let hb = (qh[idx / 8] >> (idx % 8)) & 1;

                let q5 = lo4 | (hb << 4); // 5-bit value, range 0..31
                output.push(sc * f32::from(q5) - m);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_from_u32() {
        assert_eq!(GgufDType::from_u32(0), Some(GgufDType::F32));
        assert_eq!(GgufDType::from_u32(2), Some(GgufDType::Q4_0));
        assert_eq!(GgufDType::from_u32(8), Some(GgufDType::Q8_0));
        assert_eq!(GgufDType::from_u32(12), Some(GgufDType::Q4K));
        assert_eq!(GgufDType::from_u32(255), None);
    }

    #[test]
    fn test_q8_0_dequant_zero_scale() {
        // Block with zero scale: all outputs should be 0.0.
        let mut block = vec![0u8; 34];
        // scale = f16(0.0)
        block[0] = 0;
        block[1] = 0;
        // Fill quantized values with arbitrary data.
        for i in 0..32 {
            block[2 + i] = 42;
        }
        let result = dequantize_q8_0(&block, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_0_dequant_roundtrip() {
        // Build a Q4_0 block with scale=1.0, all nibbles=8 (centered)
        // dequant: (8-8)*1.0 = 0.0 for all values
        let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        // Fill with 0x88 -> lo=8, hi=8 -> both dequant to 0.0
        for i in 0..16 {
            block[2 + i] = 0x88;
        }
        let result = dequantize_q4_0(&block, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(GgufDType::Q4_0.block_size(), 32);
        assert_eq!(GgufDType::Q8_0.block_size(), 32);
        assert_eq!(GgufDType::Q4K.block_size(), 256);
        assert_eq!(GgufDType::F32.block_size(), 1);
    }

    #[test]
    fn test_q4k_block_size_and_type_size() {
        assert_eq!(GgufDType::Q4K.block_size(), 256);
        assert_eq!(GgufDType::Q4K.type_size(), 144);
    }

    #[test]
    fn test_q4k_dequant_zero_scale_produces_zeros() {
        // Q4_K block with d=0, dmin=0 should produce all zeros regardless
        // of the quantized values and scales.
        let mut block = vec![0u8; 144];
        // d = f16(0.0), dmin = f16(0.0) -- first 4 bytes all zero.
        // Fill scales section with nonzero data.
        for i in 4..16 {
            block[i] = 0xFF;
        }
        // Fill quantized values with nonzero data.
        for i in 16..144 {
            block[i] = 0xAB;
        }
        let result = dequantize_q4_k(&block, 256);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0, "all values should be 0 when d=0 and dmin=0");
        }
    }

    #[test]
    fn test_q4k_dequant_known_values() {
        // Build a Q4_K block where:
        // d = 1.0, dmin = 0.0
        // Sub-block 0: scale=2, min=0, all q values = 3
        // Expected output for sub-block 0: d * scale * q - dmin * min
        //   = 1.0 * 2 * 3 - 0 = 6.0 for all 32 elements
        let mut block = vec![0u8; 144];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // dmin = 0.0 (bytes 2,3 already zero)

        // Scales section (12 bytes at offset 4..16):
        // scales[0] = 2 (low 4 bits), all others = 0
        // mins[*] = 0
        // Byte 0 (offset 4): low nibble for scales[0], hi nibble for scales[1]
        block[4] = 0x02; // scales[0]=2, scales[1]=0

        // Quantized values: sub-block 0 occupies bytes 16..32 (16 bytes).
        // Each byte: lo=3, hi=3 -> 0x33.
        for i in 0..16 {
            block[16 + i] = 0x33;
        }

        let result = dequantize_q4_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0 (first 32 elements): d * 2 * 3 - 0 = 6.0
        for i in 0..32 {
            assert!(
                (result[i] - 6.0).abs() < 1e-4,
                "sub-block 0 element {i}: expected 6.0, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..7: scale=0, min=0, so all values = 0.
        for i in 32..256 {
            assert!(
                result[i].abs() < 1e-6,
                "sub-block {} element {}: expected 0.0, got {}",
                i / 32,
                i % 32,
                result[i]
            );
        }
    }

    #[test]
    fn test_q4k_dequant_with_dmin() {
        // Test the dmin (minimum offset) term.
        // d = 1.0, dmin = 0.5
        // Sub-block 0: scale=1, min=4, q=0
        // Expected: 1.0 * 1 * 0 - 0.5 * 4 = -2.0
        let mut block = vec![0u8; 144];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];

        // scales[0] = 1 (low 4 bits at byte 0, lo nibble)
        block[4] = 0x01;
        // mins[0] = 4 (low 4 bits at byte 4 of scales_data, i.e. offset 8)
        block[8] = 0x04;

        // All quantized values = 0x00 (already zero)

        let result = dequantize_q4_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: d * 1 * 0 - dmin * 4 = -2.0
        for i in 0..32 {
            assert!(
                (result[i] - (-2.0)).abs() < 1e-3,
                "element {i}: expected -2.0, got {}",
                result[i]
            );
        }
    }

    // --- Q4_1 tests ---

    #[test]
    fn test_q4_1_dequant_zero_scale() {
        // d=0, m=0.5: all outputs should be m=0.5 regardless of quantized values.
        let mut block = vec![0u8; 20];
        let m_bytes = half::f16::from_f32(0.5).to_le_bytes();
        // d = 0.0 (first 2 bytes already zero)
        block[2] = m_bytes[0];
        block[3] = m_bytes[1];
        for i in 0..16 {
            block[4 + i] = 0xFF; // arbitrary quant values
        }
        let result = dequantize_q4_1(&block, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-4, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_q4_1_dequant_known_values() {
        // d=1.0, m=2.0, all q=3 -> val = 1.0 * 3 + 2.0 = 5.0
        let mut block = vec![0u8; 20];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let m_bytes = half::f16::from_f32(2.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = m_bytes[0];
        block[3] = m_bytes[1];
        // 0x33 -> lo=3, hi=3
        for i in 0..16 {
            block[4 + i] = 0x33;
        }
        let result = dequantize_q4_1(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-3, "element {i}: expected 5.0, got {v}");
        }
    }

    // --- Q5_0 tests ---

    #[test]
    fn test_q5_0_dequant_zero_scale() {
        // scale=0: all outputs = 0.0
        let block = vec![0u8; 22];
        let result = dequantize_q5_0(&block, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q5_0_dequant_centered() {
        // scale=1.0, all low nibbles=0, all high bits=1 -> q = 0 | (1<<4) = 16
        // val = 1.0 * (16 - 16) = 0.0
        let mut block = vec![0u8; 22];
        let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        // Set all 32 high bits to 1 -> u32 = 0xFFFFFFFF
        block[2] = 0xFF;
        block[3] = 0xFF;
        block[4] = 0xFF;
        block[5] = 0xFF;
        // Low nibbles all 0 (already zero).
        let result = dequantize_q5_0(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!((v).abs() < 1e-4, "element {i}: expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q5_0_dequant_known_values() {
        // scale=0.5, lo nibble=5 for all, high bit=0 -> q=5, val=0.5*(5-16)=-5.5
        let mut block = vec![0u8; 22];
        let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        // high bits all 0 (already zero)
        // low nibbles: 0x55 -> lo=5, hi=5
        for i in 0..16 {
            block[6 + i] = 0x55;
        }
        let result = dequantize_q5_0(&block, 32);
        assert_eq!(result.len(), 32);
        let expected = 0.5 * (5.0 - 16.0);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-3,
                "element {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_q5_0_dequant_two_blocks() {
        // Verify multi-block dequantization (2 blocks = 64 elements).
        // Block 0: scale=1.0, all zero (q=0, val = 1.0*(0-16) = -16.0)
        // Block 1: scale=2.0, all zero (q=0, val = 2.0*(0-16) = -32.0)
        let mut data = vec![0u8; 22 * 2];
        let s1 = half::f16::from_f32(1.0).to_le_bytes();
        let s2 = half::f16::from_f32(2.0).to_le_bytes();
        data[0] = s1[0];
        data[1] = s1[1];
        data[22] = s2[0];
        data[22 + 1] = s2[1];
        // All qh and qs are 0 -> q=0 for every value.

        let result = dequantize_q5_0(&data, 64);
        assert_eq!(result.len(), 64);
        for i in 0..32 {
            assert!(
                (result[i] - (-16.0)).abs() < 1e-3,
                "block 0 element {i}: expected -16.0, got {}",
                result[i]
            );
        }
        for i in 32..64 {
            assert!(
                (result[i] - (-32.0)).abs() < 1e-3,
                "block 1 element {}: expected -32.0, got {}",
                i - 32,
                result[i]
            );
        }
    }

    #[test]
    fn test_q5_0_dequant_mixed_high_bits() {
        // scale=1.0, lo nibble=0, high bits alternate 0 and 1.
        // Even positions: q = 0 | (0<<4) = 0, val = 1.0*(0-16) = -16.0
        // Odd positions:  q = 0 | (1<<4) = 16, val = 1.0*(16-16) = 0.0
        //
        // qh bit pattern: bit 0 (value 0) = 0, bit 1 (value 1) = 1,
        //                 bit 2 (value 2) = 0, bit 3 (value 3) = 1, ...
        // That is 0xAAAAAAAA.
        let mut block = vec![0u8; 22];
        let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        // qh = 0xAAAAAAAA (odd bits set)
        let qh_bytes = 0xAAAAAAAAu32.to_le_bytes();
        block[2] = qh_bytes[0];
        block[3] = qh_bytes[1];
        block[4] = qh_bytes[2];
        block[5] = qh_bytes[3];
        // qs all zero (lo nibbles = 0)

        let result = dequantize_q5_0(&block, 32);
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            // In the loop, j=0..16, each j produces two values at positions 2*j and 2*j+1.
            // High bit for position 2*j comes from qh bit 2*j.
            // High bit for position 2*j+1 comes from qh bit 2*j+1.
            // qh = 0xAAAAAAAA means even bits=0, odd bits=1.
            // Position 2*j: bit index 2*j (even) -> 0 -> q=0, val=-16
            // Position 2*j+1: bit index 2*j+1 (odd) -> 1 -> q=16, val=0
            let expected = if i % 2 == 0 { -16.0 } else { 0.0 };
            assert!(
                (result[i] - expected).abs() < 1e-3,
                "element {i}: expected {expected}, got {}",
                result[i]
            );
        }
    }

    // --- Q5_1 tests ---

    #[test]
    fn test_q5_1_dequant_zero_scale() {
        // d=0, m=3.0: all outputs should be m=3.0
        let mut block = vec![0u8; 24];
        let m_bytes = half::f16::from_f32(3.0).to_le_bytes();
        block[2] = m_bytes[0];
        block[3] = m_bytes[1];
        for i in 0..16 {
            block[8 + i] = 0xAA; // arbitrary quant values
        }
        let result = dequantize_q5_1(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 3.0).abs() < 1e-3, "element {i}: expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_q5_1_dequant_known_values() {
        // d=1.0, m=0.0, lo nibble=7, high bit=1 -> q = 7 | (1<<4) = 23
        // val = 1.0 * 23 + 0.0 = 23.0
        let mut block = vec![0u8; 24];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // m = 0 (already zero)
        // high bits all 1 -> u32 = 0xFFFFFFFF
        block[4] = 0xFF;
        block[5] = 0xFF;
        block[6] = 0xFF;
        block[7] = 0xFF;
        // lo nibbles: 0x77 -> lo=7, hi=7
        for i in 0..16 {
            block[8 + i] = 0x77;
        }
        let result = dequantize_q5_1(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 23.0).abs() < 1e-2,
                "element {i}: expected 23.0, got {v}"
            );
        }
    }

    #[test]
    fn test_q5_1_dequant_with_minimum() {
        // d=0.5, m=10.0, lo nibble=0, high bits=0 -> q=0
        // val = 0.5 * 0 + 10.0 = 10.0
        let mut block = vec![0u8; 24];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        let m_bytes = half::f16::from_f32(10.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = m_bytes[0];
        block[3] = m_bytes[1];
        // qh = 0, qs = 0 (already zero)
        let result = dequantize_q5_1(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 10.0).abs() < 1e-2,
                "element {i}: expected 10.0, got {v}"
            );
        }
    }

    #[test]
    fn test_q5_1_dequant_two_blocks() {
        // Block 0: d=1.0, m=0.0, all q=0 -> val = 0.0
        // Block 1: d=1.0, m=5.0, all q=0 -> val = 5.0
        let mut data = vec![0u8; 24 * 2];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let m2_bytes = half::f16::from_f32(5.0).to_le_bytes();
        // Block 0: d=1.0, m=0.0
        data[0] = d_bytes[0];
        data[1] = d_bytes[1];
        // Block 1: d=1.0, m=5.0
        data[24] = d_bytes[0];
        data[24 + 1] = d_bytes[1];
        data[24 + 2] = m2_bytes[0];
        data[24 + 3] = m2_bytes[1];

        let result = dequantize_q5_1(&data, 64);
        assert_eq!(result.len(), 64);
        for i in 0..32 {
            assert!(
                result[i].abs() < 1e-3,
                "block 0 element {i}: expected 0.0, got {}",
                result[i]
            );
        }
        for i in 32..64 {
            assert!(
                (result[i] - 5.0).abs() < 1e-2,
                "block 1 element {}: expected 5.0, got {}",
                i - 32,
                result[i]
            );
        }
    }

    #[test]
    fn test_q5_1_dequant_max_quant() {
        // d=1.0, m=0.0, lo nibble=15, high bit=1 -> q = 15 | (1<<4) = 31
        // val = 1.0 * 31 + 0.0 = 31.0
        let mut block = vec![0u8; 24];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // m = 0 (already zero)
        // qh = 0xFFFFFFFF (all high bits set)
        block[4] = 0xFF;
        block[5] = 0xFF;
        block[6] = 0xFF;
        block[7] = 0xFF;
        // qs: lo=15, hi=15 -> 0xFF
        for i in 0..16 {
            block[8 + i] = 0xFF;
        }
        let result = dequantize_q5_1(&block, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 31.0).abs() < 1e-2,
                "element {i}: expected 31.0, got {v}"
            );
        }
    }

    // --- Q6_K tests ---

    #[test]
    fn test_q6k_block_size_and_type_size() {
        assert_eq!(GgufDType::Q6K.block_size(), 256);
        assert_eq!(GgufDType::Q6K.type_size(), 210);
    }

    #[test]
    fn test_q6k_dequant_zero_scale_produces_zeros() {
        // d=0: all outputs should be 0.0 regardless of scales and quant values.
        let mut block = vec![0u8; 210];
        // Fill ql, qh, scales with nonzero data.
        for i in 0..192 {
            block[i] = 0xAB;
        }
        for i in 192..208 {
            block[i] = 0x7F; // scales = 127
        }
        // d = f16(0.0) at bytes 208..210 (already zero)
        let result = dequantize_q6_k(&block, 256);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0, "all values should be 0 when d=0");
        }
    }

    #[test]
    fn test_q6k_dequant_zero_scales_produces_zeros() {
        // d=1.0, but all per-sub-block scales=0: output should be 0.0.
        let mut block = vec![0u8; 210];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        // scales[0..16] = 0 (already zero)
        // Fill ql, qh with nonzero.
        for i in 0..192 {
            block[i] = 0xFF;
        }
        let result = dequantize_q6_k(&block, 256);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0, "all values should be 0 when per-sub-block scales=0");
        }
    }

    #[test]
    fn test_q6k_dequant_centered_q32() {
        // d=1.0, scale=1 (int8), all q6 values = 32 -> (32 - 32) = 0 -> output 0.0.
        // q6=32: lo4 = 32 & 0x0F = 0, hi2 = (32 >> 4) = 2.
        // So ql bytes = 0x00, qh crumbs = 2 (0b10).
        let mut block = vec![0u8; 210];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        // scales[0..16] = 1 (as i8)
        for i in 192..208 {
            block[i] = 1;
        }
        // ql[0..128] = 0x00 (low nibble = 0 for all) — already zero.
        // qh[0..64]: each byte holds 4 crumbs. Each crumb = 2 (0b10).
        // 0b10_10_10_10 = 0xAA.
        for i in 128..192 {
            block[i] = 0xAA;
        }
        let result = dequantize_q6_k(&block, 256);
        assert_eq!(result.len(), 256);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-4, "element {i}: expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q6k_dequant_known_values() {
        // d=0.5, sub-block 0: scale=2, q6=35 for all 16 elements.
        // val = 0.5 * 2 * (35 - 32) = 0.5 * 2 * 3 = 3.0
        //
        // q6=35: lo4 = 35 & 0x0F = 3, hi2 = (35 >> 4) = 2.
        // ql nibbles: each pair = (3, 3) -> 0x33.
        // qh crumbs: each crumb = 2 -> 0xAA per byte.
        let mut block = vec![0u8; 210];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // scales[0] = 2 (i8), rest = 0.
        block[192] = 2;

        // Sub-block 0 occupies values 0..16.
        // ql for values 0..16: bytes ql[0..8] (2 nibbles per byte).
        for i in 0..8 {
            block[i] = 0x33; // lo nibble=3, hi nibble=3
        }
        // qh for values 0..16: bytes qh[0..4] (4 crumbs per byte).
        for i in 0..4 {
            block[128 + i] = 0xAA; // each crumb = 2
        }

        let result = dequantize_q6_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: val = 0.5 * 2 * (35 - 32) = 3.0
        for i in 0..16 {
            assert!(
                (result[i] - 3.0).abs() < 1e-3,
                "sub-block 0 element {i}: expected 3.0, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..16: scale=0 -> all 0.0 (regardless of q6 values).
        for i in 16..256 {
            assert!(
                result[i].abs() < 1e-4,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q6k_dequant_negative_scale() {
        // d=1.0, scale=-3 (i8 = 0xFD), q6=32 -> (32-32)=0 -> 0.0 regardless of scale.
        // q6=33: lo4=1, hi2=2 -> val = 1.0 * (-3) * (33-32) = -3.0.
        let mut block = vec![0u8; 210];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // scales[0] = -3 as i8 = 0xFD.
        block[192] = 0xFD_u8;

        // Sub-block 0: q6=33 -> lo4=1, hi2=2.
        // ql[0..8]: lo nibble=1, hi nibble=1 -> 0x11.
        for i in 0..8 {
            block[i] = 0x11;
        }
        // qh[0..4]: crumb=2 -> 0xAA.
        for i in 0..4 {
            block[128 + i] = 0xAA;
        }

        let result = dequantize_q6_k(&block, 256);
        for i in 0..16 {
            assert!(
                (result[i] - (-3.0)).abs() < 1e-3,
                "element {i}: expected -3.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q6k_dequant_two_blocks() {
        // Verify multi-block dequantization (2 blocks = 512 elements).
        let mut data = vec![0u8; 210 * 2];
        let d1_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let d2_bytes = half::f16::from_f32(2.0).to_le_bytes();
        // Block 0: d=1.0, all scales=0 -> all zeros.
        data[208] = d1_bytes[0];
        data[209] = d1_bytes[1];
        // Block 1: d=2.0, all scales=0 -> all zeros.
        data[210 + 208] = d2_bytes[0];
        data[210 + 209] = d2_bytes[1];

        let result = dequantize_q6_k(&data, 512);
        assert_eq!(result.len(), 512);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q6k_dequant_max_quant() {
        // q6=63 (max 6-bit): lo4=15, hi2=3. val = d * scale * (63 - 32) = d * scale * 31.
        // d=0.5, sub-block 0: scale=4, q6=63
        // val = 0.5 * 4 * 31 = 62.0
        let mut block = vec![0u8; 210];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // scales[0] = 4 (i8)
        block[192] = 4;

        // Sub-block 0: values 0..16.
        // ql[0..8]: lo nibble=15, hi nibble=15 -> 0xFF.
        for i in 0..8 {
            block[i] = 0xFF;
        }
        // qh[0..4]: crumb=3 -> 0b11_11_11_11 = 0xFF.
        for i in 0..4 {
            block[128 + i] = 0xFF;
        }

        let result = dequantize_q6_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: 0.5 * 4 * (63 - 32) = 62.0
        for i in 0..16 {
            assert!(
                (result[i] - 62.0).abs() < 1e-2,
                "element {i}: expected 62.0, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..16: scale=0 -> 0.0
        for i in 16..256 {
            assert!(
                result[i].abs() < 1e-4,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q6k_dequant_min_quant() {
        // q6=0 (min 6-bit): lo4=0, hi2=0. val = d * scale * (0 - 32) = d * scale * (-32).
        // d=1.0, sub-block 0: scale=2, q6=0
        // val = 1.0 * 2 * (-32) = -64.0
        let mut block = vec![0u8; 210];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // scales[0] = 2 (i8)
        block[192] = 2;

        // ql and qh all zero (q6=0) -- already zero from vec init.

        let result = dequantize_q6_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: 1.0 * 2 * (0 - 32) = -64.0
        for i in 0..16 {
            assert!(
                (result[i] - (-64.0)).abs() < 1e-2,
                "element {i}: expected -64.0, got {}",
                result[i]
            );
        }
    }

    // --- Q2_K tests ---

    #[test]
    fn test_q2k_block_size_and_type_size() {
        assert_eq!(GgufDType::Q2K.block_size(), 256);
        assert_eq!(GgufDType::Q2K.type_size(), 84);
    }

    #[test]
    fn test_q2k_dequant_zero_d_and_dmin_produces_zeros() {
        // d=0, dmin=0: all outputs = 0.0.
        let mut block = vec![0u8; 84];
        // Fill scales and qs with nonzero data.
        for i in 0..16 {
            block[i] = 0xFF;
        }
        for i in 16..80 {
            block[i] = 0xFF;
        }
        // d=0, dmin=0 at bytes 80..84 (already zero).
        let result = dequantize_q2_k(&block, 256);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0, "all values should be 0 when d=0 and dmin=0");
        }
    }

    #[test]
    fn test_q2k_dequant_known_values() {
        // d=1.0, dmin=0.0
        // Sub-block 0: scale=3, min=0, all q2 values = 2
        // val = d * scale * q - dmin * min = 1.0 * 3 * 2 - 0 = 6.0
        let mut block = vec![0u8; 84];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[80] = d_bytes[0];
        block[81] = d_bytes[1];
        // dmin = 0.0 (bytes 82..84 already zero).

        // scales[0]: lo4 = 3 (scale), hi4 = 0 (min).
        block[0] = 0x03;

        // Sub-block 0 values 0..16: all q2=2.
        // qs: byte idx/4, each byte holds 4 crumbs.
        // Values 0..16 -> qs bytes 0..4.
        // Each crumb = 2 -> 0b10_10_10_10 = 0xAA.
        for i in 0..4 {
            block[16 + i] = 0xAA;
        }

        let result = dequantize_q2_k(&block, 256);
        assert_eq!(result.len(), 256);

        for i in 0..16 {
            assert!(
                (result[i] - 6.0).abs() < 1e-3,
                "sub-block 0 element {i}: expected 6.0, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..16: scale=0, min=0 -> all 0.0.
        for i in 16..256 {
            assert!(
                result[i].abs() < 1e-6,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q2k_dequant_with_dmin() {
        // d=1.0, dmin=0.5
        // Sub-block 0: scale=1, min=4, q2=0
        // val = 1.0 * 1 * 0 - 0.5 * 4 = -2.0
        let mut block = vec![0u8; 84];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[80] = d_bytes[0];
        block[81] = d_bytes[1];
        block[82] = dmin_bytes[0];
        block[83] = dmin_bytes[1];

        // scales[0]: lo4 = 1 (scale), hi4 = 4 (min) -> 0x41.
        block[0] = 0x41;

        // All qs = 0 (q2=0 for all, already zero).

        let result = dequantize_q2_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: d * 1 * 0 - dmin * 4 = -2.0
        for i in 0..16 {
            assert!(
                (result[i] - (-2.0)).abs() < 1e-3,
                "element {i}: expected -2.0, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..16: scale=0, min=0 -> 0.0.
        for i in 16..256 {
            assert!(
                result[i].abs() < 1e-6,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q2k_dequant_max_quant() {
        // d=2.0, dmin=1.0
        // Sub-block 0: scale=15, min=15, all q2=3 (max 2-bit value)
        // val = 2.0 * 15 * 3 - 1.0 * 15 = 90.0 - 15.0 = 75.0
        let mut block = vec![0u8; 84];
        let d_bytes = half::f16::from_f32(2.0).to_le_bytes();
        let dmin_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[80] = d_bytes[0];
        block[81] = d_bytes[1];
        block[82] = dmin_bytes[0];
        block[83] = dmin_bytes[1];

        // scales[0]: lo4=15, hi4=15 -> 0xFF.
        block[0] = 0xFF;

        // q2=3 for sub-block 0: 0b11_11_11_11 = 0xFF.
        for i in 0..4 {
            block[16 + i] = 0xFF;
        }

        let result = dequantize_q2_k(&block, 256);

        for i in 0..16 {
            assert!(
                (result[i] - 75.0).abs() < 1e-1,
                "element {i}: expected 75.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q2k_dequant_two_blocks() {
        // Verify multi-block dequantization (2 blocks = 512 elements).
        let mut data = vec![0u8; 84 * 2];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        // Block 0: d=1.0, dmin=0, all scales/qs=0 -> all zeros.
        data[80] = d_bytes[0];
        data[81] = d_bytes[1];
        // Block 1: d=1.0, dmin=0, all scales/qs=0 -> all zeros.
        data[84 + 80] = d_bytes[0];
        data[84 + 81] = d_bytes[1];

        let result = dequantize_q2_k(&data, 512);
        assert_eq!(result.len(), 512);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    // --- Q3_K tests ---

    #[test]
    fn test_q3k_block_size_and_type_size() {
        assert_eq!(GgufDType::Q3K.block_size(), 256);
        assert_eq!(GgufDType::Q3K.type_size(), 110);
    }

    #[test]
    fn test_q3k_dequant_zero_d_produces_zeros() {
        // d=0: all outputs should be 0.0 regardless of scales and quant values.
        let mut block = vec![0u8; 110];
        // Fill hmask, qs, scales with nonzero data.
        for i in 0..108 {
            block[i] = 0xFF;
        }
        // d = f16(0.0) at bytes 108..110 (already zero).
        let result = dequantize_q3_k(&block, 256);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0, "all values should be 0 when d=0");
        }
    }

    #[test]
    fn test_q3k_dequant_centered_q4() {
        // All q3=4 (centered): val = d * scale * (4 - 4) = 0.0 for any d, scale.
        // q3=4: qs_2bit=0 (4 & 0x3 = 0), hmask_bit=1 (4 >> 2 = 1).
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // Set all scales to nonzero (6-bit unsigned 33 - 32 = 1 as signed).
        // lo4=1, hi2=2 => 6-bit = 1 | (2 << 4) = 33, signed = 33 - 32 = 1.
        for k in 0..8 {
            // lo4: scale byte k/2, nibble k%2. We want lo4=1 for all 16 sub-blocks.
            // Bytes 0..8: 0x11 -> lo nibble=1, hi nibble=1.
            block[96 + k] = 0x11;
        }
        // hi2=2 for all: bytes 8..12 of scales.
        // Each byte holds 4 sub-blocks. hi2=2 => 0b10_10_10_10 = 0xAA.
        for k in 0..4 {
            block[96 + 8 + k] = 0xAA;
        }

        // hmask: all bits set (high bit = 1 for all 256 values).
        for i in 0..32 {
            block[i] = 0xFF;
        }
        // qs: all zero (low 2 bits = 0 for all 256 values).
        // Already zero from vec init.

        let result = dequantize_q3_k(&block, 256);
        assert_eq!(result.len(), 256);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-4, "element {i}: expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q3k_dequant_known_values() {
        // d=0.5, sub-block 0: scale signed = 1, all q3=5.
        // val = 0.5 * 1 * (5 - 4) = 0.5
        //
        // q3=5: qs_2bit = 5 & 3 = 1, hmask_bit = 5 >> 2 = 1.
        // scale: we need signed scale = 1, so 6-bit unsigned = 33, 33 - 32 = 1.
        // lo4=1, hi2=2. lo4=1 means byte 0 of scales_raw lo nibble = 1.
        // hi2=2 means byte 8 of scales_raw bits [0:1] = 0b10.
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // Scale for sub-block 0: lo4=1, hi2=2.
        block[96] = 0x01; // byte 0: lo nibble for sub-block 0 = 1, hi nibble for sub-block 1 = 0.
        block[96 + 8] = 0x02; // byte 8: hi2 for sub-block 0 = 2 (bits 0..1), sub-block 1 = 0, etc.

        // Sub-block 0: values 0..16.
        // hmask: set bit for each of values 0..15. Values 0..7 are in byte 0, values 8..15 in byte 1.
        block[0] = 0xFF; // bits 0..7 set
        block[1] = 0xFF; // bits 8..15 set (only 8..15 used for sub-block 0)

        // qs: values 0..15. Each crumb = 1 (qs_2bit=1). 4 per byte.
        // 0b01_01_01_01 = 0x55.
        for i in 0..4 {
            block[32 + i] = 0x55;
        }

        let result = dequantize_q3_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0 (values 0..16): 0.5 * 1 * (5 - 4) = 0.5.
        for i in 0..16 {
            assert!(
                (result[i] - 0.5).abs() < 1e-3,
                "sub-block 0 element {i}: expected 0.5, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..16 have scale=0-32=-32 by default (all scales_raw bytes are 0
        // except what we set), and q3 values depend on the zero-initialized data.
        // With qs=0 and hmask=0 (for sub-blocks >= 2): q3=0, val = d * (-32) * (0-4) = 0.5 * (-32) * (-4) = 64.0.
        // But sub-block 1 has hmask bytes 0,1 partially overlapping.
        // For sub-blocks with zero scale_raw bytes: 6-bit unsigned = 0, signed = 0-32 = -32.
        // So let's just verify sub-block 0 and skip the rest — the important test is known-value correctness.
    }

    #[test]
    fn test_q3k_dequant_zero_scales_with_nonzero_d() {
        // d=1.0, all scale bytes = 0 → 6-bit unsigned=0, signed=0-32=-32.
        // All qs=0, hmask=0 → q3=0, val = 1.0 * (-32) * (0-4) = 128.0.
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];
        // Everything else is zero.

        let result = dequantize_q3_k(&block, 256);
        assert_eq!(result.len(), 256);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 128.0).abs() < 1e-1,
                "element {i}: expected 128.0, got {v}"
            );
        }
    }

    #[test]
    fn test_q3k_dequant_all_q3_max() {
        // q3=7 (max): qs_2bit=3, hmask_bit=1.
        // val = d * scale * (7 - 4) = d * scale * 3.
        // d=1.0, set all scales to signed 1 (6-bit unsigned 33).
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // All scales: lo4=1, hi2=2 → 6-bit=33 → signed=1.
        for k in 0..8 {
            block[96 + k] = 0x11;
        }
        for k in 0..4 {
            block[96 + 8 + k] = 0xAA;
        }

        // hmask: all set.
        for i in 0..32 {
            block[i] = 0xFF;
        }

        // qs: all crumbs = 3 → 0xFF.
        for i in 0..64 {
            block[32 + i] = 0xFF;
        }

        let result = dequantize_q3_k(&block, 256);
        assert_eq!(result.len(), 256);
        // val = 1.0 * 1 * (7 - 4) = 3.0
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 3.0).abs() < 1e-3, "element {i}: expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_q3k_dequant_all_q3_min() {
        // q3=0 (min): qs_2bit=0, hmask_bit=0.
        // val = d * scale * (0 - 4) = d * scale * (-4).
        // d=0.5, all scales signed=2 (6-bit unsigned 34).
        // lo4=2, hi2=2 → 34 → 34-32=2.
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // scales: lo4=2 for all 16 sub-blocks.
        for k in 0..8 {
            block[96 + k] = 0x22; // lo4=2 for even sub-block, lo4=2 for odd
        }
        // hi2=2 for all.
        for k in 0..4 {
            block[96 + 8 + k] = 0xAA;
        }

        // hmask: all zero, qs: all zero → q3=0.
        // Already zero from vec init.

        let result = dequantize_q3_k(&block, 256);
        assert_eq!(result.len(), 256);
        // val = 0.5 * 2 * (0 - 4) = -4.0
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-4.0)).abs() < 1e-3,
                "element {i}: expected -4.0, got {v}"
            );
        }
    }

    #[test]
    fn test_q3k_dequant_negative_scale() {
        // Signed scale = -1 (6-bit unsigned = 31).
        // lo4 = 31 & 0x0F = 15, hi2 = (31 >> 4) = 1.
        // d=1.0, q3=5 → val = 1.0 * (-1) * (5-4) = -1.0.
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // Scale for sub-block 0 only: lo4=15, hi2=1.
        block[96] = 0x0F; // byte 0: lo nibble for sub-block 0 = 15
        block[96 + 8] = 0x01; // byte 8: hi2 for sub-block 0 = 1

        // Sub-block 0: values 0..15.
        // q3=5: qs_2bit=1, hmask_bit=1.
        block[0] = 0xFF; // hmask bits 0..7
        block[1] = 0xFF; // hmask bits 8..15
        for i in 0..4 {
            block[32 + i] = 0x55; // qs crumbs = 1
        }

        let result = dequantize_q3_k(&block, 256);

        // Sub-block 0: 1.0 * (-1) * (5 - 4) = -1.0
        for i in 0..16 {
            assert!(
                (result[i] - (-1.0)).abs() < 1e-3,
                "element {i}: expected -1.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q3k_dequant_two_blocks() {
        // Two blocks = 512 elements. Both with d=0 → all zeros.
        let mut data = vec![0u8; 110 * 2];
        // d=0 for both blocks (already zero).
        // Fill with nonzero noise.
        for i in 0..108 {
            data[i] = 0xAB;
            data[110 + i] = 0xCD;
        }
        let result = dequantize_q3_k(&data, 512);
        assert_eq!(result.len(), 512);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q3k_dequant_mixed_high_bits() {
        // Test that hmask correctly selects the high bit per value.
        // d=1.0, sub-block 0: scale signed=1 (6-bit unsigned 33).
        // Values 0..7: hmask=0, qs_2bit=0 → q3=0, val=1*(0-4)=-4.
        // Values 8..15: hmask=1, qs_2bit=0 → q3=4, val=1*(4-4)=0.
        let mut block = vec![0u8; 110];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // Scale for sub-block 0: lo4=1, hi2=2.
        block[96] = 0x01;
        block[96 + 8] = 0x02;

        // hmask: values 0..7 are bits 0..7 of byte 0 → zero.
        // Values 8..15 are bits 0..7 of byte 1 → set.
        block[0] = 0x00; // values 0..7: hmask=0
        block[1] = 0xFF; // values 8..15: hmask=1

        // qs: all zero (already zero).

        let result = dequantize_q3_k(&block, 256);

        // Values 0..7 in sub-block 0: q3=0, val=-4.0.
        for i in 0..8 {
            assert!(
                (result[i] - (-4.0)).abs() < 1e-3,
                "element {i}: expected -4.0, got {}",
                result[i]
            );
        }
        // Values 8..15 in sub-block 0: q3=4, val=0.0.
        for i in 8..16 {
            assert!(
                result[i].abs() < 1e-3,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    // --- Q5_K tests ---

    #[test]
    fn test_q5k_block_size_and_type_size() {
        assert_eq!(GgufDType::Q5K.block_size(), 256);
        assert_eq!(GgufDType::Q5K.type_size(), 176);
    }

    #[test]
    fn test_q5k_dequant_zero_d_and_dmin_produces_zeros() {
        // d=0, dmin=0: all outputs should be 0.0.
        let mut block = vec![0u8; 176];
        // Fill scales, qh, qs with nonzero data.
        for i in 4..176 {
            block[i] = 0xFF;
        }
        // d=0, dmin=0 at bytes 0..4 (already zero).
        let result = dequantize_q5_k(&block, 256);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0, "all values should be 0 when d=0 and dmin=0");
        }
    }

    #[test]
    fn test_q5k_dequant_known_values() {
        // d=1.0, dmin=0.0.
        // Sub-block 0: scale=2, min=0, all q5=3 (lo4=3, hb=0).
        // val = 1.0 * 2 * 3 - 0 = 6.0 for all 32 elements.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // dmin = 0.0 (bytes 2,3 already zero).

        // scales[0] = 2 (low 4 bits at byte 0 lo nibble of scales_data).
        block[4] = 0x02;

        // qh: all zero (high bits = 0 for all values). Already zero.

        // qs: sub-block 0 occupies values 0..31, bytes qs[0..16] (2 nibbles per byte).
        // Each nibble = 3 → 0x33.
        for i in 0..16 {
            block[48 + i] = 0x33;
        }

        let result = dequantize_q5_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: d * 2 * 3 - 0 = 6.0
        for i in 0..32 {
            assert!(
                (result[i] - 6.0).abs() < 1e-3,
                "sub-block 0 element {i}: expected 6.0, got {}",
                result[i]
            );
        }

        // Sub-blocks 1..7: scale=0, min=0 → 0.0.
        for i in 32..256 {
            assert!(
                result[i].abs() < 1e-6,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q5k_dequant_with_dmin() {
        // d=1.0, dmin=0.5.
        // Sub-block 0: scale=1, min=4, q5=0.
        // val = 1.0 * 1 * 0 - 0.5 * 4 = -2.0.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];

        // scales[0] = 1.
        block[4] = 0x01;
        // mins[0] = 4 (low 4 bits at byte 4 of scales_data, i.e. offset 8).
        block[8] = 0x04;

        // All q5=0 (qs=0, qh=0). Already zero.

        let result = dequantize_q5_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: -2.0.
        for i in 0..32 {
            assert!(
                (result[i] - (-2.0)).abs() < 1e-3,
                "element {i}: expected -2.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q5k_dequant_with_high_bit() {
        // Test qh (high bit) contribution.
        // d=1.0, dmin=0.0, sub-block 0: scale=1, min=0.
        // All lo4=0, all hb=1 → q5 = 0 | (1<<4) = 16.
        // val = 1.0 * 1 * 16 - 0 = 16.0.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];

        // scales[0] = 1.
        block[4] = 0x01;

        // qh: set high bits for all 256 values.
        for i in 0..32 {
            block[16 + i] = 0xFF;
        }

        // qs: all zero (lo4=0 for all). Already zero.

        let result = dequantize_q5_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: val = 1.0 * 1 * 16 = 16.0.
        for i in 0..32 {
            assert!(
                (result[i] - 16.0).abs() < 1e-3,
                "element {i}: expected 16.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q5k_dequant_max_quant() {
        // q5=31 (max): lo4=15, hb=1.
        // d=0.5, dmin=0.0, sub-block 0: scale=4, min=0.
        // val = 0.5 * 4 * 31 = 62.0.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];

        // scales[0] = 4.
        block[4] = 0x04;

        // qh: all high bits set.
        for i in 0..32 {
            block[16 + i] = 0xFF;
        }

        // qs: all nibbles = 15 → 0xFF.
        for i in 0..128 {
            block[48 + i] = 0xFF;
        }

        let result = dequantize_q5_k(&block, 256);
        assert_eq!(result.len(), 256);

        // Sub-block 0: 0.5 * 4 * 31 = 62.0
        for i in 0..32 {
            assert!(
                (result[i] - 62.0).abs() < 1e-1,
                "element {i}: expected 62.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q5k_dequant_mixed_high_bits() {
        // d=1.0, dmin=0.0, sub-block 0: scale=1, min=0.
        // Values 0..7: lo4=0, hb=0 → q5=0, val=0.
        // Values 8..15: lo4=0, hb=1 → q5=16, val=16.
        // Values 16..31: lo4=0, hb=0 → q5=0, val=0.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[4] = 0x01; // scales[0]=1

        // qh byte layout: byte i of qh holds bits for values (i*8)..(i*8+7).
        // Values 8..15 → qh byte 1 (bits for values 8..15).
        block[16 + 1] = 0xFF; // set high bits for values 8..15

        let result = dequantize_q5_k(&block, 256);

        for i in 0..8 {
            assert!(
                result[i].abs() < 1e-4,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
        for i in 8..16 {
            assert!(
                (result[i] - 16.0).abs() < 1e-3,
                "element {i}: expected 16.0, got {}",
                result[i]
            );
        }
        for i in 16..32 {
            assert!(
                result[i].abs() < 1e-4,
                "element {i}: expected 0.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q5k_dequant_two_blocks() {
        // Verify multi-block (2 blocks = 512 elements).
        // Block 0: d=1.0, dmin=0, all scales/qs/qh=0 → val = 0.
        // Block 1: d=2.0, dmin=0, all scales/qs/qh=0 → val = 0.
        let mut data = vec![0u8; 176 * 2];
        let d1 = half::f16::from_f32(1.0).to_le_bytes();
        let d2 = half::f16::from_f32(2.0).to_le_bytes();
        data[0] = d1[0];
        data[1] = d1[1];
        data[176] = d2[0];
        data[176 + 1] = d2[1];

        let result = dequantize_q5_k(&data, 512);
        assert_eq!(result.len(), 512);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q5k_dequant_scale_packing_matches_q4k() {
        // Verify the scale/min packing is identical to Q4_K by constructing
        // the same scale pattern and checking a known sub-block value.
        // d=1.0, dmin=1.0.
        // Sub-block 0: scale=2, min=3, q5=0.
        // val = 1.0 * 2 * 0 - 1.0 * 3 = -3.0.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let dmin_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];

        // scales[0] = 2 (lo4 of byte 0 lo nibble).
        block[4] = 0x02;
        // mins[0] = 3 (lo4 of byte 4 of scales_data, i.e. offset 8 lo nibble).
        block[8] = 0x03;

        let result = dequantize_q5_k(&block, 256);

        for i in 0..32 {
            assert!(
                (result[i] - (-3.0)).abs() < 1e-3,
                "element {i}: expected -3.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_q5k_dequant_high_scale_bits() {
        // Test the high 2-bit scale packing (bytes 8..12 of scales_data).
        // Sub-block 0: scale with hi2 set → scale = lo4 | (hi2 << 4).
        // lo4=3, hi2=2 → scale=3 + 32 = 35.
        // d=0.1, dmin=0.0, q5=1.
        // val = 0.1 * 35 * 1 = 3.5.
        let mut block = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(0.1).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];

        // scales[0] lo4=3.
        block[4] = 0x03;
        // scales[0] hi2=2 → byte 8 of scales_data, bits [0:1] = 2.
        block[4 + 8] = 0x02;

        // All qs nibbles=1 → 0x11 for sub-block 0 (16 bytes = 32 nibbles).
        for i in 0..16 {
            block[48 + i] = 0x11;
        }
        // qh: all zero (no high bit). Already zero.

        let result = dequantize_q5_k(&block, 256);

        let expected = half::f16::from_f32(0.1).to_f32() * 35.0 * 1.0;
        for i in 0..32 {
            assert!(
                (result[i] - expected).abs() < 1e-2,
                "element {i}: expected {expected}, got {}",
                result[i]
            );
        }
    }
}
