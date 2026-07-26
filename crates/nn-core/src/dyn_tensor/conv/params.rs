// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Named-field parameter structs for convolution operations.
//!
//! Replaces bare `usize` parameter soup that caused P1 bug #1484
//! (silent stride↔output_padding swap in `TrackedTensor::conv_transpose1d`).

use crate::error::{Result, TensorError};

/// Parameters for 1D convolution.
///
/// Named fields prevent parameter-order mistakes that the compiler cannot catch
/// when all parameters are bare `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Conv1dParams {
    /// Zero-padding added to both sides of the input.
    pub padding: usize,
    /// Stride of the convolution.
    pub stride: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for Conv1dParams {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl Conv1dParams {
    /// Validate that stride, dilation, and groups are non-zero.
    pub fn validate(&self) -> Result<()> {
        if self.stride == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.dilation == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "dilation",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        Ok(())
    }
}

/// Parameters for 2D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Conv2dParams {
    /// Zero-padding added to both sides of each spatial dimension.
    pub padding: usize,
    /// Stride of the convolution.
    pub stride: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl Conv2dParams {
    /// Validate that stride, dilation, and groups are non-zero.
    pub fn validate(&self) -> Result<()> {
        if self.stride == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.dilation == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "dilation",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        Ok(())
    }
}

/// Parameters for transposed 1D convolution (deconvolution).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ConvTranspose1dParams {
    /// Zero-padding subtracted from both sides of the output.
    pub padding: usize,
    /// Additional size added to one side of the output.
    pub output_padding: usize,
    /// Stride of the convolution.
    pub stride: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for ConvTranspose1dParams {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl ConvTranspose1dParams {
    /// Validate that stride, dilation, and groups are non-zero,
    /// and output_padding < stride.
    pub fn validate(&self) -> Result<()> {
        if self.stride == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.dilation == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "dilation",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.output_padding >= self.stride {
            return Err(TensorError::ConvParameterInvalid {
                param: "output_padding",
                value: self.output_padding,
                reason: "must be < stride",
            });
        }
        Ok(())
    }
}

/// Parameters for transposed 2D convolution (deconvolution).
///
/// Spatial parameters use `[usize; 2]` for `[height, width]` to support
/// non-square transposed convolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ConvTranspose2dParams {
    /// Zero-padding subtracted from both sides of each spatial dimension `[pad_h, pad_w]`.
    pub padding: [usize; 2],
    /// Additional size added to one side of each spatial dimension `[opad_h, opad_w]`.
    pub output_padding: [usize; 2],
    /// Stride of the convolution `[stride_h, stride_w]`.
    pub stride: [usize; 2],
    /// Spacing between kernel elements `[dil_h, dil_w]`.
    pub dilation: [usize; 2],
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for ConvTranspose2dParams {
    fn default() -> Self {
        Self {
            padding: [0, 0],
            output_padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

impl ConvTranspose2dParams {
    /// Validate that stride, dilation, and groups are non-zero,
    /// and output_padding < stride (per dimension).
    pub fn validate(&self) -> Result<()> {
        if self.stride[0] == 0 || self.stride[1] == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.dilation[0] == 0 || self.dilation[1] == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "dilation",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.output_padding[0] >= self.stride[0] || self.output_padding[1] >= self.stride[1] {
            return Err(TensorError::ConvParameterInvalid {
                param: "output_padding",
                value: self.output_padding[0].max(self.output_padding[1]),
                reason: "must be < stride (per dimension)",
            });
        }
        Ok(())
    }
}

/// Parameters for 3D convolution.
///
/// Spatial parameters use `[usize; 3]` for `[depth, height, width]` to support
/// non-cubic convolutions (e.g., Qwen3-VL 3D patch embedding with temporal+spatial dims).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Conv3dParams {
    /// Zero-padding added to both sides of each spatial dimension `[pad_d, pad_h, pad_w]`.
    pub padding: [usize; 3],
    /// Stride of the convolution `[stride_d, stride_h, stride_w]`.
    pub stride: [usize; 3],
    /// Spacing between kernel elements `[dil_d, dil_h, dil_w]`.
    pub dilation: [usize; 3],
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for Conv3dParams {
    fn default() -> Self {
        Self {
            padding: [0, 0, 0],
            stride: [1, 1, 1],
            dilation: [1, 1, 1],
            groups: 1,
        }
    }
}

impl Conv3dParams {
    /// Validate that stride, dilation, and groups are non-zero.
    pub fn validate(&self) -> Result<()> {
        if self.stride[0] == 0 || self.stride[1] == 0 || self.stride[2] == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.dilation[0] == 0 || self.dilation[1] == 0 || self.dilation[2] == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "dilation",
                value: 0,
                reason: "must be > 0",
            });
        }
        if self.groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_params_default() {
        let p = Conv1dParams::default();
        assert_eq!(p.padding, 0);
        assert_eq!(p.stride, 1);
        assert_eq!(p.dilation, 1);
        assert_eq!(p.groups, 1);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_conv1d_params_zero_stride() {
        let p = Conv1dParams {
            stride: 0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv1d_params_zero_groups() {
        let p = Conv1dParams {
            groups: 0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv_transpose1d_params_default() {
        let p = ConvTranspose1dParams::default();
        assert_eq!(p.padding, 0);
        assert_eq!(p.output_padding, 0);
        assert_eq!(p.stride, 1);
        assert_eq!(p.dilation, 1);
        assert_eq!(p.groups, 1);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_conv_transpose1d_params_output_padding_too_large() {
        let p = ConvTranspose1dParams {
            stride: 2,
            output_padding: 2,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv2d_params_default() {
        let p = Conv2dParams::default();
        assert_eq!(p.padding, 0);
        assert_eq!(p.stride, 1);
        assert_eq!(p.dilation, 1);
        assert_eq!(p.groups, 1);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_conv2d_params_zero_dilation() {
        let p = Conv2dParams {
            dilation: 0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv_transpose2d_params_default() {
        let p = ConvTranspose2dParams::default();
        assert_eq!(p.padding, [0, 0]);
        assert_eq!(p.output_padding, [0, 0]);
        assert_eq!(p.stride, [1, 1]);
        assert_eq!(p.dilation, [1, 1]);
        assert_eq!(p.groups, 1);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_conv_transpose2d_params_output_padding_too_large() {
        let p = ConvTranspose2dParams {
            stride: [2, 2],
            output_padding: [2, 2],
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv_transpose2d_params_zero_stride() {
        let p = ConvTranspose2dParams {
            stride: [0, 0],
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv3d_params_default() {
        let p = Conv3dParams::default();
        assert_eq!(p.padding, [0, 0, 0]);
        assert_eq!(p.stride, [1, 1, 1]);
        assert_eq!(p.dilation, [1, 1, 1]);
        assert_eq!(p.groups, 1);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_conv3d_params_zero_stride() {
        let p = Conv3dParams {
            stride: [0, 1, 1],
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv3d_params_zero_dilation() {
        let p = Conv3dParams {
            dilation: [1, 0, 1],
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_conv3d_params_zero_groups() {
        let p = Conv3dParams {
            groups: 0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }
}
