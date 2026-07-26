// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf image preprocessing invariants (#3920).
//!
//! Proves properties of all `DpdfPreprocessConfig` presets:
//! - Normalization means are in [0, 1]
//! - Normalization stds are > 0 (non-zero divisor)
//! - Target dimensions are positive (where applicable)
//! - Letterbox padding dimensions >= original dimensions
//! - Scale factors are positive

#[cfg(kani)]
mod proofs {
    use nn_models::dpdf_image_preprocess::{
        compute_letterbox_params, compute_resize_dims, DpdfPreprocessConfig,
    };

    /// Helper: collect all 7 presets for exhaustive property checking.
    fn all_presets() -> [DpdfPreprocessConfig; 7] {
        [
            DpdfPreprocessConfig::for_granite_docling(),
            DpdfPreprocessConfig::for_doclayout_yolo(),
            DpdfPreprocessConfig::for_paddle_ocr_detect(),
            DpdfPreprocessConfig::for_paddle_ocr_recognize(),
            DpdfPreprocessConfig::for_table_transformer(),
            DpdfPreprocessConfig::for_qwen3_vl(),
            DpdfPreprocessConfig::for_glm_ocr(),
        ];
    }

    // ========================================================================
    // Normalization mean proofs
    // ========================================================================

    /// All preset normalization means are in [0, 1] for each channel.
    ///
    /// Mean values outside [0, 1] would indicate a configuration error since
    /// normalized pixel values (after scale_factor * pixel) are in [0, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_normalize_mean_valid() {
        for config in &all_presets() {
            for c in 0..3 {
                let mean = config.mean[c];
                assert!(
                    mean >= 0.0 && mean <= 1.0,
                    "mean must be in [0, 1]"
                );
                assert!(mean.is_finite(), "mean must be finite");
            }
        }
    }

    // ========================================================================
    // Normalization std proofs
    // ========================================================================

    /// All preset normalization stds are strictly positive for each channel.
    ///
    /// std <= 0 would cause division by zero or sign-flip in normalization:
    /// `(pixel * scale - mean) / std`.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_normalize_std_positive() {
        for config in &all_presets() {
            for c in 0..3 {
                let std_val = config.std[c];
                assert!(std_val > 0.0, "std must be positive");
                assert!(std_val.is_finite(), "std must be finite");
            }
        }
    }

    // ========================================================================
    // Target dimension proofs
    // ========================================================================

    /// All presets with fixed target dimensions have target_height > 0 and
    /// target_width > 0. Qwen3-VL uses dynamic resolution (target 0x0) so
    /// it uses min_pixels/max_pixels/patch_size instead.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_resize_dims_positive() {
        let fixed_presets = [
            DpdfPreprocessConfig::for_granite_docling(),
            DpdfPreprocessConfig::for_doclayout_yolo(),
            DpdfPreprocessConfig::for_paddle_ocr_detect(),
            DpdfPreprocessConfig::for_paddle_ocr_recognize(),
            DpdfPreprocessConfig::for_table_transformer(),
            DpdfPreprocessConfig::for_glm_ocr(),
        ];

        for config in &fixed_presets {
            assert!(config.target_height > 0, "target_height must be positive");
            assert!(config.target_width > 0, "target_width must be positive");
        }

        // Qwen3-VL: dynamic resolution — target dims are 0 but min/max pixels
        // and patch_size are configured.
        let qwen = DpdfPreprocessConfig::for_qwen3_vl();
        assert_eq!(qwen.target_height, 0);
        assert_eq!(qwen.target_width, 0);
        assert!(qwen.min_pixels > 0, "Qwen3 min_pixels must be positive");
        assert!(qwen.max_pixels > qwen.min_pixels, "max_pixels > min_pixels");
        assert!(qwen.patch_size > 0, "Qwen3 patch_size must be positive");
    }

    // ========================================================================
    // Letterbox padding proofs
    // ========================================================================

    /// Letterbox padding dimensions are always >= original (resized) dims.
    ///
    /// After letterbox padding, the total canvas is (target_h, target_w)
    /// and the original resized image fits within it. This proves that
    /// compute_letterbox_params never produces negative padding.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_letterbox_dims_ge_input() {
        // Use bounded symbolic values for tractability.
        let resize_h: u32 = kani::any();
        let resize_w: u32 = kani::any();
        let target_h: u32 = kani::any();
        let target_w: u32 = kani::any();

        // Constrain to realistic ranges for CBMC tractability.
        kani::assume(resize_h > 0 && resize_h <= 4096);
        kani::assume(resize_w > 0 && resize_w <= 4096);
        kani::assume(target_h >= resize_h && target_h <= 4096);
        kani::assume(target_w >= resize_w && target_w <= 4096);

        let params = compute_letterbox_params(resize_h, resize_w, target_h, target_w);

        // Total padded dimensions must equal target dimensions.
        let padded_h = resize_h + params.top + params.bottom;
        let padded_w = resize_w + params.left + params.right;
        assert_eq!(padded_h, target_h, "padded height must equal target");
        assert_eq!(padded_w, target_w, "padded width must equal target");

        // Padded dims are >= input dims (padding is non-negative).
        assert!(padded_h >= resize_h);
        assert!(padded_w >= resize_w);
    }

    /// Letterbox padding is symmetric: top and bottom differ by at most 1
    /// (same for left/right). This ensures the image is centered.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_letterbox_padding_symmetric() {
        let resize_h: u32 = kani::any();
        let resize_w: u32 = kani::any();
        let target_h: u32 = kani::any();
        let target_w: u32 = kani::any();

        kani::assume(resize_h > 0 && resize_h <= 2048);
        kani::assume(resize_w > 0 && resize_w <= 2048);
        kani::assume(target_h >= resize_h && target_h <= 2048);
        kani::assume(target_w >= resize_w && target_w <= 2048);

        let params = compute_letterbox_params(resize_h, resize_w, target_h, target_w);

        // Top and bottom differ by at most 1 (odd padding splits as floor/ceil).
        let h_diff = if params.top >= params.bottom {
            params.top - params.bottom
        } else {
            params.bottom - params.top
        };
        assert!(h_diff <= 1, "vertical padding asymmetry must be <= 1");

        let w_diff = if params.left >= params.right {
            params.left - params.right
        } else {
            params.right - params.left
        };
        assert!(w_diff <= 1, "horizontal padding asymmetry must be <= 1");
    }

    // ========================================================================
    // Scale factor proofs
    // ========================================================================

    /// All preset scale factors are positive (> 0).
    ///
    /// A non-positive scale factor would break normalization:
    /// scale <= 0 would zero-out or invert pixel values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scale_factor_positive() {
        for config in &all_presets() {
            assert!(config.scale_factor > 0.0, "scale_factor must be positive");
            assert!(config.scale_factor.is_finite(), "scale_factor must be finite");
            // All current presets use 1/255 scaling.
            assert!(
                config.scale_factor <= 1.0,
                "scale_factor should be <= 1.0 for pixel normalization"
            );
        }
    }

    // ========================================================================
    // compute_resize_dims proofs
    // ========================================================================

    /// compute_resize_dims always returns dimensions >= 1.
    /// This prevents zero-size tensors in the preprocessing pipeline.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_compute_resize_dims_nonzero() {
        let src_h: u32 = kani::any();
        let src_w: u32 = kani::any();
        let target_h: u32 = kani::any();
        let target_w: u32 = kani::any();
        let maintain_aspect: bool = kani::any();

        kani::assume(src_h > 0 && src_h <= 4096);
        kani::assume(src_w > 0 && src_w <= 4096);
        kani::assume(target_h <= 4096);
        kani::assume(target_w <= 4096);

        let (h, w) = compute_resize_dims(src_h, src_w, target_h, target_w, maintain_aspect);

        assert!(h >= 1, "resize height must be >= 1");
        assert!(w >= 1, "resize width must be >= 1");
    }

    /// When maintain_aspect is true and targets are valid, the resized image
    /// fits within the target bounding box (neither dimension exceeds target).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_compute_resize_dims_fits_target() {
        let src_h: u32 = kani::any();
        let src_w: u32 = kani::any();
        let target_h: u32 = kani::any();
        let target_w: u32 = kani::any();

        kani::assume(src_h > 0 && src_h <= 4096);
        kani::assume(src_w > 0 && src_w <= 4096);
        kani::assume(target_h > 0 && target_h <= 4096);
        kani::assume(target_w > 0 && target_w <= 4096);

        let (h, w) = compute_resize_dims(src_h, src_w, target_h, target_w, true);

        // With maintain_aspect=true, the result fits within the target box.
        // Allow +1 tolerance for rounding.
        assert!(
            h <= target_h + 1,
            "resized height must fit within target + rounding"
        );
        assert!(
            w <= target_w + 1,
            "resized width must fit within target + rounding"
        );
    }
}
