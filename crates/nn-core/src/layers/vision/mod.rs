// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vision layers: ViT encoder, spatial upsampling, pixel shuffle, SE block, MBConv,
//! CNN detection blocks (SPPF, C2f, PAN, DetectHead, NMS), ResNet backbone.
//!
//! - [`VitEncoder`] / [`VitEncoderBlock`] / [`PatchEmbedding`] — Vision Transformer
//! - [`Upsample2d`] — nearest-neighbor / bilinear 2D upsampling
//! - [`PixelShuffle`] / [`PixelUnshuffle`] — sub-pixel convolution
//! - [`SqueezeExcitation`] — channel attention (SE block)
//! - [`MBConv`] — EfficientNet mobile inverted bottleneck
//! - [`ConvBnAct`] — Conv2d + BatchNorm + Activation fused block
//! - [`Sppf`] — Spatial Pyramid Pooling - Fast (YOLO)
//! - [`C2f`] / [`Bottleneck`] — Cross-Stage Partial bottleneck (YOLO)
//! - [`PanNeck`] — Path Aggregation Network for multi-scale feature fusion
//! - [`DetectHead`] — Anchor-free detection head with DFL regression
//! - [`DetrDecoder`] — DETR transformer decoder with object queries
//! - [`nms`] — Non-Maximum Suppression for detection post-processing
//! - [`ImageProcessor`] — Image preprocessing (resize, normalize, HWC->CHW)

// -- Upsample2d (nearest-neighbor / bilinear 2D upsampling) -------------------
mod upsample;
pub use upsample::{Upsample2d, Upsample2dToSize, UpsampleMode};

// -- PixelShuffle / PixelUnshuffle (sub-pixel convolution) --------------------
mod pixel_shuffle;
pub use pixel_shuffle::{PixelShuffle, PixelUnshuffle};

// -- Squeeze-and-Excitation (SE) block (channel attention) --------------------
mod se_block;
pub use se_block::SqueezeExcitation;

// -- 1D Squeeze-and-Excitation for temporal features (ECAPA-TDNN) ------------
mod se_block_1d;
pub use se_block_1d::SqueezeExcitation1d;

// -- Res2Net multi-scale feature extraction (ECAPA-TDNN) ----------------------
mod res2net;
pub use res2net::Res2NetBlock;

// -- Attentive Statistics Pooling (ECAPA-TDNN speaker verification) -----------
mod asp;
pub use asp::AttentiveStatisticsPooling;

// -- MBConv (EfficientNet mobile inverted bottleneck) -------------------------
mod mbconv;
pub use mbconv::{MBConv, MBConvConfig};

// -- Vision Transformer (ViT) -------------------------------------------------
mod vit;
pub use vit::{PatchEmbedding, PoolingStrategy, VitConfig, VitEncoder, VitEncoderBlock};

// -- Qwen2.5-VL / Qwen3-VL ViT config (window attention) ---------------------
mod qwen2vl_config;
pub use qwen2vl_config::{Qwen2VLVitConfig, Qwen3VLVitConfig};

// -- SigLIP2 Vision Encoder (dpdf #2418) --------------------------------------
mod siglip2;
pub use siglip2::{SigLip2Config, SigLip2VisionEncoder};

// -- Window attention ViT encoder (Qwen2.5-VL / Qwen3-VL, #2421 / #3857) -----
mod window_vit;
pub use window_vit::{WindowVitConfig, WindowVitEncoder, WindowVitEncoderBlock};

// -- DeepStack: multi-level ViT feature fusion (Qwen3-VL, dpdf #2433) --------
mod deep_stack;
pub use deep_stack::DeepStackFusion;

// -- Conv2d + BatchNorm + Activation fused block (YOLO building block) --------
mod conv_bn;
pub use conv_bn::ConvBnAct;

// -- SPPF: Spatial Pyramid Pooling — Fast (YOLO) -----------------------------
mod sppf;
pub use sppf::Sppf;

// -- C2f: Cross-Stage Partial bottleneck (YOLO) -------------------------------
mod csp;
pub use csp::{Bottleneck, C2f};

// -- PAN: Path Aggregation Network for multi-scale feature fusion -------------
mod pan;
pub use pan::PanNeck;

// -- Anchor-free detection head (YOLOv8 Detect) -------------------------------
mod detect_head;
pub use detect_head::{make_anchor_grid, DetectHead, ScaleOutput};

// -- DETR decoder (object queries + cross-attention to encoder features) ------
mod detr_decoder;
pub use detr_decoder::{DetrDecoder, DetrDecoderLayer, DetrOutput};

// -- NMS: Non-Maximum Suppression for detection post-processing ---------------
pub mod nms;
pub use nms::{nms as nms_filter, Detection};

// -- Image preprocessing (resize, normalize, HWC->CHW) -----------------------
mod image_processor;
pub use image_processor::ImageProcessor;

// -- Tensor-based image preprocessing (rescale, normalize, CHW) ---------------
mod image_preprocess;
pub use image_preprocess::ImagePreprocessor;

// -- ResNet backbone (ResNet-18 for Table Transformer / DETR) -----------------
pub mod resnet;
pub use resnet::{BasicBlock, ResNet18};

// -- HuggingFace-compatible ResNet-18 backbone (RT-DETR) ----------------------
pub mod resnet_hf;
pub use resnet_hf::ResNet18Hf;

// -- Kani proof harnesses for vision module safety (#3606) --------------------
#[cfg(kani)]
#[path = "kani_vision_proofs.rs"]
mod kani_vision_proofs;

// -- Kani proof harnesses for SigLIP2 encoder (#3711) -------------------------
#[cfg(kani)]
#[path = "kani_siglip2_proofs.rs"]
mod kani_siglip2_proofs;

// -- Kani proof harnesses for spatial transforms (#4071) ----------------------
#[cfg(kani)]
#[path = "kani_spatial_transform_proofs.rs"]
mod kani_spatial_transform_proofs;

// -- Kani proof harnesses for detection postprocessing (#4067) -----------------
#[cfg(kani)]
#[path = "kani_detection_proofs.rs"]
mod kani_detection_proofs;

// -- Kani proof harnesses for ResNet backbone and DETR decoder (#4070) ---------
#[cfg(kani)]
#[path = "kani_resnet_detr_proofs.rs"]
mod kani_resnet_detr_proofs;
// -- Kani proof harnesses for CNN building blocks (#4068) ---------------------
#[cfg(kani)]
#[path = "kani_cnn_block_proofs.rs"]
mod kani_cnn_block_proofs;

#[cfg(kani)]
mod kani_vit_issue_3730;

// -- Kani proof harnesses for image preprocessing (#4069) ---------------------
#[cfg(kani)]
#[path = "kani_image_preprocess_proofs.rs"]
mod kani_image_preprocess_proofs;

// -- Kani proof harnesses for Qwen2VL/Qwen3VL config + SigLIP2 (#4092) -------
#[cfg(kani)]
#[path = "kani_qwen2vl_siglip2_proofs.rs"]
mod kani_qwen2vl_siglip2_proofs;
// -- Kani proof harnesses for WindowViT encoder and DeepStack fusion (#4091) --
#[cfg(kani)]
#[path = "kani_window_vit_deep_stack_proofs.rs"]
mod kani_window_vit_deep_stack_proofs;

// -- Kani proof harnesses for PixelShuffle/PixelUnshuffle (#4155) --------------
#[cfg(kani)]
#[path = "kani_pixel_shuffle_proofs.rs"]
mod kani_pixel_shuffle_proofs;
// -- Kani proof harnesses for MBConv shape safety (#4156) ---------------------
#[cfg(kani)]
#[path = "kani_mbconv_proofs.rs"]
mod kani_mbconv_proofs;

// -- Kani proof harnesses for Upsample2d safety (#4161) -----------------------
#[cfg(kani)]
#[path = "kani_upsample_proofs.rs"]
mod kani_upsample_proofs;

// -- Kani proof harnesses for Conv2d extended safety (#4193) ------------------
#[cfg(kani)]
#[path = "kani_conv2d_extended_proofs.rs"]
mod kani_conv2d_extended_proofs;

// -- Kani proof harnesses for dpdf decoder pipeline composition (#4271) --------
#[cfg(kani)]
#[path = "kani_dpdf_decoder_proofs.rs"]
mod kani_dpdf_decoder_proofs;

// -- DocLayout-YOLO compile-time static tests (#3845) -------------------------
#[cfg(test)]
#[path = "doclayout_yolo_static_tests.rs"]
mod doclayout_yolo_static_tests;

// -- Granite-Docling-258M compile-time static tests (#3846) -------------------
#[cfg(test)]
#[path = "granite_docling_static_tests.rs"]
mod granite_docling_static_tests;

// -- OCR model architecture static tests (PaddleOCR-VL, FireRed-OCR) ----------
#[cfg(test)]
#[path = "ocr_model_static_tests.rs"]
mod ocr_model_static_tests;

// -- Qwen3-VL MoE compile-time static tests (#3847) --------------------------
#[cfg(test)]
#[path = "qwen3vl_static_tests.rs"]
mod qwen3vl_static_tests;
