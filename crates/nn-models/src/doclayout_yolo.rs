// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DocLayout-YOLO end-to-end model builder for document layout detection.
//!
//! Assembles [`ConvBnAct`] + [`C2f`] + [`Sppf`] + [`PanNeck`] + [`DetectHead`]
//! into a complete anchor-free detection model for 10 document layout classes.
//!
//! Architecture: YOLOv8-nano backbone (~16M params) with:
//! - 5-stage backbone: progressive channel expansion 3 → 16 → 32 → 64 → 128 → 256
//! - PAN neck: top-down + bottom-up multi-scale fusion
//! - DetectHead: decoupled classification + DFL regression, 3 detection scales
//!
//! # Classes
//!
//! ```text
//! 0: caption    1: footnote      2: formula    3: list-item    4: page-footer
//! 5: page-header  6: picture     7: section-header  8: table   9: text
//! ```
//!
//! Reference: Zhao et al. 2024, "DocLayout-YOLO: Enhancing Document Layout
//! Analysis through Diverse Synthetic Data and Global-to-Local Adaptive
//! Perception", arXiv:2410.12628.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::{C2f, ConvBnAct, DetectHead, Detection, PanNeck, Sppf};
use nn_core::layers::{Activation, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

/// Number of document layout classes.
pub const NUM_CLASSES: usize = 10;

/// DFL regression bin count (standard YOLOv8).
pub const REG_MAX: usize = 16;

/// Typical input resolution (square).
pub const INPUT_SIZE: usize = 800;

/// Human-readable class names indexed by class_id.
pub const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "caption",
    "footnote",
    "formula",
    "list-item",
    "page-footer",
    "page-header",
    "picture",
    "section-header",
    "table",
    "text",
];

/// Detection strides for the 3 output scales.
const STRIDES: [usize; 3] = [8, 16, 32];

/// Detection head hidden channel count.
const HEAD_HIDDEN: usize = 256;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the DocLayout-YOLO model.
#[derive(Debug, Clone)]
pub struct DocLayoutYoloConfig {
    /// Input image channels (default 3 for RGB).
    pub input_channels: usize,
    /// Backbone channel widths per stage (default `[16, 32, 64, 128, 256]`).
    pub backbone_channels: [usize; 5],
    /// Number of object classes (default 10).
    pub num_classes: usize,
    /// DFL regression bins (default 16).
    pub reg_max: usize,
    /// Minimum confidence to keep a detection (default 0.25).
    pub conf_threshold: f32,
    /// IoU threshold for NMS suppression (default 0.45).
    pub iou_threshold: f32,
}

impl Default for DocLayoutYoloConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            backbone_channels: [16, 32, 64, 128, 256],
            num_classes: NUM_CLASSES,
            reg_max: REG_MAX,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        }
    }
}

impl DocLayoutYoloConfig {
    /// Channel counts for the 3 PAN/DetectHead scales (P3=stride 8, P4=16, P5=32).
    #[must_use]
    pub fn neck_channels(&self) -> [usize; 3] {
        let c = self.backbone_channels;
        [c[2], c[3], c[4]] // 64, 128, 256
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// DocLayout-YOLO document layout detection model.
///
/// Backbone → PAN neck → DetectHead → NMS pipeline for 10 document classes.
///
/// # Weight names
///
/// Expects VarBuilder scoped to the model root:
/// - `"backbone.stage0.*"` through `"backbone.stage4.*"` — backbone stages
/// - `"neck.*"` — PAN neck
/// - `"head.*"` — detection head
#[derive(Debug, Clone)]
pub struct DocLayoutYolo {
    config: DocLayoutYoloConfig,
    // Backbone stages
    stem: ConvBnAct,        // stage 0: 3→16, stride 2
    stage1_conv: ConvBnAct, // 16→32, stride 2
    stage1_c2f: C2f,        // 32→32, 1 bottleneck
    stage2_conv: ConvBnAct, // 32→64, stride 2
    stage2_c2f: C2f,        // 64→64, 2 bottlenecks
    stage3_conv: ConvBnAct, // 64→128, stride 2
    stage3_c2f: C2f,        // 128→128, 2 bottlenecks
    stage4_conv: ConvBnAct, // 128→256, stride 2
    stage4_c2f: C2f,        // 256→256, 1 bottleneck
    stage4_sppf: Sppf,      // 256→256
    // Neck + head
    neck: PanNeck,
    head: DetectHead,
}

impl DocLayoutYolo {
    /// Load the model from a VarBuilder using PyTorch weight naming.
    pub fn load(vb: impl AsRef<VarBuilder>, config: DocLayoutYoloConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let [c0, c1, c2, c3, c4] = config.backbone_channels;
        let act = Some(Activation::Silu);

        // -- Backbone --
        let bb = vb.pp("backbone");
        let stem = ConvBnAct::load(bb.pp("stage0"), config.input_channels, c0, 3, 2, act)?;

        let stage1_conv = ConvBnAct::load(bb.pp("stage1.conv"), c0, c1, 3, 2, act)?;
        let stage1_c2f = C2f::load(bb.pp("stage1.c2f"), c1, c1, 1, true)?;

        let stage2_conv = ConvBnAct::load(bb.pp("stage2.conv"), c1, c2, 3, 2, act)?;
        let stage2_c2f = C2f::load(bb.pp("stage2.c2f"), c2, c2, 2, true)?;

        let stage3_conv = ConvBnAct::load(bb.pp("stage3.conv"), c2, c3, 3, 2, act)?;
        let stage3_c2f = C2f::load(bb.pp("stage3.c2f"), c3, c3, 2, true)?;

        let stage4_conv = ConvBnAct::load(bb.pp("stage4.conv"), c3, c4, 3, 2, act)?;
        let stage4_c2f = C2f::load(bb.pp("stage4.c2f"), c4, c4, 1, true)?;
        let stage4_sppf = Sppf::load(bb.pp("stage4.sppf"), c4, 5)?;

        // -- Neck --
        let neck = PanNeck::load(vb.pp("neck"), config.neck_channels(), 1)?;

        // -- Detection head --
        let [nc2, nc3, nc4] = config.neck_channels();
        let head = DetectHead::load(
            vb.pp("head"),
            &[nc2, nc3, nc4],
            config.num_classes,
            config.reg_max,
            HEAD_HIDDEN,
        )?;

        Ok(Self {
            config,
            stem,
            stage1_conv,
            stage1_c2f,
            stage2_conv,
            stage2_c2f,
            stage3_conv,
            stage3_c2f,
            stage4_conv,
            stage4_c2f,
            stage4_sppf,
            neck,
            head,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &DocLayoutYoloConfig {
        &self.config
    }

    /// Backbone forward pass producing 3 multi-scale feature maps.
    ///
    /// Input: `[B, 3, H, W]` image tensor.
    /// Returns `(p3, p4, p5)` at strides 8, 16, 32.
    pub fn forward_backbone(&self, x: &DynTensor) -> Result<(DynTensor, DynTensor, DynTensor)> {
        // Stage 0: stride 2 → H/2
        let x = self.stem.forward(x)?;
        // Stage 1: stride 2 → H/4
        let x = self.stage1_conv.forward(&x)?;
        let x = self.stage1_c2f.forward(&x)?;
        // Stage 2: stride 2 → H/8 = P3
        let x = self.stage2_conv.forward(&x)?;
        let p3 = self.stage2_c2f.forward(&x)?;
        // Stage 3: stride 2 → H/16 = P4
        let x = self.stage3_conv.forward(&p3)?;
        let p4 = self.stage3_c2f.forward(&x)?;
        // Stage 4: stride 2 → H/32 = P5
        let x = self.stage4_conv.forward(&p4)?;
        let x = self.stage4_c2f.forward(&x)?;
        let p5 = self.stage4_sppf.forward(&x)?;

        Ok((p3, p4, p5))
    }

    /// PAN neck forward pass fusing 3 multi-scale features.
    pub fn forward_neck(
        &self,
        p3: &DynTensor,
        p4: &DynTensor,
        p5: &DynTensor,
    ) -> Result<(DynTensor, DynTensor, DynTensor)> {
        self.neck.forward_multi(p3, p4, p5)
    }

    /// Full forward pass: backbone → neck → head → NMS.
    ///
    /// Input: `[B, 3, H, W]` image tensor (typically 800×800).
    /// Returns filtered [`Detection`] objects with class IDs mapped to
    /// [`CLASS_NAMES`].
    pub fn forward(&self, image: &DynTensor) -> Result<Vec<Detection>> {
        let dims = image.dims();
        if dims.len() != 4 || dims[1] != self.config.input_channels {
            return Err(TensorError::shape_mismatch(
                vec![0, self.config.input_channels, 0, 0],
                dims.to_vec(),
            ));
        }
        let img_h = dims[2];
        let img_w = dims[3];

        let (p3, p4, p5) = self.forward_backbone(image)?;
        let (n3, n4, n5) = self.forward_neck(&p3, &p4, &p5)?;

        let scale_outputs = self.head.forward_multi(&[&n3, &n4, &n5])?;
        self.head.decode_detections(
            &scale_outputs,
            &STRIDES,
            (img_h, img_w),
            self.config.conf_threshold,
            self.config.iou_threshold,
        )
    }

    /// Map raw detections to human-readable `(class_name, confidence, [x1, y1, x2, y2])`.
    #[must_use]
    pub fn label_detections(detections: &[Detection]) -> Vec<(&'static str, f32, [f32; 4])> {
        detections
            .iter()
            .filter_map(|d| {
                let idx = d.class_id as usize;
                CLASS_NAMES
                    .get(idx)
                    .map(|name| (*name, d.confidence, [d.x1, d.y1, d.x2, d.y2]))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "doclayout_yolo_tests.rs"]
mod tests;
