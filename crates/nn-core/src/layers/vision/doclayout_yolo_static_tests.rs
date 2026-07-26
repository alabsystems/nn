// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compile-time static tests for the DocLayout-YOLO detection model architecture.
//!
//! DocLayout-YOLO is a YOLOv8-based model for document layout analysis. It
//! detects 10 document element classes (title, text, list, table, figure,
//! caption, header, footer, reference, equation) using a standard
//! CSPDarknet backbone → SPPF → PAN neck → DetectHead pipeline.
//!
//! These tests validate architectural invariants at compile time (const
//! assertions) and test time (f32 threshold checks) to catch configuration
//! errors before any weights are loaded or inference is run.
//!
//! Reference: <https://arxiv.org/abs/2410.12628>

// ============================================================================
// DocLayout-YOLO configuration constants
// ============================================================================

/// Number of document layout element classes.
/// (title, text, list, table, figure, caption, header, footer, reference, equation)
const NUM_CLASSES: usize = 10;

/// DFL regression bin count (standard YOLOv8 value).
const REG_MAX: usize = 16;

/// Number of detection scales (P3/stride-8, P4/stride-16, P5/stride-32).
const NUM_SCALES: usize = 3;

/// Backbone output channels per scale level: [P3, P4, P5].
const BACKBONE_CHANNELS: [usize; 3] = [256, 512, 1024];

/// PAN neck output channels per scale level (matches backbone channels).
const PAN_OUTPUT_CHANNELS: [usize; 3] = [256, 512, 1024];

/// Detection head hidden channel count (intermediate conv branches).
const DETECT_HIDDEN: usize = 256;

/// SPPF max-pool kernel size (must be odd for symmetric padding).
const SPPF_KERNEL: usize = 5;

/// Number of C2f bottleneck blocks per CSP stage.
const CSP_DEPTH: usize = 3;

/// Number of C2f bottleneck blocks in the PAN neck fusion stages.
const PAN_DEPTH: usize = 3;

/// NMS confidence threshold for document layout detection.
const NMS_CONFIDENCE_THRESHOLD: f32 = 0.25;

/// NMS IoU threshold for document layout detection.
const NMS_IOU_THRESHOLD: f32 = 0.45;

/// Detection head regression output channels per scale = 4 * REG_MAX.
const REG_OUTPUT_CHANNELS: usize = 4 * REG_MAX;

/// Detection head classification output channels per scale = NUM_CLASSES.
const CLS_OUTPUT_CHANNELS: usize = NUM_CLASSES;

/// Backbone strides for the 3 detection scales.
const BACKBONE_STRIDES: [usize; 3] = [8, 16, 32];

// ============================================================================
// Compile-time const assertions
// ============================================================================

// --- FPN / PAN channel compatibility ---

// PAN top-down path: P5 upsampled + P4 → C2f → N4 (c4 channels).
// The concat input = P5_channels + P4_channels, output = P4_channels.
const _: () = assert!(
    BACKBONE_CHANNELS[2] + BACKBONE_CHANNELS[1] > BACKBONE_CHANNELS[1],
    "PAN top-down P5+P4 concat must exceed P4 output channels"
);

// PAN top-down path: N4 upsampled + P3 → C2f → N3 (c3 channels).
const _: () = assert!(
    BACKBONE_CHANNELS[1] + BACKBONE_CHANNELS[0] > BACKBONE_CHANNELS[0],
    "PAN top-down N4+P3 concat must exceed P3 output channels"
);

// PAN bottom-up path: N3 downsampled + N4 → C2f → N4'.
// The downsampled N3 has P3 channels, concat with N4 (P4 channels).
const _: () = assert!(
    BACKBONE_CHANNELS[0] + BACKBONE_CHANNELS[1] > BACKBONE_CHANNELS[1],
    "PAN bottom-up N3+N4 concat must exceed N4 output channels"
);

// PAN bottom-up path: N4' downsampled + P5 → C2f → N5'.
// The downsampled N4' has P4 channels, concat with P5 (P5 channels).
const _: () = assert!(
    BACKBONE_CHANNELS[1] + BACKBONE_CHANNELS[2] > BACKBONE_CHANNELS[2],
    "PAN bottom-up N4+P5 concat must exceed P5 output channels"
);

// PAN output channels match backbone channels (lateral connections).
const _: () = assert!(
    PAN_OUTPUT_CHANNELS[0] == BACKBONE_CHANNELS[0],
    "PAN output P3 channels must match backbone P3 channels"
);
const _: () = assert!(
    PAN_OUTPUT_CHANNELS[1] == BACKBONE_CHANNELS[1],
    "PAN output P4 channels must match backbone P4 channels"
);
const _: () = assert!(
    PAN_OUTPUT_CHANNELS[2] == BACKBONE_CHANNELS[2],
    "PAN output P5 channels must match backbone P5 channels"
);

// --- Detection head output dimensions ---

// Regression branch outputs 4 bbox coordinates * REG_MAX bins.
const _: () = assert!(
    REG_OUTPUT_CHANNELS == 4 * REG_MAX,
    "Detection head regression output must be 4 * reg_max"
);

// Classification branch outputs one score per class.
const _: () = assert!(
    CLS_OUTPUT_CHANNELS == NUM_CLASSES,
    "Detection head classification output must equal num_classes"
);

// Detection head output = num_classes + 4 bbox coords (after DFL decode).
const _: () = assert!(
    NUM_CLASSES + 4 == 14,
    "Total detection output per anchor = num_classes(10) + 4 bbox = 14"
);

// --- SPPF kernel size must be odd (symmetric padding) ---

const _: () = assert!(SPPF_KERNEL % 2 == 1, "SPPF kernel size must be odd");
const _: () = assert!(SPPF_KERNEL >= 3, "SPPF kernel size must be at least 3");

// --- CSP block depths must be > 0 ---

const _: () = assert!(CSP_DEPTH > 0, "CSP bottleneck depth must be > 0");
const _: () = assert!(PAN_DEPTH > 0, "PAN C2f bottleneck depth must be > 0");

// --- REG_MAX must be positive ---

const _: () = assert!(REG_MAX > 0, "REG_MAX must be positive");

// --- NUM_CLASSES must be positive ---

const _: () = assert!(NUM_CLASSES > 0, "NUM_CLASSES must be positive");

// --- NUM_SCALES consistency ---

const _: () = assert!(
    NUM_SCALES == 3,
    "DocLayout-YOLO uses exactly 3 detection scales"
);

// --- Backbone strides must be powers of 2 and strictly increasing ---

const _: () = assert!(
    BACKBONE_STRIDES[0] < BACKBONE_STRIDES[1],
    "Backbone strides must be strictly increasing (P3 < P4)"
);
const _: () = assert!(
    BACKBONE_STRIDES[1] < BACKBONE_STRIDES[2],
    "Backbone strides must be strictly increasing (P4 < P5)"
);

// Strides must be powers of 2.
const _: () = assert!(
    BACKBONE_STRIDES[0].is_power_of_two(),
    "P3 stride must be a power of 2"
);
const _: () = assert!(
    BACKBONE_STRIDES[1].is_power_of_two(),
    "P4 stride must be a power of 2"
);
const _: () = assert!(
    BACKBONE_STRIDES[2].is_power_of_two(),
    "P5 stride must be a power of 2"
);

// --- Backbone channels must be positive and increasing ---

const _: () = assert!(
    BACKBONE_CHANNELS[0] > 0,
    "P3 backbone channels must be positive"
);
const _: () = assert!(
    BACKBONE_CHANNELS[1] > BACKBONE_CHANNELS[0],
    "P4 channels must exceed P3 channels"
);
const _: () = assert!(
    BACKBONE_CHANNELS[2] > BACKBONE_CHANNELS[1],
    "P5 channels must exceed P4 channels"
);

// --- C2f hidden channels (out_c / 2) must be positive for all scales ---

const _: () = assert!(
    BACKBONE_CHANNELS[0] / 2 > 0,
    "C2f hidden channels at P3 must be positive"
);
const _: () = assert!(
    BACKBONE_CHANNELS[1] / 2 > 0,
    "C2f hidden channels at P4 must be positive"
);
const _: () = assert!(
    BACKBONE_CHANNELS[2] / 2 > 0,
    "C2f hidden channels at P5 must be positive"
);

// --- Backbone channels must be even (C2f splits into two halves) ---

const _: () = assert!(
    BACKBONE_CHANNELS[0].is_multiple_of(2),
    "P3 channels must be even for C2f split"
);
const _: () = assert!(
    BACKBONE_CHANNELS[1].is_multiple_of(2),
    "P4 channels must be even for C2f split"
);
const _: () = assert!(
    BACKBONE_CHANNELS[2].is_multiple_of(2),
    "P5 channels must be even for C2f split"
);

// --- SPPF channel reduction: channels / 2 must be positive ---

const _: () = assert!(
    BACKBONE_CHANNELS[2] / 2 > 0,
    "SPPF hidden (channels/2) must be positive"
);

// --- SPPF output concat = 4 * hidden channels ---
// SPPF concatenates [y1, y2, y3, y4] each with hidden = channels/2 channels.
// Total concat = 4 * (channels/2) = 2 * channels, then projected back.

const _: () = assert!(
    4 * (BACKBONE_CHANNELS[2] / 2) == 2 * BACKBONE_CHANNELS[2],
    "SPPF concat channels must equal 2 * input channels"
);

// --- Detection head hidden channels must be positive ---

const _: () = assert!(
    DETECT_HIDDEN > 0,
    "Detection head hidden channels must be positive"
);

// ============================================================================
// Runtime tests (f32 comparisons cannot be const)
// ============================================================================

#[test]
fn test_nms_confidence_threshold_in_valid_range() {
    assert!(
        NMS_CONFIDENCE_THRESHOLD >= 0.0 && NMS_CONFIDENCE_THRESHOLD <= 1.0,
        "NMS confidence_threshold must be in [0.0, 1.0], got {NMS_CONFIDENCE_THRESHOLD}"
    );
    assert!(
        NMS_CONFIDENCE_THRESHOLD.is_finite(),
        "NMS confidence_threshold must be finite"
    );
}

#[test]
fn test_nms_iou_threshold_in_valid_range() {
    assert!(
        NMS_IOU_THRESHOLD >= 0.0 && NMS_IOU_THRESHOLD <= 1.0,
        "NMS iou_threshold must be in [0.0, 1.0], got {NMS_IOU_THRESHOLD}"
    );
    assert!(
        NMS_IOU_THRESHOLD.is_finite(),
        "NMS iou_threshold must be finite"
    );
}

#[test]
fn test_nms_confidence_strictly_below_iou() {
    // Standard heuristic: confidence threshold < iou threshold.
    // Confidence filters weak detections; IoU filters overlapping ones.
    assert!(
        NMS_CONFIDENCE_THRESHOLD < NMS_IOU_THRESHOLD,
        "NMS confidence threshold ({NMS_CONFIDENCE_THRESHOLD}) should be less than IoU threshold ({NMS_IOU_THRESHOLD})"
    );
}

#[test]
fn test_detection_output_dimensions_per_scale() {
    // For each scale, the detection head produces:
    // - cls: [B, NUM_CLASSES, H, W]
    // - reg: [B, 4 * REG_MAX, H, W]
    // After DFL decode: reg becomes [B, 4, H, W]
    // Total per-anchor output = NUM_CLASSES + 4
    let total_per_anchor = NUM_CLASSES + 4;
    assert_eq!(total_per_anchor, 14);

    // Raw regression output before DFL
    let raw_reg = 4 * REG_MAX;
    assert_eq!(raw_reg, 64);
}

#[test]
fn test_pan_channel_arithmetic() {
    let [c3, c4, c5] = BACKBONE_CHANNELS;

    // Top-down: P5 upsampled concat P4 → input to C2f
    let td1_in = c5 + c4;
    assert_eq!(td1_in, 1536, "P5+P4 concat = 1024+512");

    // Top-down: N4 upsampled concat P3 → input to C2f
    let td2_in = c4 + c3;
    assert_eq!(td2_in, 768, "N4+P3 concat = 512+256");

    // Bottom-up: N3 downsampled concat N4 → input to C2f
    let bu1_in = c3 + c4;
    assert_eq!(bu1_in, 768, "N3+N4 concat = 256+512");

    // Bottom-up: N4' downsampled concat P5 → input to C2f
    let bu2_in = c4 + c5;
    assert_eq!(bu2_in, 1536, "N4+P5 concat = 512+1024");
}

#[test]
fn test_c2f_cat_channels() {
    // C2f concatenates: [chunk0, chunk1, out0, ..., out_{n-1}]
    // Total = (2 + n_bottlenecks) * hidden, where hidden = out_c / 2.
    for &out_c in &BACKBONE_CHANNELS {
        let hidden = out_c / 2;
        let cat_channels = (2 + CSP_DEPTH) * hidden;
        assert!(
            cat_channels > 0,
            "C2f concat channels must be positive for out_c={out_c}"
        );
        // The final cv2 projects cat_channels → out_c
        assert!(
            cat_channels > out_c,
            "C2f concat ({cat_channels}) should exceed output ({out_c}) for compression"
        );
    }
}

#[test]
fn test_sppf_padding_preserves_spatial() {
    // SPPF max-pool uses kernel=SPPF_KERNEL, stride=1, padding=kernel/2.
    // This preserves spatial dimensions: output_size = (input + 2*pad - kernel)/1 + 1
    // = input + 2*(kernel/2) - kernel + 1 = input (when kernel is odd).
    let pad = SPPF_KERNEL / 2;
    // For any spatial dim S: (S + 2*pad - SPPF_KERNEL) / 1 + 1 == S
    // Simplifies to: 2*pad - SPPF_KERNEL + 1 == 0
    let residual = 2 * pad + 1;
    assert_eq!(
        residual, SPPF_KERNEL,
        "SPPF padding must preserve spatial dimensions"
    );
}

#[test]
fn test_backbone_stride_ratios() {
    // Adjacent strides should have ratio 2 (standard FPN).
    assert_eq!(BACKBONE_STRIDES[1] / BACKBONE_STRIDES[0], 2);
    assert_eq!(BACKBONE_STRIDES[2] / BACKBONE_STRIDES[1], 2);
}
