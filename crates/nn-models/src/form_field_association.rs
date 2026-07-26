// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Form field association: key-value pairing from LayoutLMv3 token labels.
//!
//! LayoutLMv3 produces per-token entity labels using a BIO tagging scheme:
//! `B-QUESTION`, `I-QUESTION`, `B-ANSWER`, `I-ANSWER`, `B-HEADER`, `I-HEADER`,
//! `O` (other). This module groups consecutive tokens into spans, then pairs
//! question spans (keys) with their nearest answer spans (values) using
//! spatial proximity and reading order heuristics.
//!
//! # Architecture
//!
//! 1. **BIO decoding**: Group consecutive tokens sharing the same entity into
//!    labeled spans with merged bounding boxes.
//! 2. **Spatial pairing**: For each key span, find the nearest value span by
//!    reading-order distance (right-of or below, with configurable bias).
//! 3. **Output**: A list of [`FormField`] key-value pairs with bounding boxes.

// ---------------------------------------------------------------------------
// Entity label types
// ---------------------------------------------------------------------------

/// BIO entity tag from LayoutLMv3 sequence labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityTag {
    /// Beginning of a question / key field.
    BQuestion,
    /// Inside a question / key field.
    IQuestion,
    /// Beginning of an answer / value field.
    BAnswer,
    /// Inside an answer / value field.
    IAnswer,
    /// Beginning of a header field.
    BHeader,
    /// Inside a header field.
    IHeader,
    /// Other (non-entity token).
    Other,
}

impl EntityTag {
    /// Parse from a label index (FUNSD convention).
    ///
    /// 0=Other, 1=B-Question, 2=I-Question, 3=B-Answer, 4=I-Answer,
    /// 5=B-Header, 6=I-Header. Out-of-range maps to `Other`.
    #[must_use]
    pub fn from_label_id(id: usize) -> Self {
        match id {
            1 => Self::BQuestion,
            2 => Self::IQuestion,
            3 => Self::BAnswer,
            4 => Self::IAnswer,
            5 => Self::BHeader,
            6 => Self::IHeader,
            _ => Self::Other,
        }
    }

    /// Whether this tag begins a new entity span.
    #[must_use]
    pub fn is_begin(&self) -> bool {
        matches!(self, Self::BQuestion | Self::BAnswer | Self::BHeader)
    }

    /// The entity type this tag belongs to, if any.
    #[must_use]
    pub fn entity_type(&self) -> Option<EntityType> {
        match self {
            Self::BQuestion | Self::IQuestion => Some(EntityType::Question),
            Self::BAnswer | Self::IAnswer => Some(EntityType::Answer),
            Self::BHeader | Self::IHeader => Some(EntityType::Header),
            Self::Other => None,
        }
    }
}

/// High-level entity category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    /// A form key / question.
    Question,
    /// A form value / answer.
    Answer,
    /// A section header.
    Header,
}

// ---------------------------------------------------------------------------
// Token and span types
// ---------------------------------------------------------------------------

/// A labeled token with spatial information.
#[derive(Debug, Clone)]
pub struct LabeledToken {
    /// Token text.
    pub text: String,
    /// Bounding box `[x1, y1, x2, y2]` in pixel coordinates.
    pub bbox: [f32; 4],
    /// Entity tag from the model.
    pub tag: EntityTag,
}

/// A contiguous span of tokens sharing the same entity type.
#[derive(Debug, Clone)]
pub struct EntitySpan {
    /// Concatenated text of all tokens in this span.
    pub text: String,
    /// Enclosing bounding box `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// Entity type.
    pub entity_type: EntityType,
    /// Confidence (average of token confidences if available; 1.0 default).
    pub confidence: f32,
}

/// A paired key-value form field.
#[derive(Debug, Clone)]
pub struct FormField {
    /// The key / question span.
    pub key: EntitySpan,
    /// The value / answer span, if a match was found.
    pub value: Option<EntitySpan>,
}

/// Unpaired header detected on the form.
#[derive(Debug, Clone)]
pub struct FormHeader {
    /// Header span text and bbox.
    pub span: EntitySpan,
}

/// Complete form extraction result.
#[derive(Debug, Clone)]
pub struct FormExtractionResult {
    /// Paired key-value fields.
    pub fields: Vec<FormField>,
    /// Unpaired headers.
    pub headers: Vec<FormHeader>,
    /// Answer spans that could not be paired to any key.
    pub orphan_values: Vec<EntitySpan>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for form field association.
#[derive(Debug, Clone)]
pub struct FormAssociationConfig {
    /// Maximum pixel distance for key-value pairing (default 200.0).
    pub max_pair_distance: f32,
    /// Bias factor for horizontal proximity vs vertical (default 1.5).
    /// Higher values prefer answers that are to the right of the key
    /// over answers that are below.
    pub horizontal_bias: f32,
    /// Whether to allow one value to pair with multiple keys (default false).
    pub allow_shared_values: bool,
}

impl Default for FormAssociationConfig {
    fn default() -> Self {
        Self {
            max_pair_distance: 200.0,
            horizontal_bias: 1.5,
            allow_shared_values: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BIO decoding
// ---------------------------------------------------------------------------

/// Decode BIO-tagged tokens into contiguous entity spans.
///
/// Consecutive `I-*` tokens following a `B-*` of the same type are merged.
/// Orphan `I-*` tokens (without a preceding `B-*`) start a new span.
#[must_use]
pub fn decode_bio_spans(tokens: &[LabeledToken]) -> Vec<EntitySpan> {
    let mut spans: Vec<EntitySpan> = Vec::new();
    let mut current: Option<EntitySpan> = None;

    for token in tokens {
        let entity_type = token.tag.entity_type();

        match (&mut current, entity_type, token.tag.is_begin()) {
            // Continue current span with matching I-tag.
            (Some(ref mut span), Some(etype), false) if span.entity_type == etype => {
                span.text.push(' ');
                span.text.push_str(&token.text);
                span.bbox = merge_bbox(&span.bbox, &token.bbox);
            }
            // Begin tag or mismatched I-tag: close current and start new.
            (_, Some(etype), _) => {
                if let Some(span) = current.take() {
                    spans.push(span);
                }
                current = Some(EntitySpan {
                    text: token.text.clone(),
                    bbox: token.bbox,
                    entity_type: etype,
                    confidence: 1.0,
                });
            }
            // Other tag: close current span.
            (_, None, _) => {
                if let Some(span) = current.take() {
                    spans.push(span);
                }
            }
        }
    }

    if let Some(span) = current {
        spans.push(span);
    }

    spans
}

// ---------------------------------------------------------------------------
// Spatial pairing
// ---------------------------------------------------------------------------

/// Pair question spans with their nearest answer spans.
///
/// For each question, computes a biased distance to all answers and picks
/// the closest one within `config.max_pair_distance`. Unless
/// `allow_shared_values` is true, each answer can only be paired once.
#[must_use]
pub fn pair_key_value(
    spans: &[EntitySpan],
    config: &FormAssociationConfig,
) -> FormExtractionResult {
    let questions: Vec<(usize, &EntitySpan)> = spans
        .iter()
        .enumerate()
        .filter(|(_, s)| s.entity_type == EntityType::Question)
        .collect();

    let answers: Vec<(usize, &EntitySpan)> = spans
        .iter()
        .enumerate()
        .filter(|(_, s)| s.entity_type == EntityType::Answer)
        .collect();

    let headers: Vec<&EntitySpan> = spans
        .iter()
        .filter(|s| s.entity_type == EntityType::Header)
        .collect();

    let mut used_answers = vec![false; answers.len()];
    let mut fields = Vec::with_capacity(questions.len());

    for (_qi, question) in &questions {
        let mut best_dist = f32::INFINITY;
        let mut best_idx: Option<usize> = None;

        for (ai, (_ans_idx, answer)) in answers.iter().enumerate() {
            if !config.allow_shared_values && used_answers[ai] {
                continue;
            }
            let dist = biased_distance(&question.bbox, &answer.bbox, config.horizontal_bias);
            if dist < best_dist && dist <= config.max_pair_distance {
                best_dist = dist;
                best_idx = Some(ai);
            }
        }

        let value = best_idx.map(|ai| {
            used_answers[ai] = true;
            answers[ai].1.clone()
        });

        fields.push(FormField {
            key: (*question).clone(),
            value,
        });
    }

    let orphan_values: Vec<EntitySpan> = answers
        .iter()
        .enumerate()
        .filter(|(ai, _)| !used_answers[*ai])
        .map(|(_, (_, span))| (*span).clone())
        .collect();

    let headers: Vec<FormHeader> = headers
        .iter()
        .map(|s| FormHeader { span: (*s).clone() })
        .collect();

    FormExtractionResult {
        fields,
        headers,
        orphan_values,
    }
}

/// Full form extraction: BIO decode + spatial pairing.
#[must_use]
pub fn extract_form_fields(
    tokens: &[LabeledToken],
    config: &FormAssociationConfig,
) -> FormExtractionResult {
    let spans = decode_bio_spans(tokens);
    pair_key_value(&spans, config)
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Biased reading-order distance between two bounding boxes.
///
/// Favors answers that are to the right of (or slightly below) the key.
/// Horizontal distance is divided by `horizontal_bias` to prefer
/// right-aligned answers in form-like layouts.
fn biased_distance(key_bbox: &[f32; 4], value_bbox: &[f32; 4], horizontal_bias: f32) -> f32 {
    let key_cx = (key_bbox[0] + key_bbox[2]) * 0.5;
    let key_cy = (key_bbox[1] + key_bbox[3]) * 0.5;
    let val_cx = (value_bbox[0] + value_bbox[2]) * 0.5;
    let val_cy = (value_bbox[1] + value_bbox[3]) * 0.5;

    let dx = val_cx - key_cx;
    let dy = val_cy - key_cy;

    // Penalize answers that are above or far to the left of the key.
    let effective_dx = if dx >= 0.0 {
        dx / horizontal_bias.max(0.01)
    } else {
        dx.abs() * 2.0 // penalty for leftward answers
    };

    let effective_dy = if dy >= 0.0 {
        dy
    } else {
        dy.abs() * 3.0 // strong penalty for answers above the key
    };

    effective_dx.hypot(effective_dy)
}

/// Merge two bounding boxes into their enclosing union.
fn merge_bbox(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

#[cfg(test)]
#[path = "form_field_association_tests.rs"]
mod tests;
