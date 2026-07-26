// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Document output export: JSON, HTML, Markdown, and CSV structured output.
//!
//! Provides [`DocumentExporter`] trait and four concrete exporters for
//! converting [`DocumentOutput`] (from [`super::dpdf_pipeline`]) into
//! portable formats suitable for downstream consumption, archival, or
//! human review.
//!
//! # Exporters
//!
//! - [`JsonExporter`] — structured JSON via `serde_json`
//! - [`HtmlExporter`] — semantic HTML (h1, p, table, figure, ul)
//! - [`MarkdownExporter`] — Markdown with headers, pipe tables, code blocks
//! - [`CsvTableExporter`] — CSV for table cells only (row, col, text, confidence)

use crate::dpdf_pipeline::{DocumentOutput, DocumentRegion, PageOutput};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during document export.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExportError {
    /// JSON serialization failed.
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// The document has no pages to export.
    #[error("document has no pages")]
    EmptyDocument,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for exporting [`DocumentOutput`] to a string format.
pub trait DocumentExporter {
    /// Export the full document to a string representation.
    ///
    /// # Errors
    ///
    /// Returns [`ExportError`] if serialization fails or the document is
    /// unsuitable for this format.
    fn export(&self, output: &DocumentOutput) -> Result<String, ExportError>;
}

// ---------------------------------------------------------------------------
// JSON exporter
// ---------------------------------------------------------------------------

/// Exports [`DocumentOutput`] as structured JSON.
///
/// Each page becomes a JSON object with `page_index`, `width`, `height`,
/// and a `regions` array. Each region includes its type, bounding box,
/// confidence, and content fields.
#[derive(Debug, Clone, Default)]
pub struct JsonExporter {
    /// Pretty-print the JSON output (default: false).
    pub pretty: bool,
}

impl JsonExporter {
    /// Create a new JSON exporter with compact output.
    #[must_use]
    pub fn new() -> Self {
        Self { pretty: false }
    }

    /// Create a new JSON exporter with pretty-printed output.
    #[must_use]
    pub fn pretty() -> Self {
        Self { pretty: true }
    }
}

impl DocumentExporter for JsonExporter {
    fn export(&self, output: &DocumentOutput) -> Result<String, ExportError> {
        let pages: Vec<serde_json::Value> = output
            .pages
            .iter()
            .enumerate()
            .map(|(i, page)| page_to_json(i, page))
            .collect();

        let doc = serde_json::json!({
            "pages": pages,
            "page_count": output.pages.len(),
        });

        if self.pretty {
            Ok(serde_json::to_string_pretty(&doc)?)
        } else {
            Ok(serde_json::to_string(&doc)?)
        }
    }
}

/// Convert a single page to a JSON value.
fn page_to_json(page_index: usize, page: &PageOutput) -> serde_json::Value {
    let regions: Vec<serde_json::Value> = page
        .reading_order
        .iter()
        .map(|&idx| region_to_json(&page.regions[idx]))
        .collect();

    serde_json::json!({
        "page_index": page_index,
        "width": page.width,
        "height": page.height,
        "region_count": regions.len(),
        "regions": regions,
    })
}

/// Convert a single region to a JSON value.
fn region_to_json(region: &DocumentRegion) -> serde_json::Value {
    let bbox = region.bbox();
    let mut obj = serde_json::json!({
        "type": region.class_name(),
        "confidence": region.confidence(),
        "bbox": {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
        },
    });

    // Add content-specific fields.
    match region {
        DocumentRegion::Text { content, .. }
        | DocumentRegion::SectionHeader { content, .. }
        | DocumentRegion::PageHeader { content, .. }
        | DocumentRegion::PageFooter { content, .. }
        | DocumentRegion::Caption { content, .. }
        | DocumentRegion::ListItem { content, .. }
        | DocumentRegion::Footnote { content, .. } => {
            obj["content"] = serde_json::Value::String(content.clone());
        }
        DocumentRegion::Table { cells, .. } => {
            let rows: Vec<serde_json::Value> = cells
                .iter()
                .map(|row| {
                    serde_json::Value::Array(
                        row.iter()
                            .map(|c| serde_json::Value::String(c.clone()))
                            .collect(),
                    )
                })
                .collect();
            obj["cells"] = serde_json::Value::Array(rows);
        }
        DocumentRegion::Figure { caption, .. } => {
            obj["caption"] = match caption {
                Some(c) => serde_json::Value::String(c.clone()),
                None => serde_json::Value::Null,
            };
        }
        DocumentRegion::Formula { latex, .. } => {
            obj["latex"] = match latex {
                Some(l) => serde_json::Value::String(l.clone()),
                None => serde_json::Value::Null,
            };
        }
    }

    obj
}

// ---------------------------------------------------------------------------
// HTML exporter
// ---------------------------------------------------------------------------

/// Exports [`DocumentOutput`] as semantic HTML.
///
/// Region type mapping:
/// - `SectionHeader` -> `<h1>`
/// - `Text` -> `<p>`
/// - `Table` -> `<table>`
/// - `Figure` -> `<figure>`
/// - `ListItem` -> collected into `<ul><li>` groups
/// - `Formula` -> `<pre class="formula">`
/// - `Caption` -> `<p class="caption">`
/// - `Footnote` -> `<aside class="footnote">`
/// - `PageHeader` / `PageFooter` -> `<header>` / `<footer>`
#[derive(Debug, Clone, Default)]
pub struct HtmlExporter;

impl HtmlExporter {
    /// Create a new HTML exporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DocumentExporter for HtmlExporter {
    fn export(&self, output: &DocumentOutput) -> Result<String, ExportError> {
        let mut html = String::with_capacity(4096);
        html.push_str("<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"></head>\n<body>\n");

        for (i, page) in output.pages.iter().enumerate() {
            html.push_str(&format!(
                "<section class=\"page\" data-page=\"{}\" data-width=\"{}\" data-height=\"{}\">\n",
                i, page.width, page.height,
            ));
            export_page_html(&mut html, page);
            html.push_str("</section>\n");
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }
}

/// Append HTML for a single page to the buffer.
fn export_page_html(html: &mut String, page: &PageOutput) {
    for &idx in &page.reading_order {
        let region = &page.regions[idx];
        region_to_html(html, region);
    }
}

/// Append HTML for a single region to the buffer.
fn region_to_html(html: &mut String, region: &DocumentRegion) {
    match region {
        DocumentRegion::SectionHeader { content, .. } => {
            html.push_str("<h1>");
            push_html_escaped(html, content);
            html.push_str("</h1>\n");
        }
        DocumentRegion::Text { content, .. } => {
            html.push_str("<p>");
            push_html_escaped(html, content);
            html.push_str("</p>\n");
        }
        DocumentRegion::Table { cells, .. } => {
            html.push_str("<table>\n");
            for (i, row) in cells.iter().enumerate() {
                html.push_str("  <tr>");
                let tag = if i == 0 { "th" } else { "td" };
                for cell in row {
                    html.push_str(&format!("<{tag}>"));
                    push_html_escaped(html, cell);
                    html.push_str(&format!("</{tag}>"));
                }
                html.push_str("</tr>\n");
            }
            html.push_str("</table>\n");
        }
        DocumentRegion::Figure { caption, .. } => {
            html.push_str("<figure>\n");
            html.push_str("  <figcaption>");
            if let Some(cap) = caption {
                push_html_escaped(html, cap);
            }
            html.push_str("</figcaption>\n");
            html.push_str("</figure>\n");
        }
        DocumentRegion::ListItem { content, .. } => {
            html.push_str("<ul><li>");
            push_html_escaped(html, content);
            html.push_str("</li></ul>\n");
        }
        DocumentRegion::Formula { latex, .. } => {
            html.push_str("<pre class=\"formula\">");
            if let Some(l) = latex {
                push_html_escaped(html, l);
            }
            html.push_str("</pre>\n");
        }
        DocumentRegion::Caption { content, .. } => {
            html.push_str("<p class=\"caption\">");
            push_html_escaped(html, content);
            html.push_str("</p>\n");
        }
        DocumentRegion::Footnote { content, .. } => {
            html.push_str("<aside class=\"footnote\">");
            push_html_escaped(html, content);
            html.push_str("</aside>\n");
        }
        DocumentRegion::PageHeader { content, .. } => {
            html.push_str("<header>");
            push_html_escaped(html, content);
            html.push_str("</header>\n");
        }
        DocumentRegion::PageFooter { content, .. } => {
            html.push_str("<footer>");
            push_html_escaped(html, content);
            html.push_str("</footer>\n");
        }
    }
}

/// Push HTML-escaped text into a string buffer.
fn push_html_escaped(buf: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '<' => buf.push_str("&lt;"),
            '>' => buf.push_str("&gt;"),
            '&' => buf.push_str("&amp;"),
            '"' => buf.push_str("&quot;"),
            _ => buf.push(ch),
        }
    }
}

// ---------------------------------------------------------------------------
// Markdown exporter
// ---------------------------------------------------------------------------

/// Exports [`DocumentOutput`] as Markdown.
///
/// Region type mapping:
/// - `SectionHeader` -> `# heading`
/// - `Text` -> paragraph
/// - `Table` -> pipe-syntax Markdown table
/// - `Figure` -> `![caption]()`
/// - `ListItem` -> `- item`
/// - `Formula` -> `` `$latex$` `` code block
/// - `Caption` -> italic paragraph
/// - `Footnote` -> `[^N]: text`
/// - `PageHeader` / `PageFooter` -> bold text
#[derive(Debug, Clone, Default)]
pub struct MarkdownExporter;

impl MarkdownExporter {
    /// Create a new Markdown exporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DocumentExporter for MarkdownExporter {
    fn export(&self, output: &DocumentOutput) -> Result<String, ExportError> {
        let mut md = String::with_capacity(4096);

        for (i, page) in output.pages.iter().enumerate() {
            if i > 0 {
                md.push_str("\n---\n\n");
            }
            export_page_markdown(&mut md, page);
        }

        // Trim trailing whitespace.
        let trimmed = md.trim_end().to_string();
        Ok(trimmed)
    }
}

/// Append Markdown for a single page.
fn export_page_markdown(md: &mut String, page: &PageOutput) {
    let mut footnote_counter = 0usize;

    for (i, &idx) in page.reading_order.iter().enumerate() {
        if i > 0 {
            md.push_str("\n\n");
        }
        let region = &page.regions[idx];
        region_to_markdown(md, region, &mut footnote_counter);
    }
}

/// Append Markdown for a single region.
fn region_to_markdown(md: &mut String, region: &DocumentRegion, footnote_counter: &mut usize) {
    match region {
        DocumentRegion::SectionHeader { content, .. } => {
            md.push_str("# ");
            md.push_str(content);
        }
        DocumentRegion::Text { content, .. } => {
            md.push_str(content);
        }
        DocumentRegion::Table { cells, .. } => {
            if cells.is_empty() {
                md.push_str("[table]");
            } else {
                md.push_str(&table_to_pipe_markdown(cells));
            }
        }
        DocumentRegion::Figure { caption, .. } => {
            let cap = caption.as_deref().unwrap_or("Figure");
            md.push_str(&format!("![{cap}]()"));
        }
        DocumentRegion::ListItem { content, .. } => {
            md.push_str("- ");
            md.push_str(content);
        }
        DocumentRegion::Formula { latex, .. } => match latex {
            Some(l) => {
                md.push('$');
                md.push_str(l);
                md.push('$');
            }
            None => md.push_str("[formula]"),
        },
        DocumentRegion::Caption { content, .. } => {
            md.push('*');
            md.push_str(content);
            md.push('*');
        }
        DocumentRegion::Footnote { content, .. } => {
            *footnote_counter += 1;
            md.push_str(&format!("[^{footnote_counter}]: {content}"));
        }
        DocumentRegion::PageHeader { content, .. } => {
            md.push_str("**");
            md.push_str(content);
            md.push_str("**");
        }
        DocumentRegion::PageFooter { content, .. } => {
            md.push_str("**");
            md.push_str(content);
            md.push_str("**");
        }
    }
}

/// Render a cell grid as a Markdown pipe table.
fn table_to_pipe_markdown(cells: &[Vec<String>]) -> String {
    if cells.is_empty() {
        return String::new();
    }
    let mut lines = Vec::with_capacity(cells.len() + 1);
    // Header row.
    let header = &cells[0];
    lines.push(format!("| {} |", header.join(" | ")));
    // Separator.
    let sep: Vec<&str> = (0..header.len()).map(|_| "---").collect();
    lines.push(format!("| {} |", sep.join(" | ")));
    // Data rows.
    for row in cells.iter().skip(1) {
        lines.push(format!("| {} |", row.join(" | ")));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// CSV table exporter
// ---------------------------------------------------------------------------

/// Exports table cells from [`DocumentOutput`] as CSV.
///
/// Output format: `page,region_index,row,col,text,confidence`
///
/// Only `Table` regions are included. Non-table regions are skipped.
#[derive(Debug, Clone, Default)]
pub struct CsvTableExporter;

impl CsvTableExporter {
    /// Create a new CSV table exporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DocumentExporter for CsvTableExporter {
    fn export(&self, output: &DocumentOutput) -> Result<String, ExportError> {
        let mut csv = String::with_capacity(1024);
        csv.push_str("page,region_index,row,col,text,confidence\n");

        for (page_idx, page) in output.pages.iter().enumerate() {
            for (region_idx, region) in page.regions.iter().enumerate() {
                if let DocumentRegion::Table {
                    cells, confidence, ..
                } = region
                {
                    for (r, row) in cells.iter().enumerate() {
                        for (c, cell_text) in row.iter().enumerate() {
                            csv.push_str(&format!(
                                "{},{},{},{},{},{:.4}\n",
                                page_idx,
                                region_idx,
                                r,
                                c,
                                csv_escape(cell_text),
                                confidence,
                            ));
                        }
                    }
                }
            }
        }

        Ok(csv)
    }
}

/// Escape a CSV field: wrap in quotes if it contains commas, quotes, or newlines.
fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "dpdf_export_tests.rs"]
mod tests;
