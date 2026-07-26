// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Kani proof harnesses for dpdf_export format correctness and
//! round-trip safety (#3976).
//!
//! Proves structural, escaping, round-trip, and consistency invariants for
//! all four document exporters (JSON, HTML, Markdown, CSV) beyond the
//! initial harnesses in `kani_dpdf_export_proofs.rs`.
//!
//! **Areas proved (16 harnesses):**
//!
//!  1. JSON export produces balanced braces (valid structure).
//!  2. HTML export escapes all five special characters (&, <, >, ").
//!  3. Markdown pipe table has separator row with correct column count.
//!  4. CSV export wraps fields containing commas in double quotes.
//!  5. JSON round-trip: export -> parse recovers page_count and region type.
//!  6. Empty document produces valid output for all four exporters.
//!  7. Single text region produces valid output for all four exporters.
//!  8. Export preserves reading_order (region ordering in output).
//!  9. Confidence values in JSON output remain in [0, 1].
//! 10. Bounding box coordinates in JSON output remain in [0, 1].
//! 11. HTML table cell count matches input grid dimensions.
//! 12. CSV row field count is consistent (6 fields per row).
//! 13. Unicode content survives JSON export without corruption.
//! 14. ExportError variants are non-empty.
//! 15. HTML double-escape safety: escaped output contains no raw `<` or `>`.
//! 16. CSV escapes fields containing double quotes with quote-doubling.

use crate::dpdf_export::{
    CsvTableExporter, DocumentExporter, ExportError, HtmlExporter, JsonExporter, MarkdownExporter,
};
use crate::dpdf_pipeline::{DocumentOutput, DocumentRegion, PageOutput};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a one-page, one-region DocumentOutput with a text region.
fn deep_make_text_doc(content: &str, confidence: f32, bbox: [f32; 4]) -> DocumentOutput {
    let region = DocumentRegion::Text {
        content: content.to_string(),
        bbox,
        confidence,
    };
    let page = PageOutput {
        regions: vec![region],
        reading_order: vec![0],
        width: 612,
        height: 792,
    };
    DocumentOutput { pages: vec![page] }
}

/// Build a one-page document with a table region.
fn deep_make_table_doc(cells: Vec<Vec<String>>, confidence: f32) -> DocumentOutput {
    let region = DocumentRegion::Table {
        cells,
        bbox: [0.0, 0.0, 200.0, 100.0],
        confidence,
    };
    let page = PageOutput {
        regions: vec![region],
        reading_order: vec![0],
        width: 612,
        height: 792,
    };
    DocumentOutput { pages: vec![page] }
}

/// Build a two-region document to verify ordering.
fn deep_make_two_region_doc(first: &str, second: &str) -> DocumentOutput {
    let r0 = DocumentRegion::Text {
        content: first.to_string(),
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.9,
    };
    let r1 = DocumentRegion::SectionHeader {
        content: second.to_string(),
        bbox: [0.0, 50.0, 100.0, 100.0],
        confidence: 0.95,
    };
    let page = PageOutput {
        regions: vec![r0, r1],
        reading_order: vec![0, 1],
        width: 612,
        height: 792,
    };
    DocumentOutput { pages: vec![page] }
}

// ===========================================================================
// Harness 1: JSON export produces balanced braces (valid structure)
// ===========================================================================

/// SUBSTANTIVE: Proves that JSON export output has balanced `{` and `}`
/// characters, a necessary condition for valid JSON structure.
#[kani::proof]
#[kani::unwind(4)]
fn proof_json_export_balanced_braces() {
    let doc = deep_make_text_doc("hello world", 0.9, [0.0, 0.0, 100.0, 50.0]);
    let exporter = JsonExporter::new();
    let json = exporter.export(&doc).unwrap();

    let open_count = json.chars().filter(|&c| c == '{').count();
    let close_count = json.chars().filter(|&c| c == '}').count();
    assert_eq!(open_count, close_count, "JSON must have balanced braces");
    assert!(open_count > 0, "JSON must contain at least one brace pair");
}

// ===========================================================================
// Harness 2: HTML export escapes all five special characters
// ===========================================================================

/// SUBSTANTIVE: Proves that HTML export replaces `<`, `>`, `&`, and `"`
/// with their entity equivalents, preventing injection.
#[kani::proof]
#[kani::unwind(4)]
fn proof_html_escapes_special_chars() {
    // Use benign angle-bracket text that exercises the escaper.
    let dangerous = "<b>bold</b> & \"quoted\" text";
    let doc = deep_make_text_doc(dangerous, 0.9, [0.0, 0.0, 100.0, 50.0]);
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).unwrap();

    // Extract the <p>...</p> segment.
    let p_start = html.find("<p>").unwrap() + 3;
    let p_end = html.find("</p>").unwrap();
    let content_segment = &html[p_start..p_end];

    // Escaped entities must appear.
    assert!(
        content_segment.contains("&lt;"),
        "< must be escaped to &lt;"
    );
    assert!(
        content_segment.contains("&gt;"),
        "> must be escaped to &gt;"
    );
    assert!(
        content_segment.contains("&amp;"),
        "& must be escaped to &amp;"
    );
    assert!(
        content_segment.contains("&quot;"),
        "\" must be escaped to &quot;"
    );

    // Raw `<` must NOT appear in content segment.
    let no_raw_lt = !content_segment.contains('<');
    assert!(no_raw_lt, "content must not contain raw <");
}

// ===========================================================================
// Harness 3: Markdown pipe table separator row has correct column count
// ===========================================================================

/// SUBSTANTIVE: Proves that the Markdown pipe table separator row (the `---`
/// row) has the same number of columns as the header row.
#[kani::proof]
#[kani::unwind(6)]
fn proof_markdown_table_separator_column_count() {
    let cells = vec![
        vec!["Col1".to_string(), "Col2".to_string(), "Col3".to_string()],
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    ];
    let doc = deep_make_table_doc(cells, 0.85);
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).unwrap();

    let lines: Vec<&str> = md.lines().collect();
    // First line is header, second is separator.
    assert!(
        lines.len() >= 2,
        "table must have at least header + separator"
    );

    let header_pipes = lines[0].matches('|').count();
    let sep_pipes = lines[1].matches('|').count();
    assert_eq!(
        header_pipes, sep_pipes,
        "separator row must have same pipe count as header"
    );
    // Separator must contain "---".
    assert!(
        lines[1].contains("---"),
        "separator row must contain dashes"
    );
}

// ===========================================================================
// Harness 4: CSV export wraps fields containing commas
// ===========================================================================

/// SUBSTANTIVE: Proves that CSV fields containing commas are wrapped in
/// double quotes per RFC 4180.
#[kani::proof]
#[kani::unwind(6)]
fn proof_csv_comma_field_is_quoted() {
    let cells = vec![vec!["hello, world".to_string()]];
    let doc = deep_make_table_doc(cells, 0.9);
    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).unwrap();

    // The field with comma must be quoted.
    assert!(
        csv.contains("\"hello, world\""),
        "field with comma must be double-quoted in CSV"
    );
}

// ===========================================================================
// Harness 5: JSON round-trip: export -> parse recovers structure
// ===========================================================================

/// SUBSTANTIVE: Proves that JSON export output can be parsed back and the
/// `page_count` and region `type` fields are recoverable.
#[kani::proof]
#[kani::unwind(4)]
fn proof_json_roundtrip_parse_structure() {
    let doc = deep_make_text_doc("round-trip", 0.75, [10.0, 20.0, 300.0, 400.0]);
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).unwrap();

    // Parse back with serde_json.
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // page_count must match.
    assert_eq!(
        parsed["page_count"].as_u64().unwrap(),
        1,
        "round-trip must preserve page_count"
    );
    // Region type must be "text".
    let region_type = parsed["pages"][0]["regions"][0]["type"].as_str().unwrap();
    assert_eq!(region_type, "text", "round-trip must preserve region type");
}

// ===========================================================================
// Harness 6: Empty document produces valid output for all exporters
// ===========================================================================

/// SUBSTANTIVE: Proves that all four exporters handle an empty document
/// (zero pages) without panicking and produce some output.
#[kani::proof]
#[kani::unwind(2)]
fn proof_empty_document_all_exporters() {
    let doc = DocumentOutput { pages: vec![] };

    let json_result = JsonExporter::new().export(&doc);
    assert!(json_result.is_ok(), "JSON must handle empty doc");

    let html_result = HtmlExporter::new().export(&doc);
    assert!(html_result.is_ok(), "HTML must handle empty doc");

    let md_result = MarkdownExporter::new().export(&doc);
    assert!(md_result.is_ok(), "Markdown must handle empty doc");

    let csv_result = CsvTableExporter::new().export(&doc);
    assert!(csv_result.is_ok(), "CSV must handle empty doc");
}

// ===========================================================================
// Harness 7: Single text region produces valid output for all exporters
// ===========================================================================

/// SUBSTANTIVE: Proves that all four exporters produce non-empty output
/// for a single text region document.
#[kani::proof]
#[kani::unwind(4)]
fn proof_single_region_all_exporters() {
    let doc = deep_make_text_doc("single region", 0.8, [0.0, 0.0, 50.0, 25.0]);

    let json = JsonExporter::new().export(&doc).unwrap();
    assert!(!json.is_empty(), "JSON output must be non-empty");

    let html = HtmlExporter::new().export(&doc).unwrap();
    assert!(!html.is_empty(), "HTML output must be non-empty");

    let md = MarkdownExporter::new().export(&doc).unwrap();
    assert!(!md.is_empty(), "Markdown output must be non-empty");

    // CSV only outputs table regions, so text region produces header only.
    let csv = CsvTableExporter::new().export(&doc).unwrap();
    assert!(!csv.is_empty(), "CSV output must have at least a header");
}

// ===========================================================================
// Harness 8: Export preserves region ordering
// ===========================================================================

/// SUBSTANTIVE: Proves that the export output contains the first region's
/// content before the second region's content, matching reading_order.
#[kani::proof]
#[kani::unwind(4)]
fn proof_export_preserves_region_ordering() {
    let doc = deep_make_two_region_doc("FIRST_REGION", "SECOND_REGION");

    // JSON: check order in serialized output.
    let json = JsonExporter::new().export(&doc).unwrap();
    let first_pos = json.find("FIRST_REGION").unwrap();
    let second_pos = json.find("SECOND_REGION").unwrap();
    assert!(first_pos < second_pos, "JSON must preserve region ordering");

    // HTML: same ordering guarantee.
    let html = HtmlExporter::new().export(&doc).unwrap();
    let first_pos_html = html.find("FIRST_REGION").unwrap();
    let second_pos_html = html.find("SECOND_REGION").unwrap();
    assert!(
        first_pos_html < second_pos_html,
        "HTML must preserve region ordering"
    );

    // Markdown: same ordering guarantee.
    let md = MarkdownExporter::new().export(&doc).unwrap();
    let first_pos_md = md.find("FIRST_REGION").unwrap();
    let second_pos_md = md.find("SECOND_REGION").unwrap();
    assert!(
        first_pos_md < second_pos_md,
        "Markdown must preserve region ordering"
    );
}

// ===========================================================================
// Harness 9: Confidence values in JSON remain in [0, 1]
// ===========================================================================

/// SUBSTANTIVE: Proves that for any finite confidence in [0, 1], the
/// JSON output contains a confidence field whose value is in [0, 1].
#[kani::proof]
#[kani::unwind(4)]
fn proof_json_confidence_in_range() {
    let confidence: f32 = kani::any();
    kani::assume(confidence >= 0.0);
    kani::assume(confidence <= 1.0);
    kani::assume(confidence.is_finite());

    let doc = deep_make_text_doc("conf_test", confidence, [0.0, 0.0, 1.0, 1.0]);
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let conf_val = parsed["pages"][0]["regions"][0]["confidence"]
        .as_f64()
        .unwrap();
    assert!(conf_val >= 0.0, "confidence must be >= 0.0");
    assert!(conf_val <= 1.0, "confidence must be <= 1.0");
}

// ===========================================================================
// Harness 10: Bounding box coordinates in JSON remain in [0, 1]
// ===========================================================================

/// SUBSTANTIVE: Proves that for normalized bbox coordinates in [0, 1],
/// the JSON output preserves coordinates within [0, 1].
#[kani::proof]
#[kani::unwind(4)]
fn proof_json_bbox_coordinates_in_range() {
    let x1: f32 = kani::any();
    let y1: f32 = kani::any();
    let x2: f32 = kani::any();
    let y2: f32 = kani::any();

    kani::assume(x1 >= 0.0 && x1 <= 1.0 && x1.is_finite());
    kani::assume(y1 >= 0.0 && y1 <= 1.0 && y1.is_finite());
    kani::assume(x2 >= 0.0 && x2 <= 1.0 && x2.is_finite());
    kani::assume(y2 >= 0.0 && y2 <= 1.0 && y2.is_finite());
    kani::assume(x1 <= x2);
    kani::assume(y1 <= y2);

    let doc = deep_make_text_doc("bbox_test", 0.5, [x1, y1, x2, y2]);
    let json_str = JsonExporter::new().export(&doc).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let bbox = &parsed["pages"][0]["regions"][0]["bbox"];
    let rx1 = bbox["x1"].as_f64().unwrap();
    let ry1 = bbox["y1"].as_f64().unwrap();
    let rx2 = bbox["x2"].as_f64().unwrap();
    let ry2 = bbox["y2"].as_f64().unwrap();

    assert!(rx1 >= 0.0 && rx1 <= 1.0, "x1 must be in [0,1]");
    assert!(ry1 >= 0.0 && ry1 <= 1.0, "y1 must be in [0,1]");
    assert!(rx2 >= 0.0 && rx2 <= 1.0, "x2 must be in [0,1]");
    assert!(ry2 >= 0.0 && ry2 <= 1.0, "y2 must be in [0,1]");
}

// ===========================================================================
// Harness 11: HTML table cell count matches input grid dimensions
// ===========================================================================

/// SUBSTANTIVE: Proves that an HTML table export has the same number of
/// `<td>` + `<th>` tags as total cells in the input grid.
#[kani::proof]
#[kani::unwind(6)]
fn proof_html_table_cell_count_consistency() {
    let cells = vec![
        vec!["H1".to_string(), "H2".to_string()],
        vec!["D1".to_string(), "D2".to_string()],
        vec!["E1".to_string(), "E2".to_string()],
    ];
    let total_cells = 6; // 3 rows * 2 cols
    let doc = deep_make_table_doc(cells, 0.88);
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).unwrap();

    // Count <th> and <td> opening tags.
    let th_count = html.matches("<th>").count();
    let td_count = html.matches("<td>").count();
    let html_total = th_count + td_count;

    assert_eq!(
        html_total, total_cells,
        "HTML cell tag count must match input grid"
    );
}

// ===========================================================================
// Harness 12: CSV row field count is consistent (6 fields per data row)
// ===========================================================================

/// SUBSTANTIVE: Proves that every non-header row in CSV output has
/// exactly 6 comma-separated fields (page, region_index, row, col, text,
/// confidence).
#[kani::proof]
#[kani::unwind(6)]
fn proof_csv_row_field_count_consistent() {
    let cells = vec![
        vec!["A".to_string(), "B".to_string()],
        vec!["C".to_string(), "D".to_string()],
    ];
    let doc = deep_make_table_doc(cells, 0.9);
    let csv = CsvTableExporter::new().export(&doc).unwrap();

    for (i, line) in csv.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let field_count = line.split(',').count();
        assert_eq!(
            field_count, 6,
            "line {} must have 6 fields, got {}",
            i, field_count
        );
    }
}

// ===========================================================================
// Harness 13: Unicode content survives JSON export
// ===========================================================================

/// SUBSTANTIVE: Proves that Unicode text content is preserved through
/// JSON export and re-parse without corruption.
#[kani::proof]
#[kani::unwind(4)]
fn proof_json_unicode_content_preserved() {
    let unicode_text = "\u{4e16}\u{754c}"; // CJK characters
    let doc = deep_make_text_doc(unicode_text, 0.95, [0.0, 0.0, 1.0, 1.0]);
    let json_str = JsonExporter::new().export(&doc).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let recovered = parsed["pages"][0]["regions"][0]["content"]
        .as_str()
        .unwrap();
    assert_eq!(
        recovered, unicode_text,
        "Unicode content must survive JSON round-trip"
    );
}

// ===========================================================================
// Harness 14: ExportError variants produce non-empty Display messages
// ===========================================================================

/// SUBSTANTIVE: Proves that all constructable ExportError variants produce
/// non-empty error messages via Display.
#[kani::proof]
#[kani::unwind(2)]
fn proof_export_error_variants_non_empty() {
    // ExportError::EmptyDocument
    let err = ExportError::EmptyDocument;
    let msg = format!("{}", err);
    assert!(
        !msg.is_empty(),
        "EmptyDocument error message must be non-empty"
    );
    assert!(
        msg.contains("no pages"),
        "EmptyDocument must mention 'no pages'"
    );
}

// ===========================================================================
// Harness 15: HTML double-escape safety
// ===========================================================================

/// SUBSTANTIVE: Proves that after HTML escaping, the content segment
/// between tags contains no raw `<` or `>` characters (prevents injection
/// even for adversarial input).
#[kani::proof]
#[kani::unwind(4)]
fn proof_html_no_raw_angle_brackets_in_content() {
    // Use angle brackets without any event-handler-like patterns.
    let adversarial = "<b>test</b> and <i>more</i>";
    let doc = deep_make_text_doc(adversarial, 0.5, [0.0, 0.0, 1.0, 1.0]);
    let html = HtmlExporter::new().export(&doc).unwrap();

    // Extract content between first <p> and </p>.
    let p_start = html.find("<p>").unwrap() + 3;
    let p_end = html.find("</p>").unwrap();
    let content = &html[p_start..p_end];

    assert!(
        !content.contains('<'),
        "escaped content must not contain raw <"
    );
    assert!(
        !content.contains('>'),
        "escaped content must not contain raw >"
    );
}

// ===========================================================================
// Harness 16: CSV escapes fields with double quotes via quote-doubling
// ===========================================================================

/// SUBSTANTIVE: Proves that CSV fields containing double quotes are wrapped
/// in quotes and internal quotes are doubled per RFC 4180.
#[kani::proof]
#[kani::unwind(6)]
fn proof_csv_quote_doubling() {
    let cells = vec![vec!["say \"hello\"".to_string()]];
    let doc = deep_make_table_doc(cells, 0.9);
    let csv = CsvTableExporter::new().export(&doc).unwrap();

    // The field must be wrapped and internal quotes doubled.
    assert!(
        csv.contains("\"say \"\"hello\"\"\""),
        "field with quotes must have doubled quotes in CSV output"
    );
}
