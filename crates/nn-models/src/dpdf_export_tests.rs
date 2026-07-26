// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dpdf document output export module.

use super::*;
use crate::dpdf_pipeline::{DocumentOutput, DocumentRegion, PageOutput};

/// Build a simple test document with a variety of region types.
fn sample_document() -> DocumentOutput {
    let regions = vec![
        DocumentRegion::PageHeader {
            content: "Page Header".to_string(),
            bbox: [0.0, 0.0, 600.0, 30.0],
            confidence: 0.99,
        },
        DocumentRegion::SectionHeader {
            content: "Introduction".to_string(),
            bbox: [10.0, 40.0, 300.0, 70.0],
            confidence: 0.95,
        },
        DocumentRegion::Text {
            content: "Hello world.".to_string(),
            bbox: [10.0, 80.0, 580.0, 120.0],
            confidence: 0.92,
        },
        DocumentRegion::Table {
            cells: vec![
                vec!["Name".to_string(), "Age".to_string()],
                vec!["Alice".to_string(), "30".to_string()],
            ],
            bbox: [10.0, 130.0, 580.0, 250.0],
            confidence: 0.88,
        },
        DocumentRegion::Figure {
            caption: Some("Diagram 1".to_string()),
            bbox: [10.0, 260.0, 580.0, 400.0],
            confidence: 0.85,
        },
        DocumentRegion::ListItem {
            content: "First item".to_string(),
            bbox: [20.0, 410.0, 300.0, 430.0],
            confidence: 0.90,
        },
        DocumentRegion::Formula {
            latex: Some("E = mc^2".to_string()),
            bbox: [10.0, 440.0, 200.0, 470.0],
            confidence: 0.93,
        },
        DocumentRegion::Caption {
            content: "Table 1 caption".to_string(),
            bbox: [10.0, 475.0, 300.0, 490.0],
            confidence: 0.87,
        },
        DocumentRegion::Footnote {
            content: "See references.".to_string(),
            bbox: [10.0, 700.0, 300.0, 720.0],
            confidence: 0.80,
        },
        DocumentRegion::PageFooter {
            content: "Page 1".to_string(),
            bbox: [0.0, 750.0, 600.0, 780.0],
            confidence: 0.98,
        },
    ];

    let reading_order: Vec<usize> = (0..regions.len()).collect();
    let page = PageOutput {
        regions,
        reading_order,
        width: 612,
        height: 792,
    };

    DocumentOutput { pages: vec![page] }
}

/// Build a minimal document with a single text region.
fn minimal_document() -> DocumentOutput {
    let regions = vec![DocumentRegion::Text {
        content: "Simple text".to_string(),
        bbox: [10.0, 20.0, 300.0, 80.0],
        confidence: 0.95,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 612,
        height: 792,
    };
    DocumentOutput { pages: vec![page] }
}

/// Build a document with no pages.
fn empty_document() -> DocumentOutput {
    DocumentOutput { pages: vec![] }
}

// ---------------------------------------------------------------------------
// JSON exporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_json_exporter_compact_roundtrips() {
    let doc = sample_document();
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export should succeed");

    // Parse back to verify valid JSON.
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("output should be valid JSON");

    assert_eq!(parsed["page_count"], 1);
    let pages = parsed["pages"].as_array().expect("pages should be array");
    assert_eq!(pages.len(), 1);
    let page = &pages[0];
    assert_eq!(page["width"], 612);
    assert_eq!(page["height"], 792);

    let regions = page["regions"].as_array().expect("regions array");
    assert_eq!(regions.len(), 10);

    // Check first region is page-header.
    assert_eq!(regions[0]["type"], "page-header");
    assert_eq!(regions[0]["content"], "Page Header");
}

#[test]
fn test_json_exporter_pretty_has_newlines() {
    let doc = minimal_document();
    let exporter = JsonExporter::pretty();
    let json_str = exporter.export(&doc).expect("pretty JSON export");
    assert!(
        json_str.contains('\n'),
        "pretty JSON should contain newlines"
    );
}

#[test]
fn test_json_exporter_table_cells() {
    let doc = sample_document();
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let regions = parsed["pages"][0]["regions"].as_array().unwrap();
    let table_region = regions.iter().find(|r| r["type"] == "table").unwrap();
    let cells = table_region["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 2);
    assert_eq!(cells[0][0], "Name");
    assert_eq!(cells[1][1], "30");
}

#[test]
fn test_json_exporter_figure_caption() {
    let doc = sample_document();
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let regions = parsed["pages"][0]["regions"].as_array().unwrap();
    let figure = regions.iter().find(|r| r["type"] == "picture").unwrap();
    assert_eq!(figure["caption"], "Diagram 1");
}

#[test]
fn test_json_exporter_formula_latex() {
    let doc = sample_document();
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let regions = parsed["pages"][0]["regions"].as_array().unwrap();
    let formula = regions.iter().find(|r| r["type"] == "formula").unwrap();
    assert_eq!(formula["latex"], "E = mc^2");
}

#[test]
fn test_json_exporter_empty_document() {
    let doc = empty_document();
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("empty doc JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["page_count"], 0);
}

#[test]
fn test_json_exporter_bbox_fields() {
    let doc = minimal_document();
    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let bbox = &parsed["pages"][0]["regions"][0]["bbox"];
    assert_eq!(bbox["x1"], 10.0);
    assert_eq!(bbox["y1"], 20.0);
    assert_eq!(bbox["x2"], 300.0);
    assert_eq!(bbox["y2"], 80.0);
}

// ---------------------------------------------------------------------------
// HTML exporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_html_exporter_valid_structure() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");

    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("<html>"));
    assert!(html.contains("</html>"));
    assert!(html.contains("<body>"));
    assert!(html.contains("</body>"));
}

#[test]
fn test_html_exporter_section_header_as_h1() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<h1>Introduction</h1>"));
}

#[test]
fn test_html_exporter_text_as_p() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<p>Hello world.</p>"));
}

#[test]
fn test_html_exporter_table_structure() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<table>"));
    assert!(html.contains("</table>"));
    assert!(html.contains("<th>Name</th>"));
    assert!(html.contains("<td>Alice</td>"));
}

#[test]
fn test_html_exporter_figure() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<figure>"));
    assert!(html.contains("<figcaption>Diagram 1</figcaption>"));
}

#[test]
fn test_html_exporter_list_item() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<ul><li>First item</li></ul>"));
}

#[test]
fn test_html_exporter_formula() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<pre class=\"formula\">E = mc^2</pre>"));
}

#[test]
fn test_html_exporter_page_attributes() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("data-page=\"0\""));
    assert!(html.contains("data-width=\"612\""));
    assert!(html.contains("data-height=\"792\""));
}

#[test]
fn test_html_exporter_escapes_special_chars() {
    let regions = vec![DocumentRegion::Text {
        content: "<script>alert('xss')</script>".to_string(),
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.9,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(!html.contains("<script>"));
    assert!(html.contains("&lt;script&gt;"));
}

#[test]
fn test_html_exporter_header_footer() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<header>Page Header</header>"));
    assert!(html.contains("<footer>Page 1</footer>"));
}

#[test]
fn test_html_exporter_footnote() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<aside class=\"footnote\">See references.</aside>"));
}

#[test]
fn test_html_exporter_caption() {
    let doc = sample_document();
    let exporter = HtmlExporter::new();
    let html = exporter.export(&doc).expect("HTML export");
    assert!(html.contains("<p class=\"caption\">Table 1 caption</p>"));
}

// ---------------------------------------------------------------------------
// Markdown exporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_markdown_exporter_section_header() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("# Introduction"));
}

#[test]
fn test_markdown_exporter_text_paragraph() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("Hello world."));
}

#[test]
fn test_markdown_exporter_pipe_table() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("| Name | Age |"));
    assert!(md.contains("| --- | --- |"));
    assert!(md.contains("| Alice | 30 |"));
}

#[test]
fn test_markdown_exporter_figure() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("![Diagram 1]()"));
}

#[test]
fn test_markdown_exporter_list_item() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("- First item"));
}

#[test]
fn test_markdown_exporter_formula() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("$E = mc^2$"));
}

#[test]
fn test_markdown_exporter_caption() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("*Table 1 caption*"));
}

#[test]
fn test_markdown_exporter_footnote() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("[^1]: See references."));
}

#[test]
fn test_markdown_exporter_page_header_footer_bold() {
    let doc = sample_document();
    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("**Page Header**"));
    assert!(md.contains("**Page 1**"));
}

#[test]
fn test_markdown_exporter_page_separator() {
    let page = PageOutput {
        regions: vec![DocumentRegion::Text {
            content: "Page text".to_string(),
            bbox: [0.0, 0.0, 100.0, 50.0],
            confidence: 0.9,
        }],
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput {
        pages: vec![page.clone(), page],
    };

    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(
        md.contains("---"),
        "multi-page documents should have separator"
    );
}

#[test]
fn test_markdown_exporter_empty_table() {
    let regions = vec![DocumentRegion::Table {
        cells: vec![],
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.9,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("[table]"));
}

// ---------------------------------------------------------------------------
// CSV table exporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_csv_exporter_header_row() {
    let doc = sample_document();
    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).expect("CSV export");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "page,region_index,row,col,text,confidence");
}

#[test]
fn test_csv_exporter_table_cells() {
    let doc = sample_document();
    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).expect("CSV export");
    let lines: Vec<&str> = csv.lines().collect();

    // Header + 4 data rows (2x2 table).
    assert_eq!(lines.len(), 5);
    assert!(lines[1].contains("Name"));
    assert!(lines[2].contains("Age"));
    assert!(lines[3].contains("Alice"));
    assert!(lines[4].contains("30"));
}

#[test]
fn test_csv_exporter_no_tables() {
    let doc = minimal_document(); // only text, no tables
    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).expect("CSV export");
    let lines: Vec<&str> = csv.lines().collect();
    // Only header row.
    assert_eq!(lines.len(), 1);
}

#[test]
fn test_csv_exporter_escapes_commas() {
    let regions = vec![DocumentRegion::Table {
        cells: vec![vec!["hello, world".to_string()]],
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.9,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).expect("CSV export");
    assert!(
        csv.contains("\"hello, world\""),
        "commas in cells should be quoted"
    );
}

#[test]
fn test_csv_exporter_confidence_precision() {
    let doc = sample_document();
    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).expect("CSV export");
    // Table confidence is 0.88 -> "0.8800"
    assert!(csv.contains("0.8800"));
}

#[test]
fn test_csv_exporter_empty_document() {
    let doc = empty_document();
    let exporter = CsvTableExporter::new();
    let csv = exporter.export(&doc).expect("CSV export");
    let lines: Vec<&str> = csv.lines().collect();
    // Only header row.
    assert_eq!(lines.len(), 1);
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_formula_without_latex_json() {
    let regions = vec![DocumentRegion::Formula {
        latex: None,
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.8,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed["pages"][0]["regions"][0]["latex"].is_null());
}

#[test]
fn test_figure_without_caption_json() {
    let regions = vec![DocumentRegion::Figure {
        caption: None,
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.8,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = JsonExporter::new();
    let json_str = exporter.export(&doc).expect("JSON export");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed["pages"][0]["regions"][0]["caption"].is_null());
}

#[test]
fn test_figure_without_caption_markdown() {
    let regions = vec![DocumentRegion::Figure {
        caption: None,
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.8,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("![Figure]()"));
}

#[test]
fn test_formula_without_latex_markdown() {
    let regions = vec![DocumentRegion::Formula {
        latex: None,
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.8,
    }];
    let page = PageOutput {
        regions,
        reading_order: vec![0],
        width: 100,
        height: 100,
    };
    let doc = DocumentOutput { pages: vec![page] };

    let exporter = MarkdownExporter::new();
    let md = exporter.export(&doc).expect("Markdown export");
    assert!(md.contains("[formula]"));
}
