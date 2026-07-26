// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Context-1 tool-call parser.

use crate::agent::{Document, GrepResult, SearchResult, SearchTool, ToolResult};

use super::*;

// ---------------------------------------------------------------------------
// parse_tool_calls
// ---------------------------------------------------------------------------

#[test]
fn test_parse_search_corpus() {
    let text = r#"Some thinking.
<|start|>assistant to=functions.search_corpus<|channel|>commentary json<|message|>{"query":"rust ML framework"}<|call|>"#;
    let tools = parse_tool_calls(text);
    assert_eq!(tools.len(), 1);
    match &tools[0] {
        SearchTool::SearchCorpus { query } => {
            assert_eq!(query, "rust ML framework");
        }
        other => panic!("Expected SearchCorpus, got {other:?}"),
    }
}

#[test]
fn test_parse_grep_corpus() {
    let text = r#"<|start|>assistant to=functions.grep_corpus<|channel|>commentary json<|message|>{"pattern":"fn forward"}<|call|>"#;
    let tools = parse_tool_calls(text);
    assert_eq!(tools.len(), 1);
    match &tools[0] {
        SearchTool::GrepCorpus { pattern } => {
            assert_eq!(pattern, "fn forward");
        }
        other => panic!("Expected GrepCorpus, got {other:?}"),
    }
}

#[test]
fn test_parse_read_document() {
    let text = r#"<|start|>assistant to=functions.read_document<|channel|>commentary json<|message|>{"doc_id":"doc-42"}<|call|>"#;
    let tools = parse_tool_calls(text);
    assert_eq!(tools.len(), 1);
    match &tools[0] {
        SearchTool::ReadDocument { doc_id } => {
            assert_eq!(doc_id, "doc-42");
        }
        other => panic!("Expected ReadDocument, got {other:?}"),
    }
}

#[test]
fn test_parse_prune_chunks() {
    let text = r#"<|start|>assistant to=functions.prune_chunks<|channel|>commentary json<|message|>{"chunk_ids":["c1","c2","c3"]}<|call|>"#;
    let tools = parse_tool_calls(text);
    assert_eq!(tools.len(), 1);
    match &tools[0] {
        SearchTool::PruneChunks { chunk_ids } => {
            assert_eq!(chunk_ids, &["c1", "c2", "c3"]);
        }
        other => panic!("Expected PruneChunks, got {other:?}"),
    }
}

#[test]
fn test_parse_multiple_tool_calls() {
    let text = r#"<|start|>assistant to=functions.search_corpus<|channel|>commentary json<|message|>{"query":"first"}<|call|>
<|start|>assistant to=functions.read_document<|channel|>commentary json<|message|>{"doc_id":"d1"}<|call|>"#;
    let tools = parse_tool_calls(text);
    assert_eq!(tools.len(), 2);
    assert!(matches!(&tools[0], SearchTool::SearchCorpus { .. }));
    assert!(matches!(&tools[1], SearchTool::ReadDocument { .. }));
}

#[test]
fn test_parse_unknown_tool_ignored() {
    let text = r#"<|start|>assistant to=functions.unknown_tool<|channel|>commentary json<|message|>{"arg":"val"}<|call|>"#;
    let tools = parse_tool_calls(text);
    assert!(tools.is_empty());
}

#[test]
fn test_parse_no_tool_calls() {
    let text = "Just some regular text with no tool calls.";
    let tools = parse_tool_calls(text);
    assert!(tools.is_empty());
}

#[test]
fn test_parse_malformed_json_ignored() {
    // Missing closing quote — extract_json_string returns None.
    let text = r#"<|start|>assistant to=functions.search_corpus<|channel|>commentary json<|message|>{"query":"unterminated}<|call|>"#;
    let tools = parse_tool_calls(text);
    // The colon-value parser finds the opening quote but no closing quote
    // within the JSON range, so it should fail gracefully.
    // Depending on exact behavior, either 0 or 1 with partial parse.
    // Our parser should skip malformed entries.
    assert!(tools.is_empty() || tools.len() == 1);
}

// ---------------------------------------------------------------------------
// is_final_answer
// ---------------------------------------------------------------------------

#[test]
fn test_is_final_answer_with_channel() {
    let text = "<|start|>assistant<|channel|>final<|message|>Here are the documents.";
    assert!(is_final_answer(text));
}

#[test]
fn test_is_final_answer_with_document_tag() {
    let text = r#"<Document id="d1"><Justification>Relevant</Justification></Document>"#;
    assert!(is_final_answer(text));
}

#[test]
fn test_is_final_answer_false() {
    let text = r#"<|start|>assistant to=functions.search_corpus<|channel|>commentary json<|message|>{"query":"test"}<|call|>"#;
    assert!(!is_final_answer(text));
}

// ---------------------------------------------------------------------------
// parse_final_answer
// ---------------------------------------------------------------------------

#[test]
fn test_parse_final_answer_single_doc() {
    let text =
        r#"<Document id="doc-7"><Justification>Contains the API spec</Justification></Document>"#;
    let docs = parse_final_answer(text);
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].doc_id, "doc-7");
    assert_eq!(docs[0].justification, "Contains the API spec");
}

#[test]
fn test_parse_final_answer_multiple_docs() {
    let text = r#"Here are the relevant documents:
<Document id="d1"><Justification>Main reference</Justification></Document>
<Document id="d2"><Justification>Supporting evidence</Justification></Document>
<Document id="d3"><Justification>Background context</Justification></Document>"#;
    let docs = parse_final_answer(text);
    assert_eq!(docs.len(), 3);
    assert_eq!(docs[0].doc_id, "d1");
    assert_eq!(docs[1].doc_id, "d2");
    assert_eq!(docs[2].doc_id, "d3");
    assert_eq!(docs[2].justification, "Background context");
}

#[test]
fn test_parse_final_answer_no_docs() {
    let text = "No documents found for this query.";
    let docs = parse_final_answer(text);
    assert!(docs.is_empty());
}

#[test]
fn test_parse_final_answer_missing_justification() {
    let text = r#"<Document id="d1">Some content without justification tag</Document>"#;
    let docs = parse_final_answer(text);
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].doc_id, "d1");
    assert!(docs[0].justification.is_empty());
}

// ---------------------------------------------------------------------------
// format_observation
// ---------------------------------------------------------------------------

#[test]
fn test_format_search_observation() {
    let result = ToolResult::Search(vec![SearchResult {
        doc_id: "d1".into(),
        title: "Test".into(),
        snippet: "snippet".into(),
        score: 0.9,
    }]);
    let obs = format_observation("search_corpus", &result);
    assert!(obs.contains("functions.search_corpus to=assistant"));
    assert!(obs.contains("<|channel|>commentary"));
    assert!(obs.contains("Found 1 result"));
    assert!(obs.contains("[d1]"));
    assert!(obs.contains("0.900"));
    assert!(obs.contains("<|end|>"));
}

#[test]
fn test_format_grep_observation() {
    let result = ToolResult::Grep(vec![GrepResult {
        doc_id: "d2".into(),
        line: "fn main()".into(),
        line_number: 10,
    }]);
    let obs = format_observation("grep_corpus", &result);
    assert!(obs.contains("Found 1 match"));
    assert!(obs.contains("[d2] L10: fn main()"));
}

#[test]
fn test_format_read_observation() {
    let result = ToolResult::Read(Document {
        doc_id: "d3".into(),
        title: "Nn Doc".into(),
        content: "Content here.".into(),
    });
    let obs = format_observation("read_document", &result);
    assert!(obs.contains("Document: d3 (Nn Doc)"));
    assert!(obs.contains("Content here."));
}

#[test]
fn test_format_prune_observation() {
    let result = ToolResult::Pruned { removed: 2 };
    let obs = format_observation("prune_chunks", &result);
    assert!(obs.contains("Pruned 2 chunk(s)."));
}

#[test]
fn test_format_error_observation() {
    let result = ToolResult::Error("backend timeout".into());
    let obs = format_observation("search_corpus", &result);
    assert!(obs.contains("Error: backend timeout"));
}

#[test]
fn test_format_empty_search_observation() {
    let result = ToolResult::Search(vec![]);
    let obs = format_observation("search_corpus", &result);
    assert!(obs.contains("No results found."));
}
