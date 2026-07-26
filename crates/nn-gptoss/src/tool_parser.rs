// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tool-call parser for Context-1 agentic search.
//!
//! Context-1 uses a structured format defined by its chat template:
//!
//! - **Tool call:** `<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{"arg":"val"}<|call|>`
//! - **Tool result:** `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>RESULT<|end|>`
//! - **Final answer:** `<|start|>assistant<|channel|>final<|message|>...<Document id="X">...<|return|>`
//!
//! This module parses model output text into [`SearchTool`] variants and
//! formats [`ToolResult`]s back into the prompt format the model expects.

use crate::agent::{GrepResult, RetrievedDocument, SearchResult, SearchTool, ToolResult};

#[cfg(test)]
#[path = "tool_parser_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Tool call parsing
// ---------------------------------------------------------------------------

/// Parse tool calls from model output text.
///
/// Scans for `to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>`
/// patterns and returns the corresponding [`SearchTool`] variants.
pub(crate) fn parse_tool_calls(text: &str) -> Vec<SearchTool> {
    let mut tools = Vec::new();
    let mut remaining = text;

    while let Some(start) = remaining.find("to=functions.") {
        remaining = &remaining[start + "to=functions.".len()..];

        // Extract tool name (up to the next `<`).
        let name_end = match remaining.find('<') {
            Some(i) => i,
            None => continue,
        };
        let tool_name = remaining[..name_end].trim();
        remaining = &remaining[name_end..];

        // Find the JSON payload between `<|message|>` and `<|call|>`.
        let msg_marker = "<|message|>";
        let call_marker = "<|call|>";
        let msg_start = match remaining.find(msg_marker) {
            Some(i) => i + msg_marker.len(),
            None => continue,
        };
        let call_end = match remaining[msg_start..].find(call_marker) {
            Some(i) => msg_start + i,
            None => continue,
        };
        let json_str = remaining[msg_start..call_end].trim();
        remaining = &remaining[call_end + call_marker.len()..];

        if let Some(tool) = parse_single_tool(tool_name, json_str) {
            tools.push(tool);
        }
    }

    tools
}

/// Parse a single tool call from its name and JSON arguments.
fn parse_single_tool(name: &str, json_str: &str) -> Option<SearchTool> {
    match name {
        "search_corpus" => {
            let query = extract_json_string(json_str, "query")?;
            Some(SearchTool::SearchCorpus { query })
        }
        "grep_corpus" => {
            let pattern = extract_json_string(json_str, "pattern")?;
            Some(SearchTool::GrepCorpus { pattern })
        }
        "read_document" => {
            let doc_id = extract_json_string(json_str, "doc_id")?;
            Some(SearchTool::ReadDocument { doc_id })
        }
        "prune_chunks" => {
            let ids = extract_json_string_array(json_str, "chunk_ids")?;
            Some(SearchTool::PruneChunks { chunk_ids: ids })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Final answer parsing
// ---------------------------------------------------------------------------

/// Check if the model output contains a final answer.
///
/// A final answer is signalled by the `<|channel|>final` marker or by
/// `<Document` tags in the output.
pub(crate) fn is_final_answer(text: &str) -> bool {
    text.contains("<|channel|>final") || text.contains("<Document ")
}

/// Parse final answer documents from model output.
///
/// Extracts `<Document id="..."><Justification>...</Justification></Document>`
/// blocks.
pub(crate) fn parse_final_answer(text: &str) -> Vec<RetrievedDocument> {
    let mut docs = Vec::new();
    let mut remaining = text;

    while let Some(tag_start) = remaining.find("<Document ") {
        remaining = &remaining[tag_start..];

        // Extract id attribute.
        let id = match extract_xml_attr(remaining, "id") {
            Some(id) => id,
            None => {
                remaining = &remaining["<Document ".len()..];
                continue;
            }
        };

        // Find closing tag.
        let close_tag = "</Document>";
        let close_pos = match remaining.find(close_tag) {
            Some(i) => i,
            None => {
                remaining = &remaining["<Document ".len()..];
                continue;
            }
        };

        let inner = &remaining[..close_pos];
        remaining = &remaining[close_pos + close_tag.len()..];

        // Extract justification.
        let justification = extract_xml_content(inner, "Justification").unwrap_or_default();

        docs.push(RetrievedDocument {
            doc_id: id,
            justification,
        });
    }

    docs
}

// ---------------------------------------------------------------------------
// Observation formatting
// ---------------------------------------------------------------------------

/// Format a tool result as an observation prompt for the model.
///
/// Returns the `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>...<|end|>`
/// block.
pub(crate) fn format_observation(tool_name: &str, result: &ToolResult) -> String {
    let body = match result {
        ToolResult::Search(results) => format_search_results(results),
        ToolResult::Grep(results) => format_grep_results(results),
        ToolResult::Read(doc) => format_read_result(doc),
        ToolResult::Pruned { removed } => format!("Pruned {removed} chunk(s)."),
        ToolResult::Error(msg) => format!("Error: {msg}"),
    };
    format!(
        "<|start|>functions.{tool_name} to=assistant\
         <|channel|>commentary<|message|>{body}<|end|>"
    )
}

/// Format search results as text.
fn format_search_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }
    let mut out = format!("Found {} result(s):\n", results.len());
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. [{}] {} (score: {:.3})\n   {}\n",
            i + 1,
            r.doc_id,
            r.title,
            r.score,
            r.snippet,
        ));
    }
    out
}

/// Format grep results as text.
fn format_grep_results(results: &[GrepResult]) -> String {
    if results.is_empty() {
        return "No matches found.".to_string();
    }
    let mut out = format!("Found {} match(es):\n", results.len());
    for r in results {
        out.push_str(&format!(
            "  [{}] L{}: {}\n",
            r.doc_id, r.line_number, r.line,
        ));
    }
    out
}

/// Format a read result as text.
fn format_read_result(doc: &crate::agent::Document) -> String {
    format!(
        "Document: {} ({})\n---\n{}",
        doc.doc_id, doc.title, doc.content,
    )
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no serde dependency)
// ---------------------------------------------------------------------------

/// Extract a string value from a simple JSON object by key.
///
/// This is a lightweight parser — it handles `{"key": "value"}` patterns
/// without pulling in serde_json as a runtime dependency. It does NOT
/// handle escaped quotes inside values.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    // Look for `"key"` followed by `:` and a quoted value.
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];

    // Skip whitespace and colon.
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    // Read quoted string.
    if !after_colon.starts_with('"') {
        return None;
    }
    let value_start = 1; // skip opening quote
    let value_end = after_colon[value_start..].find('"')?;
    Some(after_colon[value_start..value_start + value_end].to_string())
}

/// Extract a string array from a simple JSON object by key.
///
/// Handles `{"key": ["a", "b"]}` patterns.
fn extract_json_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];

    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    if !after_colon.starts_with('[') {
        return None;
    }
    let bracket_end = after_colon.find(']')?;
    let array_content = &after_colon[1..bracket_end];

    let mut items = Vec::new();
    let mut remaining = array_content;
    while let Some(q_start) = remaining.find('"') {
        remaining = &remaining[q_start + 1..];
        let q_end = remaining.find('"')?;
        items.push(remaining[..q_end].to_string());
        remaining = &remaining[q_end + 1..];
    }
    Some(items)
}

/// Extract an XML attribute value: `<Tag attr="value"` -> `value`.
fn extract_xml_attr(tag_text: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = tag_text.find(&pattern)? + pattern.len();
    let end = tag_text[start..].find('"')?;
    Some(tag_text[start..start + end].to_string())
}

/// Extract inner content of an XML element: `<Tag>content</Tag>` -> `content`.
fn extract_xml_content(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text[start..].find(&close)?;
    Some(text[start..start + end].trim().to_string())
}
