// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the agentic search harness.

use super::*;

// ---------------------------------------------------------------------------
// AgentConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_agent_config_defaults() {
    let cfg = AgentConfig::default();
    assert_eq!(cfg.token_budget, 32_768);
    assert_eq!(cfg.soft_threshold, 24_576);
    assert_eq!(cfg.max_turns, 128);
    assert_eq!(cfg.search_top_k, 10);
    assert_eq!(cfg.grep_max_matches, 5);
}

#[test]
fn test_agent_config_builder() {
    let cfg = AgentConfig::new()
        .with_token_budget(16_384)
        .with_soft_threshold(12_288)
        .with_max_turns(64);
    assert_eq!(cfg.token_budget, 16_384);
    assert_eq!(cfg.soft_threshold, 12_288);
    assert_eq!(cfg.max_turns, 64);
    // Unchanged defaults.
    assert_eq!(cfg.search_top_k, 10);
    assert_eq!(cfg.grep_max_matches, 5);
}

// ---------------------------------------------------------------------------
// ContextManager tests
// ---------------------------------------------------------------------------

#[test]
fn test_context_manager_add_chunk() {
    let cfg = AgentConfig::new().with_token_budget(100);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "hello world".into(), 10)
        .expect("should add");
    assert_eq!(cm.token_count(), 10);
    assert_eq!(cm.chunk_count(), 1);
    assert!(cm.has_seen("c1"));
    assert!(!cm.has_seen("c2"));
}

#[test]
fn test_context_manager_deduplication() {
    let cfg = AgentConfig::new().with_token_budget(100);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "first".into(), 10).unwrap();
    // Adding the same ID again should be a no-op.
    cm.add_chunk("c1".into(), "second".into(), 20).unwrap();
    assert_eq!(cm.token_count(), 10);
    assert_eq!(cm.chunk_count(), 1);
}

#[test]
fn test_context_manager_budget_enforcement() {
    let cfg = AgentConfig::new().with_token_budget(50);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "a".into(), 30).unwrap();
    // This should exceed the budget.
    let result = cm.add_chunk("c2".into(), "b".into(), 25);
    assert!(result.is_err());
    // Original state preserved.
    assert_eq!(cm.token_count(), 30);
    assert_eq!(cm.chunk_count(), 1);
}

#[test]
fn test_context_manager_prune() {
    let cfg = AgentConfig::new().with_token_budget(200);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "aaa".into(), 10).unwrap();
    cm.add_chunk("c2".into(), "bbb".into(), 20).unwrap();
    cm.add_chunk("c3".into(), "ccc".into(), 30).unwrap();
    assert_eq!(cm.token_count(), 60);
    assert_eq!(cm.chunk_count(), 3);

    cm.prune(&["c2".into()]);
    assert_eq!(cm.token_count(), 40);
    assert_eq!(cm.chunk_count(), 2);

    // c2 is still "seen" — re-adding should be a no-op.
    assert!(cm.has_seen("c2"));
    cm.add_chunk("c2".into(), "bbb again".into(), 20).unwrap();
    assert_eq!(cm.token_count(), 40); // unchanged
}

#[test]
fn test_context_manager_prune_unknown_id() {
    let cfg = AgentConfig::new().with_token_budget(100);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "data".into(), 10).unwrap();
    // Pruning an unknown ID should be a no-op.
    cm.prune(&["nonexistent".into()]);
    assert_eq!(cm.token_count(), 10);
    assert_eq!(cm.chunk_count(), 1);
}

#[test]
fn test_context_manager_soft_threshold() {
    let cfg = AgentConfig::new()
        .with_token_budget(100)
        .with_soft_threshold(50);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "x".into(), 40).unwrap();
    assert!(!cm.is_over_soft_threshold());
    cm.add_chunk("c2".into(), "y".into(), 15).unwrap();
    assert!(cm.is_over_soft_threshold());
    assert!(!cm.is_over_budget());
}

#[test]
fn test_context_manager_build_context() {
    let cfg = AgentConfig::new().with_token_budget(200);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("doc1".into(), "Hello".into(), 5).unwrap();
    cm.add_chunk("doc2".into(), "World".into(), 5).unwrap();
    let ctx = cm.build_context();
    assert!(ctx.contains("[chunk:doc1]"));
    assert!(ctx.contains("Hello"));
    assert!(ctx.contains("[chunk:doc2]"));
    assert!(ctx.contains("World"));
}

#[test]
fn test_context_manager_chunks_iterator() {
    let cfg = AgentConfig::new().with_token_budget(200);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("a".into(), "alpha".into(), 5).unwrap();
    cm.add_chunk("b".into(), "beta".into(), 4).unwrap();
    let pairs: Vec<_> = cm.chunks().collect();
    assert_eq!(pairs, vec![("a", "alpha"), ("b", "beta")]);
}

// ---------------------------------------------------------------------------
// execute_tool tests (with mock backend)
// ---------------------------------------------------------------------------

struct MockBackend;

impl SearchBackend for MockBackend {
    fn search(&self, query: &str, _top_k: usize) -> Result<Vec<SearchResult>> {
        Ok(vec![SearchResult {
            doc_id: "d1".into(),
            title: format!("Result for '{query}'"),
            snippet: "snippet text".into(),
            score: 0.95,
        }])
    }

    fn grep(&self, pattern: &str, _max_matches: usize) -> Result<Vec<GrepResult>> {
        Ok(vec![GrepResult {
            doc_id: "d1".into(),
            line: format!("match: {pattern}"),
            line_number: 42,
        }])
    }

    fn read_document(&self, doc_id: &str) -> Result<Document> {
        Ok(Document {
            doc_id: doc_id.to_string(),
            title: "Test Doc".into(),
            content: "This is the document content.".into(),
        })
    }
}

#[test]
fn test_execute_search_tool() {
    let backend = MockBackend;
    let cfg = AgentConfig::default();
    let mut ctx = ContextManager::new(cfg.clone());
    let tool = SearchTool::SearchCorpus {
        query: "test query".into(),
    };
    let result = execute_tool(&tool, &backend, &mut ctx, &cfg);
    match result {
        ToolResult::Search(results) => {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "d1");
        }
        other => panic!("Expected Search result, got {other:?}"),
    }
}

#[test]
fn test_execute_read_adds_to_context() {
    let backend = MockBackend;
    let cfg = AgentConfig::default();
    let mut ctx = ContextManager::new(cfg.clone());
    let tool = SearchTool::ReadDocument {
        doc_id: "d1".into(),
    };
    let result = execute_tool(&tool, &backend, &mut ctx, &cfg);
    assert!(matches!(result, ToolResult::Read(_)));
    assert!(ctx.has_seen("d1"));
    assert!(ctx.token_count() > 0);
}

#[test]
fn test_execute_prune_tool() {
    let backend = MockBackend;
    let cfg = AgentConfig::default();
    let mut ctx = ContextManager::new(cfg.clone());

    // First read a document so there's something to prune.
    let read_tool = SearchTool::ReadDocument {
        doc_id: "d1".into(),
    };
    execute_tool(&read_tool, &backend, &mut ctx, &cfg);
    assert_eq!(ctx.chunk_count(), 1);

    let prune_tool = SearchTool::PruneChunks {
        chunk_ids: vec!["d1".into()],
    };
    let result = execute_tool(&prune_tool, &backend, &mut ctx, &cfg);
    match result {
        ToolResult::Pruned { removed } => assert_eq!(removed, 1),
        other => panic!("Expected Pruned, got {other:?}"),
    }
    assert_eq!(ctx.chunk_count(), 0);
}

// ---------------------------------------------------------------------------
// Prompt formatting tests
// ---------------------------------------------------------------------------

#[test]
fn test_search_tool_definitions_contains_all_tools() {
    let defs = search_tool_definitions();
    assert!(defs.contains("search_corpus"));
    assert!(defs.contains("grep_corpus"));
    assert!(defs.contains("read_document"));
    assert!(defs.contains("prune_chunks"));
    assert!(defs.contains("namespace functions"));
}

#[test]
fn test_format_tool_call_prompt() {
    let prompt = format_tool_call_prompt("search_corpus", r#"{"query":"rust ml"}"#);
    assert!(prompt.contains("to=functions.search_corpus"));
    assert!(prompt.contains("<|channel|>commentary json"));
    assert!(prompt.contains(r#"{"query":"rust ml"}"#));
    assert!(prompt.contains("<|call|>"));
}

#[test]
fn test_format_tool_result_prompt() {
    let prompt = format_tool_result_prompt("search_corpus", "Found 3 results.");
    assert!(prompt.contains("functions.search_corpus to=assistant"));
    assert!(prompt.contains("<|channel|>commentary"));
    assert!(prompt.contains("Found 3 results."));
    assert!(prompt.contains("<|end|>"));
}

// ---------------------------------------------------------------------------
// Agent loop integration test (with mock model)
// ---------------------------------------------------------------------------

#[test]
fn test_agent_loop_final_answer_first_turn() {
    let backend = MockBackend;
    let call_count = std::cell::Cell::new(0);
    let model_fn = |_prompt: &str| -> Result<String> {
        call_count.set(call_count.get() + 1);
        Ok(r#"<|start|>assistant<|channel|>final<|message|>
<Document id="d1"><Justification>Relevant to query</Justification></Document>
<|return|>"#
            .to_string())
    };
    let cfg = AgentConfig::new().with_max_turns(10);
    let output = run_agent_loop("test", &backend, model_fn, cfg).unwrap();
    assert!(output.completed);
    assert_eq!(output.turns, 1);
    assert_eq!(output.documents.len(), 1);
    assert_eq!(output.documents[0].doc_id, "d1");
}

#[test]
fn test_agent_loop_tool_then_answer() {
    let backend = MockBackend;
    let call_count = std::cell::Cell::new(0);
    let model_fn = |_prompt: &str| -> Result<String> {
        let n = call_count.get();
        call_count.set(n + 1);
        if n == 0 {
            // First turn: search.
            Ok(r#"<|start|>assistant to=functions.search_corpus<|channel|>commentary json<|message|>{"query":"test"}<|call|>"#.to_string())
        } else {
            // Second turn: final answer.
            Ok(
                r#"<Document id="d1"><Justification>Found via search</Justification></Document>"#
                    .to_string(),
            )
        }
    };
    let cfg = AgentConfig::new().with_max_turns(10);
    let output = run_agent_loop("test", &backend, model_fn, cfg).unwrap();
    assert!(output.completed);
    assert_eq!(output.turns, 2);
    assert_eq!(output.documents.len(), 1);
}

#[test]
fn test_agent_loop_max_turns() {
    let backend = MockBackend;
    let model_fn = |_prompt: &str| -> Result<String> {
        // Always emit a tool call, never a final answer.
        Ok(r#"<|start|>assistant to=functions.search_corpus<|channel|>commentary json<|message|>{"query":"loop"}<|call|>"#.to_string())
    };
    let cfg = AgentConfig::new().with_max_turns(3);
    let output = run_agent_loop("test", &backend, model_fn, cfg).unwrap();
    assert!(!output.completed);
    assert_eq!(output.turns, 3);
    assert!(output.documents.is_empty());
}
