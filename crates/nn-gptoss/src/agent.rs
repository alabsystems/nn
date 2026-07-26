// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Agentic search harness for Context-1.
//!
//! Implements the observe-reason-act loop for iterative document retrieval.
//! The model generates tool calls (search, grep, read, prune) and the harness
//! executes them against a [`SearchBackend`], managing a fixed token budget
//! via [`ContextManager`].
//!
//! The protocol follows the Context-1 chat template: tool calls use
//! `<|start|>assistant to=functions.NAME ...` and results return via
//! `<|start|>functions.NAME to=assistant ...`. The agent loop runs until
//! the model emits a final answer on the `final` channel or `max_turns`
//! is reached.

use std::collections::HashSet;

use nn_core::Result;

use crate::tool_parser;
use crate::GptOssError;

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Tool types
// ---------------------------------------------------------------------------

/// Tool calls the model can emit.
#[derive(Debug, Clone)]
pub enum SearchTool {
    /// Full-text search over the corpus.
    SearchCorpus { query: String },
    /// Regex/literal grep across documents.
    GrepCorpus { pattern: String },
    /// Fetch the full content of a document by ID.
    ReadDocument { doc_id: String },
    /// Remove chunks from the active context window.
    PruneChunks { chunk_ids: Vec<String> },
}

/// A single search hit.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    pub title: String,
    pub snippet: String,
    pub score: f32,
}

/// A single grep match.
#[derive(Debug, Clone)]
pub struct GrepResult {
    pub doc_id: String,
    pub line: String,
    pub line_number: usize,
}

/// A full document returned by read.
#[derive(Debug, Clone)]
pub struct Document {
    pub doc_id: String,
    pub title: String,
    pub content: String,
}

/// A retrieved document with model-generated justification.
#[derive(Debug, Clone)]
pub struct RetrievedDocument {
    pub doc_id: String,
    pub justification: String,
}

// ---------------------------------------------------------------------------
// Tool result wrapper (internal)
// ---------------------------------------------------------------------------

/// Result of executing a single tool call.
#[derive(Debug, Clone)]
pub(crate) enum ToolResult {
    Search(Vec<SearchResult>),
    Grep(Vec<GrepResult>),
    Read(Document),
    Pruned { removed: usize },
    Error(String),
}

// ---------------------------------------------------------------------------
// SearchBackend trait
// ---------------------------------------------------------------------------

/// Backend for executing search operations against a document corpus.
///
/// Implementors provide the retrieval index. The agent harness calls
/// these methods in response to model-generated tool calls.
pub trait SearchBackend: Send + Sync {
    /// Full-text search, returning up to `top_k` results ordered by relevance.
    fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>>;

    /// Grep for `pattern` across the corpus, returning up to `max_matches`.
    fn grep(&self, pattern: &str, max_matches: usize) -> Result<Vec<GrepResult>>;

    /// Read the full content of a document by its ID.
    fn read_document(&self, doc_id: &str) -> Result<Document>;
}

// ---------------------------------------------------------------------------
// AgentConfig
// ---------------------------------------------------------------------------

/// Configuration for the search agent loop.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AgentConfig {
    /// Hard token budget for the context window (default: 32768).
    pub token_budget: usize,
    /// Soft threshold at which the model is advised to prune (default: 24576).
    pub soft_threshold: usize,
    /// Maximum turns before forced termination (default: 128).
    pub max_turns: usize,
    /// Number of results per search call (default: 10).
    pub search_top_k: usize,
    /// Maximum grep matches per call (default: 5).
    pub grep_max_matches: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            token_budget: 32_768,
            soft_threshold: 24_576,
            max_turns: 128,
            search_top_k: 10,
            grep_max_matches: 5,
        }
    }
}

impl AgentConfig {
    /// Create with all defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set hard token budget.
    #[must_use]
    pub fn with_token_budget(mut self, budget: usize) -> Self {
        self.token_budget = budget;
        self
    }

    /// Builder: set soft threshold.
    #[must_use]
    pub fn with_soft_threshold(mut self, threshold: usize) -> Self {
        self.soft_threshold = threshold;
        self
    }

    /// Builder: set max turns.
    #[must_use]
    pub fn with_max_turns(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }
}

// ---------------------------------------------------------------------------
// AgentOutput
// ---------------------------------------------------------------------------

/// Output from an agent search session.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    /// Retrieved documents with justifications.
    pub documents: Vec<RetrievedDocument>,
    /// Number of turns taken.
    pub turns: usize,
    /// Whether the agent terminated naturally (vs. hitting `max_turns`).
    pub completed: bool,
}

// ---------------------------------------------------------------------------
// ContextManager
// ---------------------------------------------------------------------------

/// Token-budget-aware context manager.
///
/// Tracks active chunks, deduplicates by ID, and enforces the hard token
/// budget. When the soft threshold is exceeded, callers should prompt the
/// model to prune.
pub struct ContextManager {
    /// Active chunks: (chunk_id, content).
    chunks: Vec<(String, String)>,
    /// Token count per chunk, parallel to `chunks`.
    chunk_tokens: Vec<usize>,
    /// Set of all chunk IDs ever added (deduplication).
    seen_ids: HashSet<String>,
    /// Current total token count.
    token_count: usize,
    /// Configuration.
    config: AgentConfig,
}

impl ContextManager {
    /// Create a new context manager with the given configuration.
    #[must_use]
    pub fn new(config: AgentConfig) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_tokens: Vec::new(),
            seen_ids: HashSet::new(),
            token_count: 0,
            config,
        }
    }

    /// Add a chunk. Returns error if it would exceed the hard token budget.
    ///
    /// Duplicate `id`s are silently skipped (idempotent).
    pub fn add_chunk(&mut self, id: String, content: String, tokens: usize) -> Result<()> {
        if self.seen_ids.contains(&id) {
            return Ok(());
        }
        if self.token_count + tokens > self.config.token_budget {
            return Err(GptOssError::InvalidInput {
                reason: format!(
                    "adding chunk '{id}' ({tokens} tokens) would exceed budget \
                     ({} + {tokens} > {})",
                    self.token_count, self.config.token_budget
                ),
            }
            .into());
        }
        self.seen_ids.insert(id.clone());
        self.chunks.push((id, content));
        self.chunk_tokens.push(tokens);
        self.token_count += tokens;
        Ok(())
    }

    /// Remove chunks by ID. Unknown IDs are silently ignored.
    pub fn prune(&mut self, chunk_ids: &[String]) {
        let remove_set: HashSet<&String> = chunk_ids.iter().collect();
        let mut i = 0;
        while i < self.chunks.len() {
            if remove_set.contains(&self.chunks[i].0) {
                let (_id, _content) = self.chunks.remove(i);
                let tokens = self.chunk_tokens.remove(i);
                self.token_count = self.token_count.saturating_sub(tokens);
                // Note: we do NOT remove from seen_ids — the chunk stays
                // "seen" so re-adding it is a no-op.
            } else {
                i += 1;
            }
        }
    }

    /// Whether the current token count exceeds the soft threshold.
    #[must_use]
    pub fn is_over_soft_threshold(&self) -> bool {
        self.token_count > self.config.soft_threshold
    }

    /// Whether the current token count exceeds the hard budget.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.token_count > self.config.token_budget
    }

    /// Current total token count.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Whether a chunk ID has already been seen (added at least once).
    #[must_use]
    pub fn has_seen(&self, id: &str) -> bool {
        self.seen_ids.contains(id)
    }

    /// Number of active (non-pruned) chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Iterate over active chunks as `(id, content)` pairs.
    pub fn chunks(&self) -> impl Iterator<Item = (&str, &str)> {
        self.chunks.iter().map(|(id, c)| (id.as_str(), c.as_str()))
    }

    /// Build the aggregated context string for the model prompt.
    #[must_use]
    pub fn build_context(&self) -> String {
        let mut out = String::new();
        for (id, content) in &self.chunks {
            out.push_str(&format!("[chunk:{id}]\n{content}\n\n"));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Agent execution
// ---------------------------------------------------------------------------

/// Execute a tool call against the backend, returning a `ToolResult`.
pub(crate) fn execute_tool(
    tool: &SearchTool,
    backend: &dyn SearchBackend,
    context: &mut ContextManager,
    config: &AgentConfig,
) -> ToolResult {
    match tool {
        SearchTool::SearchCorpus { query } => match backend.search(query, config.search_top_k) {
            Ok(results) => ToolResult::Search(results),
            Err(e) => ToolResult::Error(e.to_string()),
        },
        SearchTool::GrepCorpus { pattern } => {
            match backend.grep(pattern, config.grep_max_matches) {
                Ok(results) => ToolResult::Grep(results),
                Err(e) => ToolResult::Error(e.to_string()),
            }
        }
        SearchTool::ReadDocument { doc_id } => match backend.read_document(doc_id) {
            Ok(doc) => {
                // Estimate tokens as chars/4 (rough BPE approximation).
                let est_tokens = doc.content.len() / 4;
                if let Err(e) =
                    context.add_chunk(doc.doc_id.clone(), doc.content.clone(), est_tokens)
                {
                    return ToolResult::Error(e.to_string());
                }
                ToolResult::Read(doc)
            }
            Err(e) => ToolResult::Error(e.to_string()),
        },
        SearchTool::PruneChunks { chunk_ids } => {
            let before = context.chunk_count();
            context.prune(chunk_ids);
            let after = context.chunk_count();
            ToolResult::Pruned {
                removed: before - after,
            }
        }
    }
}

/// Format tool definitions for the Context-1 developer prompt.
///
/// Returns the TypeScript-style namespace block that the chat template
/// expects in the `tools` section.
#[must_use]
pub fn search_tool_definitions() -> String {
    r#"## functions

namespace functions {

// Search the document corpus for relevant results.
type search_corpus = (_: {
// The search query
query: string,
}) => any;

// Grep the document corpus for a pattern.
type grep_corpus = (_: {
// The grep pattern (literal or regex)
pattern: string,
}) => any;

// Read the full content of a document by its ID.
type read_document = (_: {
// The document identifier
doc_id: string,
}) => any;

// Remove chunks from the active context to free token budget.
type prune_chunks = (_: {
// List of chunk IDs to remove
chunk_ids: string[],
}) => any;

} // namespace functions"#
        .to_string()
}

/// Build the system prompt prefix that instructs the model on the search task.
#[must_use]
pub fn build_search_system_prompt(query: &str, config: &AgentConfig) -> String {
    format!(
        concat!(
            "user query by iteratively searching, reading, and pruning documents.\n\n",
            "Token budget: {} (soft threshold: {}). When you approach the soft ",
            "threshold, prune less-relevant chunks.\n\n",
            "Available tools: search_corpus, grep_corpus, read_document, prune_chunks.\n\n",
            "When you have found all relevant documents, output your final answer ",
            "using <Document> tags:\n",
            "<Document id=\"DOC_ID\"><Justification>Why this document is relevant</Justification></Document>\n\n",
            "User query: {query}",
        ),
        config.token_budget,
        config.soft_threshold,
        query = query,
    )
}

/// Format the Context-1 tool call message.
///
/// Produces the `<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>`
/// block that Context-1 uses for function invocation.
#[must_use]
pub fn format_tool_call_prompt(tool_name: &str, args_json: &str) -> String {
    format!(
        concat!(
            "<|start|>assistant to=functions.{tool_name}",
            "<|channel|>commentary json<|message|>{args_json}<|call|>",
        ),
        tool_name = tool_name,
        args_json = args_json,
    )
}

/// Format the Context-1 tool result message.
///
/// Produces the `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>RESULT<|end|>`
/// block.
#[must_use]
pub fn format_tool_result_prompt(tool_name: &str, result: &str) -> String {
    format!(
        concat!(
            "<|start|>functions.{tool_name} to=assistant",
            "<|channel|>commentary<|message|>{result}<|end|>",
        ),
        tool_name = tool_name,
        result = result,
    )
}

/// Run the full agent loop.
///
/// Drives the observe-reason-act cycle: the `model_fn` generates text
/// given the current prompt, the harness parses tool calls, executes them
/// against `backend`, and feeds results back. Terminates when the model
/// emits a final answer or `max_turns` is reached.
///
/// # Arguments
///
/// * `query` - The user search query.
/// * `backend` - Corpus search backend.
/// * `model_fn` - Closure that takes a prompt string and returns model output text.
/// * `config` - Agent configuration.
pub fn run_agent_loop<F>(
    query: &str,
    backend: &dyn SearchBackend,
    mut model_fn: F,
    config: AgentConfig,
) -> Result<AgentOutput>
where
    F: FnMut(&str) -> Result<String>,
{
    let mut context = ContextManager::new(config.clone());
    let mut conversation = build_search_system_prompt(query, &config);

    for turn in 0..config.max_turns {
        let output = model_fn(&conversation)?;

        // Check for final answer.
        if tool_parser::is_final_answer(&output) {
            let documents = tool_parser::parse_final_answer(&output);
            return Ok(AgentOutput {
                documents,
                turns: turn + 1,
                completed: true,
            });
        }

        // Parse and execute tool calls.
        let tools = tool_parser::parse_tool_calls(&output);
        if tools.is_empty() {
            // Model produced neither tool calls nor a final answer —
            // treat as implicit completion with no results.
            return Ok(AgentOutput {
                documents: Vec::new(),
                turns: turn + 1,
                completed: true,
            });
        }

        for tool in &tools {
            let tool_name = match tool {
                SearchTool::SearchCorpus { .. } => "search_corpus",
                SearchTool::GrepCorpus { .. } => "grep_corpus",
                SearchTool::ReadDocument { .. } => "read_document",
                SearchTool::PruneChunks { .. } => "prune_chunks",
            };
            let result = execute_tool(tool, backend, &mut context, &config);
            let observation = tool_parser::format_observation(tool_name, &result);
            conversation.push_str(&observation);
        }

        // Append soft-threshold advisory if needed.
        if context.is_over_soft_threshold() {
            conversation
                .push_str("\n[system: token budget is at soft threshold — consider pruning]\n");
        }
    }

    // Reached max_turns without a final answer.
    Ok(AgentOutput {
        documents: Vec::new(),
        turns: config.max_turns,
        completed: false,
    })
}
