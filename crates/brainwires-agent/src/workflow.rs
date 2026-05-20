//! Workflow Graph Builder — Declarative DAG-based workflow pipelines
//!
//! Provides a [`WorkflowBuilder`] API for defining multi-step workflows as
//! directed acyclic graphs (DAGs).  Workflows compile down to `TaskSpec`
//! vectors and execute via the existing `TaskOrchestrator`.
//!
//! # Example
//!
//! ```rust,ignore
//! use brainwires_agent::workflow::{WorkflowBuilder, WorkflowContext};
//!
//! let workflow = WorkflowBuilder::new("review-pipeline")
//!     .node("fetch", |ctx| Box::pin(async move {
//!         ctx.set("code", serde_json::json!("fn main() {}")).await;
//!         Ok(serde_json::json!({"status": "fetched"}))
//!     }))
//!     .node("lint", |ctx| Box::pin(async move {
//!         let code = ctx.get("code").await;
//!         Ok(serde_json::json!({"lint": "passed"}))
//!     }))
//!     .node("review", |ctx| Box::pin(async move {
//!         Ok(serde_json::json!({"review": "approved"}))
//!     }))
//!     .edge("fetch", "lint")
//!     .edge("fetch", "review")   // lint + review run in parallel
//!     .edge("lint", "summarize")
//!     .edge("review", "summarize")
//!     .node("summarize", |ctx| Box::pin(async move {
//!         Ok(serde_json::json!({"summary": "all good"}))
//!     }))
//!     .build()
//!     .unwrap();
//!
//! let results = workflow.run().await.unwrap();
//! ```

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use petgraph::algo::is_cyclic_directed;
use petgraph::graph::{DiGraph, NodeIndex};
use serde_json::Value;
use tokio::sync::RwLock;

// ── Shared workflow state ────────────────────────────────────────────────────

/// Shared state accessible to all workflow nodes during execution.
///
/// Nodes read and write values to this shared map to pass data between
/// pipeline stages.
#[derive(Clone)]
pub struct WorkflowContext {
    state: Arc<RwLock<HashMap<String, Value>>>,
    /// Per-node results, keyed by node name.
    results: Arc<RwLock<HashMap<String, Value>>>,
}

impl WorkflowContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set a value in the shared state.
    pub async fn set(&self, key: impl Into<String>, value: Value) {
        self.state.write().await.insert(key.into(), value);
    }

    /// Get a value from the shared state.
    pub async fn get(&self, key: &str) -> Option<Value> {
        self.state.read().await.get(key).cloned()
    }

    /// Remove a value from the shared state.
    pub async fn remove(&self, key: &str) -> Option<Value> {
        self.state.write().await.remove(key)
    }

    /// Get the result of a previously completed node.
    pub async fn node_result(&self, node_name: &str) -> Option<Value> {
        self.results.read().await.get(node_name).cloned()
    }

    /// Store a node's result (called internally by the executor).
    async fn store_result(&self, node_name: impl Into<String>, value: Value) {
        self.results.write().await.insert(node_name.into(), value);
    }

    /// Get all results as a map.
    pub async fn all_results(&self) -> HashMap<String, Value> {
        self.results.read().await.clone()
    }
}

impl Default for WorkflowContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── Node function type ───────────────────────────────────────────────────────

/// A boxed async function that a workflow node executes.
pub type NodeFn = Box<
    dyn Fn(WorkflowContext) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync,
>;

/// A conditional edge function that returns the name of the next node(s)
/// to activate based on the current node's result.
pub type ConditionalFn = Box<dyn Fn(&Value) -> Vec<String> + Send + Sync>;

// ── Internal node representation ─────────────────────────────────────────────

struct WorkflowNode {
    name: String,
    handler: NodeFn,
}

enum EdgeType {
    /// Always-active edge from `from` to `to`.
    Direct { from: String, to: String },
    /// Conditional edge: `evaluator` returns which downstream nodes to activate.
    Conditional {
        from: String,
        evaluator: ConditionalFn,
    },
}

// ── WorkflowBuilder ──────────────────────────────────────────────────────────

/// Builder for constructing workflow DAGs.
///
/// Add nodes with [`node`][Self::node], wire them with [`edge`][Self::edge]
/// or [`conditional`][Self::conditional], then call [`build`][Self::build].
pub struct WorkflowBuilder {
    name: String,
    nodes: Vec<WorkflowNode>,
    node_names: HashSet<String>,
    edges: Vec<EdgeType>,
}

impl WorkflowBuilder {
    /// Create a new workflow builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            node_names: HashSet::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the workflow.
    ///
    /// The handler receives a [`WorkflowContext`] and returns `Result<Value>`.
    /// Nodes with no incoming edges are considered entry points and run first.
    pub fn node<F, Fut>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(WorkflowContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let name = name.into();
        self.node_names.insert(name.clone());
        self.nodes.push(WorkflowNode {
            name,
            handler: Box::new(move |ctx| Box::pin(handler(ctx))),
        });
        self
    }

    /// Add a direct edge from one node to another.
    ///
    /// The `to` node will only execute after `from` completes successfully.
    /// Multiple edges into the same node create a join (all predecessors must
    /// complete).
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(EdgeType::Direct {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// Add a conditional edge from a node.
    ///
    /// After `from` completes, `evaluator` is called with the node's result
    /// value. It returns a list of downstream node names to activate. Nodes
    /// not in the returned list are skipped (treated as completed with a
    /// null result).
    pub fn conditional<F>(mut self, from: impl Into<String>, evaluator: F) -> Self
    where
        F: Fn(&Value) -> Vec<String> + Send + Sync + 'static,
    {
        self.edges.push(EdgeType::Conditional {
            from: from.into(),
            evaluator: Box::new(evaluator),
        });
        self
    }

    /// Validate and build the workflow.
    ///
    /// Returns an error if:
    /// - An edge references a node that does not exist
    /// - The graph contains a cycle
    /// - There are no nodes
    pub fn build(self) -> Result<Workflow> {
        if self.nodes.is_empty() {
            return Err(anyhow!("Workflow '{}' has no nodes", self.name));
        }

        // Build petgraph for validation
        let mut graph = DiGraph::<String, ()>::new();
        let mut name_to_idx: HashMap<String, NodeIndex> = HashMap::new();

        for node in &self.nodes {
            let idx = graph.add_node(node.name.clone());
            name_to_idx.insert(node.name.clone(), idx);
        }

        // Validate and collect edges
        let mut direct_edges: Vec<(String, String)> = Vec::new();
        let mut conditional_edges: Vec<(String, ConditionalFn)> = Vec::new();

        for edge in self.edges {
            match edge {
                EdgeType::Direct { from, to } => {
                    if !name_to_idx.contains_key(&from) {
                        return Err(anyhow!("Edge references unknown source node '{}'", from));
                    }
                    if !name_to_idx.contains_key(&to) {
                        return Err(anyhow!("Edge references unknown target node '{}'", to));
                    }
                    graph.add_edge(name_to_idx[&from], name_to_idx[&to], ());
                    direct_edges.push((from, to));
                }
                EdgeType::Conditional { from, evaluator } => {
                    if !name_to_idx.contains_key(&from) {
                        return Err(anyhow!(
                            "Conditional edge references unknown source node '{}'",
                            from
                        ));
                    }
                    conditional_edges.push((from, evaluator));
                }
            }
        }

        if is_cyclic_directed(&graph) {
            return Err(anyhow!("Workflow '{}' contains a cycle", self.name));
        }

        // Identify entry nodes (no incoming direct edges)
        let targets: HashSet<&str> = direct_edges.iter().map(|(_, t)| t.as_str()).collect();
        let entry_nodes: Vec<String> = self
            .nodes
            .iter()
            .map(|n| &n.name)
            .filter(|n| !targets.contains(n.as_str()))
            .cloned()
            .collect();

        if entry_nodes.is_empty() {
            return Err(anyhow!(
                "Workflow '{}' has no entry nodes (every node has an incoming edge)",
                self.name
            ));
        }

        // Build handler map
        let mut handlers: HashMap<String, NodeFn> = HashMap::new();
        for node in self.nodes {
            handlers.insert(node.name, node.handler);
        }

        Ok(Workflow {
            name: self.name,
            handlers: Arc::new(handlers),
            direct_edges,
            conditional_edges: Arc::new(conditional_edges),
            entry_nodes,
            all_nodes: self.node_names,
        })
    }
}

// ── Compiled Workflow ────────────────────────────────────────────────────────

/// A compiled workflow ready for execution.
///
/// Created by [`WorkflowBuilder::build`]. Execute with [`run`][Self::run].
///
/// Note: `Debug` is implemented manually because the handler functions
/// are not `Debug`.
pub struct Workflow {
    name: String,
    handlers: Arc<HashMap<String, NodeFn>>,
    direct_edges: Vec<(String, String)>,
    conditional_edges: Arc<Vec<(String, ConditionalFn)>>,
    entry_nodes: Vec<String>,
    all_nodes: HashSet<String>,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workflow")
            .field("name", &self.name)
            .field("entry_nodes", &self.entry_nodes)
            .field("all_nodes", &self.all_nodes)
            .field("direct_edges", &self.direct_edges)
            .field("handlers", &format!("<{} handlers>", self.handlers.len()))
            .finish()
    }
}

/// Result of a completed workflow execution.
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    /// Workflow name.
    pub name: String,
    /// Whether all executed nodes succeeded.
    pub success: bool,
    /// Per-node results (only includes nodes that actually ran).
    pub node_results: HashMap<String, Value>,
    /// Nodes that were skipped (via conditional edges).
    pub skipped_nodes: Vec<String>,
    /// Nodes that failed, with their error messages.
    pub failed_nodes: HashMap<String, String>,
}

impl Workflow {
    /// Execute the workflow.
    ///
    /// Entry nodes (no incoming edges) run first. Downstream nodes run as
    /// their dependencies complete. Nodes sharing the same set of completed
    /// dependencies run concurrently via [`tokio::spawn`].
    pub async fn run(&self) -> Result<WorkflowResult> {
        self.run_with_context(WorkflowContext::new()).await
    }

    /// Execute the workflow with a pre-populated context.
    pub async fn run_with_context(&self, ctx: WorkflowContext) -> Result<WorkflowResult> {
        let completed: Arc<RwLock<HashSet<String>>> = Arc::new(RwLock::new(HashSet::new()));
        let failed: Arc<RwLock<HashMap<String, String>>> = Arc::new(RwLock::new(HashMap::new()));
        let skipped: Arc<RwLock<HashSet<String>>> = Arc::new(RwLock::new(HashSet::new()));

        // Build dependency map: node -> set of predecessors
        let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
        for node in &self.all_nodes {
            deps.insert(node.clone(), HashSet::new());
        }
        for (from, to) in &self.direct_edges {
            deps.entry(to.clone()).or_default().insert(from.clone());
        }

        loop {
            // First, propagate failures: any pending node whose predecessor
            // failed gets skipped.
            {
                let done = completed.read().await;
                let fail = failed.read().await;
                let skip = skipped.read().await;
                let mut to_skip = Vec::new();
                for (name, predecessors) in &deps {
                    if done.contains(name) || fail.contains_key(name) || skip.contains(name) {
                        continue;
                    }
                    if predecessors.iter().any(|p| fail.contains_key(p)) {
                        to_skip.push(name.clone());
                    }
                }
                drop(done);
                drop(fail);
                drop(skip);
                if !to_skip.is_empty() {
                    let mut skip_guard = skipped.write().await;
                    for name in to_skip {
                        skip_guard.insert(name);
                    }
                }
            }

            let ready: Vec<String> = {
                let done = completed.read().await;
                let fail = failed.read().await;
                let skip = skipped.read().await;
                deps.iter()
                    .filter(|(name, predecessors)| {
                        !done.contains(*name)
                            && !fail.contains_key(*name)
                            && !skip.contains(*name)
                            && predecessors
                                .iter()
                                .all(|p| done.contains(p) || skip.contains(p))
                    })
                    .map(|(name, _)| name.clone())
                    .collect()
            };

            if ready.is_empty() {
                break;
            }

            // Spawn all ready nodes concurrently
            let mut handles = Vec::new();
            for name in ready {
                let ctx = ctx.clone();
                let handlers = Arc::clone(&self.handlers);
                let completed = Arc::clone(&completed);
                let failed = Arc::clone(&failed);
                let conditional_edges = Arc::clone(&self.conditional_edges);
                let node_name = name.clone();

                let handle = tokio::spawn(async move {
                    if let Some(handler) = handlers.get(&node_name) {
                        match handler(ctx.clone()).await {
                            Ok(result) => {
                                ctx.store_result(&node_name, result.clone()).await;

                                // Evaluate conditional edges from this node and store
                                // the activated set for the main loop to process.
                                for (from, evaluator) in conditional_edges.iter() {
                                    if from == &node_name {
                                        let activated = evaluator(&result);
                                        ctx.set(
                                            format!("__conditional_activated_{}", node_name),
                                            serde_json::json!(activated),
                                        )
                                        .await;
                                    }
                                }

                                completed.write().await.insert(node_name);
                            }
                            Err(e) => {
                                failed.write().await.insert(node_name, e.to_string());
                            }
                        }
                    } else {
                        failed
                            .write()
                            .await
                            .insert(node_name, "Handler not found".to_string());
                    }
                });
                handles.push(handle);
            }

            // Wait for all concurrent nodes to complete
            for handle in handles {
                let _ = handle.await;
            }

            // Process conditional edge results: skip non-activated downstream nodes
            {
                let ctx_state = ctx.state.read().await;
                let mut skip_guard = skipped.write().await;
                for (from, _) in self.conditional_edges.iter() {
                    let key = format!("__conditional_activated_{}", from);
                    if let Some(activated_val) = ctx_state.get(&key)
                        && let Some(activated) = activated_val.as_array()
                    {
                        let activated_set: HashSet<String> = activated
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                        // Find all direct-edge targets from this node
                        for (edge_from, edge_to) in &self.direct_edges {
                            if edge_from == from && !activated_set.contains(edge_to) {
                                skip_guard.insert(edge_to.clone());
                            }
                        }
                    }
                }
            }
        }

        let node_results = ctx.all_results().await;
        let failed_map = failed.read().await.clone();
        let skipped_vec: Vec<String> = skipped.read().await.iter().cloned().collect();
        let success = failed_map.is_empty();

        Ok(WorkflowResult {
            name: self.name.clone(),
            success,
            node_results,
            skipped_nodes: skipped_vec,
            failed_nodes: failed_map,
        })
    }

    /// Get the workflow name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the entry node names.
    pub fn entry_nodes(&self) -> &[String] {
        &self.entry_nodes
    }

    /// Get all node names.
    pub fn node_names(&self) -> &HashSet<String> {
        &self.all_nodes
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_linear_workflow() {
        let workflow = WorkflowBuilder::new("linear")
            .node("a", |ctx| {
                Box::pin(async move {
                    ctx.set("counter", serde_json::json!(1)).await;
                    Ok(serde_json::json!({"step": "a"}))
                })
            })
            .node("b", |ctx| {
                Box::pin(async move {
                    let val = ctx.get("counter").await.unwrap();
                    let n = val.as_i64().unwrap();
                    ctx.set("counter", serde_json::json!(n + 1)).await;
                    Ok(serde_json::json!({"step": "b"}))
                })
            })
            .edge("a", "b")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert!(result.success);
        assert_eq!(result.node_results.len(), 2);
        assert!(result.failed_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_workflow() {
        let workflow = WorkflowBuilder::new("parallel")
            .node("start", |_ctx| {
                Box::pin(async move { Ok(serde_json::json!("started")) })
            })
            .node("branch_a", |_ctx| {
                Box::pin(async move { Ok(serde_json::json!("a_done")) })
            })
            .node("branch_b", |_ctx| {
                Box::pin(async move { Ok(serde_json::json!("b_done")) })
            })
            .node("join", |ctx| {
                Box::pin(async move {
                    let a = ctx.node_result("branch_a").await;
                    let b = ctx.node_result("branch_b").await;
                    Ok(serde_json::json!({"a": a, "b": b}))
                })
            })
            .edge("start", "branch_a")
            .edge("start", "branch_b")
            .edge("branch_a", "join")
            .edge("branch_b", "join")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert!(result.success);
        assert_eq!(result.node_results.len(), 4);
    }

    #[tokio::test]
    async fn test_diamond_workflow() {
        let workflow = WorkflowBuilder::new("diamond")
            .node("a", |_| Box::pin(async { Ok(serde_json::json!(1)) }))
            .node("b", |_| Box::pin(async { Ok(serde_json::json!(2)) }))
            .node("c", |_| Box::pin(async { Ok(serde_json::json!(3)) }))
            .node("d", |ctx| {
                Box::pin(async move {
                    let b = ctx.node_result("b").await.unwrap();
                    let c = ctx.node_result("c").await.unwrap();
                    Ok(serde_json::json!(b.as_i64().unwrap() + c.as_i64().unwrap()))
                })
            })
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert!(result.success);
        assert_eq!(result.node_results["d"], serde_json::json!(5));
    }

    #[tokio::test]
    async fn test_conditional_workflow() {
        let workflow = WorkflowBuilder::new("conditional")
            .node("check", |_| {
                Box::pin(async { Ok(serde_json::json!({"route": "fast"})) })
            })
            .node("fast_path", |_| {
                Box::pin(async { Ok(serde_json::json!("fast_done")) })
            })
            .node("slow_path", |_| {
                Box::pin(async { Ok(serde_json::json!("slow_done")) })
            })
            .edge("check", "fast_path")
            .edge("check", "slow_path")
            .conditional("check", |result| {
                let route = result
                    .get("route")
                    .and_then(|v| v.as_str())
                    .unwrap_or("fast");
                if route == "fast" {
                    vec!["fast_path".to_string()]
                } else {
                    vec!["slow_path".to_string()]
                }
            })
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert!(result.success);
        assert!(result.node_results.contains_key("fast_path"));
        assert!(result.skipped_nodes.contains(&"slow_path".to_string()));
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let result = WorkflowBuilder::new("cyclic")
            .node("a", |_| Box::pin(async { Ok(serde_json::json!(1)) }))
            .node("b", |_| Box::pin(async { Ok(serde_json::json!(2)) }))
            .edge("a", "b")
            .edge("b", "a")
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cycle"));
    }

    #[tokio::test]
    async fn test_unknown_node_in_edge() {
        let result = WorkflowBuilder::new("bad")
            .node("a", |_| Box::pin(async { Ok(serde_json::json!(1)) }))
            .edge("a", "nonexistent")
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown target"));
    }

    #[tokio::test]
    async fn test_empty_workflow() {
        let result = WorkflowBuilder::new("empty").build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no nodes"));
    }

    #[tokio::test]
    async fn test_single_node_workflow() {
        let workflow = WorkflowBuilder::new("single")
            .node("only", |_| {
                Box::pin(async { Ok(serde_json::json!("done")) })
            })
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert!(result.success);
        assert_eq!(result.node_results.len(), 1);
    }

    #[tokio::test]
    async fn test_node_failure_skips_dependents() {
        let workflow = WorkflowBuilder::new("fail")
            .node("a", |_| Box::pin(async { Err(anyhow::anyhow!("boom")) }))
            .node("b", |_| {
                Box::pin(async { Ok(serde_json::json!("should not run")) })
            })
            .edge("a", "b")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert!(!result.success);
        assert!(result.failed_nodes.contains_key("a"));
        assert!(result.skipped_nodes.contains(&"b".to_string()));
    }

    #[tokio::test]
    async fn test_pre_populated_context() {
        let ctx = WorkflowContext::new();
        ctx.set("input", serde_json::json!("hello")).await;

        let workflow = WorkflowBuilder::new("with-ctx")
            .node("use_input", |ctx| {
                Box::pin(async move {
                    let input = ctx.get("input").await.unwrap();
                    Ok(serde_json::json!({"received": input}))
                })
            })
            .build()
            .unwrap();

        let result = workflow.run_with_context(ctx).await.unwrap();
        assert!(result.success);
        assert_eq!(
            result.node_results["use_input"],
            serde_json::json!({"received": "hello"})
        );
    }
}
