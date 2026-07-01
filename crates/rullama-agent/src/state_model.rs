//! Three-State Model for comprehensive state tracking
//!
//! Based on SagaLLM's Three-State Architecture, this module maintains three
//! separate state domains:
//!
//! 1. **Application State** - Domain logic (what resources exist, their current values)
//! 2. **Operation State** - Execution logs, timing, agent actions
//! 3. **Dependency State** - Constraint graphs, resource relationships
//!
//! This separation enables:
//! - Better debugging through complete operation history
//! - Deadlock detection via dependency graph analysis
//! - Validation of operations against current state
//! - Saga-style compensation using operation logs

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use petgraph::Direction;
use petgraph::algo::is_cyclic_directed;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Three-State Model for comprehensive state tracking
pub struct ThreeStateModel {
    /// Domain-level resource tracking.
    pub application_state: Arc<ApplicationState>,
    /// Execution logging and history.
    pub operation_state: Arc<OperationState>,
    /// Resource relationship graph.
    pub dependency_state: Arc<DependencyState>,
}

impl ThreeStateModel {
    /// Create a new three-state model
    pub fn new() -> Self {
        Self {
            application_state: Arc::new(ApplicationState::new()),
            operation_state: Arc::new(OperationState::new()),
            dependency_state: Arc::new(DependencyState::new()),
        }
    }

    /// Validate that a proposed operation is consistent with current state
    pub async fn validate_operation(&self, op: &StateModelProposedOperation) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check application state - do required resources exist?
        for resource in &op.resources_needed {
            if !self.application_state.resource_exists(resource).await {
                // Not necessarily an error - resource might be created
                warnings.push(format!("Resource '{}' does not exist yet", resource));
            }
        }

        // Check operation state - any conflicting running operations?
        let active_ops = self.operation_state.get_active_operations().await;
        for active_op in active_ops {
            // Check if any resources overlap
            let active_resources: HashSet<_> = active_op
                .resources_needed
                .iter()
                .chain(active_op.resources_produced.iter())
                .collect();

            let proposed_resources: HashSet<_> = op
                .resources_needed
                .iter()
                .chain(op.resources_produced.iter())
                .collect();

            let overlap: Vec<_> = active_resources.intersection(&proposed_resources).collect();

            if !overlap.is_empty() {
                errors.push(format!(
                    "Conflict with running operation '{}': shared resources {:?}",
                    active_op.id,
                    overlap.iter().map(|s| s.as_str()).collect::<Vec<_>>()
                ));
            }
        }

        // Check dependency state - would this create a deadlock?
        if self
            .dependency_state
            .would_deadlock(&op.agent_id, &op.resources_needed)
            .await
        {
            errors.push("Operation would create a deadlock".to_string());
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Record state change from an operation
    pub async fn record_state_change(&self, change: StateChange) {
        // Update application state
        for app_change in &change.application_changes {
            match app_change {
                ApplicationChange::FileModified { path, new_hash } => {
                    self.application_state
                        .update_file(path.clone(), new_hash.clone())
                        .await;
                }
                ApplicationChange::ArtifactInvalidated { artifact_id } => {
                    self.application_state
                        .invalidate_artifact(artifact_id)
                        .await;
                }
                ApplicationChange::GitStateChanged { new_state } => {
                    self.application_state
                        .update_git_state(new_state.clone())
                        .await;
                }
                ApplicationChange::ResourceCreated { resource_id } => {
                    self.application_state
                        .mark_resource_exists(resource_id)
                        .await;
                }
                ApplicationChange::ResourceDeleted { resource_id } => {
                    self.application_state
                        .mark_resource_deleted(resource_id)
                        .await;
                }
            }
        }

        // Update dependency state
        for (from, to, edge) in &change.new_dependencies {
            self.dependency_state
                .add_dependency(from, to, edge.clone())
                .await;
        }
    }

    /// Get a snapshot of current state for validation
    pub async fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            files: self.application_state.get_all_files().await,
            locks: self.dependency_state.get_current_holders().await,
            git_state: self.application_state.get_git_state().await,
            active_operations: self.operation_state.get_active_operation_ids().await,
        }
    }
}

impl Default for ThreeStateModel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Application State
// =============================================================================

/// Application State: Domain-level resource tracking
pub struct ApplicationState {
    /// Files and their current status
    files: RwLock<HashMap<PathBuf, FileStatus>>,
    /// Build artifacts and their validity
    build_artifacts: RwLock<HashMap<String, ArtifactStatus>>,
    /// Git repository state
    git_state: RwLock<GitState>,
    /// Generic resource existence tracking
    resources: RwLock<HashSet<String>>,
}

impl ApplicationState {
    /// Create a new empty application state.
    pub fn new() -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            build_artifacts: RwLock::new(HashMap::new()),
            git_state: RwLock::new(GitState::default()),
            resources: RwLock::new(HashSet::new()),
        }
    }

    /// Check if a resource exists
    pub async fn resource_exists(&self, resource_id: &str) -> bool {
        // Check files first
        let path = PathBuf::from(resource_id);
        if self.files.read().await.contains_key(&path) {
            return true;
        }

        // Check generic resources
        self.resources.read().await.contains(resource_id)
    }

    /// Mark a resource as existing
    pub async fn mark_resource_exists(&self, resource_id: &str) {
        self.resources.write().await.insert(resource_id.to_string());
    }

    /// Mark a resource as deleted
    pub async fn mark_resource_deleted(&self, resource_id: &str) {
        self.resources.write().await.remove(resource_id);
    }

    /// Update file status
    pub async fn update_file(&self, path: PathBuf, content_hash: String) {
        let mut files = self.files.write().await;
        let status = files.entry(path).or_insert_with(|| FileStatus {
            exists: true,
            content_hash: String::new(),
            last_modified: SystemTime::now(),
            locked_by: None,
            dirty: false,
        });
        status.content_hash = content_hash;
        status.last_modified = SystemTime::now();
        status.dirty = true;
        status.exists = true;
    }

    /// Get all file statuses
    pub async fn get_all_files(&self) -> HashMap<PathBuf, FileStatus> {
        self.files.read().await.clone()
    }

    /// Invalidate a build artifact
    pub async fn invalidate_artifact(&self, artifact_id: &str) {
        if let Some(artifact) = self.build_artifacts.write().await.get_mut(artifact_id) {
            artifact.valid = false;
        }
    }

    /// Update git state
    pub async fn update_git_state(&self, state: GitState) {
        *self.git_state.write().await = state;
    }

    /// Get current git state
    pub async fn get_git_state(&self) -> GitState {
        self.git_state.read().await.clone()
    }

    /// Mark file as locked by agent
    pub async fn lock_file(&self, path: &PathBuf, agent_id: &str) {
        if let Some(file) = self.files.write().await.get_mut(path) {
            file.locked_by = Some(agent_id.to_string());
        }
    }

    /// Release file lock
    pub async fn unlock_file(&self, path: &PathBuf) {
        if let Some(file) = self.files.write().await.get_mut(path) {
            file.locked_by = None;
        }
    }

    /// Mark all source files as clean after successful build
    pub async fn mark_files_clean(&self) {
        for file in self.files.write().await.values_mut() {
            file.dirty = false;
        }
    }

    /// Record a build artifact
    pub async fn record_artifact(&self, artifact_id: String, source_hash: String) {
        self.build_artifacts.write().await.insert(
            artifact_id,
            ArtifactStatus {
                valid: true,
                built_from_hash: source_hash,
                build_time: Instant::now(),
            },
        );
    }
}

impl Default for ApplicationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Status of a tracked file in application state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStatus {
    /// Whether the file exists on disk.
    pub exists: bool,
    /// Hash of the file contents.
    pub content_hash: String,
    /// Last modification time.
    #[serde(skip, default = "default_system_time")]
    pub last_modified: SystemTime,
    /// Agent currently holding a lock on this file.
    pub locked_by: Option<String>,
    /// Whether the file has uncommitted changes.
    pub dirty: bool,
}

fn default_system_time() -> SystemTime {
    SystemTime::UNIX_EPOCH
}

/// Status of a build artifact.
#[derive(Debug, Clone)]
pub struct ArtifactStatus {
    /// Whether the artifact is still valid.
    pub valid: bool,
    /// Hash of the source that produced this artifact.
    pub built_from_hash: String,
    /// When the artifact was built.
    pub build_time: Instant,
}

/// Git repository state snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GitState {
    /// Current branch name.
    pub current_branch: String,
    /// HEAD commit hash.
    pub head_commit: String,
    /// Files staged for commit.
    pub staged_files: Vec<String>,
    /// Modified but unstaged files.
    pub modified_files: Vec<String>,
    /// Whether there are merge conflicts.
    pub has_conflicts: bool,
}

// =============================================================================
// Operation State
// =============================================================================

/// Operation State: Execution logging and history
pub struct OperationState {
    /// All operations by ID
    operations: RwLock<HashMap<String, OperationLog>>,
    /// Operations by agent
    agent_operations: RwLock<HashMap<String, Vec<String>>>,
    /// Current active operations
    active_operations: RwLock<HashSet<String>>,
    /// Operation ID counter
    next_id: RwLock<u64>,
}

impl OperationState {
    /// Create a new empty operation state.
    pub fn new() -> Self {
        Self {
            operations: RwLock::new(HashMap::new()),
            agent_operations: RwLock::new(HashMap::new()),
            active_operations: RwLock::new(HashSet::new()),
            next_id: RwLock::new(1),
        }
    }

    /// Generate a new unique operation ID
    pub async fn generate_id(&self) -> String {
        let mut id = self.next_id.write().await;
        let op_id = format!("op-{}", *id);
        *id += 1;
        op_id
    }

    /// Start tracking a new operation
    pub async fn start_operation(&self, log: OperationLog) -> String {
        let id = log.id.clone();

        // Add to operations map
        self.operations
            .write()
            .await
            .insert(id.clone(), log.clone());

        // Add to agent's operations
        self.agent_operations
            .write()
            .await
            .entry(log.agent_id.clone())
            .or_default()
            .push(id.clone());

        // Mark as active
        self.active_operations.write().await.insert(id.clone());

        id
    }

    /// Complete an operation
    pub async fn complete_operation(
        &self,
        operation_id: &str,
        success: bool,
        outputs: Option<serde_json::Value>,
        error: Option<String>,
    ) {
        // Remove from active
        self.active_operations.write().await.remove(operation_id);

        // Update operation log
        if let Some(op) = self.operations.write().await.get_mut(operation_id) {
            op.completed_at = Some(Instant::now());
            op.status = if success {
                OperationLogStatus::Completed
            } else {
                OperationLogStatus::Failed
            };
            op.outputs = outputs;
            op.error = error;
        }
    }

    /// Mark an operation as compensated
    pub async fn mark_compensated(&self, operation_id: &str) {
        if let Some(op) = self.operations.write().await.get_mut(operation_id) {
            op.status = OperationLogStatus::Compensated;
        }
    }

    /// Get active operations
    pub async fn get_active_operations(&self) -> Vec<OperationLog> {
        let active_ids = self.active_operations.read().await.clone();
        let operations = self.operations.read().await;

        active_ids
            .iter()
            .filter_map(|id| operations.get(id).cloned())
            .collect()
    }

    /// Get active operation IDs
    pub async fn get_active_operation_ids(&self) -> Vec<String> {
        self.active_operations
            .read()
            .await
            .iter()
            .cloned()
            .collect()
    }

    /// Get operation by ID
    pub async fn get_operation(&self, operation_id: &str) -> Option<OperationLog> {
        self.operations.read().await.get(operation_id).cloned()
    }

    /// Get all operations for an agent
    pub async fn get_agent_operations(&self, agent_id: &str) -> Vec<OperationLog> {
        let op_ids = self
            .agent_operations
            .read()
            .await
            .get(agent_id)
            .cloned()
            .unwrap_or_default();

        let operations = self.operations.read().await;
        op_ids
            .iter()
            .filter_map(|id| operations.get(id).cloned())
            .collect()
    }

    /// Add child operation to parent
    pub async fn add_child_operation(&self, parent_id: &str, child_id: &str) {
        let mut operations = self.operations.write().await;

        if let Some(parent) = operations.get_mut(parent_id) {
            parent.child_operations.push(child_id.to_string());
        }

        if let Some(child) = operations.get_mut(child_id) {
            child.parent_operation = Some(parent_id.to_string());
        }
    }
}

impl Default for OperationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Log entry for a tracked operation.
#[derive(Debug, Clone)]
pub struct OperationLog {
    /// Unique operation identifier.
    pub id: String,
    /// ID of the agent performing the operation.
    pub agent_id: String,
    /// Type of operation (e.g., "build", "test").
    pub operation_type: String,
    /// When the operation started.
    pub started_at: Instant,
    /// When the operation completed (if finished).
    pub completed_at: Option<Instant>,
    /// Current status of the operation.
    pub status: OperationLogStatus,
    /// Input parameters for the operation.
    pub inputs: serde_json::Value,
    /// Output data (if completed).
    pub outputs: Option<serde_json::Value>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// IDs of child operations spawned by this one.
    pub child_operations: Vec<String>,
    /// ID of the parent operation (if this is a child).
    pub parent_operation: Option<String>,
    /// Resources required by this operation.
    pub resources_needed: Vec<String>,
    /// Resources produced by this operation.
    pub resources_produced: Vec<String>,
}

impl OperationLog {
    /// Create a new operation log entry.
    pub fn new(
        id: String,
        agent_id: String,
        operation_type: String,
        inputs: serde_json::Value,
    ) -> Self {
        Self {
            id,
            agent_id,
            operation_type,
            started_at: Instant::now(),
            completed_at: None,
            status: OperationLogStatus::Running,
            inputs,
            outputs: None,
            error: None,
            child_operations: Vec::new(),
            parent_operation: None,
            resources_needed: Vec::new(),
            resources_produced: Vec::new(),
        }
    }

    /// Set the resources needed and produced by this operation.
    pub fn with_resources(mut self, needed: Vec<String>, produced: Vec<String>) -> Self {
        self.resources_needed = needed;
        self.resources_produced = produced;
        self
    }

    /// Get the duration of the operation (if completed).
    pub fn duration(&self) -> Option<std::time::Duration> {
        self.completed_at
            .map(|end| end.duration_since(self.started_at))
    }
}

/// Status of an operation in its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationLogStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Rolled back via saga compensation.
    Compensated,
}

// =============================================================================
// Dependency State
// =============================================================================

/// Dependency State: Resource relationship graph
pub struct DependencyState {
    /// Resource dependency graph
    graph: RwLock<DiGraph<ResourceNode, DependencyEdge>>,
    /// Index: resource name -> node index
    resource_index: RwLock<HashMap<String, NodeIndex>>,
}

impl DependencyState {
    /// Create a new empty dependency state.
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(DiGraph::new()),
            resource_index: RwLock::new(HashMap::new()),
        }
    }

    /// Add or get a resource node
    async fn ensure_node(&self, resource_id: &str, resource_type: ResourceNodeType) -> NodeIndex {
        let mut index = self.resource_index.write().await;
        let mut graph = self.graph.write().await;

        if let Some(&node_idx) = index.get(resource_id) {
            return node_idx;
        }

        let node = ResourceNode {
            resource_id: resource_id.to_string(),
            resource_type,
            current_holder: None,
        };

        let node_idx = graph.add_node(node);
        index.insert(resource_id.to_string(), node_idx);
        node_idx
    }

    /// Add a dependency between resources
    ///
    /// For a "BlockedBy" relationship: `add_dependency("op-a", "op-b", BlockedBy)` means
    /// "op-a is blocked by op-b", so op-b must execute before op-a.
    /// The edge direction is from→to (to comes before from in execution order).
    pub async fn add_dependency(&self, from: &str, to: &str, edge: DependencyEdge) {
        let from_idx = self.ensure_node(from, ResourceNodeType::Generic).await;
        let to_idx = self.ensure_node(to, ResourceNodeType::Generic).await;

        // Edge from "to" to "from" because "from" depends on "to"
        // In graph terms: to → from (to must be processed before from)
        self.graph.write().await.add_edge(to_idx, from_idx, edge);
    }

    /// Remove a dependency
    pub async fn remove_dependency(&self, from: &str, to: &str) {
        let index = self.resource_index.read().await;
        let mut graph = self.graph.write().await;

        // Edge was added from to→from, so we need to find it that way
        if let (Some(&from_idx), Some(&to_idx)) = (index.get(from), index.get(to))
            && let Some(edge) = graph.find_edge(to_idx, from_idx)
        {
            graph.remove_edge(edge);
        }
    }

    /// Check if acquiring resources would create a deadlock
    pub async fn would_deadlock(&self, agent_id: &str, resources: &[String]) -> bool {
        let mut graph = self.graph.write().await;
        let mut index = self.resource_index.write().await;

        // Create temporary nodes for the agent's wait-for relationship
        let agent_node_id = format!("agent:{}", agent_id);

        // Check if agent node exists, create if not
        let agent_idx = if let Some(&idx) = index.get(&agent_node_id) {
            idx
        } else {
            let node = ResourceNode {
                resource_id: agent_node_id.clone(),
                resource_type: ResourceNodeType::Agent(agent_id.to_string()),
                current_holder: None,
            };
            let idx = graph.add_node(node);
            index.insert(agent_node_id.clone(), idx);
            idx
        };

        // Temporarily add edges from agent to requested resources
        let mut temp_edges = Vec::new();
        for resource in resources {
            if let Some(&resource_idx) = index.get(resource) {
                let edge = graph.add_edge(
                    agent_idx,
                    resource_idx,
                    DependencyEdge {
                        dependency_type: DependencyType::WaitsFor,
                        strength: DependencyStrength::Hard,
                    },
                );
                temp_edges.push(edge);
            }
        }

        // Check for cycles
        let has_cycle = is_cyclic_directed(&*graph);

        // Remove temporary edges
        for edge in temp_edges {
            graph.remove_edge(edge);
        }

        has_cycle
    }

    /// Get all resources that must be released before agent can acquire resource
    pub async fn get_blocking_resources(&self, resource_id: &str) -> Vec<String> {
        let graph = self.graph.read().await;
        let index = self.resource_index.read().await;

        let mut blocking = Vec::new();

        if let Some(&node_idx) = index.get(resource_id) {
            // Find all incoming edges (resources that this resource depends on)
            for edge_ref in graph.edges_directed(node_idx, Direction::Incoming) {
                if let Some(source_node) = graph.node_weight(edge_ref.source())
                    && source_node.current_holder.is_some()
                {
                    blocking.push(source_node.resource_id.clone());
                }
            }
        }

        blocking
    }

    /// Set the current holder of a resource
    pub async fn set_holder(&self, resource_id: &str, agent_id: Option<&str>) {
        let index = self.resource_index.read().await;
        let mut graph = self.graph.write().await;

        if let Some(&node_idx) = index.get(resource_id)
            && let Some(node) = graph.node_weight_mut(node_idx)
        {
            node.current_holder = agent_id.map(String::from);
        }
    }

    /// Get current resource holders
    pub async fn get_current_holders(&self) -> HashMap<String, String> {
        let graph = self.graph.read().await;

        graph
            .node_weights()
            .filter_map(|node| {
                node.current_holder
                    .as_ref()
                    .map(|holder| (node.resource_id.clone(), holder.clone()))
            })
            .collect()
    }

    /// Get resources held by an agent
    pub async fn get_agent_resources(&self, agent_id: &str) -> Vec<String> {
        let graph = self.graph.read().await;

        graph
            .node_weights()
            .filter_map(|node| {
                if node.current_holder.as_deref() == Some(agent_id) {
                    Some(node.resource_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Topological sort of operations respecting dependencies
    pub async fn get_execution_order(&self, operation_ids: &[String]) -> Vec<String> {
        let graph = self.graph.read().await;
        let index = self.resource_index.read().await;

        // Simple implementation: return in dependency order
        // More sophisticated implementation would use Kahn's algorithm
        let mut ordered = Vec::new();
        let mut remaining: HashSet<_> = operation_ids.iter().cloned().collect();

        while !remaining.is_empty() {
            let mut made_progress = false;

            for op_id in remaining.clone() {
                if let Some(&node_idx) = index.get(&op_id) {
                    // Check if all dependencies are satisfied
                    let all_deps_satisfied = graph
                        .edges_directed(node_idx, Direction::Incoming)
                        .all(|edge| {
                            graph
                                .node_weight(edge.source())
                                .map(|n| !remaining.contains(&n.resource_id))
                                .unwrap_or(true)
                        });

                    if all_deps_satisfied {
                        ordered.push(op_id.clone());
                        remaining.remove(&op_id);
                        made_progress = true;
                    }
                } else {
                    // No dependencies, can be added
                    ordered.push(op_id.clone());
                    remaining.remove(&op_id);
                    made_progress = true;
                }
            }

            if !made_progress {
                // Cycle detected or remaining items have unsatisfied deps
                // Just add remaining in arbitrary order
                ordered.extend(remaining.drain());
                break;
            }
        }

        ordered
    }
}

impl Default for DependencyState {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the resource dependency graph.
#[derive(Debug, Clone)]
pub struct ResourceNode {
    /// Unique resource identifier.
    pub resource_id: String,
    /// Type of this resource.
    pub resource_type: ResourceNodeType,
    /// Agent currently holding this resource.
    pub current_holder: Option<String>,
}

/// Type of resource node in the dependency graph.
#[derive(Debug, Clone)]
pub enum ResourceNodeType {
    /// A file resource.
    File(PathBuf),
    /// Build system lock.
    BuildLock,
    /// Test system lock.
    TestLock,
    /// Git index lock.
    GitIndex,
    /// Git branch resource.
    GitBranch(String),
    /// An agent node (for wait-for graphs).
    Agent(String),
    /// Generic resource.
    Generic,
}

/// An edge in the dependency graph between resources.
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// Type of dependency relationship.
    pub dependency_type: DependencyType,
    /// How strictly the dependency must be respected.
    pub strength: DependencyStrength,
}

/// Type of dependency relationship between resources
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyType {
    /// A depends on B (A needs B to complete first)
    BlockedBy,
    /// A produces B (completing A makes B available)
    Produces,
    /// A and B conflict (cannot run concurrently)
    ConflictsWith,
    /// A reads B (A needs B in consistent state)
    Reads,
    /// A writes B (A will modify B)
    Writes,
    /// A waits for B (used in deadlock detection)
    WaitsFor,
}

/// How strictly a dependency must be respected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyStrength {
    /// Must be respected
    Hard,
    /// Preferred but can be violated
    Soft,
    /// Information only
    Advisory,
}

// =============================================================================
// Shared Types
// =============================================================================

/// Result of validating an operation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the operation passed validation.
    pub valid: bool,
    /// Validation errors (if any).
    pub errors: Vec<String>,
    /// Validation warnings (if any).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a passing validation result.
    pub fn ok() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a failing validation result with a single error.
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            valid: false,
            errors: vec![msg.into()],
            warnings: Vec::new(),
        }
    }
}

/// A proposed operation to validate (for the three-state model)
#[derive(Debug, Clone)]
pub struct StateModelProposedOperation {
    /// ID of the agent proposing the operation.
    pub agent_id: String,
    /// Type of proposed operation.
    pub operation_type: String,
    /// Resources needed by the operation.
    pub resources_needed: Vec<String>,
    /// Resources that will be produced.
    pub resources_produced: Vec<String>,
}

/// State change to record
#[derive(Debug, Clone)]
pub struct StateChange {
    /// ID of the operation that caused this change.
    pub operation_id: String,
    /// Application-level changes.
    pub application_changes: Vec<ApplicationChange>,
    /// New dependency relationships (from, to, edge).
    pub new_dependencies: Vec<(String, String, DependencyEdge)>,
}

/// Types of application state changes
#[derive(Debug, Clone)]
pub enum ApplicationChange {
    /// A file was modified.
    FileModified {
        /// Path of the modified file.
        path: PathBuf,
        /// New content hash.
        new_hash: String,
    },
    /// A build artifact was invalidated.
    ArtifactInvalidated {
        /// ID of the invalidated artifact.
        artifact_id: String,
    },
    /// Git repository state changed.
    GitStateChanged {
        /// New git state.
        new_state: GitState,
    },
    /// A new resource was created.
    ResourceCreated {
        /// ID of the created resource.
        resource_id: String,
    },
    /// A resource was deleted.
    ResourceDeleted {
        /// ID of the deleted resource.
        resource_id: String,
    },
}

/// Snapshot of current state
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// All tracked files and their status.
    pub files: HashMap<PathBuf, FileStatus>,
    /// Current resource locks (resource -> holder agent).
    pub locks: HashMap<String, String>,
    /// Current git repository state.
    pub git_state: GitState,
    /// IDs of currently active operations.
    pub active_operations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_three_state_model_creation() {
        let model = ThreeStateModel::new();
        let snapshot = model.snapshot().await;

        assert!(snapshot.files.is_empty());
        assert!(snapshot.locks.is_empty());
        assert!(snapshot.active_operations.is_empty());
    }

    #[tokio::test]
    async fn test_application_state_file_tracking() {
        let app_state = ApplicationState::new();

        let path = PathBuf::from("/test/file.rs");
        app_state
            .update_file(path.clone(), "hash123".to_string())
            .await;

        let files = app_state.get_all_files().await;
        assert!(files.contains_key(&path));
        assert_eq!(files[&path].content_hash, "hash123");
        assert!(files[&path].dirty);
    }

    #[tokio::test]
    async fn test_operation_state_lifecycle() {
        let op_state = OperationState::new();

        let log = OperationLog::new(
            "op-1".to_string(),
            "agent-1".to_string(),
            "build".to_string(),
            serde_json::json!({}),
        );

        let id = op_state.start_operation(log).await;
        assert_eq!(id, "op-1");

        let active = op_state.get_active_operations().await;
        assert_eq!(active.len(), 1);

        op_state.complete_operation(&id, true, None, None).await;

        let active = op_state.get_active_operations().await;
        assert!(active.is_empty());

        let op = op_state.get_operation(&id).await.unwrap();
        assert_eq!(op.status, OperationLogStatus::Completed);
    }

    #[tokio::test]
    async fn test_dependency_state_deadlock_detection() {
        let dep_state = DependencyState::new();

        // Create a simple dependency: resource-a -> resource-b
        dep_state
            .add_dependency(
                "resource-a",
                "resource-b",
                DependencyEdge {
                    dependency_type: DependencyType::BlockedBy,
                    strength: DependencyStrength::Hard,
                },
            )
            .await;

        // Set holder for resource-a
        dep_state.set_holder("resource-a", Some("agent-1")).await;

        // Agent-1 trying to acquire resource-b shouldn't deadlock
        let would_deadlock = dep_state
            .would_deadlock("agent-1", &["resource-b".to_string()])
            .await;

        // This shouldn't cause a deadlock in this simple case
        // (more complex scenarios would involve actual wait-for graphs)
        assert!(!would_deadlock);
    }

    #[tokio::test]
    async fn test_validate_operation_conflict_detection() {
        let model = ThreeStateModel::new();

        // Start a running operation
        let log = OperationLog::new(
            "op-1".to_string(),
            "agent-1".to_string(),
            "build".to_string(),
            serde_json::json!({}),
        )
        .with_resources(vec!["resource-a".to_string()], vec![]);

        model.operation_state.start_operation(log).await;

        // Try to validate an operation that conflicts
        let proposed = StateModelProposedOperation {
            agent_id: "agent-2".to_string(),
            operation_type: "build".to_string(),
            resources_needed: vec!["resource-a".to_string()],
            resources_produced: vec![],
        };

        let result = model.validate_operation(&proposed).await;
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_state_change_recording() {
        let model = ThreeStateModel::new();

        let change = StateChange {
            operation_id: "op-1".to_string(),
            application_changes: vec![
                ApplicationChange::FileModified {
                    path: PathBuf::from("/test/file.rs"),
                    new_hash: "newhash".to_string(),
                },
                ApplicationChange::ResourceCreated {
                    resource_id: "build-artifact".to_string(),
                },
            ],
            new_dependencies: vec![],
        };

        model.record_state_change(change).await;

        let snapshot = model.snapshot().await;
        assert!(snapshot.files.contains_key(&PathBuf::from("/test/file.rs")));
        assert!(
            model
                .application_state
                .resource_exists("build-artifact")
                .await
        );
    }

    #[tokio::test]
    async fn test_execution_order() {
        let dep_state = DependencyState::new();

        // A depends on B
        dep_state
            .add_dependency(
                "op-a",
                "op-b",
                DependencyEdge {
                    dependency_type: DependencyType::BlockedBy,
                    strength: DependencyStrength::Hard,
                },
            )
            .await;

        let order = dep_state
            .get_execution_order(&["op-a".to_string(), "op-b".to_string()])
            .await;

        // op-b should come before op-a since op-a depends on op-b
        let pos_a = order.iter().position(|x| x == "op-a").unwrap();
        let pos_b = order.iter().position(|x| x == "op-b").unwrap();
        assert!(pos_b < pos_a);
    }
}
