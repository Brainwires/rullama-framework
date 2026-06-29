use serde::{Deserialize, Serialize};

/// Request to find the definition of a symbol at a given location
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FindDefinitionRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (0-based)
    pub column: usize,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
}

impl FindDefinitionRequest {
    /// Validate the find definition request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.line == 0 {
            return Err("line must be 1-based (cannot be 0)".to_string());
        }
        Ok(())
    }
}

/// Response from find_definition
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FindDefinitionResponse {
    /// The found definition, if any
    pub definition: Option<crate::code_analysis::DefinitionResult>,
    /// Precision level of the result
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

fn default_references_limit() -> usize {
    100
}

fn default_include_definition() -> bool {
    true
}

/// Request to find all references to a symbol
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FindReferencesRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (0-based)
    pub column: usize,
    /// Maximum number of references to return
    #[serde(default = "default_references_limit")]
    pub limit: usize,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Include the definition itself in results
    #[serde(default = "default_include_definition")]
    pub include_definition: bool,
}

impl FindReferencesRequest {
    /// Validate the find references request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.line == 0 {
            return Err("line must be 1-based (cannot be 0)".to_string());
        }
        const MAX_LIMIT: usize = 10000;
        if self.limit > MAX_LIMIT {
            return Err(format!(
                "limit too large: {} (max: {})",
                self.limit, MAX_LIMIT
            ));
        }
        Ok(())
    }
}

/// Response from find_references
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FindReferencesResponse {
    /// The symbol being referenced
    pub symbol_name: Option<String>,
    /// List of found references
    pub references: Vec<crate::code_analysis::ReferenceResult>,
    /// Total count (may be higher than returned if limit applied)
    pub total_count: usize,
    /// Precision level of the results
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

fn default_call_graph_depth() -> usize {
    2
}

fn default_true() -> bool {
    true
}

/// Request to get call graph for a function
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetCallGraphRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (0-based)
    pub column: usize,
    /// Maximum depth to traverse (default: 2)
    #[serde(default = "default_call_graph_depth")]
    pub depth: usize,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Include callers (functions that call this function)
    #[serde(default = "default_true")]
    pub include_callers: bool,
    /// Include callees (functions this function calls)
    #[serde(default = "default_true")]
    pub include_callees: bool,
}

impl GetCallGraphRequest {
    /// Validate the get call graph request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.line == 0 {
            return Err("line must be 1-based (cannot be 0)".to_string());
        }
        const MAX_DEPTH: usize = 10;
        if self.depth > MAX_DEPTH {
            return Err(format!(
                "depth too large: {} (max: {})",
                self.depth, MAX_DEPTH
            ));
        }
        Ok(())
    }
}

/// Response from get_call_graph
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GetCallGraphResponse {
    /// The root symbol (function/method at the requested location)
    pub root_symbol: Option<crate::code_analysis::SymbolInfo>,
    /// Functions/methods that call this symbol (incoming calls)
    pub callers: Vec<crate::code_analysis::CallGraphNode>,
    /// Functions/methods called by this symbol (outgoing calls)
    pub callees: Vec<crate::code_analysis::CallGraphNode>,
    /// Precision level of the results
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}
