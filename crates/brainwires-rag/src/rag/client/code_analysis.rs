//! Code-navigation methods for [`RagClient`]: find definition, references, call graph.
//!
//! All items in this file are gated on the `code-analysis` feature.

use super::RagClient;
use crate::code_analysis::{DefinitionResult, ReferenceResult, RelationsProvider};
use crate::rag::types::*;
use anyhow::{Context, Result};
use std::time::Instant;

impl RagClient {
    /// Find the definition of a symbol at a given file location
    ///
    /// This method looks up the symbol at the specified location and returns
    /// its definition information if found.
    ///
    /// # Arguments
    ///
    /// * `request` - The find definition request containing file path, line, and column
    ///
    /// # Returns
    ///
    /// A response containing the definition if found, along with precision info
    pub async fn find_definition(
        &self,
        request: FindDefinitionRequest,
    ) -> Result<FindDefinitionResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the definition at the requested position
        let definition = definitions.into_iter().find(|def| {
            request.line >= def.symbol_id.start_line
                && request.line <= def.end_line
                && (request.column == 0 || request.column >= def.symbol_id.start_col)
        });

        let result = definition.map(|def| DefinitionResult::from(&def));

        Ok(FindDefinitionResponse {
            definition: result,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Find all references to a symbol at a given file location
    ///
    /// This method finds all locations where the symbol at the given position
    /// is referenced throughout the indexed codebase.
    ///
    /// # Arguments
    ///
    /// * `request` - The find references request containing file path, line, column, and limit
    ///
    /// # Returns
    ///
    /// A response containing the list of references found
    pub async fn find_references(
        &self,
        request: FindReferencesRequest,
    ) -> Result<FindReferencesResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file to find the symbol at the position
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the symbol at the requested position
        let target_symbol = definitions.iter().find(|def| {
            request.line >= def.symbol_id.start_line
                && request.line <= def.end_line
                && (request.column == 0 || request.column >= def.symbol_id.start_col)
        });

        let symbol_name = target_symbol.map(|def| def.symbol_id.name.clone());

        // If no symbol found at position, return empty result
        if symbol_name.is_none() {
            return Ok(FindReferencesResponse {
                symbol_name: None,
                references: Vec::new(),
                total_count: 0,
                precision: format!("{:?}", precision).to_lowercase(),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        let symbol_name_str = symbol_name
            .as_ref()
            .expect("checked is_none above and returned early");

        // Build symbol index from definitions
        let mut symbol_index: std::collections::HashMap<
            String,
            Vec<crate::code_analysis::Definition>,
        > = std::collections::HashMap::new();
        for def in definitions {
            symbol_index
                .entry(def.symbol_id.name.clone())
                .or_default()
                .push(def);
        }

        // Find references in the same file
        let references = self
            .relations_provider
            .extract_references(&file_info, &symbol_index)
            .context("Failed to extract references")?;

        // Filter to references matching our target symbol
        let matching_refs: Vec<ReferenceResult> = references
            .iter()
            .filter(|r| {
                // Check if this reference points to our target symbol
                r.target_symbol_id.contains(symbol_name_str.as_str())
            })
            .take(request.limit)
            .map(ReferenceResult::from)
            .collect();

        let total_count = matching_refs.len();

        Ok(FindReferencesResponse {
            symbol_name,
            references: matching_refs,
            total_count,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Get the call graph for a function at a given file location
    ///
    /// This method returns the callers (incoming calls) and callees (outgoing calls)
    /// for the function at the specified location.
    ///
    /// # Arguments
    ///
    /// * `request` - The call graph request containing file path, line, column, and depth
    ///
    /// # Returns
    ///
    /// A response containing the root symbol and its call graph
    pub async fn get_call_graph(
        &self,
        request: GetCallGraphRequest,
    ) -> Result<GetCallGraphResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file to find the function at the position
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the function at the requested position
        let target_function = definitions.iter().find(|def| {
            // Only consider functions/methods
            matches!(
                def.symbol_id.kind,
                crate::code_analysis::SymbolKind::Function
                    | crate::code_analysis::SymbolKind::Method
            ) && request.line >= def.symbol_id.start_line
                && request.line <= def.end_line
                && (request.column == 0 || request.column >= def.symbol_id.start_col)
        });

        // If no function found at position, return empty result
        let root_symbol = match target_function {
            Some(func) => crate::code_analysis::SymbolInfo {
                name: func.symbol_id.name.clone(),
                kind: func.symbol_id.kind,
                file_path: request.file_path.clone(),
                start_line: func.symbol_id.start_line,
                end_line: func.end_line,
                signature: func.signature.clone(),
            },
            None => {
                return Ok(GetCallGraphResponse {
                    root_symbol: None,
                    callers: Vec::new(),
                    callees: Vec::new(),
                    precision: format!("{:?}", precision).to_lowercase(),
                    duration_ms: start.elapsed().as_millis() as u64,
                });
            }
        };

        let function_name = root_symbol.name.clone();

        // Build symbol index from definitions
        let mut symbol_index: std::collections::HashMap<
            String,
            Vec<crate::code_analysis::Definition>,
        > = std::collections::HashMap::new();
        for def in &definitions {
            symbol_index
                .entry(def.symbol_id.name.clone())
                .or_default()
                .push(def.clone());
        }

        // Find references in the same file to identify callers
        let references = self
            .relations_provider
            .extract_references(&file_info, &symbol_index)
            .context("Failed to extract references")?;

        // Find callers (references with Call kind pointing to our function)
        let mut seen_callers = std::collections::HashSet::new();
        let callers: Vec<crate::code_analysis::CallGraphNode> = references
            .iter()
            .filter(|r| {
                r.reference_kind == crate::code_analysis::ReferenceKind::Call
                    && r.target_symbol_id.contains(&function_name)
            })
            .filter_map(|r| {
                // Try to find which function contains this call
                definitions.iter().find(|def| {
                    matches!(
                        def.symbol_id.kind,
                        crate::code_analysis::SymbolKind::Function
                            | crate::code_analysis::SymbolKind::Method
                    ) && r.start_line >= def.symbol_id.start_line
                        && r.start_line <= def.end_line
                })
            })
            .filter(|def| seen_callers.insert(def.symbol_id.name.clone()))
            .map(|def| crate::code_analysis::CallGraphNode {
                name: def.symbol_id.name.clone(),
                kind: def.symbol_id.kind,
                file_path: request.file_path.clone(),
                line: def.symbol_id.start_line,
                children: Vec::new(),
            })
            .collect();

        // Find callees (calls made from within our function)
        let target_func = target_function.expect("early return on None above guarantees Some");
        let mut seen_callees = std::collections::HashSet::new();
        let callees: Vec<crate::code_analysis::CallGraphNode> = references
            .iter()
            .filter(|r| {
                r.reference_kind == crate::code_analysis::ReferenceKind::Call
                    && r.start_line >= target_func.symbol_id.start_line
                    && r.start_line <= target_func.end_line
            })
            .filter_map(|r| {
                // Extract the called function name from target_symbol_id
                let parts: Vec<&str> = r.target_symbol_id.split(':').collect();
                if parts.len() >= 2 {
                    Some(parts[1].to_string())
                } else {
                    None
                }
            })
            .filter(|name| seen_callees.insert(name.clone()))
            .filter_map(|name| {
                // Find the definition of the called function
                symbol_index
                    .get(&name)
                    .and_then(|defs| defs.first())
                    .cloned()
            })
            .map(|def| crate::code_analysis::CallGraphNode {
                name: def.symbol_id.name.clone(),
                kind: def.symbol_id.kind,
                file_path: request.file_path.clone(),
                line: def.symbol_id.start_line,
                children: Vec::new(),
            })
            .collect();

        Ok(GetCallGraphResponse {
            root_symbol: Some(root_symbol),
            callers,
            callees,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}
