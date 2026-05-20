//! Relationship Graph Storage
//!
//! Stores and queries entity relationships for enhanced context retrieval.
//! Uses an in-memory graph with optional persistence to LanceDB.

use crate::knowledge::entity::{Entity, EntityStore, EntityType, Relationship};
use std::collections::{HashMap, HashSet, VecDeque};

// Re-export graph types from core (canonical definitions)
pub use brainwires_core::graph::{EdgeType, GraphEdge, GraphNode};

/// Relationship graph for entity context
#[derive(Debug, Default)]
pub struct RelationshipGraph {
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    adjacency: HashMap<String, Vec<usize>>, // node -> edge indices
}

impl RelationshipGraph {
    /// Create a new empty relationship graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build graph from entity store
    pub fn from_entity_store(store: &EntityStore) -> Self {
        let mut graph = Self::new();

        // Add nodes from top entities
        for entity in store.get_top_entities(100) {
            graph.add_node(GraphNode {
                entity_name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                message_ids: entity.message_ids.clone(),
                mention_count: entity.mention_count,
                importance: Self::calculate_importance(entity),
            });
        }

        graph
    }

    /// Calculate importance score for an entity.
    ///
    /// Combines log-scaled mention count, entity-type bonus, and message-spread
    /// proxy into a score in `[0.0, 1.0]`.
    ///
    /// **Known limitation**: `ln(1) = 0`, so the mention-count component
    /// contributes nothing for entities seen exactly once. The type bonus and
    /// message-spread proxy still apply, so the score is non-zero, but a
    /// single-mention entity is scored identically regardless of how many
    /// times it was seen (just once vs. genuinely once). Use
    /// `ln(mention_count + 1)` to remove this discontinuity if needed.
    pub fn calculate_importance(entity: &Entity) -> f32 {
        let mut score = 0.0;

        // Base score from mentions
        score += (entity.mention_count as f32).ln().max(0.0) * 0.3;

        // Type-based importance
        score += match entity.entity_type {
            EntityType::File => 0.4,
            EntityType::Function => 0.3,
            EntityType::Type => 0.35,
            EntityType::Error => 0.25,
            EntityType::Concept => 0.2,
            EntityType::Variable => 0.1,
            EntityType::Command => 0.15,
        };

        // Recency (would need timestamp context)
        // For now, use message count as proxy
        score += (entity.message_ids.len() as f32 * 0.05).min(0.2);

        score.clamp(0.0, 1.0)
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        let name = node.entity_name.clone();
        if !self.adjacency.contains_key(&name) {
            self.adjacency.insert(name.clone(), Vec::new());
        }
        self.nodes.insert(name, node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        let idx = self.edges.len();

        // Update adjacency list for both directions
        if let Some(adj) = self.adjacency.get_mut(&edge.from) {
            adj.push(idx);
        }
        if let Some(adj) = self.adjacency.get_mut(&edge.to) {
            adj.push(idx);
        }

        self.edges.push(edge);
    }

    /// Add relationship as edge
    pub fn add_relationship(&mut self, rel: &Relationship) {
        let (from, to, edge_type, message_id) = match rel {
            Relationship::CoOccurs {
                entity_a,
                entity_b,
                message_id,
            } => (
                entity_a.clone(),
                entity_b.clone(),
                EdgeType::CoOccurs,
                Some(message_id.clone()),
            ),
            Relationship::Contains {
                container,
                contained,
            } => (
                container.clone(),
                contained.clone(),
                EdgeType::Contains,
                None,
            ),
            Relationship::References { from, to } => {
                (from.clone(), to.clone(), EdgeType::References, None)
            }
            Relationship::DependsOn {
                dependent,
                dependency,
            } => (
                dependent.clone(),
                dependency.clone(),
                EdgeType::DependsOn,
                None,
            ),
            Relationship::Modifies {
                modifier, modified, ..
            } => (modifier.clone(), modified.clone(), EdgeType::Modifies, None),
            Relationship::Defines {
                definer, defined, ..
            } => (definer.clone(), defined.clone(), EdgeType::Defines, None),
        };

        // Only add edge if both nodes exist
        if self.nodes.contains_key(&from) && self.nodes.contains_key(&to) {
            self.add_edge(GraphEdge {
                from,
                to,
                weight: edge_type.weight(),
                edge_type,
                message_id,
            });
        }
    }

    /// Get node by name
    pub fn get_node(&self, name: &str) -> Option<&GraphNode> {
        self.nodes.get(name)
    }

    /// Get all neighbors of a node
    pub fn get_neighbors(&self, name: &str) -> Vec<&GraphNode> {
        let mut neighbors = Vec::new();

        if let Some(edge_indices) = self.adjacency.get(name) {
            for &idx in edge_indices {
                if let Some(edge) = self.edges.get(idx) {
                    let neighbor_name = if edge.from == name {
                        &edge.to
                    } else {
                        &edge.from
                    };
                    if let Some(node) = self.nodes.get(neighbor_name) {
                        neighbors.push(node);
                    }
                }
            }
        }

        neighbors
    }

    /// Get edges for a node
    pub fn get_edges(&self, name: &str) -> Vec<&GraphEdge> {
        self.adjacency
            .get(name)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.edges.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find shortest path between two entities using BFS
    pub fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return None;
        }

        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            for neighbor in self.get_neighbors(&current) {
                if !visited.contains(&neighbor.entity_name) {
                    visited.insert(neighbor.entity_name.clone());
                    parent.insert(neighbor.entity_name.clone(), current.clone());

                    if neighbor.entity_name == to {
                        // Reconstruct path
                        let mut path = vec![to.to_string()];
                        let mut node = to.to_string();
                        while let Some(p) = parent.get(&node) {
                            path.push(p.clone());
                            node = p.clone();
                        }
                        path.reverse();
                        return Some(path);
                    }

                    queue.push_back(neighbor.entity_name.clone());
                }
            }
        }

        None
    }

    /// Get all context related to an entity (traverses graph)
    pub fn get_entity_context(&self, entity: &str, max_depth: usize) -> EntityContext {
        let mut context = EntityContext {
            root: entity.to_string(),
            related_entities: Vec::new(),
            message_ids: HashSet::new(),
        };

        if let Some(node) = self.nodes.get(entity) {
            context.message_ids.extend(node.message_ids.clone());
        }

        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        queue.push_back((entity.to_string(), 0));
        visited.insert(entity.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for edge in self.get_edges(&current) {
                let neighbor = if edge.from == current {
                    &edge.to
                } else {
                    &edge.from
                };

                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());

                    if let Some(node) = self.nodes.get(neighbor) {
                        context.related_entities.push(RelatedEntity {
                            name: neighbor.clone(),
                            entity_type: node.entity_type.clone(),
                            relationship: edge.edge_type.clone(),
                            distance: depth + 1,
                            relevance: edge.weight * (0.8_f32).powi((depth + 1) as i32),
                        });
                        context.message_ids.extend(node.message_ids.clone());
                    }

                    queue.push_back((neighbor.clone(), depth + 1));
                }
            }
        }

        // Sort by relevance
        context.related_entities.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        context
    }

    /// Find entities most relevant to a query (by name matching)
    pub fn search(&self, query: &str, limit: usize) -> Vec<&GraphNode> {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<_> = query_lower.split_whitespace().collect();

        let mut scored: Vec<_> = self
            .nodes
            .values()
            .map(|node| {
                let name_lower = node.entity_name.to_lowercase();
                let mut score = 0.0;

                // Exact match
                if name_lower == query_lower {
                    score += 1.0;
                }
                // Contains query
                else if name_lower.contains(&query_lower) {
                    score += 0.7;
                }
                // Query contains name
                else if query_lower.contains(&name_lower) {
                    score += 0.5;
                }
                // Word overlap
                else {
                    let name_words: HashSet<_> =
                        name_lower.split(|c: char| !c.is_alphanumeric()).collect();
                    let overlap = query_words.intersection(&name_words).count();
                    score += overlap as f32 * 0.3;
                }

                // Boost by importance
                score *= 1.0 + node.importance * 0.5;

                (node, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(limit)
            .map(|(node, _)| node)
            .collect()
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let mut type_counts = HashMap::new();
        for node in self.nodes.values() {
            *type_counts.entry(node.entity_type.as_str()).or_insert(0) += 1;
        }

        let mut edge_type_counts = HashMap::new();
        for edge in &self.edges {
            *edge_type_counts
                .entry(format!("{:?}", edge.edge_type))
                .or_insert(0) += 1;
        }

        GraphStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            nodes_by_type: type_counts,
            edges_by_type: edge_type_counts,
        }
    }

    // ============ SEAL Integration Methods ============

    /// Get entities that would be impacted by changes to a given entity.
    /// Traverses the graph to find dependent entities up to a specified depth.
    pub fn get_impact_set(&self, entity: &str, depth: usize) -> Vec<ImpactedEntity> {
        let mut impacts = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, usize, f32)> = VecDeque::new();

        if !self.nodes.contains_key(entity) {
            return impacts;
        }

        queue.push_back((entity.to_string(), 0, 1.0));
        visited.insert(entity.to_string());

        while let Some((current, current_depth, current_impact)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            for edge in self.get_edges(&current) {
                let neighbor = if edge.from == current {
                    &edge.to
                } else {
                    &edge.from
                };

                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());

                    // Calculate impact factor based on edge type and weight
                    let impact_factor = match edge.edge_type {
                        EdgeType::DependsOn => 0.9,
                        EdgeType::Contains => 0.8,
                        EdgeType::Modifies => 0.7,
                        EdgeType::References => 0.5,
                        EdgeType::Defines => 0.6,
                        EdgeType::CoOccurs => 0.3,
                    };

                    let propagated_impact = current_impact * impact_factor * edge.weight;

                    if let Some(node) = self.nodes.get(neighbor) {
                        impacts.push(ImpactedEntity {
                            name: neighbor.clone(),
                            entity_type: node.entity_type.clone(),
                            distance: current_depth + 1,
                            impact_score: propagated_impact,
                            impact_path: vec![current.clone(), neighbor.clone()],
                        });
                    }

                    queue.push_back((neighbor.clone(), current_depth + 1, propagated_impact));
                }
            }
        }

        // Sort by impact score (highest first)
        impacts.sort_by(|a, b| {
            b.impact_score
                .partial_cmp(&a.impact_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        impacts
    }

    /// Find clusters of related entities using connected component analysis
    pub fn find_clusters(&self) -> Vec<EntityCluster> {
        let mut clusters = Vec::new();
        let mut visited = HashSet::new();

        for node_name in self.nodes.keys() {
            if visited.contains(node_name) {
                continue;
            }

            // BFS to find all connected nodes
            let mut cluster_nodes = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node_name.clone());
            visited.insert(node_name.clone());

            while let Some(current) = queue.pop_front() {
                if let Some(node) = self.nodes.get(&current) {
                    cluster_nodes.push(node.clone());
                }

                for neighbor in self.get_neighbors(&current) {
                    if !visited.contains(&neighbor.entity_name) {
                        visited.insert(neighbor.entity_name.clone());
                        queue.push_back(neighbor.entity_name.clone());
                    }
                }
            }

            if !cluster_nodes.is_empty() {
                // Calculate cluster metrics
                let total_importance: f32 = cluster_nodes.iter().map(|n| n.importance).sum();
                let avg_importance = total_importance / cluster_nodes.len() as f32;

                // Find dominant type
                let mut type_counts = HashMap::new();
                for node in &cluster_nodes {
                    *type_counts.entry(node.entity_type.clone()).or_insert(0) += 1;
                }
                let dominant_type = type_counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(t, _)| t);

                clusters.push(EntityCluster {
                    id: clusters.len(),
                    nodes: cluster_nodes,
                    avg_importance,
                    dominant_type,
                });
            }
        }

        // Sort clusters by size (largest first)
        clusters.sort_by(|a, b| b.nodes.len().cmp(&a.nodes.len()));

        clusters
    }

    /// Suggest related entities given a set of entities.
    /// Uses co-occurrence and relationship analysis.
    pub fn suggest_related(&self, entities: &[&str]) -> Vec<SuggestedEntity> {
        let mut scores: HashMap<String, f32> = HashMap::new();
        let entity_set: HashSet<_> = entities.iter().copied().collect();

        for entity in entities {
            // Get direct neighbors
            for neighbor in self.get_neighbors(entity) {
                if !entity_set.contains(neighbor.entity_name.as_str()) {
                    *scores.entry(neighbor.entity_name.clone()).or_default() += neighbor.importance;
                }
            }

            // Get second-degree neighbors with lower weight
            for first_neighbor in self.get_neighbors(entity) {
                if entity_set.contains(first_neighbor.entity_name.as_str()) {
                    continue;
                }
                for second_neighbor in self.get_neighbors(&first_neighbor.entity_name) {
                    if !entity_set.contains(second_neighbor.entity_name.as_str())
                        && second_neighbor.entity_name != *entity
                    {
                        *scores
                            .entry(second_neighbor.entity_name.clone())
                            .or_default() += second_neighbor.importance * 0.5;
                    }
                }
            }
        }

        // Convert to suggestions
        let mut suggestions: Vec<_> = scores
            .into_iter()
            .filter_map(|(name, score)| {
                self.nodes.get(&name).map(|node| SuggestedEntity {
                    name: name.clone(),
                    entity_type: node.entity_type.clone(),
                    relevance_score: score,
                    reason: self.get_suggestion_reason(&name, entities),
                })
            })
            .collect();

        // Sort by relevance
        suggestions.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suggestions.truncate(10);
        suggestions
    }

    /// Get a reason for suggesting an entity
    fn get_suggestion_reason(&self, suggested: &str, source_entities: &[&str]) -> String {
        for source in source_entities {
            // Check direct relationship
            let edges = self.get_edges(source);
            for edge in edges {
                let other = if edge.from == *source {
                    &edge.to
                } else {
                    &edge.from
                };
                if other == suggested {
                    return format!("{:?} by {}", edge.edge_type, source);
                }
            }
        }
        "Related through graph".to_string()
    }

    /// Get the most central nodes in the graph (by connectivity)
    pub fn get_central_nodes(&self, limit: usize) -> Vec<&GraphNode> {
        let mut centrality: Vec<_> = self
            .nodes
            .iter()
            .map(|(name, node)| {
                let degree = self.adjacency.get(name).map(|v| v.len()).unwrap_or(0);
                let weighted_score = node.importance * 0.7 + (degree as f32 / 10.0).min(0.3);
                (node, weighted_score)
            })
            .collect();

        centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        centrality.into_iter().take(limit).map(|(n, _)| n).collect()
    }

    // ============ Spectral Graph Methods ============

    /// Convert this graph to a dense weighted adjacency matrix.
    ///
    /// Returns `(adjacency_matrix, node_names)` where `node_names[i]` is the
    /// entity name for row/column `i`. Multi-edges between the same pair are
    /// summed.
    #[cfg(feature = "spectral")]
    fn to_adjacency_matrix(&self) -> (ndarray::Array2<f32>, Vec<String>) {
        let names: Vec<String> = self.nodes.keys().cloned().collect();
        let n = names.len();
        let idx: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let mut adj = ndarray::Array2::<f32>::zeros((n, n));
        for edge in &self.edges {
            if let (Some(&i), Some(&j)) = (idx.get(edge.from.as_str()), idx.get(edge.to.as_str())) {
                adj[[i, j]] += edge.weight;
                adj[[j, i]] += edge.weight;
            }
        }

        (adj, names)
    }

    /// Find semantic communities within connected components using spectral clustering.
    ///
    /// Unlike `find_clusters` which only finds connected components, this method
    /// discovers tightly-coupled groups *within* a connected component by analyzing
    /// the graph's spectral properties (Fiedler vector of the Laplacian).
    ///
    /// # Arguments
    ///
    /// * `k` - Number of clusters to find. If the graph has fewer natural clusters,
    ///   fewer may be returned.
    #[cfg(feature = "spectral")]
    pub fn spectral_clusters(&self, k: usize) -> Vec<EntityCluster> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let (adj, names) = self.to_adjacency_matrix();
        let assignments = match crate::spectral::graph_ops::spectral_cluster(&adj, k) {
            Some(a) => a,
            None => return self.find_clusters(), // fall back to connected components
        };

        // Group nodes by cluster assignment
        let max_cluster = assignments.iter().copied().max().unwrap_or(0);
        let mut cluster_nodes: Vec<Vec<GraphNode>> = vec![Vec::new(); max_cluster + 1];

        for (i, &cluster_id) in assignments.iter().enumerate() {
            if let Some(node) = self.nodes.get(&names[i]) {
                cluster_nodes[cluster_id].push(node.clone());
            }
        }

        // Build EntityCluster for each non-empty group
        cluster_nodes
            .into_iter()
            .enumerate()
            .filter(|(_, nodes)| !nodes.is_empty())
            .map(|(id, nodes)| {
                let avg_importance =
                    nodes.iter().map(|n| n.importance).sum::<f32>() / nodes.len() as f32;
                let mut type_counts = HashMap::new();
                for node in &nodes {
                    *type_counts.entry(node.entity_type.clone()).or_insert(0) += 1;
                }
                let dominant_type = type_counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, _)| t);

                EntityCluster {
                    id,
                    nodes,
                    avg_importance,
                    dominant_type,
                }
            })
            .collect()
    }

    /// Compute spectral centrality for all nodes.
    ///
    /// Returns nodes sorted by spectral centrality (highest first). Nodes with
    /// high centrality are structural bridges between communities — important
    /// for understanding cross-cutting concerns in the codebase.
    ///
    /// This complements `get_central_nodes` which uses degree centrality.
    /// Spectral centrality captures *structural position* rather than just
    /// connection count.
    #[cfg(feature = "spectral")]
    pub fn spectral_central_nodes(&self, limit: usize) -> Vec<(&GraphNode, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let (adj, names) = self.to_adjacency_matrix();
        let scores = crate::spectral::graph_ops::spectral_centrality(&adj);

        scores
            .into_iter()
            .filter_map(|(i, score)| self.nodes.get(&names[i]).map(|node| (node, score)))
            .take(limit)
            .collect()
    }

    /// Compute the algebraic connectivity of this graph.
    ///
    /// This is the second-smallest eigenvalue of the Laplacian, measuring how
    /// well-connected the graph is:
    /// - 0 = disconnected (multiple components)
    /// - Small = bottleneck exists (near-disconnection)
    /// - Large = well-connected
    ///
    /// Useful for monitoring knowledge graph health as entities accumulate.
    #[cfg(feature = "spectral")]
    pub fn connectivity(&self) -> f32 {
        if self.nodes.len() < 2 {
            return 0.0;
        }
        let (adj, _) = self.to_adjacency_matrix();
        crate::spectral::graph_ops::algebraic_connectivity(&adj)
    }

    /// Prune redundant edges using spectral sparsification.
    ///
    /// Removes edges that are structurally redundant (many alternative paths
    /// exist) while preserving edges that are critical for connectivity
    /// (bridges, bottlenecks).
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Approximation quality. 0.3 = aggressive pruning (~30% edges
    ///   removed), 0.1 = conservative (~10% removed). The sparsified graph
    ///   preserves spectral properties within (1 ± epsilon) of the original.
    #[cfg(feature = "spectral")]
    pub fn sparsify(&mut self, epsilon: f32) {
        if self.nodes.len() < 4 {
            return; // too small to benefit
        }

        let (adj, names) = self.to_adjacency_matrix();
        let sparse_adj = crate::spectral::graph_ops::sparsify(&adj, epsilon);

        let idx: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        // Rebuild edges: keep only those present in the sparsified adjacency
        let mut new_edges = Vec::new();
        let mut new_adjacency: HashMap<String, Vec<usize>> = HashMap::new();

        // Initialize adjacency lists
        for name in self.nodes.keys() {
            new_adjacency.insert(name.clone(), Vec::new());
        }

        for edge in &self.edges {
            if let (Some(&i), Some(&j)) = (idx.get(edge.from.as_str()), idx.get(edge.to.as_str()))
                && sparse_adj[[i, j]] > 0.0
            {
                let edge_idx = new_edges.len();
                if let Some(adj_list) = new_adjacency.get_mut(&edge.from) {
                    adj_list.push(edge_idx);
                }
                if let Some(adj_list) = new_adjacency.get_mut(&edge.to) {
                    adj_list.push(edge_idx);
                }
                new_edges.push(edge.clone());
            }
        }

        self.edges = new_edges;
        self.adjacency = new_adjacency;
    }
}

impl brainwires_core::graph::RelationshipGraphT for RelationshipGraph {
    fn get_node(&self, name: &str) -> Option<&GraphNode> {
        self.nodes.get(name)
    }

    fn get_neighbors(&self, name: &str) -> Vec<&GraphNode> {
        RelationshipGraph::get_neighbors(self, name)
    }

    fn get_edges(&self, name: &str) -> Vec<&GraphEdge> {
        RelationshipGraph::get_edges(self, name)
    }

    fn search(&self, query: &str, limit: usize) -> Vec<&GraphNode> {
        RelationshipGraph::search(self, query, limit)
    }

    fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        RelationshipGraph::find_path(self, from, to)
    }
}

/// Entity impacted by changes to another entity
#[derive(Debug, Clone)]
pub struct ImpactedEntity {
    /// Entity name.
    pub name: String,
    /// Entity type.
    pub entity_type: EntityType,
    /// Graph distance from the change source.
    pub distance: usize,
    /// Computed impact score.
    pub impact_score: f32,
    /// Path of entities from source to this entity.
    pub impact_path: Vec<String>,
}

/// A cluster of related entities
#[derive(Debug)]
pub struct EntityCluster {
    /// Cluster identifier.
    pub id: usize,
    /// Nodes in this cluster.
    pub nodes: Vec<GraphNode>,
    /// Average importance of nodes.
    pub avg_importance: f32,
    /// Most common entity type in the cluster.
    pub dominant_type: Option<EntityType>,
}

/// A suggested related entity
#[derive(Debug)]
pub struct SuggestedEntity {
    /// Entity name.
    pub name: String,
    /// Entity type.
    pub entity_type: EntityType,
    /// How relevant this suggestion is.
    pub relevance_score: f32,
    /// Why this entity was suggested.
    pub reason: String,
}

/// Context gathered for an entity
#[derive(Debug)]
pub struct EntityContext {
    /// Root entity name.
    pub root: String,
    /// Entities related to the root.
    pub related_entities: Vec<RelatedEntity>,
    /// Message IDs relevant to this context.
    pub message_ids: HashSet<String>,
}

/// A related entity with relationship info
#[derive(Debug)]
pub struct RelatedEntity {
    /// Entity name.
    pub name: String,
    /// Entity type.
    pub entity_type: EntityType,
    /// Type of relationship to the root.
    pub relationship: EdgeType,
    /// Graph distance from the root.
    pub distance: usize,
    /// Relevance score.
    pub relevance: f32,
}

/// Graph statistics
#[derive(Debug)]
pub struct GraphStats {
    /// Total number of nodes.
    pub node_count: usize,
    /// Total number of edges.
    pub edge_count: usize,
    /// Node counts grouped by entity type.
    pub nodes_by_type: HashMap<&'static str, usize>,
    /// Edge counts grouped by edge type.
    pub edges_by_type: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> RelationshipGraph {
        let mut graph = RelationshipGraph::new();

        // Add nodes
        graph.add_node(GraphNode {
            entity_name: "src/main.rs".to_string(),
            entity_type: EntityType::File,
            message_ids: vec!["msg1".to_string(), "msg2".to_string()],
            mention_count: 5,
            importance: 0.8,
        });

        graph.add_node(GraphNode {
            entity_name: "main".to_string(),
            entity_type: EntityType::Function,
            message_ids: vec!["msg1".to_string()],
            mention_count: 2,
            importance: 0.6,
        });

        graph.add_node(GraphNode {
            entity_name: "Config".to_string(),
            entity_type: EntityType::Type,
            message_ids: vec!["msg2".to_string()],
            mention_count: 3,
            importance: 0.7,
        });

        // Add edges
        graph.add_edge(GraphEdge {
            from: "src/main.rs".to_string(),
            to: "main".to_string(),
            edge_type: EdgeType::Contains,
            weight: 0.9,
            message_id: Some("msg1".to_string()),
        });

        graph.add_edge(GraphEdge {
            from: "main".to_string(),
            to: "Config".to_string(),
            edge_type: EdgeType::References,
            weight: 0.6,
            message_id: Some("msg2".to_string()),
        });

        graph
    }

    #[test]
    fn test_add_and_get_node() {
        let graph = create_test_graph();

        let node = graph.get_node("src/main.rs");
        assert!(node.is_some());
        assert_eq!(node.unwrap().mention_count, 5);
    }

    #[test]
    fn test_get_neighbors() {
        let graph = create_test_graph();

        let neighbors = graph.get_neighbors("src/main.rs");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].entity_name, "main");
    }

    #[test]
    fn test_find_path() {
        let graph = create_test_graph();

        let path = graph.find_path("src/main.rs", "Config");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "src/main.rs");
        assert_eq!(path[2], "Config");
    }

    #[test]
    fn test_get_entity_context() {
        let graph = create_test_graph();

        let context = graph.get_entity_context("src/main.rs", 2);
        assert_eq!(context.root, "src/main.rs");
        assert!(!context.related_entities.is_empty());
        assert!(!context.message_ids.is_empty());
    }

    #[test]
    fn test_search() {
        let graph = create_test_graph();

        let results = graph.search("main", 5);
        assert!(!results.is_empty());
        // Should find both main function and src/main.rs
        assert!(results.iter().any(|n| n.entity_name == "main"));
    }

    #[test]
    fn test_graph_stats() {
        let graph = create_test_graph();

        let stats = graph.stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
    }

    #[test]
    fn test_empty_path() {
        let graph = create_test_graph();

        // Add disconnected node
        let mut graph = graph;
        graph.add_node(GraphNode {
            entity_name: "isolated".to_string(),
            entity_type: EntityType::Concept,
            message_ids: vec![],
            mention_count: 1,
            importance: 0.1,
        });

        let path = graph.find_path("src/main.rs", "isolated");
        assert!(path.is_none());
    }

    // ============ SEAL Integration Tests ============

    #[test]
    fn test_get_impact_set() {
        let graph = create_test_graph();

        let impacts = graph.get_impact_set("src/main.rs", 2);
        assert!(!impacts.is_empty());

        // Should find main and Config
        let names: Vec<_> = impacts.iter().map(|i| i.name.as_str()).collect();
        assert!(names.contains(&"main"));
    }

    #[test]
    fn test_get_impact_set_empty() {
        let graph = create_test_graph();

        // Non-existent entity should return empty
        let impacts = graph.get_impact_set("nonexistent", 2);
        assert!(impacts.is_empty());
    }

    #[test]
    fn test_find_clusters() {
        let mut graph = create_test_graph();

        // Add a disconnected node to create a second cluster
        graph.add_node(GraphNode {
            entity_name: "isolated".to_string(),
            entity_type: EntityType::Concept,
            message_ids: vec![],
            mention_count: 1,
            importance: 0.1,
        });

        let clusters = graph.find_clusters();
        assert_eq!(clusters.len(), 2);

        // First cluster should be the larger connected one
        assert_eq!(clusters[0].nodes.len(), 3);
        assert_eq!(clusters[1].nodes.len(), 1);
    }

    #[test]
    fn test_suggest_related() {
        let graph = create_test_graph();

        let suggestions = graph.suggest_related(&["src/main.rs"]);

        // Should suggest main (direct neighbor)
        let suggested_names: Vec<_> = suggestions.iter().map(|s| s.name.as_str()).collect();
        assert!(suggested_names.contains(&"main"));
    }

    #[test]
    fn test_get_central_nodes() {
        let graph = create_test_graph();

        let central = graph.get_central_nodes(2);
        assert!(!central.is_empty());

        // main.rs should be among the most central (has edges)
        let names: Vec<_> = central.iter().map(|n| n.entity_name.as_str()).collect();
        assert!(names.contains(&"src/main.rs") || names.contains(&"main"));
    }
}
