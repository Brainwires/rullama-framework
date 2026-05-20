//! Semantic Query Core Extraction
//!
//! Extracts structured "query cores" from natural language questions.
//! Query cores capture the essential semantics of a question in a format
//! that can be executed against the relationship graph.
//!
//! ## S-Expression Inspired Design
//!
//! Based on the SEAL paper's approach, we extract simplified query structures:
//!
//! - `JOIN(relation, subject, object)` - Traverse a relationship
//! - `AND(expr1, expr2, ...)` - Conjunction of expressions
//! - `OR(expr1, expr2, ...)` - Disjunction of expressions
//! - `FILTER(source, predicate)` - Filter results
//! - `COUNT(expr)` - Count results
//! - `ARGMAX/ARGMIN(expr, property)` - Superlative queries
//!
//! ## Example
//!
//! ```rust,ignore
//! let extractor = QueryCoreExtractor::new();
//!
//! // "What uses main.rs?"
//! let core = extractor.extract(
//!     "What uses main.rs?",
//!     &[("main.rs".to_string(), EntityType::File)]
//! );
//!
//! // Produces: QueryCore {
//! //     question_type: Dependency,
//! //     op: Join {
//! //         relation: DependsOn,
//! //         subject: Variable("?dependent"),
//! //         object: Constant("main.rs", File)
//! //     }
//! // }
//! ```

use brainwires_core::graph::{EdgeType, EntityType, RelationshipGraphT};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// --- LazyLock regex statics for question classification patterns ---

// Definition patterns
static RE_WHAT_IS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)what\s+is\s+(\w+)").expect("valid regex"));
static RE_EXPLAIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)explain\s+(\w+)").expect("valid regex"));

// Location patterns
static RE_WHERE_IS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)where\s+is\s+(.+?)\s*(defined|declared|located)").expect("valid regex")
});
static RE_WHICH_FILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)which\s+file\s+(contains|has|defines)\s+(.+)").expect("valid regex")
});
static RE_FIND_IN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)find\s+(.+?)\s+in").expect("valid regex"));

// Dependency patterns
static RE_WHAT_USES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)what\s+(uses|depends\s+on|calls|imports)\s+(.+)").expect("valid regex")
});
static RE_WHAT_DOES_USE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)what\s+does\s+(.+?)\s+(use|depend\s+on|call|import)").expect("valid regex")
});
static RE_SHOW_DEPS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)show\s+(dependencies|usages)\s+(of|for)\s+(.+)").expect("valid regex")
});

// Count patterns
static RE_HOW_MANY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)how\s+many\s+(.+)").expect("valid regex"));
static RE_COUNT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)count\s+(.+)").expect("valid regex"));

// Superlative patterns
static RE_WHICH_MOST: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)which\s+(.+?)\s+has\s+the\s+(most|least|highest|lowest)").expect("valid regex")
});
static RE_LARGEST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(largest|smallest|biggest)\s+(.+)").expect("valid regex"));

// Enumeration patterns
static RE_LIST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)list\s+(all\s+)?(.+)").expect("valid regex"));
static RE_SHOW: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)show\s+(all\s+)?(.+)").expect("valid regex"));

// Boolean patterns
static RE_DOES_USE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)does\s+(.+?)\s+(use|depend|call|import|contain)\s+(.+)").expect("valid regex")
});
static RE_IS_USED_BY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)is\s+(.+?)\s+(used|called|imported)\s+by\s+(.+)").expect("valid regex")
});

/// Core operations in the query language (S-expression inspired)
#[derive(Debug, Clone)]
pub enum QueryOp {
    /// Join two expressions via a relationship
    Join {
        /// The relationship connecting subject and object.
        relation: RelationType,
        /// The subject expression.
        subject: Box<QueryExpr>,
        /// The object expression.
        object: Box<QueryExpr>,
    },
    /// Logical AND of expressions
    And(Vec<QueryExpr>),
    /// Logical OR of expressions
    Or(Vec<QueryExpr>),
    /// Literal values
    Values(Vec<String>),
    /// Filter results by predicate
    Filter {
        /// The expression to filter.
        source: Box<QueryExpr>,
        /// The filter predicate to apply.
        predicate: FilterPredicate,
    },
    /// Count results
    Count(Box<QueryExpr>),
    /// Superlative query (argmax/argmin)
    Superlative {
        /// The expression to evaluate.
        source: Box<QueryExpr>,
        /// The property to compare.
        property: String,
        /// Whether to find the maximum or minimum.
        direction: SuperlativeDir,
    },
}

/// A query expression (variable, constant, or operation)
#[derive(Debug, Clone)]
pub enum QueryExpr {
    /// A variable binding (e.g., ?file, ?function)
    Variable(String),
    /// A constant value with type
    Constant(String, EntityType),
    /// A complex operation
    Op(QueryOp),
}

impl QueryExpr {
    /// Create a new variable expression
    pub fn var(name: &str) -> Self {
        QueryExpr::Variable(format!("?{}", name.trim_start_matches('?')))
    }

    /// Create a new constant expression
    pub fn constant(value: &str, entity_type: EntityType) -> Self {
        QueryExpr::Constant(value.to_string(), entity_type)
    }

    /// Create a join operation
    pub fn join(relation: RelationType, subject: QueryExpr, object: QueryExpr) -> Self {
        QueryExpr::Op(QueryOp::Join {
            relation,
            subject: Box::new(subject),
            object: Box::new(object),
        })
    }

    /// Create a count operation
    pub fn count(inner: QueryExpr) -> Self {
        QueryExpr::Op(QueryOp::Count(Box::new(inner)))
    }

    /// Check if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, QueryExpr::Variable(_))
    }

    /// Get the variable name if this is a variable
    pub fn as_variable(&self) -> Option<&str> {
        match self {
            QueryExpr::Variable(name) => Some(name),
            _ => None,
        }
    }
}

/// Relation types that map to graph edge types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    /// Containment relationship (e.g., file contains function).
    Contains,
    /// Reference relationship (e.g., function references type).
    References,
    /// Dependency relationship.
    DependsOn,
    /// Modification relationship.
    Modifies,
    /// Definition relationship.
    Defines,
    /// Co-occurrence relationship.
    CoOccurs,
    /// Type annotation relationship.
    HasType,
    /// Error association relationship.
    HasError,
    /// Creation timestamp relationship.
    CreatedAt,
    /// Modification timestamp relationship.
    ModifiedAt,
    /// User-defined relationship type.
    Custom(String),
}

impl RelationType {
    /// Convert to the storage EdgeType
    pub fn to_edge_type(&self) -> Option<EdgeType> {
        match self {
            RelationType::Contains => Some(EdgeType::Contains),
            RelationType::References => Some(EdgeType::References),
            RelationType::DependsOn => Some(EdgeType::DependsOn),
            RelationType::Modifies => Some(EdgeType::Modifies),
            RelationType::Defines => Some(EdgeType::Defines),
            RelationType::CoOccurs => Some(EdgeType::CoOccurs),
            _ => None,
        }
    }

    /// Get the inverse relation (if applicable)
    pub fn inverse(&self) -> Option<RelationType> {
        match self {
            RelationType::Contains => Some(RelationType::Custom("ContainedBy".to_string())),
            RelationType::DependsOn => Some(RelationType::Custom("DependedOnBy".to_string())),
            RelationType::Defines => Some(RelationType::Custom("DefinedBy".to_string())),
            RelationType::Modifies => Some(RelationType::Custom("ModifiedBy".to_string())),
            RelationType::References => Some(RelationType::Custom("ReferencedBy".to_string())),
            _ => None,
        }
    }
}

/// Filter predicates for query results
#[derive(Debug, Clone)]
pub enum FilterPredicate {
    /// Type constraint
    HasType(EntityType),
    /// Name pattern (regex)
    NameMatches(String),
    /// Existence in set
    In(Vec<String>),
    /// Not in set
    NotIn(Vec<String>),
    /// Property comparison
    Property {
        /// The property name.
        name: String,
        /// The comparison operator.
        op: CompareOp,
        /// The value to compare against.
        value: String,
    },
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Contains substring.
    Contains,
    /// Starts with prefix.
    StartsWith,
    /// Ends with suffix.
    EndsWith,
}

/// Direction for superlative queries
#[derive(Debug, Clone)]
pub enum SuperlativeDir {
    /// Find the maximum value.
    Max,
    /// Find the minimum value.
    Min,
}

/// Question type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// "What is X?" - Definition query
    Definition,
    /// "Where is X defined?" - Location query
    Location,
    /// "What uses X?" / "What does X depend on?" - Dependency query
    Dependency,
    /// "How many X?" - Count query
    Count,
    /// "Which X has the most Y?" - Superlative query
    Superlative,
    /// "List all X that Y" - Enumeration query
    Enumeration,
    /// "Does X relate to Y?" - Boolean query
    Boolean,
    /// Complex multi-hop query
    MultiHop,
    /// Unknown question type
    Unknown,
}

/// A complete query core with metadata
#[derive(Debug, Clone)]
pub struct QueryCore {
    /// The question type
    pub question_type: QuestionType,
    /// The root query expression
    pub root: QueryExpr,
    /// Entities mentioned in the query
    pub entities: Vec<(String, EntityType)>,
    /// Original question text
    pub original: String,
    /// Resolved question text (after coreference resolution), if different from original
    pub resolved: Option<String>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl QueryCore {
    /// Create a new query core
    pub fn new(
        question_type: QuestionType,
        root: QueryExpr,
        entities: Vec<(String, EntityType)>,
        original: String,
    ) -> Self {
        Self {
            question_type,
            root,
            entities,
            original,
            resolved: None,
            confidence: 1.0,
        }
    }

    /// Set the confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the resolved query text (after coreference resolution)
    pub fn with_resolved(mut self, resolved: String) -> Self {
        self.resolved = Some(resolved);
        self
    }

    /// Convert to a human-readable string representation
    pub fn to_sexp(&self) -> String {
        Self::expr_to_sexp(&self.root)
    }

    fn expr_to_sexp(expr: &QueryExpr) -> String {
        match expr {
            QueryExpr::Variable(name) => name.clone(),
            QueryExpr::Constant(value, _) => format!("\"{}\"", value),
            QueryExpr::Op(op) => match op {
                QueryOp::Join {
                    relation,
                    subject,
                    object,
                } => {
                    format!(
                        "(JOIN {:?} {} {})",
                        relation,
                        Self::expr_to_sexp(subject),
                        Self::expr_to_sexp(object)
                    )
                }
                QueryOp::And(exprs) => {
                    let inner: Vec<_> = exprs.iter().map(Self::expr_to_sexp).collect();
                    format!("(AND {})", inner.join(" "))
                }
                QueryOp::Or(exprs) => {
                    let inner: Vec<_> = exprs.iter().map(Self::expr_to_sexp).collect();
                    format!("(OR {})", inner.join(" "))
                }
                QueryOp::Values(vals) => {
                    format!("(VALUES {})", vals.join(" "))
                }
                QueryOp::Filter { source, predicate } => {
                    format!("(FILTER {} {:?})", Self::expr_to_sexp(source), predicate)
                }
                QueryOp::Count(inner) => {
                    format!("(COUNT {})", Self::expr_to_sexp(inner))
                }
                QueryOp::Superlative {
                    source,
                    property,
                    direction,
                } => {
                    let dir = match direction {
                        SuperlativeDir::Max => "ARGMAX",
                        SuperlativeDir::Min => "ARGMIN",
                    };
                    format!("({} {} {})", dir, Self::expr_to_sexp(source), property)
                }
            },
        }
    }
}

/// Result of executing a query core
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Result values
    pub values: Vec<QueryResultValue>,
    /// Total count (if count query)
    pub count: Option<usize>,
    /// Whether the query succeeded
    pub success: bool,
    /// Error message (if any)
    pub error: Option<String>,
}

/// A single result value
#[derive(Debug, Clone)]
pub struct QueryResultValue {
    /// The value
    pub value: String,
    /// Entity type (if known)
    pub entity_type: Option<EntityType>,
    /// Score/relevance
    pub score: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for QueryResult {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            count: None,
            success: true,
            error: None,
        }
    }
}

impl QueryResult {
    /// Create an empty successful result
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create an error result
    pub fn error(msg: &str) -> Self {
        Self {
            values: Vec::new(),
            count: None,
            success: false,
            error: Some(msg.to_string()),
        }
    }

    /// Create a result with values
    pub fn with_values(values: Vec<QueryResultValue>) -> Self {
        Self {
            count: Some(values.len()),
            values,
            success: true,
            error: None,
        }
    }
}

/// Question pattern for classification
struct QuestionPattern {
    regex: &'static Regex,
    question_type: QuestionType,
    relation: Option<RelationType>,
}

/// Query core extractor
pub struct QueryCoreExtractor {
    /// Patterns for question classification
    patterns: Vec<QuestionPattern>,
}

impl QueryCoreExtractor {
    /// Create a new query core extractor
    pub fn new() -> Self {
        Self {
            patterns: Self::build_patterns(),
        }
    }

    fn build_patterns() -> Vec<QuestionPattern> {
        vec![
            // Definition patterns
            QuestionPattern {
                regex: &RE_WHAT_IS,
                question_type: QuestionType::Definition,
                relation: Some(RelationType::Defines),
            },
            QuestionPattern {
                regex: &RE_EXPLAIN,
                question_type: QuestionType::Definition,
                relation: Some(RelationType::Defines),
            },
            // Location patterns
            QuestionPattern {
                regex: &RE_WHERE_IS,
                question_type: QuestionType::Location,
                relation: Some(RelationType::Contains),
            },
            QuestionPattern {
                regex: &RE_WHICH_FILE,
                question_type: QuestionType::Location,
                relation: Some(RelationType::Contains),
            },
            QuestionPattern {
                regex: &RE_FIND_IN,
                question_type: QuestionType::Location,
                relation: Some(RelationType::Contains),
            },
            // Dependency patterns
            QuestionPattern {
                regex: &RE_WHAT_USES,
                question_type: QuestionType::Dependency,
                relation: Some(RelationType::DependsOn),
            },
            QuestionPattern {
                regex: &RE_WHAT_DOES_USE,
                question_type: QuestionType::Dependency,
                relation: Some(RelationType::DependsOn),
            },
            QuestionPattern {
                regex: &RE_SHOW_DEPS,
                question_type: QuestionType::Dependency,
                relation: Some(RelationType::DependsOn),
            },
            // Count patterns
            QuestionPattern {
                regex: &RE_HOW_MANY,
                question_type: QuestionType::Count,
                relation: None,
            },
            QuestionPattern {
                regex: &RE_COUNT,
                question_type: QuestionType::Count,
                relation: None,
            },
            // Superlative patterns
            QuestionPattern {
                regex: &RE_WHICH_MOST,
                question_type: QuestionType::Superlative,
                relation: None,
            },
            QuestionPattern {
                regex: &RE_LARGEST,
                question_type: QuestionType::Superlative,
                relation: None,
            },
            // Enumeration patterns
            QuestionPattern {
                regex: &RE_LIST,
                question_type: QuestionType::Enumeration,
                relation: None,
            },
            QuestionPattern {
                regex: &RE_SHOW,
                question_type: QuestionType::Enumeration,
                relation: None,
            },
            // Boolean patterns
            QuestionPattern {
                regex: &RE_DOES_USE,
                question_type: QuestionType::Boolean,
                relation: Some(RelationType::DependsOn),
            },
            QuestionPattern {
                regex: &RE_IS_USED_BY,
                question_type: QuestionType::Boolean,
                relation: Some(RelationType::DependsOn),
            },
        ]
    }

    /// Extract a query core from natural language
    pub fn extract(&self, query: &str, entities: &[(String, EntityType)]) -> Option<QueryCore> {
        // Classify the question
        let (question_type, relation) = self.classify_question(query);

        if question_type == QuestionType::Unknown {
            return None;
        }

        // Find mentioned entities in the query
        let mentioned: Vec<_> = entities
            .iter()
            .filter(|(name, _)| query.to_lowercase().contains(&name.to_lowercase()))
            .cloned()
            .collect();

        // Build the query expression based on question type
        let root = match question_type {
            QuestionType::Definition => {
                if let Some((name, entity_type)) = mentioned.first() {
                    QueryExpr::join(
                        RelationType::Defines,
                        QueryExpr::var("definer"),
                        QueryExpr::constant(name, entity_type.clone()),
                    )
                } else {
                    return None;
                }
            }
            QuestionType::Location => {
                if let Some((name, entity_type)) = mentioned.first() {
                    QueryExpr::join(
                        RelationType::Contains,
                        QueryExpr::var("container"),
                        QueryExpr::constant(name, entity_type.clone()),
                    )
                } else {
                    return None;
                }
            }
            QuestionType::Dependency => {
                let rel = relation.unwrap_or(RelationType::DependsOn);
                if let Some((name, entity_type)) = mentioned.first() {
                    // Determine direction based on query wording
                    if query.to_lowercase().contains("what uses")
                        || query.to_lowercase().contains("what depends on")
                    {
                        // X depends on <entity>
                        QueryExpr::join(
                            rel,
                            QueryExpr::var("dependent"),
                            QueryExpr::constant(name, entity_type.clone()),
                        )
                    } else {
                        // <entity> depends on X
                        QueryExpr::join(
                            rel,
                            QueryExpr::constant(name, entity_type.clone()),
                            QueryExpr::var("dependency"),
                        )
                    }
                } else {
                    return None;
                }
            }
            QuestionType::Count => {
                // Count query - wrap in COUNT
                if let Some((name, entity_type)) = mentioned.first() {
                    QueryExpr::count(QueryExpr::join(
                        RelationType::Contains,
                        QueryExpr::var("container"),
                        QueryExpr::constant(name, entity_type.clone()),
                    ))
                } else {
                    // General count query
                    QueryExpr::count(QueryExpr::var("entity"))
                }
            }
            QuestionType::Superlative => {
                // Superlative - determine property and direction
                let direction = if query.to_lowercase().contains("most")
                    || query.to_lowercase().contains("largest")
                    || query.to_lowercase().contains("highest")
                {
                    SuperlativeDir::Max
                } else {
                    SuperlativeDir::Min
                };

                QueryExpr::Op(QueryOp::Superlative {
                    source: Box::new(QueryExpr::var("entity")),
                    property: "mention_count".to_string(),
                    direction,
                })
            }
            QuestionType::Enumeration => {
                // List all entities of a type
                if let Some((name, entity_type)) = mentioned.first() {
                    QueryExpr::join(
                        RelationType::Contains,
                        QueryExpr::var("container"),
                        QueryExpr::constant(name, entity_type.clone()),
                    )
                } else {
                    QueryExpr::var("entity")
                }
            }
            QuestionType::Boolean => {
                if mentioned.len() >= 2 {
                    let rel = relation.unwrap_or(RelationType::DependsOn);
                    QueryExpr::join(
                        rel,
                        QueryExpr::constant(&mentioned[0].0, mentioned[0].1.clone()),
                        QueryExpr::constant(&mentioned[1].0, mentioned[1].1.clone()),
                    )
                } else {
                    return None;
                }
            }
            QuestionType::MultiHop | QuestionType::Unknown => {
                return None;
            }
        };

        Some(QueryCore::new(
            question_type,
            root,
            mentioned,
            query.to_string(),
        ))
    }

    /// Classify a question by type
    pub fn classify_question(&self, query: &str) -> (QuestionType, Option<RelationType>) {
        for pattern in &self.patterns {
            if pattern.regex.is_match(query) {
                return (pattern.question_type.clone(), pattern.relation.clone());
            }
        }
        (QuestionType::Unknown, None)
    }
}

impl Default for QueryCoreExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Query executor for running query cores against a relationship graph
pub struct QueryExecutor<'a> {
    graph: &'a dyn RelationshipGraphT,
}

impl<'a> QueryExecutor<'a> {
    /// Create a new query executor
    pub fn new(graph: &'a dyn RelationshipGraphT) -> Self {
        Self { graph }
    }

    /// Execute a query core
    pub fn execute(&self, query: &QueryCore) -> QueryResult {
        self.execute_expr(&query.root)
    }

    fn execute_expr(&self, expr: &QueryExpr) -> QueryResult {
        match expr {
            QueryExpr::Variable(_) => {
                // Return all entities as candidates
                let values: Vec<_> = self
                    .graph
                    .search("", 100)
                    .iter()
                    .map(|node| QueryResultValue {
                        value: node.entity_name.clone(),
                        entity_type: Some(node.entity_type.clone()),
                        score: node.importance,
                        metadata: HashMap::new(),
                    })
                    .collect();
                QueryResult::with_values(values)
            }
            QueryExpr::Constant(value, entity_type) => {
                // Return the constant as a single result
                QueryResult::with_values(vec![QueryResultValue {
                    value: value.clone(),
                    entity_type: Some(entity_type.clone()),
                    score: 1.0,
                    metadata: HashMap::new(),
                }])
            }
            QueryExpr::Op(op) => self.execute_op(op),
        }
    }

    fn execute_op(&self, op: &QueryOp) -> QueryResult {
        match op {
            QueryOp::Join {
                relation,
                subject,
                object,
            } => {
                // Execute the join
                let edge_type = relation.to_edge_type();

                // Determine which side is the variable
                if let QueryExpr::Constant(name, _) = object.as_ref() {
                    // Find entities that have a relationship to this constant
                    let neighbors = self.graph.get_neighbors(name);
                    let edges = self.graph.get_edges(name);

                    let values: Vec<_> = neighbors
                        .iter()
                        .zip(edges.iter())
                        .filter(|(_, edge)| {
                            edge_type.as_ref().is_none_or(|et| edge.edge_type == *et)
                        })
                        .map(|(node, edge)| QueryResultValue {
                            value: node.entity_name.clone(),
                            entity_type: Some(node.entity_type.clone()),
                            score: edge.weight,
                            metadata: HashMap::new(),
                        })
                        .collect();

                    QueryResult::with_values(values)
                } else if let QueryExpr::Constant(name, _) = subject.as_ref() {
                    // Find entities that this constant has a relationship with
                    let neighbors = self.graph.get_neighbors(name);
                    let edges = self.graph.get_edges(name);

                    let values: Vec<_> = neighbors
                        .iter()
                        .zip(edges.iter())
                        .filter(|(_, edge)| {
                            edge_type.as_ref().is_none_or(|et| edge.edge_type == *et)
                        })
                        .map(|(node, edge)| QueryResultValue {
                            value: node.entity_name.clone(),
                            entity_type: Some(node.entity_type.clone()),
                            score: edge.weight,
                            metadata: HashMap::new(),
                        })
                        .collect();

                    QueryResult::with_values(values)
                } else {
                    // Both are variables - return all edges
                    QueryResult::empty()
                }
            }
            QueryOp::And(exprs) => {
                // Intersection of results
                let mut results: Option<Vec<QueryResultValue>> = None;

                for expr in exprs {
                    let result = self.execute_expr(expr);
                    if !result.success {
                        return result;
                    }

                    if let Some(ref mut existing) = results {
                        let new_values: std::collections::HashSet<_> =
                            result.values.iter().map(|v| v.value.clone()).collect();
                        existing.retain(|v| new_values.contains(&v.value));
                    } else {
                        results = Some(result.values);
                    }
                }

                QueryResult::with_values(results.unwrap_or_default())
            }
            QueryOp::Or(exprs) => {
                // Union of results
                let mut values = Vec::new();
                let mut seen = std::collections::HashSet::new();

                for expr in exprs {
                    let result = self.execute_expr(expr);
                    for v in result.values {
                        if seen.insert(v.value.clone()) {
                            values.push(v);
                        }
                    }
                }

                QueryResult::with_values(values)
            }
            QueryOp::Values(vals) => QueryResult::with_values(
                vals.iter()
                    .map(|v| QueryResultValue {
                        value: v.clone(),
                        entity_type: None,
                        score: 1.0,
                        metadata: HashMap::new(),
                    })
                    .collect(),
            ),
            QueryOp::Filter { source, predicate } => {
                let mut result = self.execute_expr(source);

                result.values.retain(|v| match predicate {
                    FilterPredicate::HasType(t) => v.entity_type.as_ref() == Some(t),
                    FilterPredicate::NameMatches(pattern) => Regex::new(pattern)
                        .map(|r| r.is_match(&v.value))
                        .unwrap_or(false),
                    FilterPredicate::In(set) => set.contains(&v.value),
                    FilterPredicate::NotIn(set) => !set.contains(&v.value),
                    FilterPredicate::Property { name, op, value } => {
                        if let Some(prop_value) = v.metadata.get(name) {
                            match op {
                                CompareOp::Eq => prop_value == value,
                                CompareOp::Ne => prop_value != value,
                                CompareOp::Contains => prop_value.contains(value),
                                CompareOp::StartsWith => prop_value.starts_with(value),
                                CompareOp::EndsWith => prop_value.ends_with(value),
                                _ => false, // Lt, Le, Gt, Ge require numeric comparison
                            }
                        } else {
                            false
                        }
                    }
                });

                result.count = Some(result.values.len());
                result
            }
            QueryOp::Count(inner) => {
                let result = self.execute_expr(inner);
                QueryResult {
                    values: Vec::new(),
                    count: Some(result.values.len()),
                    success: result.success,
                    error: result.error,
                }
            }
            QueryOp::Superlative {
                source,
                property: _,
                direction,
            } => {
                let mut result = self.execute_expr(source);

                // Sort by score
                result.values.sort_by(|a, b| match direction {
                    SuperlativeDir::Max => b
                        .score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    SuperlativeDir::Min => a
                        .score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                });

                // Take the top result
                result.values.truncate(1);
                result.count = Some(result.values.len());
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_definition_question() {
        let extractor = QueryCoreExtractor::new();
        let (qtype, _) = extractor.classify_question("What is EntityStore?");
        assert_eq!(qtype, QuestionType::Definition);
    }

    #[test]
    fn test_classify_location_question() {
        let extractor = QueryCoreExtractor::new();
        let (qtype, _) = extractor.classify_question("Where is main defined?");
        assert_eq!(qtype, QuestionType::Location);
    }

    #[test]
    fn test_classify_dependency_question() {
        let extractor = QueryCoreExtractor::new();
        let (qtype, rel) = extractor.classify_question("What uses EntityStore?");
        assert_eq!(qtype, QuestionType::Dependency);
        assert_eq!(rel, Some(RelationType::DependsOn));
    }

    #[test]
    fn test_classify_count_question() {
        let extractor = QueryCoreExtractor::new();
        let (qtype, _) = extractor.classify_question("How many functions are there?");
        assert_eq!(qtype, QuestionType::Count);
    }

    #[test]
    fn test_extract_dependency_query() {
        let extractor = QueryCoreExtractor::new();
        let entities = vec![("main.rs".to_string(), EntityType::File)];

        let core = extractor.extract("What uses main.rs?", &entities);
        assert!(core.is_some());

        let core = core.unwrap();
        assert_eq!(core.question_type, QuestionType::Dependency);

        // Verify the S-expression output
        let sexp = core.to_sexp();
        assert!(sexp.contains("JOIN"));
        assert!(sexp.contains("DependsOn"));
    }

    #[test]
    fn test_extract_location_query() {
        let extractor = QueryCoreExtractor::new();
        let entities = vec![("process_data".to_string(), EntityType::Function)];

        let core = extractor.extract("Where is process_data defined?", &entities);
        assert!(core.is_some());

        let core = core.unwrap();
        assert_eq!(core.question_type, QuestionType::Location);
    }

    #[test]
    fn test_query_expr_helpers() {
        let var = QueryExpr::var("file");
        assert!(var.is_variable());
        assert_eq!(var.as_variable(), Some("?file"));

        let constant = QueryExpr::constant("main.rs", EntityType::File);
        assert!(!constant.is_variable());
        assert!(constant.as_variable().is_none());
    }

    #[test]
    fn test_query_result() {
        let result = QueryResult::with_values(vec![
            QueryResultValue {
                value: "test1".to_string(),
                entity_type: Some(EntityType::File),
                score: 0.9,
                metadata: HashMap::new(),
            },
            QueryResultValue {
                value: "test2".to_string(),
                entity_type: Some(EntityType::Function),
                score: 0.8,
                metadata: HashMap::new(),
            },
        ]);

        assert!(result.success);
        assert_eq!(result.count, Some(2));
        assert_eq!(result.values.len(), 2);
    }

    #[test]
    fn test_query_result_error() {
        let result = QueryResult::error("Entity not found");
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_relation_type_inverse() {
        assert!(RelationType::Contains.inverse().is_some());
        assert!(RelationType::DependsOn.inverse().is_some());
        assert!(RelationType::CoOccurs.inverse().is_none());
    }
}
