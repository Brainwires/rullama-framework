//! Result Composition
//!
//! Combines outputs from multiple subtasks into a final result
//! based on the composition function specified during decomposition.

use std::collections::HashMap;

use serde_json::Value;

use super::decomposition::CompositionFunction;
use super::error::{CompositionError, MdapResult};
use super::microagent::SubtaskOutput;

/// Result composer for combining subtask outputs
pub struct Composer {
    /// Optional custom composition handlers
    custom_handlers: HashMap<String, Box<dyn CompositionHandler>>,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new()
    }
}

impl Composer {
    /// Create a new composer
    pub fn new() -> Self {
        Self {
            custom_handlers: HashMap::new(),
        }
    }

    /// Register a custom composition handler
    pub fn register_handler(
        &mut self,
        name: impl Into<String>,
        handler: Box<dyn CompositionHandler>,
    ) {
        self.custom_handlers.insert(name.into(), handler);
    }

    /// Compose results according to the composition function
    pub fn compose(
        &self,
        results: &[SubtaskOutput],
        function: &CompositionFunction,
    ) -> MdapResult<Value> {
        if results.is_empty() {
            return Err(
                CompositionError::MissingResult("No results to compose".to_string()).into(),
            );
        }

        match function {
            CompositionFunction::Identity => Ok(results[0].value.clone()),

            CompositionFunction::Concatenate => self.concatenate(results),

            CompositionFunction::Sequence => self.sequence(results),

            CompositionFunction::ObjectMerge => self.object_merge(results),

            CompositionFunction::LastOnly => Ok(results
                .last()
                .expect("checked non-empty above")
                .value
                .clone()),

            CompositionFunction::Custom(description) => self.custom_compose(results, description),

            CompositionFunction::Reduce { operation } => self.reduce(results, operation),
        }
    }

    /// Concatenate all results as strings
    fn concatenate(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let mut combined = String::new();

        for result in results {
            match &result.value {
                Value::String(s) => {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(s);
                }
                Value::Array(arr) => {
                    for item in arr {
                        if !combined.is_empty() {
                            combined.push('\n');
                        }
                        combined.push_str(&item.to_string());
                    }
                }
                other => {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&other.to_string());
                }
            }
        }

        Ok(Value::String(combined))
    }

    /// Combine results as a sequence (array)
    fn sequence(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let items: Vec<Value> = results.iter().map(|r| r.value.clone()).collect();
        Ok(Value::Array(items))
    }

    /// Merge results into an object with subtask IDs as keys
    fn object_merge(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let mut map = serde_json::Map::new();

        for result in results {
            // If the value is already an object, merge its keys
            if let Value::Object(obj) = &result.value {
                for (k, v) in obj {
                    map.insert(k.clone(), v.clone());
                }
            } else {
                // Use subtask ID as key
                map.insert(result.subtask_id.clone(), result.value.clone());
            }
        }

        Ok(Value::Object(map))
    }

    /// Apply a reduce operation
    fn reduce(&self, results: &[SubtaskOutput], operation: &str) -> MdapResult<Value> {
        let op = operation.to_lowercase();

        match op.as_str() {
            "sum" | "add" => self.reduce_sum(results),
            "multiply" | "product" => self.reduce_product(results),
            "max" => self.reduce_max(results),
            "min" => self.reduce_min(results),
            "and" | "all" => self.reduce_and(results),
            "or" | "any" => self.reduce_or(results),
            "concat" | "join" => self.concatenate(results),
            _ => Err(CompositionError::FunctionNotFound {
                function: format!("reduce:{}", operation),
            }
            .into()),
        }
    }

    /// Sum numeric values
    fn reduce_sum(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let mut sum = 0.0f64;

        for result in results {
            let val = self.extract_number(&result.value)?;
            sum += val;
        }

        Ok(Value::Number(
            serde_json::Number::from_f64(sum)
                .ok_or_else(|| CompositionError::ExecutionFailed("Invalid number".to_string()))?,
        ))
    }

    /// Multiply numeric values
    fn reduce_product(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let mut product = 1.0f64;

        for result in results {
            let val = self.extract_number(&result.value)?;
            product *= val;
        }

        Ok(Value::Number(
            serde_json::Number::from_f64(product)
                .ok_or_else(|| CompositionError::ExecutionFailed("Invalid number".to_string()))?,
        ))
    }

    /// Find maximum value
    fn reduce_max(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let mut max = f64::NEG_INFINITY;

        for result in results {
            let val = self.extract_number(&result.value)?;
            if val > max {
                max = val;
            }
        }

        Ok(Value::Number(
            serde_json::Number::from_f64(max)
                .ok_or_else(|| CompositionError::ExecutionFailed("Invalid number".to_string()))?,
        ))
    }

    /// Find minimum value
    fn reduce_min(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        let mut min = f64::INFINITY;

        for result in results {
            let val = self.extract_number(&result.value)?;
            if val < min {
                min = val;
            }
        }

        Ok(Value::Number(
            serde_json::Number::from_f64(min)
                .ok_or_else(|| CompositionError::ExecutionFailed("Invalid number".to_string()))?,
        ))
    }

    /// Logical AND of boolean values
    fn reduce_and(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        for result in results {
            let val = self.extract_bool(&result.value)?;
            if !val {
                return Ok(Value::Bool(false));
            }
        }
        Ok(Value::Bool(true))
    }

    /// Logical OR of boolean values
    fn reduce_or(&self, results: &[SubtaskOutput]) -> MdapResult<Value> {
        for result in results {
            let val = self.extract_bool(&result.value)?;
            if val {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    }

    /// Extract number from a value
    fn extract_number(&self, value: &Value) -> MdapResult<f64> {
        match value {
            Value::Number(n) => n.as_f64().ok_or_else(|| {
                CompositionError::IncompatibleTypes {
                    type_a: "number".to_string(),
                    type_b: "invalid float".to_string(),
                }
                .into()
            }),
            Value::String(s) => s.parse::<f64>().map_err(|_| {
                CompositionError::IncompatibleTypes {
                    type_a: "number".to_string(),
                    type_b: "string".to_string(),
                }
                .into()
            }),
            _ => Err(CompositionError::IncompatibleTypes {
                type_a: "number".to_string(),
                type_b: format!("{:?}", value),
            }
            .into()),
        }
    }

    /// Extract boolean from a value
    fn extract_bool(&self, value: &Value) -> MdapResult<bool> {
        match value {
            Value::Bool(b) => Ok(*b),
            Value::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" => Ok(true),
                "false" | "no" | "0" => Ok(false),
                _ => Err(CompositionError::IncompatibleTypes {
                    type_a: "bool".to_string(),
                    type_b: "string".to_string(),
                }
                .into()),
            },
            Value::Number(n) => Ok(n.as_f64().map(|f| f != 0.0).unwrap_or(false)),
            _ => Err(CompositionError::IncompatibleTypes {
                type_a: "bool".to_string(),
                type_b: format!("{:?}", value),
            }
            .into()),
        }
    }

    /// Handle custom composition
    fn custom_compose(&self, results: &[SubtaskOutput], description: &str) -> MdapResult<Value> {
        // Check for registered handler
        if let Some(handler) = self.custom_handlers.get(description) {
            return handler.compose(results);
        }

        // Default: just concatenate with the description as context
        let mut composed = serde_json::Map::new();
        composed.insert(
            "composition".to_string(),
            Value::String(description.to_string()),
        );
        composed.insert(
            "results".to_string(),
            Value::Array(results.iter().map(|r| r.value.clone()).collect()),
        );

        Ok(Value::Object(composed))
    }
}

/// Trait for custom composition handlers
pub trait CompositionHandler: Send + Sync {
    /// Compose subtask outputs into a final result.
    fn compose(&self, results: &[SubtaskOutput]) -> MdapResult<Value>;
}

/// Trait for composing decomposition results
pub trait ResultComposer: Send + Sync {
    /// Compose results from all subtasks according to the decomposition
    fn compose(
        &self,
        decomposition: &super::decomposition::DecompositionResult,
        results: &std::collections::HashMap<String, SubtaskOutput>,
    ) -> MdapResult<Value>;
}

/// Standard result composer implementation
pub struct StandardComposer;

impl ResultComposer for StandardComposer {
    fn compose(
        &self,
        decomposition: &super::decomposition::DecompositionResult,
        results: &std::collections::HashMap<String, SubtaskOutput>,
    ) -> MdapResult<Value> {
        // Collect outputs in subtask order
        let outputs: Vec<SubtaskOutput> = decomposition
            .subtasks
            .iter()
            .filter_map(|subtask| results.get(&subtask.id).cloned())
            .collect();

        if outputs.is_empty() {
            return Err(
                CompositionError::MissingResult("No results to compose".to_string()).into(),
            );
        }

        let composer = Composer::new();
        composer.compose(&outputs, &decomposition.composition_function)
    }
}

/// Builder for composition with validation
pub struct CompositionBuilder {
    results: Vec<SubtaskOutput>,
    function: CompositionFunction,
    validate: bool,
}

impl CompositionBuilder {
    /// Create a new composition builder with the given function.
    pub fn new(function: CompositionFunction) -> Self {
        Self {
            results: Vec::new(),
            function,
            validate: true,
        }
    }

    /// Add a single subtask output to the composition.
    pub fn add_result(mut self, result: SubtaskOutput) -> Self {
        self.results.push(result);
        self
    }

    /// Add multiple subtask outputs to the composition.
    pub fn add_results(mut self, results: Vec<SubtaskOutput>) -> Self {
        self.results.extend(results);
        self
    }

    /// Disable input validation before composing.
    pub fn skip_validation(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Execute the composition and return the combined result.
    pub fn compose(self) -> MdapResult<Value> {
        if self.validate {
            self.validate_inputs()?;
        }

        let composer = Composer::new();
        composer.compose(&self.results, &self.function)
    }

    fn validate_inputs(&self) -> MdapResult<()> {
        if self.results.is_empty() {
            return Err(CompositionError::MissingResult("No results provided".to_string()).into());
        }

        // Check for type consistency in reduce operations
        if let CompositionFunction::Reduce { operation } = &self.function {
            let op = operation.to_lowercase();
            if op == "sum" || op == "multiply" || op == "max" || op == "min" {
                // All values should be numeric
                for result in &self.results {
                    if !(result.value.is_number()
                        || result.value.is_string()
                            && result
                                .value
                                .as_str()
                                .map(|s| s.parse::<f64>().is_ok())
                                .unwrap_or(false))
                    {
                        return Err(CompositionError::IncompatibleTypes {
                            type_a: "number".to_string(),
                            type_b: format!("{:?}", result.value),
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_output(id: &str, value: Value) -> SubtaskOutput {
        SubtaskOutput {
            subtask_id: id.to_string(),
            value,
            next_state: None,
        }
    }

    #[test]
    fn test_identity_composition() {
        let composer = Composer::new();
        let results = vec![make_output("a", Value::String("hello".to_string()))];

        let result = composer
            .compose(&results, &CompositionFunction::Identity)
            .unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
    }

    #[test]
    fn test_concatenate_strings() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", Value::String("hello".to_string())),
            make_output("b", Value::String("world".to_string())),
        ];

        let result = composer
            .compose(&results, &CompositionFunction::Concatenate)
            .unwrap();
        assert_eq!(result, Value::String("hello\nworld".to_string()));
    }

    #[test]
    fn test_sequence() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!(1)),
            make_output("b", serde_json::json!(2)),
            make_output("c", serde_json::json!(3)),
        ];

        let result = composer
            .compose(&results, &CompositionFunction::Sequence)
            .unwrap();
        assert_eq!(result, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_object_merge() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!({"x": 1})),
            make_output("b", serde_json::json!({"y": 2})),
        ];

        let result = composer
            .compose(&results, &CompositionFunction::ObjectMerge)
            .unwrap();
        assert_eq!(result, serde_json::json!({"x": 1, "y": 2}));
    }

    #[test]
    fn test_last_only() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!(1)),
            make_output("b", serde_json::json!(2)),
            make_output("c", serde_json::json!(3)),
        ];

        let result = composer
            .compose(&results, &CompositionFunction::LastOnly)
            .unwrap();
        assert_eq!(result, serde_json::json!(3));
    }

    #[test]
    fn test_reduce_sum() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!(10)),
            make_output("b", serde_json::json!(20)),
            make_output("c", serde_json::json!(30)),
        ];

        let result = composer
            .compose(
                &results,
                &CompositionFunction::Reduce {
                    operation: "sum".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, serde_json::json!(60.0));
    }

    #[test]
    fn test_reduce_product() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!(2)),
            make_output("b", serde_json::json!(3)),
            make_output("c", serde_json::json!(4)),
        ];

        let result = composer
            .compose(
                &results,
                &CompositionFunction::Reduce {
                    operation: "product".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, serde_json::json!(24.0));
    }

    #[test]
    fn test_reduce_max() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!(10)),
            make_output("b", serde_json::json!(50)),
            make_output("c", serde_json::json!(30)),
        ];

        let result = composer
            .compose(
                &results,
                &CompositionFunction::Reduce {
                    operation: "max".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, serde_json::json!(50.0));
    }

    #[test]
    fn test_reduce_and() {
        let composer = Composer::new();

        let all_true = vec![
            make_output("a", Value::Bool(true)),
            make_output("b", Value::Bool(true)),
        ];
        let result = composer
            .compose(
                &all_true,
                &CompositionFunction::Reduce {
                    operation: "and".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, Value::Bool(true));

        let some_false = vec![
            make_output("a", Value::Bool(true)),
            make_output("b", Value::Bool(false)),
        ];
        let result = composer
            .compose(
                &some_false,
                &CompositionFunction::Reduce {
                    operation: "and".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_reduce_or() {
        let composer = Composer::new();

        let all_false = vec![
            make_output("a", Value::Bool(false)),
            make_output("b", Value::Bool(false)),
        ];
        let result = composer
            .compose(
                &all_false,
                &CompositionFunction::Reduce {
                    operation: "or".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, Value::Bool(false));

        let some_true = vec![
            make_output("a", Value::Bool(false)),
            make_output("b", Value::Bool(true)),
        ];
        let result = composer
            .compose(
                &some_true,
                &CompositionFunction::Reduce {
                    operation: "or".to_string(),
                },
            )
            .unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_composition_builder() {
        let result = CompositionBuilder::new(CompositionFunction::Sequence)
            .add_result(make_output("a", serde_json::json!(1)))
            .add_result(make_output("b", serde_json::json!(2)))
            .compose()
            .unwrap();

        assert_eq!(result, serde_json::json!([1, 2]));
    }

    #[test]
    fn test_composition_builder_validation() {
        let result = CompositionBuilder::new(CompositionFunction::Reduce {
            operation: "sum".to_string(),
        })
        .add_result(make_output("a", Value::String("not a number".to_string())))
        .compose();

        assert!(result.is_err());
    }

    #[test]
    fn test_empty_results() {
        let composer = Composer::new();
        let results: Vec<SubtaskOutput> = vec![];

        let result = composer.compose(&results, &CompositionFunction::Identity);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_composition() {
        let composer = Composer::new();
        let results = vec![
            make_output("a", serde_json::json!(1)),
            make_output("b", serde_json::json!(2)),
        ];

        let result = composer
            .compose(
                &results,
                &CompositionFunction::Custom("test composition".to_string()),
            )
            .unwrap();

        assert!(result.is_object());
        assert_eq!(result["composition"], "test composition");
    }
}
