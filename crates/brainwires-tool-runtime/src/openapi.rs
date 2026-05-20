//! OpenAPI Tool Generation — Automatically create tools from OpenAPI 3.x specs
//!
//! Parses OpenAPI specifications and generates [`Tool`] definitions that can
//! be registered in a [`ToolRegistry`] and executed by agents.
//!
//! # Feature Gate
//!
//! This module requires the `openapi` feature:
//!
//! ```toml
//! brainwires-tools = { version = "0.10", features = ["openapi"] }
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use brainwires_tool_runtime::openapi::{openapi_to_tools, execute_openapi_tool, OpenApiAuth};
//!
//! // Parse spec and get tools
//! let spec_json = std::fs::read_to_string("openapi.json")?;
//! let api_tools = openapi_to_tools(&spec_json)?;
//!
//! // Register tools
//! for api_tool in &api_tools {
//!     registry.register(api_tool.tool.clone());
//! }
//!
//! // Execute a tool call
//! let result = execute_openapi_tool(
//!     &api_tools[0],
//!     &args,
//!     &reqwest::Client::new(),
//!     Some(&OpenApiAuth::Bearer("token".into())),
//! ).await?;
//! ```

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use openapiv3::{
    OpenAPI, Operation, Parameter, ParameterSchemaOrContent, PathItem, ReferenceOr, Schema,
    SchemaKind, Type as OApiType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use brainwires_core::{Tool, ToolInputSchema};

// ── Public types ─────────────────────────────────────────────────────────────

/// HTTP method for an OpenAPI endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    /// GET request.
    Get,
    /// POST request.
    Post,
    /// PUT request.
    Put,
    /// PATCH request.
    Patch,
    /// DELETE request.
    Delete,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Patch => write!(f, "PATCH"),
            HttpMethod::Delete => write!(f, "DELETE"),
        }
    }
}

/// Authentication configuration for OpenAPI tool execution.
#[derive(Debug, Clone)]
pub enum OpenApiAuth {
    /// Bearer token authentication.
    Bearer(String),
    /// API key in a header.
    ApiKey {
        /// Header name.
        header: String,
        /// API key value.
        key: String,
    },
    /// HTTP Basic authentication.
    Basic {
        /// Username.
        username: String,
        /// Password.
        password: String,
    },
}

/// A parsed OpenAPI endpoint with its corresponding tool definition.
#[derive(Debug, Clone)]
pub struct OpenApiTool {
    /// The generated tool definition for AI consumption.
    pub tool: Tool,
    /// The endpoint details for HTTP execution.
    pub endpoint: OpenApiEndpoint,
}

/// HTTP endpoint details extracted from an OpenAPI spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiEndpoint {
    /// HTTP method.
    pub method: HttpMethod,
    /// URL path template (e.g., "/users/{id}").
    pub path: String,
    /// Base URL for the API.
    pub base_url: String,
    /// Path parameters.
    pub path_params: Vec<OpenApiParam>,
    /// Query parameters.
    pub query_params: Vec<OpenApiParam>,
    /// Header parameters.
    pub header_params: Vec<OpenApiParam>,
    /// Whether a request body is expected.
    pub has_body: bool,
}

/// A single parameter from an OpenAPI spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiParam {
    /// Parameter name.
    pub name: String,
    /// Parameter description.
    pub description: Option<String>,
    /// Whether the parameter is required.
    pub required: bool,
    /// JSON Schema type (e.g., "string", "integer").
    pub schema_type: String,
}

// ── Parsing ──────────────────────────────────────────────────────────────────

/// Parse an OpenAPI 3.x JSON or YAML spec and generate tool definitions.
///
/// Each endpoint (method + path combination) becomes a separate [`OpenApiTool`].
/// Tool names are derived from `operationId` if present, or generated from
/// the HTTP method and path.
pub fn openapi_to_tools(spec: &str) -> Result<Vec<OpenApiTool>> {
    let openapi: OpenAPI = serde_json::from_str(spec)
        .or_else(|_| serde_yml::from_str(spec))
        .map_err(|e| anyhow!("Failed to parse OpenAPI spec: {}", e))?;

    let base_url = openapi
        .servers
        .first()
        .map(|s| s.url.trim_end_matches('/').to_string())
        .unwrap_or_default();

    let mut tools = Vec::new();

    for (path, path_item) in &openapi.paths.paths {
        if let ReferenceOr::Item(item) = path_item {
            let methods = [
                (HttpMethod::Get, &item.get),
                (HttpMethod::Post, &item.post),
                (HttpMethod::Put, &item.put),
                (HttpMethod::Patch, &item.patch),
                (HttpMethod::Delete, &item.delete),
            ];

            for (method, operation) in methods {
                if let Some(op) = operation
                    && let Some(tool) = parse_operation(&openapi, &base_url, path, method, item, op)
                {
                    tools.push(tool);
                }
            }
        }
    }

    Ok(tools)
}

fn parse_operation(
    _spec: &OpenAPI,
    base_url: &str,
    path: &str,
    method: HttpMethod,
    path_item: &PathItem,
    operation: &Operation,
) -> Option<OpenApiTool> {
    // Generate tool name from operationId or method+path
    let tool_name = operation.operation_id.clone().unwrap_or_else(|| {
        let clean_path = path
            .replace('/', "_")
            .replace(['{', '}'], "")
            .trim_matches('_')
            .to_string();
        format!("{}_{}", method.to_string().to_lowercase(), clean_path)
    });

    // Generate description
    let description = operation
        .summary
        .clone()
        .or_else(|| operation.description.clone())
        .unwrap_or_else(|| format!("{} {}", method, path));

    // Collect parameters from both path-level and operation-level
    let mut path_params = Vec::new();
    let mut query_params = Vec::new();
    let mut header_params = Vec::new();
    let mut properties: HashMap<String, Value> = HashMap::new();
    let mut required_params: Vec<String> = Vec::new();

    // Process path-level parameters
    for param_ref in &path_item.parameters {
        if let ReferenceOr::Item(param) = param_ref {
            process_parameter(
                param,
                &mut path_params,
                &mut query_params,
                &mut header_params,
                &mut properties,
                &mut required_params,
            );
        }
    }

    // Process operation-level parameters (override path-level)
    for param_ref in &operation.parameters {
        if let ReferenceOr::Item(param) = param_ref {
            process_parameter(
                param,
                &mut path_params,
                &mut query_params,
                &mut header_params,
                &mut properties,
                &mut required_params,
            );
        }
    }

    // Check for request body
    let has_body = operation.request_body.is_some();
    if has_body {
        properties.insert(
            "body".to_string(),
            json!({
                "type": "object",
                "description": "Request body (JSON object)"
            }),
        );
    }

    let input_schema = ToolInputSchema {
        schema_type: "object".to_string(),
        properties: if properties.is_empty() {
            None
        } else {
            Some(properties)
        },
        required: if required_params.is_empty() {
            None
        } else {
            Some(required_params)
        },
    };

    let tool = Tool {
        name: tool_name,
        description,
        input_schema,
        requires_approval: false,
        defer_loading: true, // Lazy-load by default
        allowed_callers: Vec::new(),
        input_examples: Vec::new(),
        serialize: false,
    };

    let endpoint = OpenApiEndpoint {
        method,
        path: path.to_string(),
        base_url: base_url.to_string(),
        path_params,
        query_params,
        header_params,
        has_body,
    };

    Some(OpenApiTool { tool, endpoint })
}

fn process_parameter(
    param: &Parameter,
    path_params: &mut Vec<OpenApiParam>,
    query_params: &mut Vec<OpenApiParam>,
    header_params: &mut Vec<OpenApiParam>,
    properties: &mut HashMap<String, Value>,
    required_params: &mut Vec<String>,
) {
    let (name, required, location, schema_or_content, description) = match param {
        Parameter::Query {
            parameter_data,
            style: _,
            allow_reserved: _,
            allow_empty_value: _,
        } => (
            &parameter_data.name,
            parameter_data.required,
            "query",
            &parameter_data.format,
            &parameter_data.description,
        ),
        Parameter::Header {
            parameter_data,
            style: _,
        } => (
            &parameter_data.name,
            parameter_data.required,
            "header",
            &parameter_data.format,
            &parameter_data.description,
        ),
        Parameter::Path {
            parameter_data,
            style: _,
        } => {
            (
                &parameter_data.name,
                true, // Path params are always required
                "path",
                &parameter_data.format,
                &parameter_data.description,
            )
        }
        Parameter::Cookie { .. } => return, // Skip cookie params
    };

    let schema_type = extract_schema_type(schema_or_content);

    let api_param = OpenApiParam {
        name: name.clone(),
        description: description.clone(),
        required,
        schema_type: schema_type.clone(),
    };

    match location {
        "path" => path_params.push(api_param),
        "query" => query_params.push(api_param),
        "header" => header_params.push(api_param),
        _ => {}
    }

    // Add to tool input schema properties
    let mut prop = json!({ "type": schema_type });
    if let Some(desc) = description {
        prop["description"] = json!(desc);
    }
    properties.insert(name.clone(), prop);

    if required && !required_params.contains(name) {
        required_params.push(name.clone());
    }
}

fn extract_schema_type(format: &ParameterSchemaOrContent) -> String {
    match format {
        ParameterSchemaOrContent::Schema(schema_ref) => {
            if let ReferenceOr::Item(schema) = schema_ref {
                schema_to_type_string(schema)
            } else {
                "string".to_string()
            }
        }
        ParameterSchemaOrContent::Content(_) => "string".to_string(),
    }
}

fn schema_to_type_string(schema: &Schema) -> String {
    match &schema.schema_kind {
        SchemaKind::Type(t) => match t {
            OApiType::String(_) => "string".to_string(),
            OApiType::Number(_) => "number".to_string(),
            OApiType::Integer(_) => "integer".to_string(),
            OApiType::Boolean(_) => "boolean".to_string(),
            OApiType::Array(_) => "array".to_string(),
            OApiType::Object(_) => "object".to_string(),
        },
        _ => "string".to_string(),
    }
}

// ── Execution ────────────────────────────────────────────────────────────────

/// Execute an OpenAPI tool by making the HTTP request.
///
/// Substitutes path parameters, adds query parameters, attaches auth,
/// and returns the response body as a string.
pub async fn execute_openapi_tool(
    api_tool: &OpenApiTool,
    args: &Value,
    client: &reqwest::Client,
    auth: Option<&OpenApiAuth>,
) -> Result<String> {
    let endpoint = &api_tool.endpoint;

    // Build URL with path parameter substitution
    let mut url_path = endpoint.path.clone();
    for param in &endpoint.path_params {
        if let Some(value) = args.get(&param.name) {
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            url_path = url_path.replace(&format!("{{{}}}", param.name), &value_str);
        } else if param.required {
            return Err(anyhow!("Missing required path parameter: {}", param.name));
        }
    }

    let url = format!("{}{}", endpoint.base_url, url_path);
    let mut request = match endpoint.method {
        HttpMethod::Get => client.get(&url),
        HttpMethod::Post => client.post(&url),
        HttpMethod::Put => client.put(&url),
        HttpMethod::Patch => client.patch(&url),
        HttpMethod::Delete => client.delete(&url),
    };

    // Add query parameters
    let mut query_pairs: Vec<(String, String)> = Vec::new();
    for param in &endpoint.query_params {
        if let Some(value) = args.get(&param.name) {
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            query_pairs.push((param.name.clone(), value_str));
        } else if param.required {
            return Err(anyhow!("Missing required query parameter: {}", param.name));
        }
    }
    if !query_pairs.is_empty() {
        request = request.query(&query_pairs);
    }

    // Add header parameters
    for param in &endpoint.header_params {
        if let Some(value) = args.get(&param.name) {
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            request = request.header(&param.name, &value_str);
        }
    }

    // Add request body
    if endpoint.has_body
        && let Some(body) = args.get("body")
    {
        request = request.json(body);
    }

    // Add authentication
    if let Some(auth) = auth {
        request = match auth {
            OpenApiAuth::Bearer(token) => request.bearer_auth(token),
            OpenApiAuth::ApiKey { header, key } => request.header(header.as_str(), key.as_str()),
            OpenApiAuth::Basic { username, password } => {
                request.basic_auth(username, Some(password))
            }
        };
    }

    // Execute request
    let response = request
        .send()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if status.is_success() {
        Ok(body)
    } else {
        Err(anyhow!(
            "HTTP {} {}: {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or(""),
            body
        ))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn petstore_spec() -> &'static str {
        r#"{
            "openapi": "3.0.0",
            "info": { "title": "Petstore", "version": "1.0.0" },
            "servers": [{ "url": "https://petstore.example.com/v1" }],
            "paths": {
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "summary": "List all pets",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": false,
                                "schema": { "type": "integer" },
                                "description": "How many items to return"
                            }
                        ],
                        "responses": { "200": { "description": "OK" } }
                    },
                    "post": {
                        "operationId": "createPet",
                        "summary": "Create a pet",
                        "requestBody": {
                            "required": true,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": { "type": "string" },
                                            "tag": { "type": "string" }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": { "201": { "description": "Created" } }
                    }
                },
                "/pets/{petId}": {
                    "get": {
                        "operationId": "showPetById",
                        "summary": "Info for a specific pet",
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": true,
                                "schema": { "type": "string" },
                                "description": "The id of the pet"
                            }
                        ],
                        "responses": { "200": { "description": "OK" } }
                    }
                }
            }
        }"#
    }

    #[test]
    fn test_parse_petstore_spec() {
        let tools = openapi_to_tools(petstore_spec()).unwrap();
        assert_eq!(tools.len(), 3);

        // Check listPets
        let list = tools.iter().find(|t| t.tool.name == "listPets").unwrap();
        assert_eq!(list.endpoint.method, HttpMethod::Get);
        assert_eq!(list.endpoint.path, "/pets");
        assert_eq!(list.endpoint.base_url, "https://petstore.example.com/v1");
        assert_eq!(list.endpoint.query_params.len(), 1);
        assert_eq!(list.endpoint.query_params[0].name, "limit");
        assert!(!list.endpoint.query_params[0].required);

        // Check createPet
        let create = tools.iter().find(|t| t.tool.name == "createPet").unwrap();
        assert_eq!(create.endpoint.method, HttpMethod::Post);
        assert!(create.endpoint.has_body);

        // Check showPetById
        let show = tools.iter().find(|t| t.tool.name == "showPetById").unwrap();
        assert_eq!(show.endpoint.method, HttpMethod::Get);
        assert_eq!(show.endpoint.path_params.len(), 1);
        assert_eq!(show.endpoint.path_params[0].name, "petId");
        assert!(show.endpoint.path_params[0].required);
    }

    #[test]
    fn test_tool_schema_generation() {
        let tools = openapi_to_tools(petstore_spec()).unwrap();
        let list = tools.iter().find(|t| t.tool.name == "listPets").unwrap();

        // Should have "limit" in properties
        let props = list.tool.input_schema.properties.as_ref().unwrap();
        assert!(props.contains_key("limit"));
        assert_eq!(props["limit"]["type"], "integer");

        // limit is not required
        assert!(list.tool.input_schema.required.is_none());
    }

    #[test]
    fn test_path_param_required() {
        let tools = openapi_to_tools(petstore_spec()).unwrap();
        let show = tools.iter().find(|t| t.tool.name == "showPetById").unwrap();

        let required = show.tool.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"petId".to_string()));
    }

    #[test]
    fn test_operation_id_fallback() {
        let spec = r#"{
            "openapi": "3.0.0",
            "info": { "title": "Test", "version": "1.0.0" },
            "servers": [{ "url": "https://api.example.com" }],
            "paths": {
                "/users/{id}/posts": {
                    "get": {
                        "summary": "Get user posts",
                        "responses": { "200": { "description": "OK" } }
                    }
                }
            }
        }"#;

        let tools = openapi_to_tools(spec).unwrap();
        assert_eq!(tools.len(), 1);
        // Without operationId, name should be generated from method + path
        assert_eq!(tools[0].tool.name, "get_users_id_posts");
    }

    #[test]
    fn test_empty_spec() {
        let spec = r#"{
            "openapi": "3.0.0",
            "info": { "title": "Empty", "version": "1.0.0" },
            "paths": {}
        }"#;

        let tools = openapi_to_tools(spec).unwrap();
        assert!(tools.is_empty());
    }

    #[test]
    fn test_invalid_spec() {
        let result = openapi_to_tools("not valid json or yaml");
        assert!(result.is_err());
    }
}
