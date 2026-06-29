/**
 * OpenAPI Tool Generation -- Automatically create tools from OpenAPI 3.x specs.
 *
 * Parses OpenAPI specifications and generates Tool definitions that can
 * be registered in a ToolRegistry and executed by agents.
 *
 * Equivalent to Rust's `rullama_tool_system::openapi` module.
 */

// deno-lint-ignore-file no-explicit-any

import type { Tool } from "@rullama/core";
import { ToolResult } from "@rullama/core";

// -- Public types -------------------------------------------------------------

/** HTTP method for an OpenAPI endpoint. */
export type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

/** A single parameter from an OpenAPI spec. */
export interface OpenApiParam {
  /** Parameter name. */
  name: string;
  /** Parameter description. */
  description?: string;
  /** Whether the parameter is required. */
  required: boolean;
  /** JSON Schema type (e.g., "string", "integer"). */
  schemaType: string;
}

/** HTTP endpoint details extracted from an OpenAPI spec. */
export interface OpenApiEndpoint {
  /** HTTP method. */
  method: HttpMethod;
  /** URL path template (e.g., "/users/{id}"). */
  path: string;
  /** Base URL for the API. */
  baseUrl: string;
  /** Path parameters. */
  pathParams: OpenApiParam[];
  /** Query parameters. */
  queryParams: OpenApiParam[];
  /** Header parameters. */
  headerParams: OpenApiParam[];
  /** Whether a request body is expected. */
  hasBody: boolean;
}

/** A parsed OpenAPI endpoint with its corresponding tool definition. */
export interface OpenApiToolDef {
  /** The generated tool definition for AI consumption. */
  tool: Tool;
  /** The endpoint details for HTTP execution. */
  endpoint: OpenApiEndpoint;
}

// -- Parsing ------------------------------------------------------------------

/**
 * Parse an OpenAPI 3.x spec object and generate Tool definitions.
 *
 * Each endpoint (method + path combination) becomes a separate tool.
 * Tool names are derived from `operationId` if present, or generated
 * from the HTTP method and path.
 */
export function openApiToTools(spec: any): Tool[] {
  if (!spec || typeof spec !== "object") {
    throw new Error("Invalid OpenAPI spec: expected an object");
  }

  const baseUrl = spec.servers?.[0]?.url?.replace(/\/+$/, "") ?? "";
  const tools: Tool[] = [];

  const paths = spec.paths ?? {};
  for (const [path, pathItem] of Object.entries<any>(paths)) {
    if (!pathItem || typeof pathItem !== "object") continue;

    const methods: [HttpMethod, string][] = [
      ["GET", "get"],
      ["POST", "post"],
      ["PUT", "put"],
      ["PATCH", "patch"],
      ["DELETE", "delete"],
    ];

    for (const [method, key] of methods) {
      const operation = pathItem[key];
      if (!operation) continue;

      const parsed = parseOperation(baseUrl, path, method, pathItem, operation);
      if (parsed) {
        tools.push(parsed.tool);
        // Stash endpoint info in tool metadata so executeOpenApiTool can use it
        (parsed.tool as any).__openapi_endpoint = parsed.endpoint;
      }
    }
  }

  return tools;
}

/**
 * Parse an OpenAPI 3.x spec object and generate OpenApiToolDef entries
 * containing both the Tool and the endpoint details.
 */
export function openApiToToolDefs(spec: any): OpenApiToolDef[] {
  if (!spec || typeof spec !== "object") {
    throw new Error("Invalid OpenAPI spec: expected an object");
  }

  const baseUrl = spec.servers?.[0]?.url?.replace(/\/+$/, "") ?? "";
  const defs: OpenApiToolDef[] = [];

  const paths = spec.paths ?? {};
  for (const [path, pathItem] of Object.entries<any>(paths)) {
    if (!pathItem || typeof pathItem !== "object") continue;

    const methods: [HttpMethod, string][] = [
      ["GET", "get"],
      ["POST", "post"],
      ["PUT", "put"],
      ["PATCH", "patch"],
      ["DELETE", "delete"],
    ];

    for (const [method, key] of methods) {
      const operation = pathItem[key];
      if (!operation) continue;

      const parsed = parseOperation(baseUrl, path, method, pathItem, operation);
      if (parsed) {
        defs.push(parsed);
      }
    }
  }

  return defs;
}

function parseOperation(
  baseUrl: string,
  path: string,
  method: HttpMethod,
  pathItem: any,
  operation: any,
): OpenApiToolDef | null {
  // Generate tool name from operationId or method+path
  const toolName = operation.operationId ??
    `${method.toLowerCase()}_${
      path.replace(/\//g, "_").replace(/[{}]/g, "").replace(/^_+|_+$/g, "")
    }`;

  // Generate description
  const description = operation.summary ??
    operation.description ??
    `${method} ${path}`;

  // Collect parameters from path-level and operation-level
  const pathParams: OpenApiParam[] = [];
  const queryParams: OpenApiParam[] = [];
  const headerParams: OpenApiParam[] = [];
  const properties: Record<string, any> = {};
  const requiredParams: string[] = [];

  // Process path-level parameters
  for (const param of pathItem.parameters ?? []) {
    processParameter(
      param,
      pathParams,
      queryParams,
      headerParams,
      properties,
      requiredParams,
    );
  }

  // Process operation-level parameters (override path-level)
  for (const param of operation.parameters ?? []) {
    processParameter(
      param,
      pathParams,
      queryParams,
      headerParams,
      properties,
      requiredParams,
    );
  }

  // Check for request body
  const hasBody = !!operation.requestBody;
  if (hasBody) {
    properties["body"] = {
      type: "object",
      description: "Request body (JSON object)",
    };
  }

  const inputSchema: any = { type: "object" };
  if (Object.keys(properties).length > 0) {
    inputSchema.properties = properties;
  }
  if (requiredParams.length > 0) {
    inputSchema.required = requiredParams;
  }

  const tool: Tool = {
    name: toolName,
    description,
    input_schema: inputSchema,
    requires_approval: false,
    defer_loading: true,
    allowed_callers: [],
    input_examples: [],
  };

  const endpoint: OpenApiEndpoint = {
    method,
    path,
    baseUrl,
    pathParams,
    queryParams,
    headerParams,
    hasBody,
  };

  return { tool, endpoint };
}

function processParameter(
  param: any,
  pathParams: OpenApiParam[],
  queryParams: OpenApiParam[],
  headerParams: OpenApiParam[],
  properties: Record<string, any>,
  requiredParams: string[],
): void {
  if (!param || typeof param !== "object" || !param.name) return;

  const name: string = param.name;
  const location: string = param.in;
  // Path params are always required
  const required: boolean = location === "path" ? true : !!param.required;
  const schemaType = extractSchemaType(param.schema);
  const description: string | undefined = param.description;

  const apiParam: OpenApiParam = { name, description, required, schemaType };

  switch (location) {
    case "path":
      pathParams.push(apiParam);
      break;
    case "query":
      queryParams.push(apiParam);
      break;
    case "header":
      headerParams.push(apiParam);
      break;
    default:
      return; // Skip cookie params etc
  }

  // Add to tool input schema properties
  const prop: any = { type: schemaType };
  if (description) {
    prop.description = description;
  }
  properties[name] = prop;

  if (required && !requiredParams.includes(name)) {
    requiredParams.push(name);
  }
}

function extractSchemaType(schema: any): string {
  if (!schema || typeof schema !== "object") return "string";
  return schema.type ?? "string";
}

// -- Execution ----------------------------------------------------------------

/**
 * Execute an OpenAPI-derived tool via fetch().
 *
 * Substitutes path parameters, adds query parameters, attaches headers
 * and request body, and returns the response as a ToolResult.
 */
// deno-lint-ignore require-await
export async function executeOpenApiTool(
  tool: Tool,
  input: any,
  baseUrl?: string,
  headers?: Record<string, string>,
): Promise<ToolResult> {
  // Get endpoint from tool metadata
  const endpoint: OpenApiEndpoint | undefined = (tool as any)
    .__openapi_endpoint;
  if (!endpoint) {
    return ToolResult.error(
      "",
      `Tool "${tool.name}" has no OpenAPI endpoint metadata`,
    );
  }

  return executeOpenApiToolWithEndpoint(endpoint, input, baseUrl, headers);
}

/**
 * Execute an OpenAPI tool using explicit endpoint details.
 */
export async function executeOpenApiToolWithEndpoint(
  endpoint: OpenApiEndpoint,
  input: any,
  baseUrlOverride?: string,
  headers?: Record<string, string>,
): Promise<ToolResult> {
  const effectiveBaseUrl = baseUrlOverride ?? endpoint.baseUrl;

  // Build URL with path parameter substitution
  let urlPath = endpoint.path;
  for (const param of endpoint.pathParams) {
    const value = input?.[param.name];
    if (value !== undefined && value !== null) {
      const valueStr = typeof value === "string" ? value : String(value);
      urlPath = urlPath.replace(
        `{${param.name}}`,
        encodeURIComponent(valueStr),
      );
    } else if (param.required) {
      return ToolResult.error(
        "",
        `Missing required path parameter: ${param.name}`,
      );
    }
  }

  const url = new URL(`${effectiveBaseUrl}${urlPath}`);

  // Add query parameters
  for (const param of endpoint.queryParams) {
    const value = input?.[param.name];
    if (value !== undefined && value !== null) {
      const valueStr = typeof value === "string" ? value : String(value);
      url.searchParams.set(param.name, valueStr);
    } else if (param.required) {
      return ToolResult.error(
        "",
        `Missing required query parameter: ${param.name}`,
      );
    }
  }

  // Build headers
  const fetchHeaders: Record<string, string> = { ...headers };

  for (const param of endpoint.headerParams) {
    const value = input?.[param.name];
    if (value !== undefined && value !== null) {
      fetchHeaders[param.name] = typeof value === "string"
        ? value
        : String(value);
    }
  }

  // Build fetch options
  const fetchOpts: RequestInit = {
    method: endpoint.method,
    headers: fetchHeaders,
  };

  // Attach body for methods that support it
  if (endpoint.hasBody && input?.body !== undefined) {
    fetchOpts.body = JSON.stringify(input.body);
    fetchHeaders["Content-Type"] = "application/json";
  }

  try {
    const response = await fetch(url.toString(), fetchOpts);
    const body = await response.text();

    if (response.ok) {
      return ToolResult.success("", body);
    } else {
      return ToolResult.error(
        "",
        `HTTP ${response.status} ${response.statusText}: ${body}`,
      );
    }
  } catch (err) {
    return ToolResult.error(
      "",
      `HTTP request failed: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
  }
}
