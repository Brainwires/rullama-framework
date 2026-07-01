/** Permission mode for tool execution.
 * Equivalent to Rust's `PermissionMode` in rullama-core. */
export type PermissionMode = "read-only" | "auto" | "full";

/** Parse a PermissionMode from a string. */
export function parsePermissionMode(s: string): PermissionMode | undefined {
  switch (s.toLowerCase()) {
    case "read-only":
    case "readonly":
      return "read-only";
    case "auto":
      return "auto";
    case "full":
      return "full";
    default:
      return undefined;
  }
}

/** Default permission mode. */
export const DEFAULT_PERMISSION_MODE: PermissionMode = "auto";
