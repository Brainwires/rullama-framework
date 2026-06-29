/** Error kind discriminator for FrameworkError. */
export type FrameworkErrorKind =
  | { type: "config"; message: string }
  | { type: "provider"; message: string }
  | { type: "provider_auth"; provider: string; message: string }
  | { type: "provider_model"; provider: string; model: string; message: string }
  | { type: "embedding_dimension"; expected: number; got: number }
  | { type: "tool_execution"; message: string }
  | { type: "agent"; message: string }
  | { type: "storage"; message: string }
  | { type: "storage_schema"; store: string; message: string }
  | { type: "training_config"; parameter: string; message: string }
  | { type: "permission_denied"; message: string }
  | { type: "serialization"; message: string }
  | { type: "other"; message: string };

/** Core framework error with typed error kinds.
 * Equivalent to Rust's `FrameworkError` in rullama-core. */
export class FrameworkError extends Error {
  readonly kind: FrameworkErrorKind;

  constructor(kind: FrameworkErrorKind) {
    // Build message from kind
    let msg: string;
    switch (kind.type) {
      case "config":
        msg = `Configuration error: ${kind.message}`;
        break;
      case "provider":
        msg = `Provider error: ${kind.message}`;
        break;
      case "provider_auth":
        msg =
          `Provider authentication failed for ${kind.provider}: ${kind.message}`;
        break;
      case "provider_model":
        msg =
          `Provider model error (${kind.provider}/${kind.model}): ${kind.message}`;
        break;
      case "embedding_dimension":
        msg =
          `Embedding dimension mismatch: expected ${kind.expected}, got ${kind.got}`;
        break;
      case "tool_execution":
        msg = `Tool execution error: ${kind.message}`;
        break;
      case "agent":
        msg = `Agent error: ${kind.message}`;
        break;
      case "storage":
        msg = `Storage error: ${kind.message}`;
        break;
      case "storage_schema":
        msg = `Storage schema error in ${kind.store}: ${kind.message}`;
        break;
      case "training_config":
        msg =
          `Training configuration error for ${kind.parameter}: ${kind.message}`;
        break;
      case "permission_denied":
        msg = `Permission denied: ${kind.message}`;
        break;
      case "serialization":
        msg = `Serialization error: ${kind.message}`;
        break;
      case "other":
        msg = kind.message;
        break;
    }
    super(msg);
    this.name = "FrameworkError";
    this.kind = kind;
  }

  /** Create a provider authentication error. */
  static providerAuth(provider: string, message: string): FrameworkError {
    return new FrameworkError({ type: "provider_auth", provider, message });
  }

  /** Create a provider model error. */
  static providerModel(
    provider: string,
    model: string,
    message: string,
  ): FrameworkError {
    return new FrameworkError({
      type: "provider_model",
      provider,
      model,
      message,
    });
  }

  /** Create an embedding dimension mismatch error. */
  static embeddingDimension(expected: number, got: number): FrameworkError {
    return new FrameworkError({ type: "embedding_dimension", expected, got });
  }

  /** Create a storage schema error. */
  static storageSchema(store: string, message: string): FrameworkError {
    return new FrameworkError({ type: "storage_schema", store, message });
  }

  /** Create a training configuration error. */
  static trainingConfig(parameter: string, message: string): FrameworkError {
    return new FrameworkError({ type: "training_config", parameter, message });
  }
}
