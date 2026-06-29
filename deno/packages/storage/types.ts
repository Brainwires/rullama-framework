/**
 * Schema, record, and filter types for the unified database layer.
 *
 * Equivalent to Rust's `databases/types.rs` in rullama-storage.
 * @module
 */

// -- Schema types -----------------------------------------------------------

/** Supported column data types. */
export type FieldType =
  | { kind: "Utf8" }
  | { kind: "Int32" }
  | { kind: "Int64" }
  | { kind: "UInt32" }
  | { kind: "UInt64" }
  | { kind: "Float32" }
  | { kind: "Float64" }
  | { kind: "Boolean" }
  | { kind: "Vector"; dimension: number };

/** Convenience constructors for FieldType. */
export const FieldTypes = {
  Utf8: { kind: "Utf8" } as FieldType,
  Int32: { kind: "Int32" } as FieldType,
  Int64: { kind: "Int64" } as FieldType,
  UInt32: { kind: "UInt32" } as FieldType,
  UInt64: { kind: "UInt64" } as FieldType,
  Float32: { kind: "Float32" } as FieldType,
  Float64: { kind: "Float64" } as FieldType,
  Boolean: { kind: "Boolean" } as FieldType,
  Vector: (dimension: number): FieldType => ({ kind: "Vector", dimension }),
} as const;

/** Definition of a single field within a table schema. */
export interface FieldDef {
  /** Column name. */
  name: string;
  /** Data type. */
  fieldType: FieldType;
  /** Whether null values are permitted. */
  nullable: boolean;
}

/** Shorthand constructor for a non-nullable field. */
export function requiredField(name: string, fieldType: FieldType): FieldDef {
  return { name, fieldType, nullable: false };
}

/** Shorthand constructor for a nullable field. */
export function optionalField(name: string, fieldType: FieldType): FieldDef {
  return { name, fieldType, nullable: true };
}

// -- Record types -----------------------------------------------------------

/** A single typed column value. */
export type FieldValue =
  | { kind: "Utf8"; value: string | null }
  | { kind: "Int32"; value: number | null }
  | { kind: "Int64"; value: number | null }
  | { kind: "UInt32"; value: number | null }
  | { kind: "UInt64"; value: number | null }
  | { kind: "Float32"; value: number | null }
  | { kind: "Float64"; value: number | null }
  | { kind: "Boolean"; value: boolean | null }
  | { kind: "Vector"; value: number[] };

/** Convenience constructors for FieldValue. */
export const FieldValues = {
  Utf8: (value: string | null): FieldValue => ({ kind: "Utf8", value }),
  Int32: (value: number | null): FieldValue => ({ kind: "Int32", value }),
  Int64: (value: number | null): FieldValue => ({ kind: "Int64", value }),
  UInt32: (value: number | null): FieldValue => ({ kind: "UInt32", value }),
  UInt64: (value: number | null): FieldValue => ({ kind: "UInt64", value }),
  Float32: (value: number | null): FieldValue => ({ kind: "Float32", value }),
  Float64: (value: number | null): FieldValue => ({ kind: "Float64", value }),
  Boolean: (value: boolean | null): FieldValue => ({ kind: "Boolean", value }),
  Vector: (value: number[]): FieldValue => ({ kind: "Vector", value }),
} as const;

/** Extract a string from a FieldValue, if it is Utf8 and non-null. */
export function fieldValueAsStr(fv: FieldValue): string | undefined {
  return fv.kind === "Utf8" && fv.value !== null ? fv.value : undefined;
}

/** Extract an i64/number from a FieldValue, if it is Int64 and non-null. */
export function fieldValueAsI64(fv: FieldValue): number | undefined {
  return fv.kind === "Int64" && fv.value !== null ? fv.value : undefined;
}

/** Extract an i32/number from a FieldValue, if it is Int32 and non-null. */
export function fieldValueAsI32(fv: FieldValue): number | undefined {
  return fv.kind === "Int32" && fv.value !== null ? fv.value : undefined;
}

/** Extract a float32 from a FieldValue. */
export function fieldValueAsF32(fv: FieldValue): number | undefined {
  return fv.kind === "Float32" && fv.value !== null ? fv.value : undefined;
}

/** Extract a float64 from a FieldValue. */
export function fieldValueAsF64(fv: FieldValue): number | undefined {
  return fv.kind === "Float64" && fv.value !== null ? fv.value : undefined;
}

/** Extract a boolean from a FieldValue. */
export function fieldValueAsBool(fv: FieldValue): boolean | undefined {
  return fv.kind === "Boolean" && fv.value !== null ? fv.value : undefined;
}

/** Extract a vector from a FieldValue, if non-empty. */
export function fieldValueAsVector(fv: FieldValue): number[] | undefined {
  return fv.kind === "Vector" && fv.value.length > 0 ? fv.value : undefined;
}

/** A generic row: ordered list of (column_name, value) pairs. */
export type Record = [string, FieldValue][];

/** Look up a field in a Record by name. */
export function recordGet(
  record: Record,
  name: string,
): FieldValue | undefined {
  const entry = record.find(([n]) => n === name);
  return entry ? entry[1] : undefined;
}

/** A record returned from vector similarity search, with a relevance score. */
export interface ScoredRecord {
  /** The matched row. */
  record: Record;
  /** Similarity score (higher is better, typically 0.0-1.0). */
  score: number;
}

// -- Filter types -----------------------------------------------------------

/** Structured query filter that backends translate into their native syntax. */
export type Filter =
  | { kind: "Eq"; field: string; value: FieldValue }
  | { kind: "Ne"; field: string; value: FieldValue }
  | { kind: "Lt"; field: string; value: FieldValue }
  | { kind: "Lte"; field: string; value: FieldValue }
  | { kind: "Gt"; field: string; value: FieldValue }
  | { kind: "Gte"; field: string; value: FieldValue }
  | { kind: "NotNull"; field: string }
  | { kind: "IsNull"; field: string }
  | { kind: "In"; field: string; values: FieldValue[] }
  | { kind: "And"; filters: Filter[] }
  | { kind: "Or"; filters: Filter[] }
  | { kind: "Raw"; expression: string };

/** Convenience constructors for Filter. */
export const Filters = {
  Eq: (field: string, value: FieldValue): Filter => ({
    kind: "Eq",
    field,
    value,
  }),
  Ne: (field: string, value: FieldValue): Filter => ({
    kind: "Ne",
    field,
    value,
  }),
  Lt: (field: string, value: FieldValue): Filter => ({
    kind: "Lt",
    field,
    value,
  }),
  Lte: (field: string, value: FieldValue): Filter => ({
    kind: "Lte",
    field,
    value,
  }),
  Gt: (field: string, value: FieldValue): Filter => ({
    kind: "Gt",
    field,
    value,
  }),
  Gte: (field: string, value: FieldValue): Filter => ({
    kind: "Gte",
    field,
    value,
  }),
  NotNull: (field: string): Filter => ({ kind: "NotNull", field }),
  IsNull: (field: string): Filter => ({ kind: "IsNull", field }),
  In: (field: string, values: FieldValue[]): Filter => ({
    kind: "In",
    field,
    values,
  }),
  And: (filters: Filter[]): Filter => ({ kind: "And", filters }),
  Or: (filters: Filter[]): Filter => ({ kind: "Or", filters }),
  Raw: (expression: string): Filter => ({ kind: "Raw", expression }),
} as const;

// -- Backend capabilities ---------------------------------------------------

/** Describes the capabilities of a database backend. */
export interface BackendCapabilities {
  /** Whether the backend supports vector similarity search. */
  vectorSearch: boolean;
}

/** Default capabilities (vector search enabled). */
export function defaultCapabilities(): BackendCapabilities {
  return { vectorSearch: true };
}
