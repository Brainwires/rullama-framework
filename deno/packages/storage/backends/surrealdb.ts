/**
 * SurrealDB backend for StorageBackend.
 *
 * Port of the Rust `rullama-storage/src/databases/surrealdb/mod.rs`.
 *
 * Uses the official SurrealDB JavaScript SDK (`npm:surrealdb`).
 * @module
 */

import Surreal from "surrealdb";
import type { StorageBackend } from "../traits.ts";
import type {
  FieldDef,
  FieldValue,
  Filter,
  Record as BwRecord,
  ScoredRecord,
} from "../types.ts";
import {
  assertIdentifier,
  assertLimit,
  assertRawAllowed,
  type FilterBuildOptions,
} from "./sql_guards.ts";

const DEFAULT_URL = "ws://localhost:8000";

// ---------------------------------------------------------------------------
// SurrealQL helpers
// ---------------------------------------------------------------------------

/** Map a FieldType kind to SurrealQL type string. */
export function fieldTypeToSurrealQL(ft: FieldDef["fieldType"]): string {
  switch (ft.kind) {
    case "Utf8":
      return "string";
    case "Int32":
    case "Int64":
    case "UInt32":
    case "UInt64":
      return "int";
    case "Float32":
    case "Float64":
      return "float";
    case "Boolean":
      return "bool";
    case "Vector":
      return `array<float, ${ft.dimension}>`;
  }
}

/** Convert a FieldValue to a JSON-compatible value for SurrealDB bindings. */
export function fieldValueToJson(fv: FieldValue): unknown {
  switch (fv.kind) {
    case "Utf8":
      return fv.value;
    case "Int32":
    case "Int64":
    case "UInt32":
    case "UInt64":
    case "Float32":
    case "Float64":
      return fv.value;
    case "Boolean":
      return fv.value;
    case "Vector":
      return fv.value;
  }
}

/**
 * Convert a Filter tree into a SurrealQL WHERE clause with named binds.
 *
 * Returns `[sql, bindings]` where bindings is `[paramName, jsonValue][]`.
 * `paramOffset` is mutated to track the next parameter index.
 */
/** Join sub-filters with `op`; `empty` is the SurrealQL for an empty list. */
function combineSurreal(
  filters: Filter[],
  op: "AND" | "OR",
  empty: string,
  paramOffset: { value: number },
  options: FilterBuildOptions | undefined,
): [string, [string, unknown][]] {
  if (filters.length === 0) return [empty, []];
  const parts: string[] = [];
  const allBinds: [string, unknown][] = [];
  for (const f of filters) {
    const [sql, binds] = filterToSurrealQL(f, paramOffset, options);
    parts.push(sql);
    allBinds.push(...binds);
  }
  return [`(${parts.join(` ${op} `)})`, allBinds];
}

export function filterToSurrealQL(
  filter: Filter,
  paramOffset: { value: number },
  options?: FilterBuildOptions,
): [string, [string, unknown][]] {
  if ("field" in filter) assertIdentifier(filter.field, "field");
  switch (filter.kind) {
    case "Eq": {
      const name = `p${paramOffset.value++}`;
      return [`${filter.field} = $${name}`, [[
        name,
        fieldValueToJson(filter.value),
      ]]];
    }
    case "Ne": {
      const name = `p${paramOffset.value++}`;
      return [`${filter.field} != $${name}`, [[
        name,
        fieldValueToJson(filter.value),
      ]]];
    }
    case "Lt": {
      const name = `p${paramOffset.value++}`;
      return [`${filter.field} < $${name}`, [[
        name,
        fieldValueToJson(filter.value),
      ]]];
    }
    case "Lte": {
      const name = `p${paramOffset.value++}`;
      return [`${filter.field} <= $${name}`, [[
        name,
        fieldValueToJson(filter.value),
      ]]];
    }
    case "Gt": {
      const name = `p${paramOffset.value++}`;
      return [`${filter.field} > $${name}`, [[
        name,
        fieldValueToJson(filter.value),
      ]]];
    }
    case "Gte": {
      const name = `p${paramOffset.value++}`;
      return [`${filter.field} >= $${name}`, [[
        name,
        fieldValueToJson(filter.value),
      ]]];
    }
    case "NotNull":
      return [`${filter.field} IS NOT NULL`, []];
    case "IsNull":
      return [`${filter.field} IS NULL`, []];
    case "In": {
      if (filter.values.length === 0) return ["false", []];
      const name = `p${paramOffset.value++}`;
      const arr = filter.values.map(fieldValueToJson);
      return [`${filter.field} IN $${name}`, [[name, arr]]];
    }
    case "And":
      return combineSurreal(
        filter.filters,
        "AND",
        "true",
        paramOffset,
        options,
      );
    case "Or":
      return combineSurreal(
        filter.filters,
        "OR",
        "false",
        paramOffset,
        options,
      );
    case "Raw":
      return [assertRawAllowed(filter.expression, options), []];
  }
}

/** Parse a JSON row from SurrealDB into a BwRecord. */
export function jsonRowToRecord(row: Record<string, unknown>): BwRecord {
  const record: BwRecord = [];
  for (const [key, val] of Object.entries(row)) {
    let fv: FieldValue;
    if (key === "id") {
      fv = {
        kind: "Utf8",
        value: typeof val === "string" ? val : JSON.stringify(val),
      };
    } else if (val === null || val === undefined) {
      fv = { kind: "Utf8", value: null };
    } else if (typeof val === "boolean") {
      fv = { kind: "Boolean", value: val };
    } else if (typeof val === "number") {
      fv = Number.isInteger(val)
        ? { kind: "Int64", value: val }
        : { kind: "Float64", value: val };
    } else if (typeof val === "string") {
      fv = { kind: "Utf8", value: val };
    } else if (Array.isArray(val)) {
      const floats = val.filter((v): v is number => typeof v === "number");
      if (floats.length === val.length && val.length > 0) {
        fv = { kind: "Vector", value: floats };
      } else {
        fv = { kind: "Utf8", value: JSON.stringify(val) };
      }
    } else {
      fv = { kind: "Utf8", value: JSON.stringify(val) };
    }
    record.push([key, fv]);
  }
  return record;
}

// ---------------------------------------------------------------------------
// SurrealDatabase
// ---------------------------------------------------------------------------

/** Configuration for SurrealDatabase connection. */
export interface SurrealConfig {
  /** WebSocket URL (default: "ws://localhost:8000"). */
  url?: string;
  /** Namespace. */
  namespace: string;
  /** Database name. */
  database: string;
  /** Username (default: "root"). */
  username?: string;
  /** Password (default: "root"). */
  password?: string;
  /** Allow `Filter.kind === "Raw"` (verbatim SurrealQL). Default: false. */
  allowRawFilters?: boolean;
}

/**
 * SurrealDB backed database implementing StorageBackend.
 */
export class SurrealDatabase implements StorageBackend {
  private db: Surreal;
  private connected = false;
  private config: SurrealConfig;

  private readonly filterOptions: FilterBuildOptions;

  /**
   * Create a SurrealDB client. The connection is opened lazily by
   * {@link SurrealDatabase.connect} (called automatically by every query).
   *
   * @param config Namespace and database are required; `url` defaults to
   *   `ws://localhost:8000`, credentials to `root`/`root`, and `Raw` filters
   *   are disabled unless `allowRawFilters` is `true`.
   */
  constructor(config: SurrealConfig) {
    this.filterOptions = { allowRaw: config.allowRawFilters ?? false };
    this.db = new Surreal();
    this.config = config;
  }

  /** Return the default connection URL. */
  static defaultUrl(): string {
    return DEFAULT_URL;
  }

  /** Connect and sign in. Must be called before other methods. */
  async connect(): Promise<void> {
    if (this.connected) return;
    const url = this.config.url ?? DEFAULT_URL;
    await this.db.connect(url);
    await this.db.signin({
      username: this.config.username ?? "root",
      password: this.config.password ?? "root",
    });
    await this.db.use({
      namespace: this.config.namespace,
      database: this.config.database,
    });
    this.connected = true;
  }

  // ── Private helpers ────────────────────────────────────────────────

  /**
   * Connect if needed, run one SurrealQL statement with `bindings`, and return
   * the first statement's rows (wrapped in an array when it is a scalar).
   */
  private async runQuery(
    query: string,
    bindings?: Record<string, unknown>,
  ): Promise<unknown[]> {
    await this.connect();
    const result = await this.db.query(query, bindings);
    // SurrealDB SDK returns an array of statement results.
    // Each entry is the result of a single statement.
    if (Array.isArray(result) && result.length > 0) {
      const first = result[0];
      if (Array.isArray(first)) return first;
      return [first];
    }
    return [];
  }

  // ── StorageBackend ─────────────────────────────────────────────────

  async ensureTable(tableName: string, schema: FieldDef[]): Promise<void> {
    assertIdentifier(tableName, "table name");
    let ddl = `DEFINE TABLE IF NOT EXISTS ${tableName} SCHEMAFULL;\n`;

    for (const field of schema) {
      const surrealType = fieldTypeToSurrealQL(field.fieldType);
      const typeExpr = field.nullable ? `option<${surrealType}>` : surrealType;
      assertIdentifier(field.name, "field");
      ddl += `DEFINE FIELD ${field.name} ON ${tableName} TYPE ${typeExpr};\n`;

      if (field.fieldType.kind === "Vector") {
        ddl +=
          `DEFINE INDEX idx_${tableName}_${field.name} ON ${tableName} FIELDS ${field.name} MTREE DIMENSION ${field.fieldType.dimension} DIST COSINE TYPE F32;\n`;
      }
    }

    await this.connect();
    await this.db.query(ddl);
  }

  async insert(tableName: string, records: BwRecord[]): Promise<void> {
    if (records.length === 0) return;
    await this.connect();

    let batch = "BEGIN TRANSACTION;\n";
    for (const record of records) {
      const obj: Record<string, unknown> = {};
      for (const [name, value] of record) {
        obj[name] = fieldValueToJson(value);
      }
      assertIdentifier(tableName, "table name");
      batch += `CREATE ${tableName} CONTENT ${JSON.stringify(obj)};\n`;
    }
    batch += "COMMIT TRANSACTION;\n";
    await this.db.query(batch);
  }

  /** ` WHERE …` (or empty) plus named bindings for an optional filter. */
  private whereClause(
    filter: Filter | undefined,
  ): [string, Record<string, unknown>] {
    if (!filter) return ["", {}];
    const [whereSql, whereBinds] = filterToSurrealQL(
      filter,
      { value: 0 },
      this.filterOptions,
    );
    return [` WHERE ${whereSql}`, Object.fromEntries(whereBinds)];
  }

  async query(
    tableName: string,
    filter?: Filter,
    limit?: number,
  ): Promise<BwRecord[]> {
    assertIdentifier(tableName, "table name");
    const [where, bindings] = this.whereClause(filter);
    let sql = `SELECT * FROM ${tableName}${where}`;
    if (limit !== undefined) {
      sql += ` LIMIT ${assertLimit(limit)}`;
    }

    const rows = await this.runQuery(sql, bindings) as Record<
      string,
      unknown
    >[];
    return rows.map(jsonRowToRecord);
  }

  async delete(tableName: string, filter: Filter): Promise<void> {
    const offset = { value: 0 };
    const [whereSql, whereBinds] = filterToSurrealQL(
      filter,
      offset,
      this.filterOptions,
    );
    const bindings: Record<string, unknown> = {};
    for (const [name, val] of whereBinds) {
      bindings[name] = val;
    }
    await this.runQuery(
      `DELETE FROM ${tableName} WHERE ${whereSql}`,
      bindings,
    );
  }

  async count(tableName: string, filter?: Filter): Promise<number> {
    assertIdentifier(tableName, "table name");
    const [where, bindings] = this.whereClause(filter);
    const sql = `SELECT count() AS total FROM ${tableName}${where} GROUP ALL`;

    const rows = await this.runQuery(sql, bindings) as Record<
      string,
      unknown
    >[];
    const first = rows[0];
    if (first && typeof first === "object" && "total" in first) {
      return Number(first.total);
    }
    return 0;
  }

  async vectorSearch(
    tableName: string,
    vectorColumn: string,
    vector: number[],
    limit: number,
    filter?: Filter,
  ): Promise<ScoredRecord[]> {
    const bindings: Record<string, unknown> = { query_vec: vector };
    let whereExtra = "";
    if (filter) {
      const offset = { value: 0 };
      const [whereSql, whereBinds] = filterToSurrealQL(
        filter,
        offset,
        this.filterOptions,
      );
      whereExtra = ` AND ${whereSql}`;
      for (const [name, val] of whereBinds) {
        bindings[name] = val;
      }
    }

    const sql =
      `SELECT *, vector::similarity::cosine(${vectorColumn}, $query_vec) AS __score ` +
      `FROM ${tableName} ` +
      `WHERE ${vectorColumn} <|${limit}|> $query_vec${whereExtra} ` +
      `ORDER BY __score DESC`;

    const rows = await this.runQuery(sql, bindings) as Record<
      string,
      unknown
    >[];
    return rows.map((row) => {
      const score = Number(row.__score ?? 0);
      const record = jsonRowToRecord(row);
      // Remove synthetic __score column from the record.
      const filtered = record.filter(([name]) => name !== "__score");
      return { record: filtered, score };
    });
  }
}
