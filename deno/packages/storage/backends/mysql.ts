/**
 * MySQL / MariaDB backend for StorageBackend.
 *
 * Port of the Rust `rullama-storage/src/databases/mysql/mod.rs`.
 *
 * Uses `npm:mysql2/promise` for async MySQL connections. Vector columns are
 * stored as JSON arrays and similarity search is performed client-side.
 * @module
 */

import mysql from "mysql2/promise";
import type { StorageBackend } from "../traits.ts";
import type {
  FieldDef,
  FieldValue,
  Filter,
  Record as BwRecord,
  ScoredRecord,
} from "../types.ts";

const DEFAULT_URL = "mysql://localhost:3306/rullama";

// ---------------------------------------------------------------------------
// MySQL SQL helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Map a FieldType kind to a MySQL column type string. */
export function mapFieldType(ft: FieldDef["fieldType"]): string {
  switch (ft.kind) {
    case "Utf8":
      return "TEXT";
    case "Int32":
      return "INT";
    case "Int64":
      return "BIGINT";
    case "UInt32":
      return "INT UNSIGNED";
    case "UInt64":
      return "BIGINT UNSIGNED";
    case "Float32":
      return "FLOAT";
    case "Float64":
      return "DOUBLE";
    case "Boolean":
      return "BOOLEAN";
    case "Vector":
      // MySQL has no native vector type; store as JSON.
      return "JSON";
  }
}

/** Extract the raw JS value from a FieldValue for use as a MySQL bind parameter. */
export function fieldValueToParam(fv: FieldValue): unknown {
  switch (fv.kind) {
    case "Utf8":
    case "Int32":
    case "Int64":
    case "Float32":
    case "Float64":
    case "Boolean":
    case "UInt32":
    case "UInt64":
      return fv.value;
    case "Vector":
      // MySQL stores vectors as JSON arrays.
      return JSON.stringify(fv.value);
  }
}

/**
 * Convert a Filter tree into a MySQL WHERE fragment with `?` placeholders.
 *
 * Returns `[sql, values]` where `values` are the bind parameters.
 */
export function filterToSql(
  filter: Filter,
): [string, FieldValue[]] {
  switch (filter.kind) {
    case "Eq":
      return [`\`${filter.field}\` = ?`, [filter.value]];
    case "Ne":
      return [`\`${filter.field}\` != ?`, [filter.value]];
    case "Lt":
      return [`\`${filter.field}\` < ?`, [filter.value]];
    case "Lte":
      return [`\`${filter.field}\` <= ?`, [filter.value]];
    case "Gt":
      return [`\`${filter.field}\` > ?`, [filter.value]];
    case "Gte":
      return [`\`${filter.field}\` >= ?`, [filter.value]];
    case "NotNull":
      return [`\`${filter.field}\` IS NOT NULL`, []];
    case "IsNull":
      return [`\`${filter.field}\` IS NULL`, []];
    case "In": {
      if (filter.values.length === 0) return ["1 = 0", []];
      const placeholders = filter.values.map(() => "?");
      return [
        `\`${filter.field}\` IN (${placeholders.join(", ")})`,
        [...filter.values],
      ];
    }
    case "And": {
      if (filter.filters.length === 0) return ["1 = 1", []];
      const parts: string[] = [];
      const allVals: FieldValue[] = [];
      for (const f of filter.filters) {
        const [sql, vals] = filterToSql(f);
        parts.push(sql);
        allVals.push(...vals);
      }
      return [`(${parts.join(" AND ")})`, allVals];
    }
    case "Or": {
      if (filter.filters.length === 0) return ["1 = 0", []];
      const parts: string[] = [];
      const allVals: FieldValue[] = [];
      for (const f of filter.filters) {
        const [sql, vals] = filterToSql(f);
        parts.push(sql);
        allVals.push(...vals);
      }
      return [`(${parts.join(" OR ")})`, allVals];
    }
    case "Raw":
      return [filter.expression, []];
  }
}

/** Build a CREATE TABLE IF NOT EXISTS DDL statement (MySQL dialect). */
export function buildCreateTable(
  tableName: string,
  schema: FieldDef[],
): string {
  const cols = schema.map((f, i) => {
    const mysqlType = mapFieldType(f.fieldType);
    const nullable = f.nullable ? "" : " NOT NULL";
    const pk = i === 0 ? " PRIMARY KEY" : "";
    return `\`${f.name}\` ${mysqlType}${nullable}${pk}`;
  });
  return `CREATE TABLE IF NOT EXISTS \`${tableName}\` (${cols.join(", ")})`;
}

/** Build an INSERT INTO statement with `?` placeholders. Returns [sql, params]. */
export function buildInsert(
  tableName: string,
  records: BwRecord[],
): [string, unknown[]] {
  if (records.length === 0) return ["", []];
  const colNames = records[0].map(([name]) => name);
  const quotedCols = colNames.map((c) => `\`${c}\``);
  const allParams: unknown[] = [];
  const rowGroups: string[] = [];
  for (const rec of records) {
    const placeholders = rec.map(() => "?");
    for (const [, fv] of rec) {
      allParams.push(fieldValueToParam(fv));
    }
    rowGroups.push(`(${placeholders.join(", ")})`);
  }
  const sql = `INSERT INTO \`${tableName}\` (${quotedCols.join(", ")}) VALUES ${
    rowGroups.join(", ")
  }`;
  return [sql, allParams];
}

/** Build a SELECT * with optional WHERE / LIMIT. Returns [sql, params]. */
export function buildSelect(
  tableName: string,
  filter?: Filter,
  limit?: number,
): [string, unknown[]] {
  let sql = `SELECT * FROM \`${tableName}\``;
  const params: unknown[] = [];
  if (filter) {
    const [whereSql, vals] = filterToSql(filter);
    sql += ` WHERE ${whereSql}`;
    params.push(...vals.map(fieldValueToParam));
  }
  if (limit !== undefined) sql += ` LIMIT ${limit}`;
  return [sql, params];
}

/** Build a DELETE FROM with WHERE. Returns [sql, params]. */
export function buildDelete(
  tableName: string,
  filter: Filter,
): [string, unknown[]] {
  const [whereSql, vals] = filterToSql(filter);
  return [
    `DELETE FROM \`${tableName}\` WHERE ${whereSql}`,
    vals.map(fieldValueToParam),
  ];
}

/** Build a SELECT COUNT(*) with optional WHERE. Returns [sql, params]. */
export function buildCount(
  tableName: string,
  filter?: Filter,
): [string, unknown[]] {
  let sql = `SELECT COUNT(*) AS cnt FROM \`${tableName}\``;
  const params: unknown[] = [];
  if (filter) {
    const [whereSql, vals] = filterToSql(filter);
    sql += ` WHERE ${whereSql}`;
    params.push(...vals.map(fieldValueToParam));
  }
  return [sql, params];
}

/** Compute cosine similarity between two number arrays. */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

/** Parse a MySQL row object into a BwRecord. */
function rowToRecord(row: Record<string, unknown>): BwRecord {
  const record: BwRecord = [];
  for (const [name, val] of Object.entries(row)) {
    let fv: FieldValue;
    if (val === null || val === undefined) {
      fv = { kind: "Utf8", value: null };
    } else if (typeof val === "string") {
      // Try to parse JSON vector.
      if (val.startsWith("[") && val.endsWith("]")) {
        try {
          const arr = JSON.parse(val);
          if (
            Array.isArray(arr) &&
            arr.every((v: unknown) => typeof v === "number")
          ) {
            fv = { kind: "Vector", value: arr as number[] };
          } else {
            fv = { kind: "Utf8", value: val };
          }
        } catch {
          fv = { kind: "Utf8", value: val };
        }
      } else {
        fv = { kind: "Utf8", value: val };
      }
    } else if (typeof val === "number") {
      if (Number.isInteger(val)) {
        fv = { kind: "Int64", value: val };
      } else {
        fv = { kind: "Float64", value: val };
      }
    } else if (typeof val === "boolean") {
      fv = { kind: "Boolean", value: val };
    } else if (
      Array.isArray(val) && val.every((v: unknown) => typeof v === "number")
    ) {
      // mysql2 may parse JSON columns into arrays directly.
      fv = { kind: "Vector", value: val as number[] };
    } else {
      fv = { kind: "Utf8", value: String(val) };
    }
    record.push([name, fv]);
  }
  return record;
}

// ---------------------------------------------------------------------------
// MySqlDatabase
// ---------------------------------------------------------------------------

/** Configuration for MySqlDatabase. */
export interface MySqlConfig {
  /** MySQL connection URI (e.g. "mysql://user:pass@host:3306/db"). */
  uri?: string;
}

/**
 * MySQL / MariaDB backed storage database implementing StorageBackend.
 *
 * Vector columns are stored as JSON arrays. Vector similarity search is
 * performed client-side via cosine similarity.
 */
export class MySqlDatabase implements StorageBackend {
  private pool: mysql.Pool;

  constructor(config?: MySqlConfig) {
    const uri = config?.uri ?? DEFAULT_URL;
    this.pool = mysql.createPool(uri);
  }

  /** Return the default connection URL. */
  static defaultUrl(): string {
    return DEFAULT_URL;
  }

  // -- StorageBackend -----------------------------------------------------

  async ensureTable(tableName: string, schema: FieldDef[]): Promise<void> {
    const ddl = buildCreateTable(tableName, schema);
    await this.pool.execute(ddl);
  }

  async insert(tableName: string, records: BwRecord[]): Promise<void> {
    if (records.length === 0) return;
    const [sql, params] = buildInsert(tableName, records);
    // deno-lint-ignore no-explicit-any
    await this.pool.execute(sql, params as any);
  }

  async query(
    tableName: string,
    filter?: Filter,
    limit?: number,
  ): Promise<BwRecord[]> {
    const [sql, params] = buildSelect(tableName, filter, limit);
    // deno-lint-ignore no-explicit-any
    const [rows] = await this.pool.execute(sql, params as any);
    return (rows as Record<string, unknown>[]).map(rowToRecord);
  }

  async delete(tableName: string, filter: Filter): Promise<void> {
    const [sql, params] = buildDelete(tableName, filter);
    // deno-lint-ignore no-explicit-any
    await this.pool.execute(sql, params as any);
  }

  async count(tableName: string, filter?: Filter): Promise<number> {
    const [sql, params] = buildCount(tableName, filter);
    // deno-lint-ignore no-explicit-any
    const [rows] = await this.pool.execute(sql, params as any);
    const first = (rows as Record<string, unknown>[])[0];
    return Number(first?.cnt ?? 0);
  }

  async vectorSearch(
    tableName: string,
    vectorColumn: string,
    vector: number[],
    limit: number,
    filter?: Filter,
  ): Promise<ScoredRecord[]> {
    // MySQL has no native vector search. Fetch all rows and score client-side.
    let sql: string;
    let params: unknown[];

    if (filter) {
      const [whereSql, vals] = filterToSql(filter);
      sql =
        `SELECT * FROM \`${tableName}\` WHERE ${whereSql} AND \`${vectorColumn}\` IS NOT NULL`;
      params = vals.map(fieldValueToParam);
    } else {
      sql =
        `SELECT * FROM \`${tableName}\` WHERE \`${vectorColumn}\` IS NOT NULL`;
      params = [];
    }

    // deno-lint-ignore no-explicit-any
    const [rows] = await this.pool.execute(sql, params as any);
    const records = (rows as Record<string, unknown>[]).map(rowToRecord);

    const scored: ScoredRecord[] = [];
    for (const record of records) {
      const vecEntry = record.find(([name]) => name === vectorColumn);
      if (!vecEntry) continue;
      const fv = vecEntry[1];
      if (fv.kind !== "Vector" || fv.value.length === 0) continue;

      const score = cosineSimilarity(vector, fv.value);
      scored.push({ record, score });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }
}
