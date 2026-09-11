/**
 * Dialect-parameterised SQL construction shared by the Postgres and MySQL
 * backends. Each dialect supplies identifier quoting, placeholder style, the
 * column-type mapping and bind-parameter conversion; everything else — filter
 * translation, DDL, INSERT/SELECT/DELETE/COUNT — is written once here.
 *
 * Identifiers are validated (never quoted-and-hoped), `LIMIT` must be an
 * integer, and `Raw` filters require an explicit opt-in (see `sql_guards.ts`).
 *
 * @module
 */

import type {
  FieldDef,
  FieldValue,
  Filter,
  Record as BwRecord,
} from "../types.ts";
import {
  assertIdentifier,
  assertLimit,
  assertRawAllowed,
  type FilterBuildOptions,
} from "./sql_guards.ts";

/** What differs between SQL dialects. */
export interface SqlDialect {
  /** Quote an (already validated) identifier. */
  quote(identifier: string): string;
  /** Placeholder for the `index`-th bind parameter (1-based, absolute). */
  placeholder(index: number): string;
  /** Column type for a field type. */
  mapFieldType(ft: FieldDef["fieldType"]): string;
  /** Bind-parameter value for a field value. */
  fieldValueToParam(fv: FieldValue): unknown;
  /** Text after `COUNT(*)` in the count query (e.g. ` AS cnt`). */
  countSuffix: string;
}

const COMPARISON: Partial<Record<Filter["kind"], string>> = {
  Eq: "=",
  Ne: "!=",
  Lt: "<",
  Lte: "<=",
  Gt: ">",
  Gte: ">=",
};

/** Join sub-filters with `op`; `empty` is the SQL for an empty list. */
function combine(
  dialect: SqlDialect,
  filters: Filter[],
  op: "AND" | "OR",
  empty: string,
  paramOffset: number,
  options: FilterBuildOptions | undefined,
): [string, FieldValue[]] {
  if (filters.length === 0) return [empty, []];
  const parts: string[] = [];
  const allVals: FieldValue[] = [];
  let offset = paramOffset;
  for (const f of filters) {
    const [sql, vals] = filterToSql(dialect, f, offset, options);
    offset += vals.length;
    parts.push(sql);
    allVals.push(...vals);
  }
  return [`(${parts.join(` ${op} `)})`, allVals];
}

/**
 * Translate a {@link Filter} into a WHERE fragment plus the values to bind,
 * numbering placeholders from `paramOffset`.
 */
export function filterToSql(
  dialect: SqlDialect,
  filter: Filter,
  paramOffset: number,
  options?: FilterBuildOptions,
): [string, FieldValue[]] {
  const col = "field" in filter
    ? dialect.quote(assertIdentifier(filter.field, "column"))
    : "";
  const cmp = COMPARISON[filter.kind];
  if (cmp !== undefined && "value" in filter) {
    return [`${col} ${cmp} ${dialect.placeholder(paramOffset)}`, [
      filter.value,
    ]];
  }
  switch (filter.kind) {
    case "NotNull":
      return [`${col} IS NOT NULL`, []];
    case "IsNull":
      return [`${col} IS NULL`, []];
    case "In": {
      if (filter.values.length === 0) return ["1 = 0", []];
      const ph = filter.values.map((_, i) =>
        dialect.placeholder(paramOffset + i)
      );
      return [`${col} IN (${ph.join(", ")})`, [...filter.values]];
    }
    case "And":
      return combine(
        dialect,
        filter.filters,
        "AND",
        "1 = 1",
        paramOffset,
        options,
      );
    case "Or":
      return combine(
        dialect,
        filter.filters,
        "OR",
        "1 = 0",
        paramOffset,
        options,
      );
    case "Raw":
      return [assertRawAllowed(filter.expression, options), []];
    default:
      throw new Error(`unsupported filter kind: ${(filter as Filter).kind}`);
  }
}

/** `CREATE TABLE IF NOT EXISTS`; the first column is the primary key. */
export function buildCreateTable(
  dialect: SqlDialect,
  tableName: string,
  schema: FieldDef[],
): string {
  const table = dialect.quote(assertIdentifier(tableName, "table name"));
  const cols = schema.map((f, i) => {
    const col = dialect.quote(assertIdentifier(f.name, "column"));
    const nullable = f.nullable ? "" : " NOT NULL";
    const pk = i === 0 ? " PRIMARY KEY" : "";
    return `${col} ${dialect.mapFieldType(f.fieldType)}${nullable}${pk}`;
  });
  return `CREATE TABLE IF NOT EXISTS ${table} (${cols.join(", ")})`;
}

/** Multi-row `INSERT`; columns are taken from the first record. */
export function buildInsert(
  dialect: SqlDialect,
  tableName: string,
  records: BwRecord[],
): [string, unknown[]] {
  if (records.length === 0) return ["", []];
  const table = dialect.quote(assertIdentifier(tableName, "table name"));
  const cols = records[0].map(([name]) =>
    dialect.quote(assertIdentifier(name, "column"))
  );
  const allParams: unknown[] = [];
  const rowGroups: string[] = [];
  let idx = 1;
  for (const rec of records) {
    const placeholders: string[] = [];
    for (const [, fv] of rec) {
      placeholders.push(dialect.placeholder(idx++));
      allParams.push(dialect.fieldValueToParam(fv));
    }
    rowGroups.push(`(${placeholders.join(", ")})`);
  }
  return [
    `INSERT INTO ${table} (${cols.join(", ")}) VALUES ${rowGroups.join(", ")}`,
    allParams,
  ];
}

/** ` WHERE …` (or empty) plus bound params for an optional filter. */
function whereClause(
  dialect: SqlDialect,
  filter: Filter | undefined,
  options: FilterBuildOptions | undefined,
): [string, unknown[]] {
  if (!filter) return ["", []];
  const [whereSql, vals] = filterToSql(dialect, filter, 1, options);
  return [` WHERE ${whereSql}`, vals.map(dialect.fieldValueToParam)];
}

/** `SELECT *` with optional WHERE / LIMIT. */
export function buildSelect(
  dialect: SqlDialect,
  tableName: string,
  filter?: Filter,
  limit?: number,
  options?: FilterBuildOptions,
): [string, unknown[]] {
  const table = dialect.quote(assertIdentifier(tableName, "table name"));
  const [where, params] = whereClause(dialect, filter, options);
  const limitSql = limit === undefined ? "" : ` LIMIT ${assertLimit(limit)}`;
  return [`SELECT * FROM ${table}${where}${limitSql}`, params];
}

/** `DELETE FROM … WHERE …`. */
export function buildDelete(
  dialect: SqlDialect,
  tableName: string,
  filter: Filter,
  options?: FilterBuildOptions,
): [string, unknown[]] {
  const table = dialect.quote(assertIdentifier(tableName, "table name"));
  const [where, params] = whereClause(dialect, filter, options);
  return [`DELETE FROM ${table}${where}`, params];
}

/** `SELECT COUNT(*)` with optional WHERE. */
export function buildCount(
  dialect: SqlDialect,
  tableName: string,
  filter?: Filter,
  options?: FilterBuildOptions,
): [string, unknown[]] {
  const table = dialect.quote(assertIdentifier(tableName, "table name"));
  const [where, params] = whereClause(dialect, filter, options);
  return [
    `SELECT COUNT(*)${dialect.countSuffix} FROM ${table}${where}`,
    params,
  ];
}
