/**
 * Guards for the parts of a query that cannot be bound as parameters:
 * identifiers (table and column names), `LIMIT` counts, and the `Raw` filter
 * escape hatch. Values are always bound; these keep the *structure* safe.
 *
 * @module
 */

/** A SQL identifier a backend will interpolate: `[A-Za-z_][A-Za-z0-9_]*`, ≤ 63 chars. */
export const SQL_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]{0,62}$/;

/** Return `name` if it is a plain identifier; throw otherwise. */
export function assertIdentifier(name: string, what = "identifier"): string {
  if (!SQL_IDENTIFIER.test(name)) {
    throw new Error(`invalid SQL ${what}: ${JSON.stringify(name)}`);
  }
  return name;
}

/** Return `limit` if it is a non-negative integer; throw otherwise. */
export function assertLimit(limit: number): number {
  if (!Number.isInteger(limit) || limit < 0) {
    throw new Error(`invalid LIMIT: ${JSON.stringify(limit)}`);
  }
  return limit;
}

/** Options accepted by the filter → SQL builders. */
export interface FilterBuildOptions {
  /**
   * Permit `Filter.kind === "Raw"`, whose expression is spliced into the query
   * verbatim. Off by default: a filter deserialized from JSON, a request body
   * or a model's tool call must never carry raw SQL.
   */
  allowRaw?: boolean;
}

/** Return the raw expression when allowed; throw otherwise. */
export function assertRawAllowed(
  expression: string,
  options: FilterBuildOptions | undefined,
): string {
  if (!options?.allowRaw) {
    throw new Error(
      "Raw filters are disabled; pass { allowRaw: true } (or construct the backend with allowRawFilters: true) to enable them",
    );
  }
  return expression;
}
