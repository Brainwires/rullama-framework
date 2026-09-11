/**
 * Shared HTTP plumbing for the speech vendor clients.
 *
 * Every client used to repeat the same block — `fetch`, check `res.ok`, read the error
 * body, throw `<vendor> API error (<status>): <body>` — with no timeout. This module is
 * that block once, with a default timeout so a hung vendor cannot hang the caller.
 *
 * @module
 */

/** Default per-request timeout. Synthesis of long text can take a while; 60 s is generous. */
export const DEFAULT_TIMEOUT_MS = 60_000;

/**
 * Send a request to a vendor API and return the successful `Response`.
 *
 * Throws `Error("<label> API error (<status>): <body>")` on a non-2xx response. A
 * timeout (`DEFAULT_TIMEOUT_MS`) is applied unless `init.signal` is already set.
 *
 * @param label Vendor + operation name used in the error message, e.g. `"Azure TTS"`.
 * @param url Request URL.
 * @param init `fetch` options (method, headers, body, optional signal).
 * @returns The response, guaranteed `ok`.
 */
export async function vendorFetch(
  label: string,
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  const res = await fetch(url, {
    ...init,
    signal: init.signal ?? AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${label} API error (${res.status}): ${body}`);
  }
  return res;
}

/**
 * {@link vendorFetch} and decode the body as JSON.
 *
 * @typeParam T The vendor's response shape (not validated at runtime).
 */
export async function vendorJson<T>(
  label: string,
  url: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await vendorFetch(label, url, init);
  return await res.json() as T;
}

/** {@link vendorFetch} and return the body as raw bytes (audio). */
export async function vendorBytes(
  label: string,
  url: string,
  init: RequestInit = {},
): Promise<Uint8Array> {
  const res = await vendorFetch(label, url, init);
  return new Uint8Array(await res.arrayBuffer());
}
