/**
 * The one HTTP call every chat provider makes: `POST` a JSON body, throw with
 * the response text on a non-2xx status, hand back the `Response`.
 *
 * @module
 */

/**
 * POST `body` as JSON to `url`.
 *
 * @param label Vendor name for the error message, e.g. `"OpenAI"`.
 * @param signal Aborts the request (and closes a streamed body) when fired.
 * @throws Error `"<label> API error (<status>): <body>"` on a non-2xx status.
 */
export async function postJson(
  label: string,
  url: string,
  headers: Record<string, string>,
  body: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`${label} API error (${response.status}): ${errorText}`);
  }
  return response;
}
