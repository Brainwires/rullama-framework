/**
 * SSRF-resistant `fetch` for URLs a model chose.
 *
 * A tool that fetches an arbitrary URL is a proxy into wherever the agent
 * runs: cloud metadata endpoints, localhost admin ports, the private network.
 * {@link safeFetch} closes that by allowing only `http`/`https`, resolving the
 * host and refusing loopback / link-local / private / CGNAT / multicast
 * addresses, re-checking every redirect hop, and bounding time and body size.
 *
 * @module
 */

/** Options for {@link safeFetch}. */
export interface SafeFetchOptions {
  /** Permit private / loopback / link-local destinations. Default: `false`. */
  allowPrivate?: boolean;
  /** Redirect hops to follow (each re-validated). Default: `5`. */
  maxRedirects?: number;
  /** Abort after this long, including redirects. Default: `30_000`. */
  timeoutMs?: number;
  /** Hostname resolver (injectable for tests). Default: `Deno.resolveDns`. */
  resolve?: (hostname: string) => Promise<string[]>;
}

/** Default body cap for {@link readCappedText}: 1 MiB. */
export const DEFAULT_MAX_BODY_BYTES = 1024 * 1024;

const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

function parseIpv4(ip: string): number[] | undefined {
  const parts = ip.split(".");
  if (parts.length !== 4) return undefined;
  const nums = parts.map((p) => /^\d{1,3}$/.test(p) ? Number(p) : NaN);
  return nums.every((n) => n >= 0 && n <= 255) ? nums : undefined;
}

/** Reserved IPv4 ranges as [first octet, second octet, prefix bits]. */
const PRIVATE_V4_CIDRS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 8], // "this" network
  [10, 0, 8], // private
  [127, 0, 8], // loopback
  [100, 64, 10], // CGNAT
  [169, 254, 16], // link-local incl. cloud metadata
  [172, 16, 12], // private
  [192, 168, 16], // private
  [224, 0, 3], // multicast + reserved + broadcast (224.0.0.0/3)
];

function inCidr(
  o: number[],
  [a, b, bits]: readonly [number, number, number],
): boolean {
  const ip = ((o[0] << 24) | (o[1] << 16) | (o[2] << 8) | o[3]) >>> 0;
  const base = ((a << 24) | (b << 16)) >>> 0;
  const mask = bits === 0 ? 0 : (0xffffffff << (32 - bits)) >>> 0;
  return (ip & mask) === (base & mask);
}

function isPrivateIpv4(o: number[]): boolean {
  return PRIVATE_V4_CIDRS.some((cidr) => inCidr(o, cidr));
}

/** `::ffff:a.b.c.d` → eight hextets, or `undefined` when not IPv4-mapped. */
function expandMappedIpv6(bare: string): string[] | undefined {
  const mapped = bare.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/i);
  if (!mapped) return undefined;
  const v4 = parseIpv4(mapped[1]);
  if (!v4) return undefined;
  const hi = ((v4[0] << 8) | v4[1]).toString(16);
  const lo = ((v4[2] << 8) | v4[3]).toString(16);
  return ["0", "0", "0", "0", "0", "ffff", hi, lo];
}

/** Expand an IPv6 literal to eight lowercase hextets, or `undefined`. */
function expandIpv6(ip: string): string[] | undefined {
  const bare = ip.replace(/^\[|\]$/g, "").split("%")[0];
  const mapped = expandMappedIpv6(bare);
  if (mapped) return mapped;
  const halves = bare.split("::");
  if (halves.length > 2) return undefined;
  const head = halves[0] ? halves[0].split(":") : [];
  const tail = halves[1] ? halves[1].split(":") : [];
  const fill = halves.length === 2 ? 8 - head.length - tail.length : 0;
  if (fill < 0 || head.length + tail.length + fill !== 8) return undefined;
  const hextets = [...head, ...Array(fill).fill("0"), ...tail];
  return hextets.every((h) => /^[0-9a-f]{1,4}$/i.test(h))
    ? hextets.map((h) => h.toLowerCase())
    : undefined;
}

function isPrivateIpv6(hextets: string[]): boolean {
  const first = parseInt(hextets[0], 16);
  const all = hextets.map((h) => parseInt(h, 16));
  if (all.every((h) => h === 0)) return true; // ::
  if (all.slice(0, 7).every((h) => h === 0) && all[7] === 1) return true; // ::1
  if (hextets[5] === "ffff" && all.slice(0, 5).every((h) => h === 0)) {
    return isPrivateIpv4([
      all[6] >> 8,
      all[6] & 0xff,
      all[7] >> 8,
      all[7] & 0xff,
    ]);
  }
  return (first & 0xfe00) === 0xfc00 || // fc00::/7 unique local
    (first & 0xffc0) === 0xfe80 || // fe80::/10 link-local
    (first & 0xff00) === 0xff00; // ff00::/8 multicast
}

/** True for loopback, unspecified, private, link-local, CGNAT and multicast addresses. */
export function isPrivateAddress(ip: string): boolean {
  const v4 = parseIpv4(ip);
  if (v4) return isPrivateIpv4(v4);
  const v6 = expandIpv6(ip);
  return v6 ? isPrivateIpv6(v6) : true; // unparseable → treat as unsafe
}

/** Reason a URL is refused, or `undefined` when it may be fetched. */
export async function checkUrl(
  url: URL,
  options: SafeFetchOptions = {},
): Promise<string | undefined> {
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return `scheme '${url.protocol}' is not allowed`;
  }
  if (options.allowPrivate) return undefined;
  const host = url.hostname.replace(/^\[|\]$/g, "");
  if (host === "localhost" || host.endsWith(".localhost")) {
    return "loopback host is not allowed";
  }
  const literal = parseIpv4(host) || expandIpv6(host);
  const addresses = literal ? [host] : await resolveHost(host, options.resolve);
  if (addresses.length === 0) return `host '${host}' does not resolve`;
  return addresses.some(isPrivateAddress)
    ? `host '${host}' resolves to a private or reserved address`
    : undefined;
}

async function resolveHost(
  host: string,
  resolve: SafeFetchOptions["resolve"],
): Promise<string[]> {
  if (resolve) return await resolve(host);
  const lookup = (type: "A" | "AAAA") =>
    Deno.resolveDns(host, type).catch(() => [] as string[]);
  const [a, aaaa] = await Promise.all([lookup("A"), lookup("AAAA")]);
  return [...a, ...aaaa];
}

/**
 * Fetch `input` with SSRF checks on the URL and on every redirect hop, and a
 * deadline covering the whole chain. Redirects are followed manually so each
 * destination is validated; a `Location` that fails the check aborts the fetch.
 *
 * @throws Error when the URL or a redirect target is refused, the redirect
 *   budget is exhausted, or the deadline passes.
 */
export async function safeFetch(
  input: string | URL,
  init: RequestInit = {},
  options: SafeFetchOptions = {},
): Promise<Response> {
  const maxRedirects = options.maxRedirects ?? 5;
  const signal = init.signal ??
    AbortSignal.timeout(options.timeoutMs ?? 30_000);
  let url = new URL(input);
  for (let hop = 0;; hop++) {
    const refusal = await checkUrl(url, options);
    if (refusal !== undefined) {
      throw new Error(`refusing to fetch ${url}: ${refusal}`);
    }
    const res = await fetch(url, { ...init, signal, redirect: "manual" });
    const location = res.headers.get("location");
    if (!REDIRECT_STATUSES.has(res.status) || location === null) return res;
    await res.body?.cancel();
    if (hop >= maxRedirects) {
      throw new Error(`too many redirects (more than ${maxRedirects})`);
    }
    url = new URL(location, url);
  }
}

/**
 * Read a response body as text, stopping after `maxBytes`. A truncated body
 * ends with a marker line so the model knows it saw a prefix.
 */
export async function readCappedText(
  res: Response,
  maxBytes = DEFAULT_MAX_BODY_BYTES,
): Promise<string> {
  if (res.body === null) return "";
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let truncated = false;
  while (!truncated) {
    const { done, value } = await reader.read();
    if (done) break;
    const room = maxBytes - total;
    if (value.byteLength > room) {
      chunks.push(value.subarray(0, room));
      truncated = true;
    } else {
      chunks.push(value);
    }
    total += Math.min(value.byteLength, room);
  }
  if (truncated) await reader.cancel();
  const joined = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    joined.set(c, offset);
    offset += c.byteLength;
  }
  const text = new TextDecoder().decode(joined);
  return truncated ? `${text}\n[truncated at ${maxBytes} bytes]` : text;
}
