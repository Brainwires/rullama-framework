/**
 * Minimal HS256 JWT helpers compatible with the gateway's
 * `brainwires_gateway::webchat` verifier. Runs in Node on the
 * `/api/auth` route and never reaches the browser.
 */

import { createHmac, timingSafeEqual } from "crypto";

export interface WebChatClaims {
  sub: string;
  exp: number;
  iat?: number;
}

function b64urlEncode(buf: Buffer | string): string {
  const b = typeof buf === "string" ? Buffer.from(buf) : buf;
  return b
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function b64urlDecode(s: string): Buffer {
  let t = s.replace(/-/g, "+").replace(/_/g, "/");
  while (t.length % 4 !== 0) t += "=";
  return Buffer.from(t, "base64");
}

export function signJwt(claims: WebChatClaims, secret: string): string {
  const header = { alg: "HS256", typ: "JWT" };
  const headerB64 = b64urlEncode(JSON.stringify(header));
  const payloadB64 = b64urlEncode(JSON.stringify(claims));
  const signingInput = `${headerB64}.${payloadB64}`;
  const sig = createHmac("sha256", secret).update(signingInput).digest();
  return `${signingInput}.${b64urlEncode(sig)}`;
}

export function verifyJwt(token: string, secret: string): WebChatClaims | null {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [headerB64, payloadB64, sigB64] = parts;
  const expected = createHmac("sha256", secret)
    .update(`${headerB64}.${payloadB64}`)
    .digest();
  const provided = b64urlDecode(sigB64);
  if (expected.length !== provided.length) return null;
  if (!timingSafeEqual(expected, provided)) return null;

  const claims = JSON.parse(b64urlDecode(payloadB64).toString("utf8")) as WebChatClaims;
  const now = Math.floor(Date.now() / 1000);
  if (typeof claims.exp !== "number" || claims.exp < now) return null;
  if (!claims.sub || typeof claims.sub !== "string") return null;
  return claims;
}
