import { NextRequest, NextResponse } from "next/server";
import { verifyJwt } from "@/lib/jwt";

export const runtime = "nodejs";

/**
 * GET /api/token
 *
 * Returns the current webchat JWT from the HttpOnly cookie — the browser
 * needs the raw token to pass as `?token=…` to the gateway WebSocket
 * endpoint, so we surface it here after verifying the cookie. This
 * endpoint is same-origin only and protected by SameSite=lax.
 */
export async function GET(req: NextRequest) {
  const secret = process.env.WEBCHAT_SECRET;
  if (!secret) {
    return NextResponse.json({ error: "server not configured" }, { status: 500 });
  }

  const cookie = req.cookies.get("webchat_jwt")?.value;
  if (!cookie) {
    return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  }

  const claims = verifyJwt(cookie, secret);
  if (!claims) {
    return NextResponse.json({ error: "invalid token" }, { status: 401 });
  }

  return NextResponse.json({ token: cookie, sub: claims.sub, exp: claims.exp });
}
