import { NextRequest, NextResponse } from "next/server";
import { createHash } from "crypto";
import { signJwt } from "@/lib/jwt";

export const runtime = "nodejs";

/**
 * POST /api/auth
 *
 * Body: `{ "admin_token": "<token>" }`
 *
 * Validates the supplied admin token against `WEBCHAT_ADMIN_TOKEN`
 * (or accepts any non-empty token in dev mode), then mints a short-lived
 * HS256 JWT using `WEBCHAT_SECRET` and sets it as an HttpOnly cookie.
 */
export async function POST(req: NextRequest) {
  const secret = process.env.WEBCHAT_SECRET;
  if (!secret) {
    return NextResponse.json(
      { error: "WEBCHAT_SECRET is not configured on the server" },
      { status: 500 },
    );
  }

  let body: { admin_token?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  const supplied = (body.admin_token ?? "").trim();
  if (!supplied) {
    return NextResponse.json({ error: "empty admin token" }, { status: 400 });
  }

  const expected = (process.env.WEBCHAT_ADMIN_TOKEN ?? "").trim();
  if (expected && supplied !== expected) {
    return NextResponse.json({ error: "invalid admin token" }, { status: 401 });
  }

  // Derive a stable, non-identifying user id from the admin token so a
  // single admin always hits the same BrainClaw session.
  const sub = createHash("sha256")
    .update(`webchat-user:${supplied}`)
    .digest("hex")
    .slice(0, 24);

  const now = Math.floor(Date.now() / 1000);
  const exp = now + 60 * 60 * 12; // 12 hours
  const token = signJwt({ sub, exp, iat: now }, secret);

  const res = NextResponse.json({ ok: true });
  res.cookies.set("webchat_jwt", token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 12,
  });
  return res;
}

export async function DELETE() {
  const res = NextResponse.json({ ok: true });
  res.cookies.delete("webchat_jwt");
  return res;
}
