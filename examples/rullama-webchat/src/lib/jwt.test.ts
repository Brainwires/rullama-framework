import { describe, expect, it } from "vitest";
import { signJwt, verifyJwt } from "./jwt";

const SECRET = "a-very-secret-value-for-tests";

function secondsFromNow(offset: number): number {
  return Math.floor(Date.now() / 1000) + offset;
}

describe("jwt", () => {
  it("sign + verify roundtrip with a valid secret returns the claims", () => {
    const claims = {
      sub: "user-123",
      exp: secondsFromNow(60 * 60),
      iat: Math.floor(Date.now() / 1000),
    };
    const token = signJwt(claims, SECRET);
    const decoded = verifyJwt(token, SECRET);
    expect(decoded).not.toBeNull();
    expect(decoded?.sub).toBe("user-123");
    expect(decoded?.exp).toBe(claims.exp);
  });

  it("verify with wrong secret fails", () => {
    const token = signJwt(
      { sub: "user-123", exp: secondsFromNow(60 * 60) },
      SECRET,
    );
    expect(verifyJwt(token, "wrong-secret")).toBeNull();
  });

  it("expired token fails", () => {
    const token = signJwt(
      { sub: "user-123", exp: secondsFromNow(-60) }, // expired 60s ago
      SECRET,
    );
    expect(verifyJwt(token, SECRET)).toBeNull();
  });

  it("malformed token (wrong number of segments) fails", () => {
    expect(verifyJwt("not.a.jwt.token", SECRET)).toBeNull();
    expect(verifyJwt("only-one-part", SECRET)).toBeNull();
    expect(verifyJwt("two.parts", SECRET)).toBeNull();
  });

  it("token with empty sub claim fails", () => {
    // Build a token with an empty sub — verifyJwt rejects empty sub.
    const token = signJwt({ exp: secondsFromNow(60 * 60), sub: "" }, SECRET);
    expect(verifyJwt(token, SECRET)).toBeNull();
  });

  it("tampered payload fails signature check", () => {
    const token = signJwt(
      { sub: "user-123", exp: secondsFromNow(60 * 60) },
      SECRET,
    );
    const [header, , sig] = token.split(".");
    // Forge a payload with a different sub but keep the original signature.
    const forged = Buffer.from(
      JSON.stringify({ sub: "attacker", exp: secondsFromNow(60 * 60) }),
    )
      .toString("base64")
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/g, "");
    const tampered = `${header}.${forged}.${sig}`;
    expect(verifyJwt(tampered, SECRET)).toBeNull();
  });
});
