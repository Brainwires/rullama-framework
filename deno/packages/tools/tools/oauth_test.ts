import { assert, assertEquals, assertFalse } from "@std/assert";
import {
  clientCredentialsConfig,
  InMemoryTokenStore,
  isTokenExpired,
  newPkceChallenge,
  type OAuthToken,
  pkceAuthorizationUrl,
} from "./oauth.ts";

Deno.test("PKCE challenge uses base64url without padding", async () => {
  const pkce = await newPkceChallenge();
  assertFalse(pkce.verifier.includes("="));
  assertFalse(pkce.challenge.includes("="));
  assertFalse(pkce.verifier.includes("+"));
  assertFalse(pkce.challenge.includes("+"));
  assertFalse(pkce.verifier.includes("/"));
  assertFalse(pkce.challenge.includes("/"));
});

Deno.test("PKCE authorization URL contains required params", async () => {
  const pkce = await newPkceChallenge();
  const url = pkceAuthorizationUrl(
    pkce,
    "https://auth.example.com/authorize",
    "client-abc",
    "https://myapp.example.com/callback",
    ["openid", "profile"],
    "random-state",
  );
  assert(url.includes("response_type=code"));
  assert(url.includes("client_id=client-abc"));
  assert(url.includes("code_challenge_method=S256"));
  assert(url.includes(pkce.challenge));
  assert(url.includes("state=random-state"));
});

Deno.test("Token without expiry is never expired", () => {
  const t: OAuthToken = {
    access_token: "tok",
    refresh_token: null,
    expires_at: null,
    scope: null,
    token_type: "Bearer",
  };
  assertFalse(isTokenExpired(t));
});

Deno.test("Token with past expiry is expired", () => {
  const t: OAuthToken = {
    access_token: "tok",
    refresh_token: null,
    expires_at: 1,
    scope: null,
    token_type: "Bearer",
  };
  assert(isTokenExpired(t));
});

Deno.test("In-memory store round-trip", async () => {
  const store = new InMemoryTokenStore();
  const token: OAuthToken = {
    access_token: "abc",
    refresh_token: null,
    expires_at: null,
    scope: null,
    token_type: "Bearer",
  };
  await store.set("user1", "github", token);
  const fetched = await store.get("user1", "github");
  assert(fetched !== null);
  assertEquals(fetched.access_token, "abc");

  await store.delete("user1", "github");
  assertEquals(await store.get("user1", "github"), null);
});

Deno.test("clientCredentialsConfig builder", () => {
  const cfg = clientCredentialsConfig(
    "https://token.example.com",
    "id",
    "secret",
    ["read", "write"],
  );
  assertEquals(cfg.scopes, ["read", "write"]);
  assertEquals(cfg.flow.kind, "client_credentials");
});
