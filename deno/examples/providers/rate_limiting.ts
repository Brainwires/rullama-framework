// Example: Rate-Limited Client with Token-Bucket Throttling
// Demonstrates RateLimiter and RateLimitedClient for controlling API request
// throughput using the token-bucket algorithm.
// Run: deno run deno/examples/providers/rate_limiting.ts

import { RateLimitedClient, RateLimiter } from "@rullama/provider";

async function main() {
  console.log("=== Rate-Limited Client Example ===\n");

  // 1. Create a standalone RateLimiter
  console.log("--- Standalone RateLimiter ---");
  const limiter = new RateLimiter(60);
  console.log(`  Max RPM: ${limiter.maxRequestsPerMinute()}`);
  console.log(`  Available tokens: ${limiter.availableTokens()}`);

  // Acquire tokens directly
  for (let i = 1; i <= 5; i++) {
    await limiter.acquire();
    console.log(
      `  Acquired token ${i} -> remaining: ${limiter.availableTokens()}`,
    );
  }
  console.log();

  // 2. Try non-blocking acquisition
  console.log("--- Non-Blocking tryAcquire ---");
  const smallLimiter = new RateLimiter(3);
  console.log(
    `  Created limiter with 3 RPM, tokens: ${smallLimiter.availableTokens()}`,
  );

  for (let i = 1; i <= 5; i++) {
    const acquired = smallLimiter.tryAcquire();
    console.log(
      `  tryAcquire attempt ${i}: ${acquired ? "OK" : "NO TOKENS"} ` +
        `(remaining: ${smallLimiter.availableTokens()})`,
    );
  }
  console.log();

  // 3. Wrap an async function with rate limiting
  console.log("--- RateLimitedClient (wrapping async function) ---");

  // Simulate an API call
  let callCount = 0;
  async function mockApiCall(
    endpoint: string,
    payload: string,
  ): Promise<string> {
    callCount++;
    return `Response #${callCount} from ${endpoint}: processed "${payload}"`;
  }

  const rateLimitedApi = new RateLimitedClient(mockApiCall, {
    requestsPerMinute: 30,
  });

  console.log(
    `  Initial tokens: ${rateLimitedApi.limiter.availableTokens()}`,
  );

  // Execute rate-limited calls
  console.log("\n  Executing rate-limited API calls:");
  for (let i = 1; i <= 5; i++) {
    const result = await rateLimitedApi.execute("/api/chat", `message-${i}`);
    console.log(
      `    Call ${i} -> ${result} (tokens: ${rateLimitedApi.limiter.availableTokens()})`,
    );
  }
  console.log();

  // 4. Demonstrate how rate limiting protects against API throttling
  console.log("--- Rate Limiting Strategy ---");
  console.log("  The RateLimiter uses a token-bucket algorithm:");
  console.log("  - Tokens refill at a fixed rate (RPM / 60 per second)");
  console.log("  - Each .acquire() consumes one token");
  console.log("  - When tokens are exhausted, .acquire() sleeps until refill");
  console.log("  - This prevents HTTP 429 (Too Many Requests) errors");
  console.log();

  // 5. Show multiple limiters for different API tiers
  console.log("--- Multiple Limiters for API Tiers ---");
  const tiers = [
    { name: "Free tier", rpm: 10 },
    { name: "Pro tier", rpm: 60 },
    { name: "Enterprise", rpm: 500 },
  ];
  for (const tier of tiers) {
    const l = new RateLimiter(tier.rpm);
    console.log(
      `  ${tier.name.padEnd(12)} -> ${l.maxRequestsPerMinute()} RPM, ` +
        `~${(tier.rpm / 60).toFixed(1)} req/sec`,
    );
  }

  console.log(
    "\nDone! In a real application, RateLimitedClient wraps your API",
  );
  console.log("calls to respect provider rate limits automatically.");
}

await main();
