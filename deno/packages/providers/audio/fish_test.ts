import { assertEquals } from "@std/assert";
import { _serializeTts, FISH_API_BASE, FishClient } from "./fish.ts";

Deno.test("client creation", () => {
  const client = new FishClient("test-key");
  assertEquals(client.base_url, FISH_API_BASE);
});

Deno.test("tts request serialization", () => {
  const json = _serializeTts({
    text: "Hello",
    reference_id: "voice-123",
    format: "wav",
  });
  assertEquals(json.text, "Hello");
  assertEquals(json.reference_id, "voice-123");
});
