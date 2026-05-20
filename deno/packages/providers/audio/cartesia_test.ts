import { assertEquals } from "@std/assert";
import { _serializeTts, CARTESIA_API_BASE, CartesiaClient } from "./cartesia.ts";

Deno.test("client creation", () => {
  const client = new CartesiaClient("test-key");
  assertEquals(client.base_url, CARTESIA_API_BASE);
});

Deno.test("tts request serialization", () => {
  const json = _serializeTts({
    model_id: "sonic-english",
    transcript: "Hello world",
    voice: { mode: "id", id: "a0e99841-438c-4a64-b679-ae501e7d6091" },
    output_format: { container: "raw", encoding: "pcm_s16le", sample_rate: 24000 },
  });
  assertEquals(json.model_id, "sonic-english");
  assertEquals(json.transcript, "Hello world");
});
