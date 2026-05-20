import { assertEquals } from "@std/assert";
import { DEEPGRAM_API_BASE, DeepgramClient, type DeepgramListenResponse } from "./deepgram.ts";

Deno.test("client creation", () => {
  const client = new DeepgramClient("test-key");
  assertEquals(client.base_url, DEEPGRAM_API_BASE);
});

Deno.test("listen response deserialization", () => {
  const json = `{
    "results": {
      "channels": [{
        "alternatives": [{
          "transcript": "hello world",
          "confidence": 0.99,
          "words": [
            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
            {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.98}
          ]
        }]
      }]
    }
  }`;
  const resp = JSON.parse(json) as DeepgramListenResponse;
  assertEquals(resp.results.channels[0].alternatives[0].transcript, "hello world");
});
