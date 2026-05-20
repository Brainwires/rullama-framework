import { assert, assertEquals } from "@std/assert";
import {
  ELEVENLABS_API_BASE,
  ElevenLabsClient,
  type ElevenLabsVoicesResponse,
  serializeTtsRequest,
} from "./elevenlabs.ts";

Deno.test("client creation", () => {
  const client = new ElevenLabsClient("test-key");
  assertEquals(client.base_url, ELEVENLABS_API_BASE);
});

Deno.test("tts request serialization skips undefined fields", () => {
  const json = serializeTtsRequest({
    text: "Hello world",
    model_id: "eleven_multilingual_v2",
    voice_settings: {
      stability: 0.5,
      similarity_boost: 0.75,
    },
    // output_format intentionally undefined
  });
  assertEquals(json.text, "Hello world");
  assertEquals(json.model_id, "eleven_multilingual_v2");
  assert(!("output_format" in json));
});

Deno.test("voices response deserialization", () => {
  const json = `{
    "voices": [
      {"voice_id": "abc123", "name": "Rachel", "labels": {"accent": "american"}}
    ]
  }`;
  const resp = JSON.parse(json) as ElevenLabsVoicesResponse;
  assertEquals(resp.voices.length, 1);
  assertEquals(resp.voices[0].name, "Rachel");
});
