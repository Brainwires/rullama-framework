import { assertEquals } from "@std/assert";
import {
  _serializeGenerate,
  MURF_API_BASE,
  MurfClient,
  type MurfGenerateResponse,
} from "./murf.ts";

Deno.test("client creation", () => {
  const client = new MurfClient("test-key");
  assertEquals(client.base_url, MURF_API_BASE);
});

Deno.test("generate request serialization", () => {
  const json = _serializeGenerate({
    voiceId: "en-US-natalie",
    text: "Hello world",
    format: "WAV",
    sampleRate: 24000,
  });
  assertEquals(json.voiceId, "en-US-natalie");
  assertEquals(json.text, "Hello world");
});

Deno.test("generate response deserialization", () => {
  const json = `{
    "audioFile": "https://cdn.murf.ai/audio/123.wav",
    "audioDuration": 2.5
  }`;
  const resp = JSON.parse(json) as MurfGenerateResponse;
  assertEquals(resp.audioFile, "https://cdn.murf.ai/audio/123.wav");
  assertEquals(resp.audioDuration, 2.5);
});
