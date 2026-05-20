import { assertEquals } from "@std/assert";
import {
  _serializeRequest,
  GOOGLE_TTS_API_BASE,
  GoogleTtsClient,
} from "./google_tts.ts";

Deno.test("client creation", () => {
  const client = new GoogleTtsClient("test-key");
  assertEquals(client.base_url, GOOGLE_TTS_API_BASE);
});

Deno.test("synthesize request serialization", () => {
  const json = _serializeRequest({
    input: { text: "Hello world" },
    voice: { languageCode: "en-US", name: "en-US-Neural2-A" },
    audioConfig: { audioEncoding: "LINEAR16", sampleRateHertz: 24000 },
  }) as {
    input: { text: string };
    voice: { languageCode: string };
    audioConfig: { audioEncoding: string; sampleRateHertz?: number };
  };
  assertEquals(json.input.text, "Hello world");
  assertEquals(json.voice.languageCode, "en-US");
  assertEquals(json.audioConfig.sampleRateHertz, 24000);
});
