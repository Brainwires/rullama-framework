import { assert, assertEquals } from "@std/assert";
import { AzureSpeechClient, type AzureSttResponse } from "./azure_speech.ts";

Deno.test("client creation", () => {
  const client = new AzureSpeechClient("test-key", "eastus");
  assert(client.ttsEndpoint().includes("eastus"));
});

Deno.test("stt response deserialization", () => {
  const json = `{
    "RecognitionStatus": "Success",
    "DisplayText": "Hello world.",
    "Offset": 0,
    "Duration": 10000000
  }`;
  const resp = JSON.parse(json) as AzureSttResponse;
  assertEquals(resp.RecognitionStatus, "Success");
  assertEquals(resp.DisplayText, "Hello world.");
});
