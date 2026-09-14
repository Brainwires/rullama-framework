/**
 * Parser for the AWS `application/vnd.amazon.eventstream` binary framing that
 * Bedrock's `invoke-with-response-stream` returns. Each frame is:
 *
 * ```
 * total length (u32) | headers length (u32) | prelude CRC (u32)
 * headers: [name length (u8) | name | type (u8) | value length (u16) | value]*
 * payload | message CRC (u32)
 * ```
 *
 * Bedrock's `chunk` events carry a JSON payload `{ "bytes": "<base64>" }` whose
 * decoded bytes are the model's own stream event (Anthropic Messages format).
 * CRCs are not verified (TLS already covers integrity).
 *
 * @module
 */

/** One decoded frame: string headers plus the raw payload. */
export interface EventStreamFrame {
  /** Header values (string-typed headers only; others are skipped). */
  headers: Record<string, string>;
  /** Raw payload bytes. */
  payload: Uint8Array;
}

const PRELUDE_BYTES = 12;
const TRAILER_BYTES = 4;
const HEADER_TYPE_STRING = 7;

/** Parse the header block of one frame. */
function parseHeaders(bytes: Uint8Array): Record<string, string> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dec = new TextDecoder();
  const headers: Record<string, string> = {};
  let pos = 0;
  while (pos < bytes.byteLength) {
    const nameLen = view.getUint8(pos);
    const name = dec.decode(bytes.subarray(pos + 1, pos + 1 + nameLen));
    pos += 1 + nameLen;
    const type = view.getUint8(pos);
    pos += 1;
    if (type !== HEADER_TYPE_STRING) break; // only string headers occur in Bedrock streams
    const valueLen = view.getUint16(pos);
    headers[name] = dec.decode(bytes.subarray(pos + 2, pos + 2 + valueLen));
    pos += 2 + valueLen;
  }
  return headers;
}

/** Decode one complete frame starting at `buf[0]`; returns `undefined` if incomplete. */
export function decodeFrame(
  buf: Uint8Array,
): { frame: EventStreamFrame; consumed: number } | undefined {
  if (buf.byteLength < PRELUDE_BYTES) return undefined;
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const total = view.getUint32(0);
  const headersLen = view.getUint32(4);
  if (buf.byteLength < total) return undefined;
  const headers = parseHeaders(
    buf.subarray(PRELUDE_BYTES, PRELUDE_BYTES + headersLen),
  );
  const payload = buf.subarray(
    PRELUDE_BYTES + headersLen,
    total - TRAILER_BYTES,
  );
  return { frame: { headers, payload }, consumed: total };
}

/** Split a byte stream into event-stream frames. */
export async function* parseEventStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<EventStreamFrame> {
  const reader = body.getReader();
  let buf = new Uint8Array(0);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      const joined = new Uint8Array(buf.byteLength + value.byteLength);
      joined.set(buf);
      joined.set(value, buf.byteLength);
      buf = joined;
      for (;;) {
        const decoded = decodeFrame(buf);
        if (!decoded) break;
        yield decoded.frame;
        buf = buf.subarray(decoded.consumed);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Yield the JSON events inside Bedrock `chunk` frames (`{"bytes": base64}`),
 * already decoded and parsed. Exception frames are surfaced as thrown errors.
 */
export async function* parseBedrockEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<Record<string, unknown>> {
  const dec = new TextDecoder();
  for await (const frame of parseEventStream(body)) {
    if (frame.headers[":message-type"] === "exception") {
      throw new Error(
        `Bedrock stream exception (${
          frame.headers[":exception-type"] ?? "unknown"
        }): ${dec.decode(frame.payload)}`,
      );
    }
    if (frame.headers[":event-type"] !== "chunk") continue;
    const wrapper = JSON.parse(dec.decode(frame.payload)) as { bytes?: string };
    if (typeof wrapper.bytes !== "string") continue;
    yield JSON.parse(atob(wrapper.bytes)) as Record<string, unknown>;
  }
}

/** Build one event-stream frame (test helper; CRCs are zero). */
export function encodeFrame(
  headers: Record<string, string>,
  payload: Uint8Array,
): Uint8Array {
  const enc = new TextEncoder();
  const headerParts: Uint8Array[] = [];
  for (const [name, value] of Object.entries(headers)) {
    const n = enc.encode(name);
    const v = enc.encode(value);
    const part = new Uint8Array(1 + n.byteLength + 1 + 2 + v.byteLength);
    const view = new DataView(part.buffer);
    part[0] = n.byteLength;
    part.set(n, 1);
    part[1 + n.byteLength] = HEADER_TYPE_STRING;
    view.setUint16(2 + n.byteLength, v.byteLength);
    part.set(v, 4 + n.byteLength);
    headerParts.push(part);
  }
  const headersLen = headerParts.reduce((s, p) => s + p.byteLength, 0);
  const total = PRELUDE_BYTES + headersLen + payload.byteLength + TRAILER_BYTES;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, total);
  view.setUint32(4, headersLen);
  let pos = PRELUDE_BYTES;
  for (const p of headerParts) {
    out.set(p, pos);
    pos += p.byteLength;
  }
  out.set(payload, pos);
  return out;
}
