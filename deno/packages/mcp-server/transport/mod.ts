/**
 * @module transport
 *
 * Transport layer re-exports: stdio, unix-socket and TCP server transports.
 */

export type { ConnectionInfo, ServerTransport } from "./traits.ts";
export { StdioServerTransport } from "./stdio.ts";
export { LineReader, MAX_LINE_BYTES } from "./framing.ts";
export {
  type AcceptedConnection,
  type PreAuthHook,
  type PreAuthResult,
  type SocketListenOptions,
  SocketServerListener,
  SocketServerTransport,
  type TcpListenOptions,
  TcpServerTransport,
  type UnixSocketListenOptions,
  UnixSocketServerTransport,
} from "./socket.ts";
