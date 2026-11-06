/**
 * @module transport
 *
 * Transport layer re-exports: stdio, unix-socket and TCP server transports,
 * plus the embeddable stream transport/listener for host-provided sockets.
 */

export type { ConnectionInfo, ServerTransport } from "./traits.ts";
export { StdioServerTransport } from "./stdio.ts";
export { LineReader, MAX_LINE_BYTES } from "./framing.ts";
export {
  type AcceptedStreamConnection,
  type ByteStreamEndpoint,
  type StreamConnection,
  type StreamListenOptions,
  StreamServerListener,
  StreamServerTransport,
  type StreamTransportOptions,
} from "./stream.ts";
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
