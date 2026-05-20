# brainwires-sandbox-proxy

Egress allowlist HTTP proxy used by `brainwires-sandbox` to enforce
`NetworkPolicy::Limited`. This crate ships a single binary —
`brainwires-sandbox-proxy` — that listens for plain HTTP requests (absolute
URI and `CONNECT`) and forwards only to hosts on its allowlist.

## Why

Docker has no built-in per-host egress controls. When a sandbox must reach
a specific short list of external services (e.g. `pypi.org`,
`api.anthropic.com`) and nothing else, the cleanest portable solution is:

1. Put the sandbox on an `internal: true` docker network so it has no route
   to the internet.
2. Run this proxy on that same network AND on the default bridge (so it can
   forward). Pass it the allowlist via `PROXY_ALLOW_HOSTS`.
3. Set `HTTP_PROXY`/`HTTPS_PROXY` inside the sandbox to point at the proxy.

`DockerSandbox::spawn()` wires all of this automatically when
`NetworkPolicy::Limited(Vec<String>)` is set.

## Configuration

| Env var              | Default        | Meaning                                           |
| -------------------- | -------------- | ------------------------------------------------- |
| `PROXY_ALLOW_HOSTS`  | _empty_        | Comma-separated hostnames. `*.example.com` wildcards supported. Empty ⇒ block everything. |
| `PROXY_LISTEN`       | `0.0.0.0:3128` | Listen socket.                                    |
| `PROXY_LOG`          | `info`         | `tracing-subscriber` EnvFilter directive.         |

Matching is case-insensitive. Ports in `CONNECT host:port` are stripped
before matching. Wildcards of the form `*.example.com` match any subdomain
and the bare apex `example.com`.

## Non-HTTP TCP

Not supported. The sandbox's internal network has no default route, so raw
TCP egress is blocked at the network level — by design. Reach for
`NetworkPolicy::Full` if you truly need raw TCP.

## Docker image

Build locally:

```
docker build -t ghcr.io/brainwires/brainwires-sandbox-proxy:latest \
  -f crates/brainwires-sandbox-proxy/Dockerfile .
```

The image is **not** published to any public registry yet. Build and push
it to your own registry (or run it locally) and set
`SandboxPolicy::proxy_image` accordingly.

## Consumed by

`brainwires-sandbox`'s `DockerSandbox` creates an ephemeral bridge network
per spawn, attaches the proxy container to both that network and the host
bridge, injects `HTTP_PROXY`/`HTTPS_PROXY` into the sandbox, and tears it
all down on `wait()` / `shutdown()`. See
`crates/brainwires-sandbox/src/docker.rs`.
