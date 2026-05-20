# syntax=docker/dockerfile:1
#
# Build: docker build -t ghcr.io/brainwires/brainwires-sandbox-proxy:latest \
#           -f crates/brainwires-sandbox-proxy/Dockerfile .
#
# Build context MUST be the workspace root so Cargo can see the full
# workspace (path dependencies, Cargo.lock).

FROM rust:1.91-alpine AS builder
RUN apk add --no-cache musl-dev pkgconfig
WORKDIR /build
COPY . .
RUN cargo build --release -p brainwires-sandbox-proxy

FROM alpine:3
RUN apk add --no-cache ca-certificates && \
    addgroup -S proxy && adduser -S -G proxy proxy
COPY --from=builder /build/target/release/brainwires-sandbox-proxy /usr/local/bin/brainwires-sandbox-proxy
USER proxy
EXPOSE 3128
ENTRYPOINT ["/usr/local/bin/brainwires-sandbox-proxy"]
