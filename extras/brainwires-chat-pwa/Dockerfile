# Multi-stage build for brainwires-chat-pwa.
#
# Build context MUST be the framework workspace root, because the wasm crate
# at extras/brainwires-chat-pwa/wasm depends on `../../../crates/brainwires-*`
# via path = "...".
#
#   docker build -f extras/brainwires-chat-pwa/Dockerfile -t brainwires-chat-pwa .

# ── Stage 1: build the static bundle ───────────────────────────────────────
FROM rust:1-bookworm AS web-builder

# Node.js 20 (LTS) for esbuild + npm. Pulled from NodeSource so the version
# is pinned and reproducible across base-image updates.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# wasm-pack drives the wasm32 build. --locked pins the version recorded in
# Cargo.lock at install time so the toolchain stays deterministic.
RUN cargo install wasm-pack --locked --version 0.13.1

# Add the wasm32 target up front; cached as its own layer.
RUN rustup target add wasm32-unknown-unknown

WORKDIR /build
COPY . .

# Install JS deps and run the existing build pipeline (wasm-pack + esbuild +
# SRI patcher). Output lands under extras/brainwires-chat-pwa/web/.
RUN cd extras/brainwires-chat-pwa/web \
    && npm ci --no-audit --no-fund \
    && ./build.sh

# ── Stage 2: serve via nginx ───────────────────────────────────────────────
FROM nginx:alpine

# envsubst for entrypoint, ca-certificates so any future fetch in the
# entrypoint can validate TLS.
RUN apk add --no-cache gettext ca-certificates

COPY extras/brainwires-chat-pwa/nginx.conf /etc/nginx/conf.d/default.conf

# Static bundle. Only the runtime files — src/, tests/, node_modules/, and
# build intermediates (sw.bundle.js) are deliberately omitted.
WORKDIR /usr/share/nginx/html
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/index.html      ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/manifest.json   ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/styles.css      ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/app.js              ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/app.js.map          ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/local-worker.js     ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/local-worker.js.map ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/sw.js               ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/build-info.js   ./
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/icons           ./icons
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/lang            ./lang
COPY --from=web-builder /build/extras/brainwires-chat-pwa/web/pkg             ./pkg

# Build metadata. DEV_MODE defaults to false; override with --build-arg or
# at runtime via the env var (entrypoint reads the env var first).
ARG BUILD_DATE=""
ARG BUILD_COMMIT=""
ARG BUILD_VERSION="0.1.0"
ARG DEV_MODE=false
ENV BRAINWIRES_BUILD_DATE="$BUILD_DATE" \
    BRAINWIRES_BUILD_COMMIT="$BUILD_COMMIT" \
    BRAINWIRES_BUILD_VERSION="$BUILD_VERSION" \
    BRAINWIRES_DEV_MODE="$DEV_MODE"

COPY extras/brainwires-chat-pwa/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
CMD ["/entrypoint.sh"]
