# ─── Stage 1: Build Rust API docs ─────────────────────────────────────────────
FROM rust:1.86-slim AS rust-docs
WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY extras/ ./extras/
COPY deprecated/ ./deprecated/
COPY deno/ ./deno/
COPY xtask/ ./xtask/
RUN cargo doc --workspace --no-deps 2>&1 || true && mkdir -p /workspace/target/doc

# ─── Stage 2: Build Next.js app ───────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app
COPY extras/brainwires-docs/package.json extras/brainwires-docs/package-lock.json* ./
RUN npm ci
COPY extras/brainwires-docs/ ./

WORKDIR /workspace-docs
COPY README.md FEATURES.md CHANGELOG.md CONTRIBUTING.md TESTING.md PUBLISHING.md ./
COPY docs/ ./docs/
COPY crates/ ./crates/
COPY extras/brainwires-cli/README.md ./extras/brainwires-cli/README.md
COPY extras/brainwires-proxy/README.md ./extras/brainwires-proxy/README.md
COPY extras/brainwires-brain-server/README.md ./extras/brainwires-brain-server/README.md
COPY extras/brainwires-rag-server/README.md ./extras/brainwires-rag-server/README.md
COPY extras/brainwires-issues/README.md ./extras/brainwires-issues/README.md
COPY extras/agent-chat/README.md ./extras/agent-chat/README.md
COPY extras/audio-demo/README.md ./extras/audio-demo/README.md
COPY extras/audio-demo-ffi/README.md ./extras/audio-demo-ffi/README.md
COPY extras/reload-daemon/README.md ./extras/reload-daemon/README.md
COPY deno/docs/ ./deno/docs/

WORKDIR /app
ENV DOCS_ROOT=/workspace-docs
ENV NODE_ENV=production
RUN npm run build

# ─── Stage 3: Production runner ───────────────────────────────────────────────
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
COPY --from=rust-docs /workspace/target/doc ./public/rustdoc

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
