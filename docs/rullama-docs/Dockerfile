# ─── Stage 1: Build Rust API docs ─────────────────────────────────────────────
FROM rust:1.86-slim AS rust-docs
WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY sdks/ ./sdks/
COPY servers/ ./servers/
COPY integrations/ ./integrations/
COPY examples/ ./examples/
COPY deno/ ./deno/
COPY xtask/ ./xtask/
RUN cargo doc --workspace --no-deps 2>&1 || true && mkdir -p /workspace/target/doc

# ─── Stage 2: Build Next.js app ───────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app
COPY docs/rullama-docs/package.json docs/rullama-docs/package-lock.json* ./
RUN npm ci
COPY docs/rullama-docs/ ./

WORKDIR /workspace-docs
COPY README.md FEATURES.md CHANGELOG.md CONTRIBUTING.md TESTING.md PUBLISHING.md ./
COPY docs/ ./docs/
COPY crates/ ./crates/
# Extras READMEs now live in sdks/ servers/ integrations/ examples/ (the old
# flat extras/ dir was removed). docs.ts resolves an extra by trying each of
# these dirs, so copy each README into its real location.
COPY sdks/rullama-proxy/README.md ./sdks/rullama-proxy/README.md
COPY servers/rullama-brain-server/README.md ./servers/rullama-brain-server/README.md
COPY servers/rullama-rag-server/README.md ./servers/rullama-rag-server/README.md
COPY servers/rullama-issues/README.md ./servers/rullama-issues/README.md
COPY examples/agent-chat/README.md ./examples/agent-chat/README.md
COPY examples/audio-demo/README.md ./examples/audio-demo/README.md
COPY examples/audio-demo-ffi/README.md ./examples/audio-demo-ffi/README.md
COPY integrations/reload-daemon/README.md ./integrations/reload-daemon/README.md
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
