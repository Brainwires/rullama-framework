# brainwires-docs

Next.js 16 documentation site for the Brainwires framework. Serves the repo's markdown/MDX from the filesystem — no CMS, no build-time content pipeline, just `fs.readFileSync` behind server components.

## What It Reads

All doc content is read from the `DOCS_ROOT` directory, defined in [`src/lib/docs.ts`](src/lib/docs.ts):

- Env override: `DOCS_ROOT`
- Default (dev): two levels up from `extras/brainwires-docs/` (i.e. the framework repo root)
- Default (Docker): `/workspace`

Concrete paths resolved inside `DOCS_ROOT`:

| Content | Path |
|---------|------|
| Crate READMEs | `crates/<name>/README.md` |
| Extras READMEs | `extras/<name>/README.md` |
| Top-level guides | `FEATURES.md`, `CONTRIBUTING.md`, `TESTING.md`, `PUBLISHING.md` |
| Extensibility | `docs/EXTENSIBILITY.md` |
| Deno SDK docs | `deno/docs/*.md` |

The slug-to-file map for `/docs/[...slug]` routes lives in `DOC_SLUG_MAP` in `src/lib/docs.ts`. Filenames are allowlisted and symlink-escape checked before read.

## Adding a Doc Page

1. Drop the `.md` / `.mdx` file under `DOCS_ROOT` (typically `docs/` or a crate / extras folder).
2. If it needs a dedicated route, register it in `DOC_SLUG_MAP` (or `DENO_SLUG_MAP`) in `src/lib/docs.ts`.
3. Add it to the navigation in [`src/lib/nav.ts`](src/lib/nav.ts) by editing the `NAV_TREE` constant. Navigation is hand-authored — it is **not** auto-scanned from the filesystem.

Crate and extras READMEs are already wired to `/crates/[name]` and `/extras/[name]` generic routes; adding a new crate only requires a `NAV_TREE` entry.

## Development

This project uses **npm** (see `package-lock.json`). Canonical commands:

```sh
npm install
npm run dev      # next dev — http://localhost:3000
npm run build    # next build
npm run start    # next start
npm run lint     # eslint
npm test         # vitest run
```

To point the site at a different repo checkout:

```sh
DOCS_ROOT=/path/to/other/brainwires-framework npm run dev
```

## Deployment

A production container is provided:

```sh
docker compose up --build
```

See `Dockerfile` and `docker-compose.yml` for the build pipeline — the container sets `DOCS_ROOT=/workspace` and mounts the repo.

## Tests

Vitest covers the docs and nav helpers. Run with `npm test`. Test files live beside their modules (e.g. `src/lib/docs.test.ts`, `src/lib/nav.test.ts`, `src/lib/search.test.ts`).

## Search

The in-app search dialog (⌘K / Ctrl+K) is backed by a JSON index built at build time and queried on the client with [fuse.js](https://www.fusejs.io/).

- **Builder**: `src/lib/search.ts` walks `NAV_TREE`, resolves each href to its backing markdown via `DOC_SLUG_MAP` / `DENO_SLUG_MAP` / the crate + extra READMEs, and emits `{ href, title, section, body }` records. Frontmatter is stripped via `gray-matter`.
- **Build script**: `scripts/build-search-index.mjs` invokes the builder and writes the result to `public/search-index.json`. Run it with `npm run build:search`, or let `npm run build` do it for you — the search index is generated before every `next build`.
- **Output**: `public/search-index.json` (gitignored — it's a regenerable artifact).
- **Runtime**: `src/components/docs/search-dialog.tsx` fetches `/search-index.json` on first open, wraps it in a `Fuse` instance, and queries as the user types.

## Notes

This is Next.js 16. APIs, conventions, and file structure differ from older versions — consult `node_modules/next/dist/docs/` before making framework-level changes. Heed deprecation notices surfaced at build time.
