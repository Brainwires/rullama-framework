// End-to-end harness for the chat-pwa quantized local-model flow.
//
// What this test does:
//   1. Boots a Playwright Chromium with WebGPU.
//   2. Navigates to http://localhost:8080/ (the local nginx).
//   3. Drives the unlock UI with a fixed test passphrase.
//   4. Activates the gemma4:e2b local model (downloads via local
//      ~/.ollama/ proxy on first run; reuses OPFS thereafter).
//   5. Sends "Hi" and waits for the assistant bubble to render.
//   6. Asserts the bubble has non-empty text content AND captures the
//      [gemma4/text] / [gemma4/perf] console lines so the bisect can
//      see the per-step trajectory without needing a human in the loop.
//
// To rerun without redownloading the model, the persistent context
// dir under .pw-profile/ keeps OPFS warm across runs.

import { test, expect, chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROFILE_DIR = path.resolve(__dirname, '../../.pw-profile');
mkdirSync(PROFILE_DIR, { recursive: true });

const PASSPHRASE = 'pw-test-passphrase-do-not-use-elsewhere';
const PROMPT_TEXT = 'Hi';

test('quantized local gemma4:e2b — generate, stream, render', async () => {
    const context = await chromium.launchPersistentContext(PROFILE_DIR, {
        headless: false,
        viewport: { width: 1280, height: 800 },
        args: [
            // Force discrete GPU pick on AMD/Intel laptops.
            '--enable-features=Vulkan',
        ],
    });
    const page = context.pages()[0] || await context.newPage();

    const consoleLines = [];
    const errors = [];
    page.on('console', (msg) => {
        const text = msg.text();
        consoleLines.push({ type: msg.type(), text, ts: Date.now() });
        // Mirror the model traces to test stdout so a tail -F gives the
        // operator the same visibility they'd have in DevTools.
        if (
            text.includes('[gemma4/text]')
            || text.includes('[gemma4/perf]')
            || text.includes('[gemma4/diag]')
            || text.includes('[wasm/gguf')
            || text.includes('[local-worker]')
            || text.includes('[bw-sw]')
        ) {
            // eslint-disable-next-line no-console
            console.log(`[browser] ${text}`);
        }
    });
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('crash', () => errors.push('page-crashed'));

    // ── Boot ─────────────────────────────────────────────────────────
    await page.goto(process.env.BW_PWA_URL || 'http://localhost:8080/');
    // Wait for the SW to claim, otherwise fetches race against
    // network-only first-load.
    await page.waitForFunction(
        () => navigator.serviceWorker && navigator.serviceWorker.controller,
        { timeout: 30_000 },
    );

    // ── Confirm WebGPU is actually available ────────────────────────
    const hasWebGPU = await page.evaluate(async () => {
        try {
            if (!navigator.gpu) return { ok: false, reason: 'no navigator.gpu' };
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return { ok: false, reason: 'no adapter' };
            const info = await adapter.requestAdapterInfo?.() || {};
            return {
                ok: true,
                vendor: info.vendor || '?',
                architecture: info.architecture || '?',
                description: info.description || '?',
            };
        } catch (e) {
            return { ok: false, reason: String(e) };
        }
    });
    console.log('[pw] webgpu adapter:', hasWebGPU);
    expect(hasWebGPU.ok, 'WebGPU adapter must be reachable').toBeTruthy();

    // ── Unlock or set passphrase ────────────────────────────────────
    // First-run flow shows a "create passphrase" form; subsequent runs
    // show "unlock". Both submit the same field. If neither is present
    // the app booted into a previously-unlocked session.
    const passphraseInput = page.locator(
        'input[type="password"], input[name="passphrase"], input[autocomplete="new-password"], input[autocomplete="current-password"]',
    ).first();
    if (await passphraseInput.count()) {
        await passphraseInput.fill(PASSPHRASE);
        // Submit on Enter; the unlock form intercepts default submit.
        await passphraseInput.press('Enter');
        // Some flows ask to confirm — fill the second box if present.
        const confirmInput = page.locator(
            'input[name="passphrase-confirm"], input[autocomplete="new-password"]',
        ).nth(1);
        if (await confirmInput.count() > 0 && await confirmInput.isVisible().catch(() => false)) {
            await confirmInput.fill(PASSPHRASE);
            await confirmInput.press('Enter');
        }
    }

    // ── Activate gemma4:e2b ────────────────────────────────────────
    // The Settings → Local Model card has a "Use" button when the
    // model is already downloaded; if not downloaded, a "Download"
    // button kicks the OPFS-write worker. The test waits for whichever
    // affordance the UI is showing and clicks through.
    //
    // Selectors are tolerant of the i18n string changes — we match by
    // role+name in lower-case, with both the english "Use" and the
    // explicit gemma4:e2b model row scope.
    await page.evaluate(() => {
        // Open settings sidebar if it isn't already.
        const opener = document.querySelector('[data-testid="open-settings"], button[aria-label="Settings"], button[aria-label="settings"]');
        if (opener) opener.click();
    });

    // Best-effort: click the Local Model "Use" button if visible. If
    // the button's already in the "✓ In use" state, the click is a
    // no-op. If the model isn't downloaded yet, the test will block
    // here on a long timer to finish the (one-time) download.
    //
    // The settings panel scopes both Edge models — the smaller
    // gemma4:e2b (Ollama Q4_K_M GGUF, ~7 GB) and "Gemma 4 E2B IT"
    // (HF BF16 safetensors, ~10 GB). We use the GGUF row — it's the
    // path the user has cached locally via ~/.ollama, the local
    // nginx proxy serves it off disk at sendfile speed, and it's
    // already activated in their fresh Playwright profile.
    const gemmaCard = page.locator('div', {
        has: page.locator('h3', { hasText: /Gemma 4 E2B \(Ollama, Q4_K_M\)/ }),
    }).first();
    const inUse = gemmaCard.locator('button:has-text("In use")').first();
    if (await inUse.count() === 0) {
        const useButton = gemmaCard.locator('button', { hasText: /^Use$/i }).first();
        if (await useButton.count() && await useButton.isEnabled().catch(() => false)) {
            await useButton.click();
        } else {
            const downloadButton = gemmaCard.locator('button', { hasText: /^Download$/i }).first();
            if (await downloadButton.count()) {
                console.log('[pw] downloading model — first-run only, this may take several minutes');
                await downloadButton.click();
                await gemmaCard.locator('button:has-text("Use"), button:has-text("In use")').first()
                    .waitFor({ timeout: 30 * 60_000 });
                const u2 = gemmaCard.locator('button', { hasText: /^Use$/i }).first();
                if (await u2.count() && await u2.isEnabled().catch(() => false)) {
                    await u2.click();
                }
            }
        }
    } else {
        console.log('[pw] gemma4:e2b already in use');
    }

    // ── Leave Settings (if we're in it) ───────────────────────────
    // Two ways to detect we're in Settings: a settings region is in
    // the DOM, OR the composer is hidden. Either way, click the Back
    // icon button in the settings header. NEVER call page.goBack() —
    // the SPA's last-active route can be Settings, and goBack walks
    // out of the SPA entry into about:blank.
    const settingsHeader = page.locator('header.settings-header').first();
    if (await settingsHeader.count()) {
        const backIcon = settingsHeader.locator('button.icon-btn').first();
        await backIcon.click();
        // The chat view mounts asynchronously; give it a moment to
        // attach the composer before the next selector wait.
        await page.waitForTimeout(250);
    }
    await page.waitForSelector('textarea.composer-input', { state: 'visible', timeout: 30_000 });

    // ── Send a chat message ────────────────────────────────────────
    const chatInput = page.locator('textarea, input[type="text"]').filter({
        hasNot: page.locator('[type="password"]'),
    }).first();
    await chatInput.waitFor({ state: 'visible', timeout: 60_000 });
    await chatInput.fill(PROMPT_TEXT);
    await chatInput.press('Enter');

    // ── Wait for streaming to complete ─────────────────────────────
    // Two-phase wait:
    //   Phase 1 — wait until the wasm pipeline emits its first
    //     [gemma4/text] step line (the prefill+first-decode forward
    //     can take 5–15 s) OR the wasm errors out. We watch the
    //     console-line buffer rather than the bubble because the
    //     bubble may legitimately stay empty (the bridge bug).
    //   Phase 2 — once generation has started, poll the bubble.
    //     A stable len > 0 for DONE_STABLE_MS = generation finished
    //     and the UI rendered it. A stable len = 0 for
    //     DONE_EMPTY_STABLE_MS after generation started =
    //     streaming-to-UI bridge is broken (the load-bearing failure
    //     mode we're trying to detect).
    const DONE_STABLE_MS = 5_000;
    const DONE_EMPTY_STABLE_MS = 30_000;
    const PHASE1_TIMEOUT_MS = 90_000;
    const PHASE2_TIMEOUT_MS = 5 * 60_000;
    const countTextLines = () => consoleLines.filter((l) => l.text.includes('[gemma4/text]')).length;
    const phase1Start = Date.now();
    while (countTextLines() < 1 && Date.now() - phase1Start < PHASE1_TIMEOUT_MS) {
        await new Promise((r) => setTimeout(r, 500));
    }
    if (countTextLines() < 1) {
        console.warn('[pw] no [gemma4/text] line within 90s — wasm forward never produced a token');
    } else {
        console.log('[pw] first token emitted by wasm — entering bubble-stable wait');
    }

    const phase2Start = Date.now();
    let lastLen = -1;
    let stableSince = Date.now();
    while (Date.now() - phase2Start < PHASE2_TIMEOUT_MS) {
        const len = await page.evaluate(() => {
            const bubbles = document.querySelectorAll('.bubble.bubble-assistant');
            if (!bubbles.length) return 0;
            const last = bubbles[bubbles.length - 1];
            const body = last.querySelector('.bubble-body, .bubble-content');
            return ((body || last).textContent || '').length;
        });
        if (len !== lastLen) {
            lastLen = len;
            stableSince = Date.now();
        } else {
            const stableFor = Date.now() - stableSince;
            const threshold = len > 0 ? DONE_STABLE_MS : DONE_EMPTY_STABLE_MS;
            if (stableFor > threshold) {
                const tokens = countTextLines();
                console.log(`[pw] bubble stable at len=${len} for ${stableFor}ms (wasm emitted ${tokens} tokens) — done`);
                break;
            }
        }
        await new Promise((r) => setTimeout(r, 500));
    }

    // ── Read back the assistant bubble text ─────────────────────────
    const lastBubbleText = await page.evaluate(() => {
        const bubbles = document.querySelectorAll('.bubble.bubble-assistant');
        if (!bubbles.length) return null;
        const last = bubbles[bubbles.length - 1];
        const body = last.querySelector('.bubble-body, .bubble-content');
        return (body || last).textContent || '';
    });
    console.log('[pw] last bubble text:', JSON.stringify(lastBubbleText));

    // ── Pull the per-step trace lines for the bisect log ───────────
    const textLines = consoleLines
        .filter((l) => l.text.includes('[gemma4/text]') || l.text.includes('[gemma4/perf]'))
        .map((l) => l.text);
    console.log(`[pw] captured ${textLines.length} model trace lines`);

    // ── Assertions ─────────────────────────────────────────────────
    expect(errors, `page errors: ${errors.join(' | ')}`).toEqual([]);
    expect(lastBubbleText, 'assistant bubble must have non-empty text').toBeTruthy();

    await context.close();
});
