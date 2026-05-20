import { defineConfig, devices } from '@playwright/test';

// Persistent profile dir keeps the OPFS-cached gemma4:e2b model (~7 GB)
// across runs — first run downloads, subsequent runs are instant.
export default defineConfig({
    testDir: '.',
    testMatch: '**/*.pw.mjs',
    fullyParallel: false,
    workers: 1,
    timeout: 30 * 60_000,
    expect: { timeout: 30_000 },
    reporter: [['list']],
    use: {
        // chat.brainwires.dev is the user's deployed instance. Override
        // with `BW_PWA_URL=http://localhost:3000` for local serve.
        baseURL: process.env.BW_PWA_URL || 'https://chat.brainwires.dev',
        trace: 'retain-on-failure',
        video: 'retain-on-failure',
        // WebGPU is on by default in Chromium 113+; AMD GCN-4 surfaces
        // through Metal on macOS, no flags needed.
        launchOptions: {
            // Chromium's default --headless mode disables WebGPU. The
            // new `--headless=new` mode supports it, but Playwright's
            // headless flag still maps to old --headless. Force
            // chrome-headless-shell off and run "headed but offscreen"
            // by setting headless: false + a virtual display via the
            // shell wrapper. For now just run headed.
            headless: false,
        },
    },
    projects: [
        {
            name: 'chromium-webgpu',
            use: {
                ...devices['Desktop Chrome'],
                channel: 'chrome',
                viewport: { width: 1280, height: 800 },
                permissions: ['clipboard-read', 'clipboard-write'],
            },
        },
    ],
});
