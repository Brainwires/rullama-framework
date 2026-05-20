// brainwires-chat-pwa — hidden developer dial-home toggle.
//
// Activates only when one of these is true:
//   * the URL has `?home=<base-url>` (e.g. `?home=http://127.0.0.1:7878`)
//   * localStorage has `bw_dial_home_url` set to a base URL
//
// When active, exposes `window.bwDialHome` with helpers a developer can
// drive from DevTools to validate the full handshake against a local
// `cargo run -p brainwires-home`. Not visible to end users; M9 wires
// the production chat-UI integration.
//
// Console workflow:
//   await bwDialHome.connect()           // resolves once data channel is open
//   await bwDialHome.ping()              // round-trips system/ping → { ok, ts }
//   await bwDialHome.send('hello')       // sends an A2A message/send
//   await bwDialHome.disconnect()
//   bwDialHome.transport                 // current HomeTransport (or null)
//   bwDialHome.signaling                 // current SignalingClient

import { SignalingClient } from './home-signaling.js';
import { HomeTransport } from './home-transport.js';

const STORAGE_KEY = 'bw_dial_home_url';

function readBaseUrl() {
    try {
        const params = new URL(location.href).searchParams;
        const q = params.get('home');
        if (q && /^https?:\/\//i.test(q)) return q;
    } catch (_) { /* SSR / older runtimes */ }
    try {
        const v = localStorage.getItem(STORAGE_KEY);
        if (v && /^https?:\/\//i.test(v)) return v;
    } catch (_) { /* private mode */ }
    return null;
}

/**
 * Wire `window.bwDialHome` if the dev flag is set. Returns true if the
 * toggle activated, false if it was skipped (no flag).
 */
export function maybeInstallDevToggle() {
    const base = readBaseUrl();
    if (!base) return false;
    if (typeof window === 'undefined') return false;

    let transport = null;
    let signaling = null;

    const api = {
        get baseUrl() { return base; },
        get transport() { return transport; },
        get signaling() { return signaling; },
        get state() { return transport ? transport.state : 'idle'; },

        async connect() {
            if (transport && transport.state !== 'closed' && transport.state !== 'failed' && transport.state !== 'idle') {
                throw new Error(`bwDialHome: already ${transport.state}`);
            }
            signaling = new SignalingClient({ baseUrl: base });
            transport = new HomeTransport({ signaling });
            await transport.connect();
            return transport;
        },

        async disconnect() {
            if (!transport) return;
            await transport.close();
        },

        async ping() {
            if (!transport || transport.state !== 'connected') {
                throw new Error('bwDialHome.ping: not connected — call connect() first');
            }
            return await transport.request('system/ping', {});
        },

        async send(text) {
            if (!transport || transport.state !== 'connected') {
                throw new Error('bwDialHome.send: not connected — call connect() first');
            }
            // Match the A2A message/send shape the home daemon understands.
            return await transport.request('message/send', {
                message: {
                    role: 'user',
                    parts: [{ kind: 'text', text: String(text) }],
                },
            });
        },

        async fetchAgentCard() {
            const sc = signaling || new SignalingClient({ baseUrl: base });
            return await sc.fetchAgentCard();
        },
    };

    // Defining as a non-enumerable property keeps it out of for-in loops
    // and `Object.keys(window)` enumerations end users might run.
    try {
        Object.defineProperty(window, 'bwDialHome', {
            value: api,
            configurable: true,
            enumerable: false,
            writable: false,
        });
    } catch (_) {
        // Some sandboxed contexts disallow defineProperty on window — fall
        // back to a plain assignment.
        window.bwDialHome = api;
    }

    console.log(`[bwDialHome] dev toggle armed against ${base} — try \`await bwDialHome.connect()\``);
    return true;
}
