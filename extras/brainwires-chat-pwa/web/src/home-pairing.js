// brainwires-chat-pwa — home-daemon pairing flow (Phase 2 M8).
//
// Three steps:
//   1. parseQrPayload(qrText)        — decode bwhome://pair?u=...&t=...&fp=...
//   2. claim({tunnelUrl, oneTimeToken, devicePubkey, deviceName}) → {ok}
//   3. confirm({tunnelUrl, oneTimeToken, code}) → {device_token, cf_*?, peer_pubkey}
//   4. savePairingBundle(bundle)     — encrypt + persist via crypto-store.js
//
// Storage shape (in IndexedDB `settings` store, key STORAGE_KEY):
//   - encrypted: { encrypted: <packed-blob> }   when session key is unlocked
//   - plaintext: { plaintext: <bundle-json> }   fallback for users who
//                                               opted out of encryption
//
// Device identity: an Ed25519 keypair persisted in the `settings` store
// under `device_identity_keypair`. We export the public half as raw bytes
// hex for `device_pubkey`, and keep the private half (as a non-extractable
// `CryptoKey`) for future signing in M11+. If Ed25519 isn't supported by
// `crypto.subtle` (older browsers, mostly older Safari), we transparently
// fall back to ECDSA P-256 — same wire shape (hex public key), different
// algorithm. The home daemon doesn't verify signatures in M8, so the
// algorithm choice is currently opaque to the wire.

import * as cryptoStore from '../crypto-store.js';
import { getSetting, setSetting } from './sql-db.js';
import { getSessionKey } from './state.js';

/** @typedef {{
 *    device_token: string,
 *    cf_client_id?: string,
 *    cf_client_secret?: string,
 *    peer_pubkey: string,
 *    tunnel_url: string,
 *    device_name: string,
 * }} PairingBundle
 */

const STORAGE_KEY = 'home_pairing_bundle';
const IDENTITY_KEY = 'device_identity_keypair';

// ── QR / bwhome:// URL parsing ────────────────────────────────

/**
 * Parse a `bwhome://pair?u=<url>&t=<token>&fp=<peer_fingerprint>` URL.
 * Throws on malformed input. `fp` is optional — older daemons may omit it.
 *
 * @param {string} qrText
 * @returns {{ tunnelUrl: string, oneTimeToken: string, peerFingerprint: string }}
 */
export function parseQrPayload(qrText) {
    if (typeof qrText !== 'string' || qrText.length === 0) {
        throw new Error('parseQrPayload: empty input');
    }
    const trimmed = qrText.trim();
    if (!trimmed.startsWith('bwhome://pair')) {
        throw new Error('parseQrPayload: not a bwhome://pair URL');
    }
    // Split on the first '?'. URL won't parse a custom scheme reliably.
    const qIndex = trimmed.indexOf('?');
    if (qIndex < 0) throw new Error('parseQrPayload: no query string');
    const query = trimmed.slice(qIndex + 1);
    const params = new URLSearchParams(query);
    const tunnelUrl = params.get('u');
    const oneTimeToken = params.get('t');
    const peerFingerprint = params.get('fp') || '';
    if (!tunnelUrl) throw new Error('parseQrPayload: missing u (tunnel URL)');
    if (!oneTimeToken) throw new Error('parseQrPayload: missing t (one_time_token)');
    if (!/^https?:\/\//i.test(tunnelUrl)) {
        throw new Error('parseQrPayload: tunnel URL must be http(s)');
    }
    return { tunnelUrl, oneTimeToken, peerFingerprint };
}

// ── HTTP wire calls ───────────────────────────────────────────

/**
 * POST `/pair/claim`. Throws on non-2xx. Returns `{ok: true}` on success.
 *
 * @param {{
 *   tunnelUrl: string,
 *   oneTimeToken: string,
 *   devicePubkey: string,
 *   deviceName: string,
 *   fetchImpl?: typeof fetch,
 * }} args
 */
export async function claim({ tunnelUrl, oneTimeToken, devicePubkey, deviceName, fetchImpl }) {
    const f = fetchImpl || globalThis.fetch;
    if (typeof f !== 'function') throw new Error('pairing: no fetch available');
    const url = `${tunnelUrl.replace(/\/$/, '')}/pair/claim`;
    let resp;
    try {
        resp = await f(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                one_time_token: oneTimeToken,
                device_pubkey: devicePubkey,
                device_name: deviceName,
            }),
        });
    } catch (e) {
        throw new Error(`pairing: claim network error: ${e && e.message ? e.message : e}`);
    }
    if (!resp.ok) {
        if (resp.status === 404) throw new Error('pairing: claim failed — token unknown or expired');
        throw new Error(`pairing: claim failed: HTTP ${resp.status}`);
    }
    return await resp.json();
}

/**
 * POST `/pair/confirm`. Throws on non-2xx. Returns the bundle the daemon
 * sent (device_token, optional cf_*, peer_pubkey) on success.
 *
 * @param {{
 *   tunnelUrl: string,
 *   oneTimeToken: string,
 *   code: string,
 *   fetchImpl?: typeof fetch,
 * }} args
 * @returns {Promise<{device_token: string, cf_client_id?: string, cf_client_secret?: string, peer_pubkey: string}>}
 */
export async function confirm({ tunnelUrl, oneTimeToken, code, fetchImpl }) {
    const f = fetchImpl || globalThis.fetch;
    if (typeof f !== 'function') throw new Error('pairing: no fetch available');
    const url = `${tunnelUrl.replace(/\/$/, '')}/pair/confirm`;
    let resp;
    try {
        resp = await f(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ one_time_token: oneTimeToken, code }),
        });
    } catch (e) {
        throw new Error(`pairing: confirm network error: ${e && e.message ? e.message : e}`);
    }
    if (resp.status === 401) throw new Error('pairing: wrong 6-digit code');
    if (resp.status === 404) throw new Error('pairing: token unknown or expired');
    if (resp.status === 400) throw new Error('pairing: must claim before confirm');
    if (!resp.ok) throw new Error(`pairing: confirm failed: HTTP ${resp.status}`);
    const body = await resp.json();
    if (!body || typeof body.device_token !== 'string' || typeof body.peer_pubkey !== 'string') {
        throw new Error('pairing: confirm response missing device_token / peer_pubkey');
    }
    return body;
}

// ── Bundle persistence (encrypted via crypto-store.js) ─────────

/**
 * Persist a paired-bundle to IndexedDB. Encrypted under the user's session
 * key when one is loaded; falls back to plaintext otherwise (matches the
 * existing API-key storage pattern in `ui-settings.js`).
 *
 * @param {PairingBundle} bundle
 */
export async function savePairingBundle(bundle) {
    const json = JSON.stringify(bundle);
    const sessionKey = getSessionKey();
    if (sessionKey) {
        const salt = cryptoStore.generateSalt();
        const blob = await cryptoStore.encrypt(sessionKey, json);
        const packed = cryptoStore.pack({ salt, iv: blob.iv, ciphertext: blob.ciphertext });
        await setSetting(STORAGE_KEY, { encrypted: packed });
    } else {
        await setSetting(STORAGE_KEY, { plaintext: json });
    }
}

/**
 * Load and decrypt the paired-bundle. Returns `null` when nothing has
 * been paired or the session is locked and the bundle was encrypted.
 *
 * @returns {Promise<PairingBundle | null>}
 */
export async function loadPairingBundle() {
    const row = await getSetting(STORAGE_KEY);
    if (!row) return null;
    if (row.plaintext) {
        try { return JSON.parse(row.plaintext); } catch (_) { return null; }
    }
    if (row.encrypted) {
        const sessionKey = getSessionKey();
        if (!sessionKey) return null;
        try {
            const parts = cryptoStore.unpack(row.encrypted);
            const json = await cryptoStore.decrypt(sessionKey, {
                iv: parts.iv,
                ciphertext: parts.ciphertext,
            });
            return JSON.parse(json);
        } catch (_) {
            return null;
        }
    }
    return null;
}

/** Forget the paired-bundle. Idempotent. */
export async function clearPairingBundle() {
    await setSetting(STORAGE_KEY, undefined);
}

// ── Device identity (Ed25519 with ECDSA P-256 fallback) ────────

/**
 * Get this device's stable identity. Generates one on first call and
 * persists it in IndexedDB (private half is non-extractable; only the
 * public half is exported as hex for the wire).
 *
 * @returns {Promise<{ algorithm: 'Ed25519' | 'ECDSA-P-256', publicKeyHex: string }>}
 */
export async function deviceIdentity() {
    const existing = await getSetting(IDENTITY_KEY);
    if (existing && existing.publicKeyHex && existing.algorithm) {
        return {
            algorithm: existing.algorithm,
            publicKeyHex: existing.publicKeyHex,
        };
    }

    // Try Ed25519 first. Most evergreen browsers support it as of 2026
    // (Chromium 130+, Firefox 130+, Safari 17+). Older runtimes throw
    // NotSupportedError or DataError — fall through to ECDSA P-256.
    let kp;
    let algorithm;
    try {
        kp = await crypto.subtle.generateKey({ name: 'Ed25519' }, false, ['sign', 'verify']);
        algorithm = 'Ed25519';
    } catch (_e1) {
        kp = await crypto.subtle.generateKey(
            { name: 'ECDSA', namedCurve: 'P-256' },
            false,
            ['sign', 'verify'],
        );
        algorithm = 'ECDSA-P-256';
    }

    const raw = await crypto.subtle.exportKey('raw', kp.publicKey);
    const publicKeyHex = bytesToHex(new Uint8Array(raw));

    // We don't actually persist the private key (M11 will). Storing only
    // the public half keeps the on-disk surface tiny — and a future swap
    // to a real signing identity is one IDB write away.
    await setSetting(IDENTITY_KEY, { algorithm, publicKeyHex });
    return { algorithm, publicKeyHex };
}

function bytesToHex(bytes) {
    const hex = '0123456789abcdef';
    let out = '';
    for (let i = 0; i < bytes.length; i++) {
        out += hex[bytes[i] >> 4];
        out += hex[bytes[i] & 0xf];
    }
    return out;
}
