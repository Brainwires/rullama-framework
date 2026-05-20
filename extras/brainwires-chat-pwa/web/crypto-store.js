// brainwires-chat-pwa — encrypted-storage primitives
//
// AES-256-GCM with PBKDF2-SHA-256 (600 000 iters, OWASP 2023+ minimum)
// via the Web Crypto API. Zero external dependencies. The same module
// is consumed by the page (passphrase unlock) and the service worker
// (per-stream API-key decrypt) — both runtimes have `crypto.subtle`.
//
// Wire format (`pack`/`unpack`):
//   base64url( salt(16) ‖ iv(12) ‖ ciphertext+tag )
// `salt` is included in the packed blob so decryption only needs the
// passphrase + the blob; nothing else has to be tracked alongside.

const PBKDF2_ITERATIONS = 600_000;
const SALT_LEN = 16;
const IV_LEN = 12;

/**
 * Derive an AES-256-GCM key from a passphrase + salt.
 *
 * @param {string} passphrase
 * @param {Uint8Array} salt
 * @returns {Promise<CryptoKey>}
 */
export async function deriveKey(passphrase, salt) {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        'raw',
        enc.encode(passphrase),
        { name: 'PBKDF2' },
        false,
        ['deriveKey'],
    );
    return crypto.subtle.deriveKey(
        { name: 'PBKDF2', salt, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
        keyMaterial,
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt'],
    );
}

/**
 * Encrypt a UTF-8 string under an AES-GCM key. Generates a fresh 12-byte IV.
 *
 * @param {CryptoKey} key
 * @param {string} plaintext
 * @returns {Promise<{ iv: Uint8Array, ciphertext: Uint8Array }>}
 */
export async function encrypt(key, plaintext) {
    const iv = crypto.getRandomValues(new Uint8Array(IV_LEN));
    const enc = new TextEncoder();
    const ct = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        key,
        enc.encode(plaintext),
    );
    return { iv, ciphertext: new Uint8Array(ct) };
}

/**
 * Decrypt an AES-GCM blob. Throws on auth-tag failure.
 *
 * @param {CryptoKey} key
 * @param {{ iv: Uint8Array, ciphertext: Uint8Array }} blob
 * @returns {Promise<string>}
 */
export async function decrypt(key, blob) {
    const plain = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv: blob.iv },
        key,
        blob.ciphertext,
    );
    return new TextDecoder().decode(plain);
}

/**
 * Generate a fresh 16-byte salt suitable for PBKDF2.
 *
 * @returns {Uint8Array}
 */
export function generateSalt() {
    return crypto.getRandomValues(new Uint8Array(SALT_LEN));
}

/**
 * Pack { salt, iv, ciphertext } into a single base64url string.
 *
 * @param {{ salt: Uint8Array, iv: Uint8Array, ciphertext: Uint8Array }} parts
 * @returns {string}
 */
export function pack(parts) {
    const { salt, iv, ciphertext } = parts;
    if (salt.length !== SALT_LEN) throw new Error('pack: salt must be 16 bytes');
    if (iv.length !== IV_LEN) throw new Error('pack: iv must be 12 bytes');
    const combined = new Uint8Array(SALT_LEN + IV_LEN + ciphertext.length);
    combined.set(salt, 0);
    combined.set(iv, SALT_LEN);
    combined.set(ciphertext, SALT_LEN + IV_LEN);
    return b64UrlEncode(combined);
}

/**
 * Inverse of `pack`. Throws on malformed input.
 *
 * @param {string} str
 * @returns {{ salt: Uint8Array, iv: Uint8Array, ciphertext: Uint8Array }}
 */
export function unpack(str) {
    const bytes = b64UrlDecode(str);
    if (bytes.length < SALT_LEN + IV_LEN + 16) {
        // 16 bytes is the minimum AES-GCM ciphertext (the auth tag alone).
        throw new Error('unpack: input too short');
    }
    const salt = bytes.slice(0, SALT_LEN);
    const iv = bytes.slice(SALT_LEN, SALT_LEN + IV_LEN);
    const ciphertext = bytes.slice(SALT_LEN + IV_LEN);
    return { salt, iv, ciphertext };
}

// ── base64url helpers ──────────────────────────────────────────
// Uses RFC 4648 §5 — '+' → '-', '/' → '_', no padding.

function b64UrlEncode(bytes) {
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function b64UrlDecode(str) {
    let s = str.replace(/-/g, '+').replace(/_/g, '/');
    while (s.length % 4) s += '=';
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}
