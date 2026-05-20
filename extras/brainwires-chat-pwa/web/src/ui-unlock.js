// brainwires-chat-pwa — first-run / unlock view
//
// Two flows:
//   - First run: ask for a passphrase and confirm. Derive + store the
//     verifier blob, set the in-memory session key, then route to chat.
//   - Returning user: ask for the passphrase, derive against the stored
//     salt, decrypt the verifier; on success route to chat.
//
// The user can also opt out of encryption — keys will be stored in
// plaintext locally and the warning is surfaced in Settings.

import { el, clear, toast } from './utils.js';
import { t } from './i18n.js';
import { getSetting, setSetting } from './sql-db.js';
import * as cryptoStore from '../crypto-store.js';
import { setSessionKey } from './state.js';
import { mount as mountView } from './views.js';

const PASSPHRASE_SETTING = 'passphraseConfig';
const ENCRYPT_OPT_OUT_SETTING = 'encryptionOptOut';

let _root = null;

export async function render(root) {
    _root = root;
    clear(root);
    const cfg = await getSetting(PASSPHRASE_SETTING);
    if (cfg && cfg.salt && cfg.verify) {
        root.appendChild(buildUnlock(cfg));
    } else {
        root.appendChild(buildFirstRun());
    }
}

export function onShow() {
    if (!_root) return;
    const input = _root.querySelector('input[type="password"]');
    if (input) try { input.focus(); } catch (_) {}
}

// ── First-run ──────────────────────────────────────────────────

function buildFirstRun() {
    const pw1 = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('unlock.passphrase'), autocomplete: 'new-password' } });
    const pw2 = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('unlock.confirm'), autocomplete: 'new-password' } });
    const err = el('div', { class: 'settings-err', attrs: { 'aria-live': 'polite' } });

    const form = el('form', {
        class: 'unlock-form',
        onSubmit: async (e) => {
            e.preventDefault();
            err.textContent = '';
            if (pw1.value.length < 8) { err.textContent = t('settings.passphrase.tooShort'); return; }
            if (pw1.value !== pw2.value) { err.textContent = t('settings.passphrase.mismatch'); return; }
            try {
                await configurePassphrase(pw1.value);
                toast(t('settings.saved'), 'success');
                mountView('chat');
            } catch (ex) {
                err.textContent = ex && ex.message ? ex.message : String(ex);
            }
        },
    },
        el('h1', { class: 'unlock-title' }, t('unlock.firstRunTitle')),
        el('p', { class: 'unlock-desc' }, t('unlock.firstRunDesc')),
        pw1,
        pw2,
        err,
        el('button', { class: 'bw-btn bw-btn-primary', attrs: { type: 'submit' } }, t('unlock.create')),
        el('button', {
            class: 'bw-btn bw-btn-link',
            attrs: { type: 'button' },
            onClick: async () => {
                await setSetting(ENCRYPT_OPT_OUT_SETTING, true);
                toast(t('settings.passphrase.skipWarn'), 'warn', 5000);
                mountView('chat');
            },
        }, t('unlock.skip')),
    );

    return el('div', { class: 'unlock-shell' }, form);
}

// ── Returning unlock ───────────────────────────────────────────

function buildUnlock(cfg) {
    const pw = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('unlock.passphrase'), autocomplete: 'current-password' } });
    const err = el('div', { class: 'settings-err', attrs: { 'aria-live': 'polite' } });

    const form = el('form', {
        class: 'unlock-form',
        onSubmit: async (e) => {
            e.preventDefault();
            err.textContent = '';
            try {
                await unlockWith(cfg, pw.value);
                mountView('chat');
            } catch (ex) {
                err.textContent = ex && ex.message ? ex.message : String(ex);
            }
        },
    },
        el('h1', { class: 'unlock-title' }, t('unlock.title')),
        el('p', { class: 'unlock-desc' }, t('unlock.unlockDesc')),
        pw,
        err,
        el('button', { class: 'bw-btn bw-btn-primary', attrs: { type: 'submit' } }, t('unlock.submit')),
        el('button', {
            class: 'bw-btn bw-btn-link',
            attrs: { type: 'button' },
            onClick: () => mountView('chat'),
        }, 'Skip for now'),
    );

    return el('div', { class: 'unlock-shell' }, form);
}

// ── Crypto helpers ─────────────────────────────────────────────

async function configurePassphrase(passphrase) {
    const salt = cryptoStore.generateSalt();
    const key = await cryptoStore.deriveKey(passphrase, salt);
    const verifyBlob = await cryptoStore.encrypt(key, 'ok');
    const verifyPacked = cryptoStore.pack({ salt, iv: verifyBlob.iv, ciphertext: verifyBlob.ciphertext });
    await setSetting(PASSPHRASE_SETTING, {
        salt: bytesToB64(salt),
        verify: verifyPacked,
    });
    setSessionKey(key);
}

async function unlockWith(cfg, passphrase) {
    const parts = cryptoStore.unpack(cfg.verify);
    const key = await cryptoStore.deriveKey(passphrase, parts.salt);
    try {
        const out = await cryptoStore.decrypt(key, { iv: parts.iv, ciphertext: parts.ciphertext });
        if (out !== 'ok') throw new Error('verify mismatch');
    } catch (_) {
        throw new Error(t('unlock.wrong'));
    }
    setSessionKey(key);
}

function bytesToB64(bytes) {
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
}
