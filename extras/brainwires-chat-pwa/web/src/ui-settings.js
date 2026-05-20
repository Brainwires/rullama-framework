// brainwires-chat-pwa — settings view
//
// Sections:
//   - Passphrase (set/change/lock)
//   - Cloud providers (per-provider card with API key + model + Test)
//   - Local model (Gemma 4 E2B card with progress mirror)
//   - Voice (TTS/STT pickers)
//   - About

import { el, clear, toast, escapeHtml } from './utils.js';
import { t, SUPPORTED_LANGS, LANG_NAMES, getCurrentLang } from './i18n.js';
import {
    getSetting,
    setSetting,
} from './sql-db.js';
import { listProviders } from './providers/index.js';
import {
    KNOWN_MODELS,
    KNOWN_OLLAMA_MODELS,
    KNOWN_EMBEDDING_MODELS,
    isDownloaded,
    cancelDownload,
    deleteModel,
    getPartialInfo,
    getKnownModelAny,
} from './model-store.js';
import * as banner from './ui-download-banner.js';
import * as cryptoStore from '../crypto-store.js';
import {
    getSessionKey,
    setSessionKey,
    isSessionUnlocked,
    getWasm,
    events as stateEvents,
} from './state.js';
import * as voice from './voice.js';
import { mount as mountView } from './views.js';
import { getTheme, setTheme } from './theme.js';
import { render as renderRagPanel } from './ui-rag-panel.js';
import { render as renderMcpPanel } from './ui-mcp-panel.js';
import { renderHomePairingCard } from './ui-home-pairing.js';
import * as homeProvider from './home-provider.js';
import * as localProvider from './providers/local.js';

const PASSPHRASE_SETTING = 'passphraseConfig'; // { salt: base64, verify: encrypted("ok") }
const ENCRYPT_OPT_OUT_SETTING = 'encryptionOptOut';

let _root = null;

// ── Public render ──────────────────────────────────────────────

export async function render(root) {
    _root = root;
    clear(root);
    root.appendChild(buildHeader());
    const main = el('div', { class: 'settings-main' });
    root.appendChild(main);

    main.appendChild(await sectionPassphrase());
    main.appendChild(await sectionTheme());
    main.appendChild(await sectionLanguage());
    main.appendChild(await sectionHomeAgent());
    main.appendChild(await sectionSync());
    main.appendChild(await sectionProviders());
    main.appendChild(await sectionLocalModel());
    main.appendChild(await sectionEmbeddingModels());
    main.appendChild(await sectionRag());
    main.appendChild(await sectionMcp());
    main.appendChild(await sectionVoice());
    main.appendChild(await sectionAbout());

    // Partial update: refresh just the affected card when its download
    // state changes (start, complete, error, cancel). This is what makes
    // the Cancel button appear/disappear and the Download button toggle
    // its disabled state in real time.
    //
    // The `download` phase fires once per chunk — gate by tracking the
    // last-seen phase per model so we only re-render on actual transitions.
    const _lastPhasePerModel = new Map();
    stateEvents.addEventListener('model_progress', (e) => {
        const d = e.detail;
        if (!d || !d.modelId) return;
        const last = _lastPhasePerModel.get(d.modelId);
        if (d.phase && d.phase !== last) {
            _lastPhasePerModel.set(d.modelId, d.phase);
            refreshCard(d.modelId);
        }
    });
    stateEvents.addEventListener('model_deleted', (e) => {
        if (e.detail && e.detail.modelId) {
            _lastPhasePerModel.delete(e.detail.modelId);
            refreshCard(e.detail.modelId);
        }
    });
}

export function onShow() { /* could refresh dynamic state here */ }

// ── Header ─────────────────────────────────────────────────────

function buildHeader() {
    return el('header', { class: 'settings-header' },
        el('button', {
            class: 'icon-btn',
            attrs: { type: 'button', 'aria-label': t('nav.back') },
            onClick: () => mountView('chat'),
        }, '←'),
        el('h1', { class: 'settings-title' }, t('settings.title')),
    );
}

function sectionWrap(title, body) {
    return el('section', { class: 'settings-section' },
        el('h2', { class: 'settings-section-title' }, title),
        body,
    );
}

// ── Passphrase ─────────────────────────────────────────────────

async function sectionPassphrase() {
    const body = el('div', { class: 'settings-card' });
    await renderPassphrase(body);
    return sectionWrap(t('settings.passphrase.title'), body);
}

async function renderPassphrase(body) {
    clear(body);
    const cfg = await getSetting(PASSPHRASE_SETTING);
    const optOut = await getSetting(ENCRYPT_OPT_OUT_SETTING);

    if (cfg && cfg.salt && cfg.verify) {
        // Configured. Show change + lock buttons.
        body.appendChild(el('p', { class: 'settings-help' },
            isSessionUnlocked() ? '✓ Unlocked' : t('settings.passphrase.locked'),
        ));

        if (!isSessionUnlocked()) {
            const pp = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('settings.passphrase.placeholder'), autocomplete: 'current-password' } });
            const err = el('div', { class: 'settings-err' });
            const unlock = el('button', {
                class: 'bw-btn bw-btn-primary',
                attrs: { type: 'button' },
                onClick: async () => {
                    err.textContent = '';
                    try {
                        await unlockPassphrase(pp.value);
                        toast('Unlocked', 'success');
                        await renderPassphrase(body);
                    } catch (e) {
                        err.textContent = e && e.message ? e.message : String(e);
                    }
                },
            }, t('settings.passphrase.unlock'));
            body.appendChild(pp);
            body.appendChild(unlock);
            body.appendChild(err);
        } else {
            body.appendChild(el('button', {
                class: 'bw-btn bw-btn-secondary',
                attrs: { type: 'button' },
                onClick: () => { setSessionKey(null); toast('Locked', 'info'); renderPassphrase(body); },
            }, t('settings.passphrase.lock')));
            // Change passphrase form (collapsed).
            body.appendChild(buildChangePassphraseForm(() => renderPassphrase(body)));
        }
    } else {
        // Not yet configured.
        if (optOut) {
            body.appendChild(el('p', { class: 'settings-help settings-warn' }, t('settings.passphrase.skipWarn')));
        }
        body.appendChild(buildSetPassphraseForm(() => renderPassphrase(body)));
        body.appendChild(el('button', {
            class: 'bw-btn bw-btn-link',
            attrs: { type: 'button' },
            onClick: async () => {
                await setSetting(ENCRYPT_OPT_OUT_SETTING, true);
                toast(t('settings.passphrase.skipWarn'), 'warn', 5000);
                await renderPassphrase(body);
            },
        }, t('settings.passphrase.skip')));
    }
}

function buildSetPassphraseForm(onDone) {
    const pw1 = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('settings.passphrase.placeholder'), autocomplete: 'new-password' } });
    const pw2 = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('settings.passphrase.confirm'), autocomplete: 'new-password' } });
    const err = el('div', { class: 'settings-err' });
    const btn = el('button', {
        class: 'bw-btn bw-btn-primary',
        attrs: { type: 'button' },
        onClick: async () => {
            err.textContent = '';
            if (pw1.value.length < 8) { err.textContent = t('settings.passphrase.tooShort'); return; }
            if (pw1.value !== pw2.value) { err.textContent = t('settings.passphrase.mismatch'); return; }
            try {
                await configurePassphrase(pw1.value);
                toast(t('settings.saved'), 'success');
                if (onDone) onDone();
            } catch (e) {
                err.textContent = e && e.message ? e.message : String(e);
            }
        },
    }, t('settings.passphrase.set'));
    return el('div', { class: 'settings-form' }, pw1, pw2, btn, err);
}

function buildChangePassphraseForm(onDone) {
    const cur = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: 'Current ' + t('settings.passphrase.placeholder'), autocomplete: 'current-password' } });
    const pw1 = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: 'New passphrase', autocomplete: 'new-password' } });
    const pw2 = el('input', { type: 'password', class: 'bw-input', attrs: { placeholder: t('settings.passphrase.confirm'), autocomplete: 'new-password' } });
    const err = el('div', { class: 'settings-err' });
    const btn = el('button', {
        class: 'bw-btn bw-btn-secondary',
        attrs: { type: 'button' },
        onClick: async () => {
            err.textContent = '';
            if (pw1.value.length < 8) { err.textContent = t('settings.passphrase.tooShort'); return; }
            if (pw1.value !== pw2.value) { err.textContent = t('settings.passphrase.mismatch'); return; }
            try {
                await unlockPassphrase(cur.value); // verify
                await configurePassphrase(pw1.value);
                toast(t('settings.saved'), 'success');
                if (onDone) onDone();
            } catch (e) {
                err.textContent = e && e.message ? e.message : String(e);
            }
        },
    }, t('settings.passphrase.change'));
    return el('details', { class: 'settings-form-collapsible' },
        el('summary', {}, t('settings.passphrase.change')),
        cur, pw1, pw2, btn, err,
    );
}

async function configurePassphrase(passphrase) {
    const salt = cryptoStore.generateSalt();
    const key = await cryptoStore.deriveKey(passphrase, salt);
    const verifyBlob = await cryptoStore.encrypt(key, 'ok');
    const verifyPacked = cryptoStore.pack({ salt, iv: verifyBlob.iv, ciphertext: verifyBlob.ciphertext });
    await setSetting(PASSPHRASE_SETTING, { salt: b64Encode(salt), verify: verifyPacked });
    setSessionKey(key);
}

async function unlockPassphrase(passphrase) {
    const cfg = await getSetting(PASSPHRASE_SETTING);
    if (!cfg) throw new Error('Passphrase not configured');
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

// Tiny helpers — full base64 not base64url since we only stash the salt.
function b64Encode(bytes) {
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
}

// ── Theme ──────────────────────────────────────────────────────

async function sectionTheme() {
    const body = el('div', { class: 'settings-card' });
    const current = getTheme();
    const sel = el('select', { class: 'bw-input', attrs: { 'aria-label': t('settings.theme.label') } });
    const options = [
        ['system', t('settings.theme.system')],
        ['light', t('settings.theme.light')],
        ['dark', t('settings.theme.dark')],
    ];
    for (const [val, label] of options) {
        const o = el('option', { attrs: { value: val } }, label);
        if (val === current) o.setAttribute('selected', '');
        sel.appendChild(o);
    }
    sel.addEventListener('change', async () => {
        try {
            await setTheme(sel.value);
            toast(t('settings.saved'), 'success', 1200);
        } catch (e) {
            toast(e && e.message ? e.message : String(e), 'error');
        }
    });
    body.appendChild(el('label', { class: 'bw-label' }, t('settings.theme.label'), sel));
    return sectionWrap(t('settings.theme.title'), body);
}

// ── Language ───────────────────────────────────────────────────

async function sectionLanguage() {
    const body = el('div', { class: 'settings-card' });
    const current = getCurrentLang();
    const sel = el('select', {
        class: 'bw-input',
        attrs: { 'aria-label': t('settings.language.label') },
    });
    for (const code of SUPPORTED_LANGS) {
        const o = el('option', { attrs: { value: code } }, LANG_NAMES[code] || code);
        if (code === current) o.setAttribute('selected', '');
        sel.appendChild(o);
    }
    sel.addEventListener('change', async () => {
        try {
            await setSetting('language', sel.value);
            location.reload();
        } catch (e) {
            toast(e && e.message ? e.message : String(e), 'error');
        }
    });
    body.appendChild(el('label', { class: 'bw-label' }, t('settings.language.label'), sel));
    return sectionWrap(t('settings.language.title'), body);
}

// ── Home agent (M8 pairing) ────────────────────────────────────

async function sectionHomeAgent() {
    const body = await renderHomePairingCard();
    return sectionWrap(t('settings.home.title'), body);
}

// ── Cross-device sync ─────────────────────────────────────────

async function sectionSync() {
    const body = el('div', { class: 'settings-card' });

    const paired = await homeProvider.isAvailable();
    if (!paired) {
        body.appendChild(el('p', { class: 'settings-help' }, t('settings.sync.requiresPairing')));
        return sectionWrap(t('settings.sync.title'), body);
    }

    const current = await getSetting('sync.enabled');
    const enabled = current === true || current === 'true';

    const checkbox = el('input', {
        type: 'checkbox',
        attrs: { 'aria-label': t('settings.sync.enable') },
    });
    checkbox.checked = enabled;

    const label = el('label', { class: 'settings-toggle' },
        checkbox,
        el('span', {}, t('settings.sync.enable')),
    );
    body.appendChild(label);
    body.appendChild(el('p', { class: 'settings-help' }, t('settings.sync.help')));

    checkbox.addEventListener('change', async () => {
        await setSetting('sync.enabled', checkbox.checked);
        toast(t('settings.saved'), 'success', 1200);
    });

    return sectionWrap(t('settings.sync.title'), body);
}

// ── Cloud providers ────────────────────────────────────────────

async function sectionProviders() {
    const body = el('div', { class: 'settings-card-list' });
    const providers = listProviders().filter((p) => p.runtime === 'cloud');
    for (const p of providers) {
        body.appendChild(await buildProviderCard(p));
    }
    return sectionWrap(t('settings.providers'), body);
}

async function buildProviderCard(p) {
    const id = p.id;
    const blob = await getSetting(`provider.${id}.apiKey`);
    const savedModel = await getSetting(`provider.${id}.model`);
    const baseUrl = id === 'ollama' ? await getSetting(`provider.${id}.baseUrl`) : null;

    const apiKeyInput = el('input', {
        type: 'password',
        class: 'bw-input',
        attrs: {
            placeholder: t('settings.apiKey'),
            autocomplete: 'off',
            'aria-label': `${p.displayName} ${t('settings.apiKey')}`,
        },
    });
    if (blob && (blob.encrypted || blob.plaintext)) {
        apiKeyInput.placeholder = '•••••••• (saved)';
    }

    const modelSelect = el('select', { class: 'bw-input', attrs: { 'aria-label': t('settings.model') } });
    for (const m of (p.models && p.models.length ? p.models : [p.defaultModel])) {
        const o = el('option', { attrs: { value: m } }, m);
        if (m === (savedModel || p.defaultModel)) o.setAttribute('selected', '');
        modelSelect.appendChild(o);
    }

    const baseUrlInput = id === 'ollama'
        ? el('input', {
            type: 'url',
            class: 'bw-input',
            value: baseUrl || 'http://localhost:11434',
            attrs: { placeholder: 'http://localhost:11434', 'aria-label': t('settings.baseUrl') },
        })
        : null;

    const status = el('div', { class: 'settings-status', attrs: { 'aria-live': 'polite' } });
    const testBtn = el('button', {
        class: 'bw-btn bw-btn-secondary',
        attrs: { type: 'button' },
        onClick: async () => {
            status.textContent = t('settings.testing');
            try {
                await testProvider(id, apiKeyInput.value, baseUrlInput ? baseUrlInput.value : null);
                status.textContent = t('settings.testOk');
                status.className = 'settings-status settings-status-ok';
            } catch (e) {
                status.textContent = t('settings.testFail', { error: e && e.message ? e.message : String(e) });
                status.className = 'settings-status settings-status-err';
            }
        },
    }, t('settings.test'));

    const saveBtn = el('button', {
        class: 'bw-btn bw-btn-primary',
        attrs: { type: 'button' },
        onClick: async () => {
            try {
                await saveProvider(id, apiKeyInput.value, modelSelect.value, baseUrlInput ? baseUrlInput.value : null);
                apiKeyInput.value = '';
                apiKeyInput.placeholder = '•••••••• (saved)';
                toast(t('settings.saved'), 'success');
            } catch (e) {
                toast(e && e.message ? e.message : String(e), 'error');
            }
        },
    }, t('settings.save'));

    const card = el('div', { class: 'settings-card' },
        el('h3', { class: 'settings-card-title' }, p.displayName),
        id !== 'ollama' && el('label', { class: 'bw-label' }, t('settings.apiKey'), apiKeyInput),
        baseUrlInput && el('label', { class: 'bw-label' }, t('settings.baseUrl'), baseUrlInput),
        el('label', { class: 'bw-label' }, t('settings.model'), modelSelect),
        el('div', { class: 'settings-actions' }, testBtn, saveBtn),
        status,
    );
    return card;
}

async function saveProvider(id, apiKey, model, baseUrl) {
    if (apiKey && apiKey.length > 0) {
        const sessionKey = getSessionKey();
        if (sessionKey) {
            const salt = cryptoStore.generateSalt();
            const blob = await cryptoStore.encrypt(sessionKey, apiKey);
            const packed = cryptoStore.pack({ salt, iv: blob.iv, ciphertext: blob.ciphertext });
            await setSetting(`provider.${id}.apiKey`, { encrypted: packed });
        } else {
            // No passphrase — store in plaintext per user opt-out.
            await setSetting(`provider.${id}.apiKey`, { plaintext: apiKey });
        }
    }
    if (model) await setSetting(`provider.${id}.model`, model);
    if (baseUrl) await setSetting(`provider.${id}.baseUrl`, baseUrl);
}

async function testProvider(id, apiKeyInline, baseUrlInline) {
    // Use the inline value if present, else the saved (decrypted) one.
    let key = apiKeyInline && apiKeyInline.length ? apiKeyInline : null;
    if (!key && id !== 'ollama') {
        const blob = await getSetting(`provider.${id}.apiKey`);
        if (blob && blob.plaintext) key = blob.plaintext;
        else if (blob && blob.encrypted) {
            const sk = getSessionKey();
            if (!sk) throw new Error(t('error.locked'));
            const parts = cryptoStore.unpack(blob.encrypted);
            key = await cryptoStore.decrypt(sk, { iv: parts.iv, ciphertext: parts.ciphertext });
        }
    }

    if (id === 'openai') {
        if (!key) throw new Error('No API key');
        const r = await fetch('https://api.openai.com/v1/models', {
            headers: { 'Authorization': `Bearer ${key}` },
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return;
    }
    if (id === 'anthropic') {
        if (!key) throw new Error('No API key');
        const r = await fetch('https://api.anthropic.com/v1/models', {
            headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01' },
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return;
    }
    if (id === 'google') {
        if (!key) throw new Error('No API key');
        const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(key)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return;
    }
    if (id === 'ollama') {
        const base = (baseUrlInline || 'http://localhost:11434').replace(/\/+$/, '');
        const r = await fetch(`${base}/api/tags`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return;
    }
    throw new Error(`Unknown provider: ${id}`);
}

// ── Local model ────────────────────────────────────────────────

async function buildLlmCard(modelId) {
    // Accept HF (KNOWN_MODELS) or Ollama (KNOWN_OLLAMA_MODELS) ids — the
    // settings card surface is identical for both, only the underlying
    // download / load paths differ (handled in model-store.js).
    const m = getKnownModelAny(modelId);
    if (!m) return el('div');
    const downloaded = await isDownloaded(modelId).catch((e) => { console.error("[bw] swallowed:", e); return false; });

    // Which downloaded LLM is currently selected for chat. The setting
    // key matches what `ui-chat.js`'s `updateAttachVisibility` already
    // reads (`provider.<id>.model`), so flipping it here is enough —
    // no separate sync needed.
    const llmSettingKey = `provider.${localProvider.id}.model`;
    const savedModel = await getSetting(llmSettingKey).catch(() => null);
    const activeLlm = savedModel || localProvider.defaultModel;
    const isActive = downloaded && activeLlm === modelId;

    const card = el('div', { class: 'settings-card', id: `model-card-${modelId}` });

    card.appendChild(el('div', { class: 'settings-card-header' },
        el('h3', { class: 'settings-card-title' }, m.displayName),
        el('span', { class: 'pill ' + (downloaded ? 'pill-ok' : 'pill-muted') },
            downloaded
                ? (isActive ? 'In use' : t('settings.localModel.ready'))
                : formatSize(m.estimatedBytes)),
    ));
    card.appendChild(el('p', { class: 'settings-help' }, m.description));

    const partial = !downloaded ? await getPartialInfo(modelId).catch(() => ({ hasData: false })) : null;

    const actions = el('div', { class: 'settings-actions' });
    if (!downloaded) {
        const downloadingThis = banner.isDownloadActive() && banner.activeModelId() === modelId;
        const anyDownloadActive = banner.isDownloadActive();
        const downloadAttrs = { type: 'button' };
        if (anyDownloadActive) downloadAttrs.disabled = '';
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-primary bw-btn-sm',
            attrs: downloadAttrs,
            onClick: async () => {
                banner.startDownload(modelId);
                await refreshCard(modelId);
            },
        }, t('settings.localModel.download')));
        if (downloadingThis) {
            actions.appendChild(el('button', {
                class: 'bw-btn bw-btn-secondary bw-btn-sm',
                attrs: { type: 'button' },
                onClick: async () => {
                    cancelDownload(modelId);
                    await refreshCard(modelId);
                },
            }, t('settings.localModel.cancel')));
        }
        if (partial && partial.hasData && !downloadingThis) {
            actions.appendChild(el('button', {
                class: 'bw-btn bw-btn-danger bw-btn-sm',
                attrs: { type: 'button' },
                onClick: async () => {
                    await deleteModel(modelId);
                    toast(`Cleared ${formatSize(partial.totalBytes)} partial data`);
                    await refreshCard(modelId);
                },
            }, `Clear partial (${formatSize(partial.totalBytes)})`));
        }
    } else {
        // Use button — clickable when this model isn't the active
        // selection, disabled (greyed) when it is.
        const useAttrs = { type: 'button' };
        if (isActive) useAttrs.disabled = '';
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-primary bw-btn-sm',
            attrs: useAttrs,
            onClick: async () => {
                await setSetting(llmSettingKey, modelId);
                toast(`${m.displayName} set as active`);
                // Refresh sibling cards too so the previously-active
                // one re-enables its Use button.
                await refreshCard('gemma-4-e2b-it');
                await refreshCard('gemma4:e2b');
            },
        }, isActive ? '✓ In use' : 'Use'));
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-danger bw-btn-sm',
            attrs: { type: 'button' },
            onClick: async () => {
                if (!confirm(t('settings.localModel.confirmDelete'))) return;
                await banner.deleteActive(modelId);
                if (isActive) {
                    // The active model is being deleted — fall back to
                    // the provider default so chat doesn't reference a
                    // missing model.
                    await setSetting(llmSettingKey, localProvider.defaultModel);
                }
                await refreshCard('gemma-4-e2b-it');
                await refreshCard('gemma4:e2b');
            },
        }, t('settings.localModel.delete')));
    }
    card.appendChild(actions);
    return card;
}

async function sectionLocalModel() {
    const body = el('div', { class: 'settings-card-list' });
    // Ollama-format Q4_K_M (Phase 4) — same model, ~6× smaller download.
    // Text-only; the GGUF doesn't carry the SigLIP vision tower.
    body.appendChild(await buildLlmCard('gemma4:e2b'));
    // HF safetensors (default) — full vision-capable model.
    body.appendChild(await buildLlmCard('gemma-4-e2b-it'));
    return sectionWrap(t('settings.localModel.title'), body);
}

// ── Embedding models ──────────────────────────────────────────

function formatSize(bytes) {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
    return `${(bytes / 1e3).toFixed(0)} KB`;
}

async function buildEmbeddingCard(m) {
    const downloaded = await isDownloaded(m.id).catch((e) => { console.error("[bw] swallowed:", e); return false; });
    const active = (await getSetting('embedding.activeModel')) === m.id;

    const card = el('div', { class: 'settings-card', id: `model-card-${m.id}` });
    card.appendChild(el('div', { class: 'settings-card-header' },
        el('h3', { class: 'settings-card-title' }, m.displayName),
        el('span', { class: 'pill ' + (downloaded ? 'pill-ok' : 'pill-muted') },
            downloaded ? (active ? 'Active' : 'Ready') : formatSize(m.estimatedBytes)),
    ));
    card.appendChild(el('p', { class: 'settings-help' },
        `${m.provider} · ${m.dimensions}-dim · ${m.maxTokens} max tokens`));
    card.appendChild(el('p', { class: 'settings-help' }, m.description));

    const partial = !downloaded ? await getPartialInfo(m.id).catch(() => ({ hasData: false })) : null;

    const actions = el('div', { class: 'settings-actions' });
    if (!downloaded) {
        const downloadingThis = banner.isDownloadActive() && banner.activeModelId() === m.id;
        const anyDownloadActive = banner.isDownloadActive();
        const downloadAttrs = { type: 'button' };
        if (anyDownloadActive) downloadAttrs.disabled = '';
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-primary bw-btn-sm',
            attrs: downloadAttrs,
            onClick: async () => {
                banner.startDownload(m.id);
                toast(`Downloading ${m.displayName}…`);
                await refreshCard(m.id);
            },
        }, 'Download'));
        if (downloadingThis) {
            actions.appendChild(el('button', {
                class: 'bw-btn bw-btn-secondary bw-btn-sm',
                attrs: { type: 'button' },
                onClick: async () => {
                    cancelDownload(m.id);
                    await refreshCard(m.id);
                },
            }, 'Cancel'));
        }
        if (partial && partial.hasData && !downloadingThis) {
            actions.appendChild(el('button', {
                class: 'bw-btn bw-btn-danger bw-btn-sm',
                attrs: { type: 'button' },
                onClick: async () => {
                    await deleteModel(m.id);
                    toast(`Cleared ${formatSize(partial.totalBytes)} partial data`);
                    await refreshCard(m.id);
                },
            }, `Clear partial (${formatSize(partial.totalBytes)})`));
        }
    } else {
        // Use button — clickable when not active, disabled-greyed when
        // this is the currently-active embedding. Mirrors the LLM
        // card pattern so both sections of the settings page have a
        // consistent "in use" affordance.
        const useAttrs = { type: 'button' };
        if (active) useAttrs.disabled = '';
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-primary bw-btn-sm',
            attrs: useAttrs,
            onClick: async () => {
                const prevActiveId = await getSetting('embedding.activeModel');
                await setSetting('embedding.activeModel', m.id);
                toast(`${m.displayName} set as active`);
                // Refresh both cards so the previously-active one
                // re-enables its Use button.
                if (prevActiveId && prevActiveId !== m.id) {
                    await refreshCard(prevActiveId);
                }
                await refreshCard(m.id);
            },
        }, active ? '✓ In use' : 'Use'));
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-danger bw-btn-sm',
            attrs: { type: 'button' },
            onClick: async () => {
                if (!confirm(`Delete ${m.displayName}?`)) return;
                await deleteModel(m.id);
                if (active) await setSetting('embedding.activeModel', '');
                toast(`${m.displayName} deleted`);
                await refreshCard(m.id);
            },
        }, 'Delete'));
    }
    card.appendChild(actions);
    return card;
}

async function sectionEmbeddingModels() {
    const body = el('div', { class: 'settings-card-list' });
    const models = Object.values(KNOWN_EMBEDDING_MODELS);
    const categories = ['small', 'medium', 'large'];
    for (const cat of categories) {
        const catModels = models.filter((m) => m.category === cat);
        if (catModels.length === 0) continue;
        const catLabel = cat === 'small' ? 'Small (< 200 MB)' : cat === 'medium' ? 'Medium (200 MB – 1 GB)' : 'Large (> 1 GB)';
        body.appendChild(el('h4', { class: 'settings-subsection' }, catLabel));
        for (const m of catModels) {
            body.appendChild(await buildEmbeddingCard(m));
        }
    }
    return sectionWrap('Embedding models (local RAG)', body);
}

// ── Private RAG ────────────────────────────────────────────────

async function sectionRag() {
    const body = await renderRagPanel();
    return sectionWrap(t('settings.rag.title'), body);
}

// ── MCP servers ────────────────────────────────────────────────

async function sectionMcp() {
    const body = await renderMcpPanel();
    return sectionWrap(t('settings.mcp.title'), body);
}

// ── Partial card refresh (swap one card, keep scroll + rest) ──

async function refreshCard(modelId) {
    const existing = document.getElementById(`model-card-${modelId}`);
    if (!existing) return;
    let newCard;
    if (KNOWN_MODELS[modelId] || KNOWN_OLLAMA_MODELS[modelId]) {
        newCard = await buildLlmCard(modelId);
    } else if (KNOWN_EMBEDDING_MODELS[modelId]) {
        newCard = await buildEmbeddingCard(KNOWN_EMBEDDING_MODELS[modelId]);
    }
    if (newCard) existing.replaceWith(newCard);
}

// ── Voice ──────────────────────────────────────────────────────

async function sectionVoice() {
    const body = el('div', { class: 'settings-card' });
    const enabled = await voice.voicePrefs.get('stt.enabled', true);

    // STT enable toggle.
    const enableLabel = el('label', { class: 'bw-label bw-label-row' },
        el('span', {}, t('settings.voice.enable')),
        el('input', {
            type: 'checkbox',
            checked: !!enabled,
            onChange: async (e) => { await voice.voicePrefs.set('stt.enabled', !!e.currentTarget.checked); toast(t('settings.saved'), 'success', 1200); },
        }),
    );
    body.appendChild(enableLabel);

    if (!voice.isSttSupported()) {
        body.appendChild(el('p', { class: 'settings-help settings-warn' }, t('settings.voice.unsupported')));
    }

    // TTS section.
    body.appendChild(el('h4', { class: 'settings-subsection' }, t('settings.voice.tts')));
    const ttsVoiceSel = el('select', { class: 'bw-input', attrs: { 'aria-label': t('settings.voice.voice') } });
    ttsVoiceSel.appendChild(el('option', { attrs: { value: '' } }, '(default)'));
    try {
        const voices = await voice.listVoices();
        const savedUri = await voice.voicePrefs.get('tts.voiceUri', '');
        for (const v of voices) {
            const opt = el('option', { attrs: { value: v.uri } }, `${v.name} — ${v.lang}`);
            if (v.uri === savedUri) opt.setAttribute('selected', '');
            ttsVoiceSel.appendChild(opt);
        }
    } catch (_err) { console.warn("[bw] caught:", _err); }
    ttsVoiceSel.addEventListener('change', () => voice.setTtsVoice(ttsVoiceSel.value || null));
    body.appendChild(el('label', { class: 'bw-label' }, t('settings.voice.voice'), ttsVoiceSel));

    body.appendChild(await buildSlider('tts.rate', t('settings.voice.rate'), 0.5, 2.0, 0.1, 1.0));
    body.appendChild(await buildSlider('tts.pitch', t('settings.voice.pitch'), 0.0, 2.0, 0.1, 1.0));
    body.appendChild(await buildSlider('tts.volume', t('settings.voice.volume'), 0.0, 1.0, 0.05, 1.0));

    body.appendChild(el('button', {
        class: 'bw-btn bw-btn-secondary',
        attrs: { type: 'button' },
        onClick: () => voice.speak(t('settings.voice.testText')).catch((e) => toast(e && e.message ? e.message : String(e), 'error')),
    }, t('settings.voice.test')));

    // STT section.
    body.appendChild(el('h4', { class: 'settings-subsection' }, t('settings.voice.stt')));
    const sttLang = el('input', {
        type: 'text',
        class: 'bw-input',
        value: await voice.voicePrefs.get('stt.lang', 'en-US'),
        attrs: { 'aria-label': t('settings.voice.lang'), placeholder: 'en-US' },
        onChange: async (e) => { await voice.setSttLang(e.currentTarget.value); toast(t('settings.saved'), 'success', 1200); },
    });
    body.appendChild(el('label', { class: 'bw-label' }, t('settings.voice.lang'), sttLang));

    const continuous = el('input', {
        type: 'checkbox',
        checked: !!(await voice.voicePrefs.get('stt.continuous', false)),
        onChange: async (e) => { await voice.voicePrefs.set('stt.continuous', !!e.currentTarget.checked); },
    });
    body.appendChild(el('label', { class: 'bw-label bw-label-row' },
        el('span', {}, t('settings.voice.continuous')), continuous));

    const interim = el('input', {
        type: 'checkbox',
        checked: !!(await voice.voicePrefs.get('stt.interim', true)),
        onChange: async (e) => { await voice.voicePrefs.set('stt.interim', !!e.currentTarget.checked); },
    });
    body.appendChild(el('label', { class: 'bw-label bw-label-row' },
        el('span', {}, t('settings.voice.interim')), interim));

    return sectionWrap(t('settings.voice.title'), body);
}

async function buildSlider(prefKey, label, min, max, step, fallback) {
    const value = await voice.voicePrefs.get(prefKey, fallback);
    const out = el('span', { class: 'bw-slider-value' }, String(value));
    const slider = el('input', {
        type: 'range',
        class: 'bw-slider',
        attrs: { min: String(min), max: String(max), step: String(step) },
        value: String(value),
        onInput: (e) => { out.textContent = e.currentTarget.value; },
        onChange: (e) => { voice.voicePrefs.set(prefKey, parseFloat(e.currentTarget.value)); },
    });
    return el('label', { class: 'bw-label' },
        el('span', {}, label),
        el('div', { class: 'bw-slider-row' }, slider, out),
    );
}

// ── About ──────────────────────────────────────────────────────

async function sectionAbout() {
    const body = el('div', { class: 'settings-card' });
    let version = 'unknown';
    try {
        const wasm = await getWasm();
        if (typeof wasm.version === 'function') version = wasm.version();
    } catch (_err) { console.warn("[bw] caught:", _err); }

    let buildTime = 'unknown';
    let buildGit = 'unknown';
    try {
        const info = await import('../build-info.js');
        buildTime = info.BUILD_TIME || buildTime;
        buildGit = info.BUILD_GIT || buildGit;
    } catch (_err) { console.warn("[bw] caught:", _err); }

    body.appendChild(el('p', {}, el('strong', {}, t('settings.about.version') + ': '), String(version)));
    body.appendChild(el('p', {}, el('strong', {}, t('settings.about.build') + ': '), `${buildTime} (${buildGit})`));
    body.appendChild(el('p', {},
        el('strong', {}, t('settings.about.source') + ': '),
        el('a', {
            attrs: {
                href: 'https://github.com/Brainwires/brainwires-framework',
                target: '_blank',
                rel: 'noopener noreferrer',
            },
        }, 'github.com/Brainwires/brainwires-framework'),
    ));
    return sectionWrap(t('settings.about.title'), body);
}

// Avoid unused warnings: escapeHtml is reserved for future use.
void escapeHtml;
