// brainwires-chat-pwa — persistent status footer
//
// Renders a strip at the bottom of the viewport that shows model
// activity (download / verify / load / ready / error) regardless of
// the active view, and stays visible when idle. Subscribes to
// `state.events`:
//   - 'model_progress' { modelId, file, fileBytesDone, totalBytesDone, totalBytesTotal,
//                        throughputBps, etaSeconds }
//   - 'model_deleted'  { modelId }
//
// State machine:
//   idle → downloading → verifying (transient) → ready (auto-dismiss)
//                       ↘ error (manual dismiss / retry)
//
// The same "fan-out" subscription updates the Settings → Local Model
// row via the inline `<div data-bw-mirror="local-model-progress">`
// element if it's present in the DOM.

import { events } from './state.js';
import {
    KNOWN_MODELS,
    KNOWN_OLLAMA_MODELS,
    KNOWN_EMBEDDING_MODELS,
    downloadModel,
    cancelDownload,
    deleteModel,
    isDownloaded,
} from './model-store.js';
import { el, clear, formatBytes, formatEta, escapeHtml, toast } from './utils.js';
import { t } from './i18n.js';
import * as cryptoStore from '../crypto-store.js';
import { setSetting, getSetting } from './sql-db.js';
import { getSessionKey } from './state.js';

const HF_TOKEN_SETTING = 'hfTokenEncrypted';

let _root = null;
let _state = 'idle';   // 'idle' | 'downloading' | 'verifying' | 'loading' | 'ready' | 'error'
let _activeModelId = null;
let _lastDetail = null;
let _retryArgs = null;
let _readyTimer = null;

/**
 * Mount the banner inside `root`. Idempotent.
 *
 * @param {HTMLElement} root  the slot element from index.html
 */
export function mountBanner(root) {
    if (!root) return;
    _root = root;
    if (!_root.querySelector('#download-banner')) {
        const banner = el('div', { id: 'download-banner', class: 'download-banner', attrs: { role: 'status', 'aria-live': 'polite' } });
        _root.appendChild(banner);
    }
    subscribe();
    render();
}

function bannerEl() {
    if (!_root) return null;
    return _root.querySelector('#download-banner');
}

function mirrorEl() {
    return document.querySelector('[data-bw-mirror="local-model-progress"]');
}

function subscribe() {
    events.addEventListener('model_progress', (e) => {
        const d = e.detail;
        if (!d) return;
        _lastDetail = d;
        _activeModelId = d.modelId;
        // Drive the state machine off `phase` when present (new schema);
        // fall back to byte-progress heuristic for older callers that
        // haven't been updated yet.
        if (d.phase === 'verifying') {
            if (_state !== 'ready' && _state !== 'error') _state = 'verifying';
        } else if (d.phase === 'loading') {
            if (_state !== 'ready' && _state !== 'error') _state = 'loading';
        } else if (d.phase === 'ready') {
            _state = 'ready';
            if (d.deviceType) _lastDetail = { ...(_lastDetail || {}), deviceType: d.deviceType };
            if (_readyTimer) clearTimeout(_readyTimer);
            _readyTimer = setTimeout(() => {
                _state = 'idle';
                _activeModelId = null;
                _lastDetail = null;
                render();
            }, 1500);
        } else if (d.phase === 'error') {
            _state = 'error';
        } else if (d.phase === 'download' || d.phase === undefined) {
            const isComplete = d.totalBytesTotal > 0 && d.totalBytesDone >= d.totalBytesTotal;
            if (isComplete && _state !== 'verifying' && _state !== 'loading' && _state !== 'ready') {
                // Optimistically transition to verifying; downloadModel
                // may still be hashing.
                _state = 'verifying';
            } else if (_state !== 'verifying' && _state !== 'loading' && _state !== 'ready' && _state !== 'error') {
                _state = 'downloading';
            }
        }
        render();
    });

    events.addEventListener('model_deleted', (e) => {
        if (e.detail && e.detail.modelId === _activeModelId) {
            _state = 'idle';
            _activeModelId = null;
            _lastDetail = null;
        }
        render();
    });
}

// ── Public API ─────────────────────────────────────────────────

/**
 * Begin (or resume) a download. Wraps `model-store.downloadModel` and
 * folds in the state-machine transitions the banner shows.
 *
 * @param {string} modelId
 * @param {object} [opts]
 * @returns {Promise<void>}
 */
export async function startDownload(modelId, opts = {}) {
    const m = (KNOWN_MODELS[modelId]
        || KNOWN_OLLAMA_MODELS[modelId]
        || KNOWN_EMBEDDING_MODELS[modelId]);
    if (!m) {
        toast(`Unknown model: ${modelId}`, 'error');
        return;
    }
    _activeModelId = modelId;
    _state = 'downloading';
    _retryArgs = { modelId, opts };
    render();

    try {
        // Pull HF token (if set) and decrypt. We swallow decrypt errors
        // and treat them as "no token" so the download surfaces an
        // HF_AUTH_REQUIRED error and re-prompts.
        let hfToken = opts.hfToken || null;
        if (!hfToken) {
            hfToken = await readDecryptedHfToken().catch((e) => { console.warn("[bw] swallowed:", e); return null; });
        }

        await downloadModel(modelId, { hfToken });
        _state = 'ready';
        render();
        // Auto-dismiss after a brief pulse.
        if (_readyTimer) clearTimeout(_readyTimer);
        _readyTimer = setTimeout(() => {
            _state = 'idle';
            _activeModelId = null;
            _lastDetail = null;
            render();
        }, 1500);
    } catch (err) {
        if (err && err.name === 'AbortError') {
            _state = 'idle';
            _activeModelId = null;
            _lastDetail = null;
            toast(t('download.cancelled'), 'info');
            render();
            return;
        }
        if (err && err.name === 'HF_AUTH_REQUIRED') {
            _state = 'error';
            render();
            promptForHfToken(modelId);
            return;
        }
        _state = 'error';
        _lastDetail = { ...(_lastDetail || {}), error: err && err.message ? err.message : String(err) };
        render();
    }
}

/** Cancel the active download (banner-driven). */
export function cancelActive() {
    if (_activeModelId) cancelDownload(_activeModelId);
}

/** Delete the model and reset banner state. */
export async function deleteActive(modelId) {
    const id = modelId || _activeModelId;
    if (!id) return;
    await deleteModel(id);
    _state = 'idle';
    _activeModelId = null;
    _lastDetail = null;
    render();
    toast(t('settings.localModel.deleted'), 'success');
}

// ── Rendering ──────────────────────────────────────────────────

function render() {
    const node = bannerEl();
    if (!node) return;
    node.dataset.state = _state;
    clear(node);
    node.appendChild(buildContent(/* compact */ false));
    const mirror = mirrorEl();
    if (mirror) {
        clear(mirror);
        // Settings mirror stays empty when there's no active task.
        if (_state !== 'idle') mirror.appendChild(buildContent(/* compact */ true));
    }
}

function buildContent(compact) {
    const m = _activeModelId
        ? (KNOWN_MODELS[_activeModelId]
            || KNOWN_OLLAMA_MODELS[_activeModelId]
            || KNOWN_EMBEDDING_MODELS[_activeModelId])
        : null;
    const name = m ? m.displayName : 'model';
    const wrap = el('div', { class: `bw-dl bw-dl-${_state}` });

    if (_state === 'idle') {
        wrap.appendChild(el('div', { class: 'bw-dl-row' },
            el('span', { class: 'bw-dl-title' }, t('download.idle')),
        ));
    } else if (_state === 'downloading') {
        const d = _lastDetail || {};
        const total = d.totalBytesTotal || 0;
        const done = d.totalBytesDone || 0;
        const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null;
        const speed = d.throughputBps ? `${formatBytes(d.throughputBps)}/s` : '';
        const eta = formatEta(d.etaSeconds);

        wrap.appendChild(el('div', { class: 'bw-dl-row' },
            el('span', { class: 'bw-dl-title' }, t('download.banner', { model: name })),
            !compact && el('button', {
                class: 'bw-dl-btn bw-dl-cancel',
                attrs: { 'aria-label': t('download.cancel'), type: 'button' },
                onClick: () => cancelActive(),
            }, t('download.cancel')),
        ));

        const bar = pct == null
            ? el('progress', { class: 'bw-dl-progress' })
            : el('progress', { class: 'bw-dl-progress', value: pct, max: 100 });
        wrap.appendChild(bar);

        const meta = el('div', { class: 'bw-dl-meta' });
        if (pct != null) meta.appendChild(el('span', {}, `${pct}%`));
        meta.appendChild(el('span', {}, `${formatBytes(done)} / ${formatBytes(total || 0)}`));
        if (speed) meta.appendChild(el('span', {}, speed));
        meta.appendChild(el('span', {}, `${t('download.eta')}: ${eta}`));
        wrap.appendChild(meta);
    } else if (_state === 'verifying') {
        wrap.appendChild(el('div', { class: 'bw-dl-row' },
            el('span', { class: 'bw-dl-title' }, t('download.verifying')),
        ));
        wrap.appendChild(el('progress', { class: 'bw-dl-progress' }));
    } else if (_state === 'loading') {
        wrap.appendChild(el('div', { class: 'bw-dl-row' },
            el('span', { class: 'bw-dl-title' }, t('download.loading')),
        ));
        wrap.appendChild(el('progress', { class: 'bw-dl-progress' }));
    } else if (_state === 'ready') {
        const deviceType = (_lastDetail && _lastDetail.deviceType) || 'cpu';
        const deviceLabel = deviceType === 'webgpu' ? 'Ready (GPU)' : t('download.ready');
        wrap.appendChild(el('div', { class: 'bw-dl-row bw-dl-ready' },
            el('span', { class: 'bw-dl-title' }, deviceLabel),
        ));
    } else if (_state === 'error') {
        let errMsg = (_lastDetail && _lastDetail.error) ? _lastDetail.error : t('error.generic');
        if (/Failed to fetch/i.test(errMsg)) errMsg = t('download.error.network');
        else if (/Cache/i.test(errMsg) && /put/i.test(errMsg)) errMsg = t('download.error.storage');
        wrap.appendChild(el('div', { class: 'bw-dl-row' },
            el('span', { class: 'bw-dl-title' }, t('download.error')),
            !compact && el('button', {
                class: 'bw-dl-btn bw-dl-retry',
                attrs: { type: 'button' },
                onClick: () => {
                    if (_retryArgs) startDownload(_retryArgs.modelId, _retryArgs.opts);
                },
            }, t('download.retry')),
        ));
        wrap.appendChild(el('div', { class: 'bw-dl-error-msg' }, errMsg));
    }
    return wrap;
}

// ── HF token modal ─────────────────────────────────────────────

function promptForHfToken(modelId) {
    // Build a tiny modal overlay.
    let overlay = document.getElementById('bw-modal-host');
    if (!overlay) {
        overlay = el('div', { id: 'bw-modal-host', class: 'bw-modal-host', attrs: { role: 'dialog', 'aria-modal': 'true' } });
        document.body.appendChild(overlay);
    }
    clear(overlay);

    const close = () => { try { overlay.remove(); } catch (_err) { console.warn("[bw] caught:", _err); } };

    const tokenInput = el('input', {
        type: 'password',
        class: 'bw-input',
        attrs: { 'aria-label': 'Hugging Face token', autocomplete: 'off', placeholder: 'hf_…' },
    });
    const errLabel = el('div', { class: 'bw-modal-err', attrs: { 'aria-live': 'polite' } });

    const card = el('div', { class: 'bw-modal-card' },
        el('h2', { class: 'bw-modal-title' }, t('download.hfTokenTitle')),
        el('p', { class: 'bw-modal-desc' }, t('download.hfTokenDesc')),
        tokenInput,
        errLabel,
        el('div', { class: 'bw-modal-actions' },
            el('button', {
                class: 'bw-btn bw-btn-secondary',
                attrs: { type: 'button' },
                onClick: close,
            }, t('download.hfTokenCancel')),
            el('button', {
                class: 'bw-btn bw-btn-primary',
                attrs: { type: 'button' },
                onClick: async () => {
                    const tok = tokenInput.value.trim();
                    if (!tok) { errLabel.textContent = 'Token required'; return; }
                    try {
                        await persistHfToken(tok);
                    } catch (e) {
                        errLabel.textContent = e && e.message ? e.message : String(e);
                        return;
                    }
                    close();
                    // Retry the download with the freshly-stored token.
                    startDownload(modelId, { hfToken: tok });
                },
            }, t('download.hfTokenSave')),
        ),
    );
    overlay.appendChild(card);
    setTimeout(() => tokenInput.focus(), 0);
}

async function persistHfToken(token) {
    const sessionKey = getSessionKey();
    if (!sessionKey) {
        // No passphrase — store plaintext (matches the user's "skip
        // encryption" choice from the unlock flow).
        await setSetting(HF_TOKEN_SETTING, { plaintext: token });
        return;
    }
    const salt = cryptoStore.generateSalt();
    // We re-encrypt with the session key, so we don't need to derive
    // again. Pack an IV-only blob.
    const blob = await cryptoStore.encrypt(sessionKey, token);
    const packed = cryptoStore.pack({ salt, iv: blob.iv, ciphertext: blob.ciphertext });
    await setSetting(HF_TOKEN_SETTING, { encrypted: packed });
}

async function readDecryptedHfToken() {
    const stored = await getSetting(HF_TOKEN_SETTING);
    if (!stored) return null;
    if (stored.plaintext) return stored.plaintext;
    if (!stored.encrypted) return null;
    const sessionKey = getSessionKey();
    if (!sessionKey) return null;
    const parts = cryptoStore.unpack(stored.encrypted);
    return cryptoStore.decrypt(sessionKey, { iv: parts.iv, ciphertext: parts.ciphertext });
}

// ── State accessors (for chat composer to disable local provider) ──

/**
 * @returns {boolean} true while a download is in progress
 */
export function isDownloadActive() {
    return _state === 'downloading' || _state === 'verifying' || _state === 'loading';
}

/** Returns the modelId of the active download (or null). */
export function activeModelId() { return _activeModelId; }

/** For unit-test parity: also expose a simple `cached()` query. */
export async function isModelReady(modelId) {
    try { return await isDownloaded(modelId); } catch (_) { return false; }
}

// Mark `escapeHtml` as used by reference so esbuild doesn't moan.
void escapeHtml;
