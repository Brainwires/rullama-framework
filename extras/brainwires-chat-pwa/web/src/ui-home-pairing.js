// brainwires-chat-pwa — Settings → Home agent section.
//
// Renders the "Home agent" card in Settings. Two states:
//   * Not paired      → "Pair this device" button opens the modal.
//   * Paired          → "Paired with home <fp>" + "Unpair" button.
//
// The modal walks the user through:
//   1. Scan QR with the BarcodeDetector API (if available) OR paste the
//      bwhome:// URL into a text input.
//   2. Confirm by entering the 6-digit code shown on the home machine.
//
// On success, the bundle is encrypted (or plaintext-fallback per opt-out)
// and stashed via crypto-store.js → home-pairing.savePairingBundle.

import { el, clear, toast } from './utils.js';
import { t } from './i18n.js';
import { appEvents } from './state.js';
import {
    parseQrPayload,
    claim,
    confirm as confirmPair,
    savePairingBundle,
    loadPairingBundle,
    clearPairingBundle,
    deviceIdentity,
} from './home-pairing.js';
import { disconnect as disconnectHome } from './home-provider.js';

/**
 * Render the "Home agent" card body. Caller wraps in a section and adds it
 * to the settings list.
 *
 * @returns {Promise<HTMLElement>}
 */
export async function renderHomePairingCard() {
    const card = el('div', { class: 'settings-card' });
    await refresh(card);
    return card;
}

/**
 * Drive the unpair side effects: best-effort disconnect of the active
 * transport, drop the encrypted bundle, then fan out a `home-unpaired`
 * event so the chat UI's provider list / status pill react.
 *
 * Exported so the unit tests can drive the side effects directly. The
 * UI button delegates here and toasts on completion.
 */
export async function performUnpair({
    _disconnect = disconnectHome,
    _clearPairingBundle = clearPairingBundle,
    _events = appEvents,
} = {}) {
    // Disconnect first so an in-flight WebRTC peer doesn't keep using
    // a bundle we're about to delete. Errors here are best-effort —
    // the user's intent is clear.
    try { await _disconnect(); } catch (_) { /* swallow */ }
    await _clearPairingBundle();
    if (_events && typeof _events.dispatchEvent === 'function') {
        _events.dispatchEvent(new CustomEvent('home-unpaired'));
    }
}

async function refresh(card) {
    clear(card);
    const bundle = await loadPairingBundle();
    if (bundle && bundle.peer_pubkey) {
        const fp = bundle.peer_pubkey.slice(0, 8);
        card.appendChild(el('p', { class: 'settings-help' },
            t('settings.home.paired', { fp })));
        card.appendChild(el('p', { class: 'settings-help' },
            bundle.tunnel_url || ''));
        const actions = el('div', { class: 'settings-actions' });
        actions.appendChild(el('button', {
            class: 'bw-btn bw-btn-danger',
            attrs: { type: 'button' },
            onClick: async () => {
                if (!confirm(t('settings.home.unpairConfirm'))) return;
                try {
                    await performUnpair();
                    toast(t('settings.home.unpaired'), 'info');
                } catch (e) {
                    toast(e && e.message ? e.message : String(e), 'error');
                }
                await refresh(card);
            },
        }, t('settings.home.removeDevice')));
        card.appendChild(actions);
    } else {
        card.appendChild(el('p', { class: 'settings-help' }, t('settings.home.notPaired')));
        card.appendChild(el('button', {
            class: 'bw-btn bw-btn-primary',
            attrs: { type: 'button' },
            onClick: () => openPairModal(card),
        }, t('settings.home.pair')));
    }
}

function openPairModal(card) {
    const overlay = el('div', { class: 'bw-modal-overlay', attrs: { role: 'dialog', 'aria-modal': 'true' } });
    const modal = el('div', { class: 'bw-modal' });
    overlay.appendChild(modal);

    let claimedQr = null; // { tunnelUrl, oneTimeToken, peerFingerprint }
    let deviceName = '';
    let scanAbort = null;

    function close() {
        if (scanAbort) try { scanAbort(); } catch (_) {}
        if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }

    function showStep1() {
        clear(modal);
        modal.appendChild(el('h2', { class: 'bw-modal-title' }, t('settings.home.scanTitle')));
        modal.appendChild(el('p', { class: 'settings-help' }, t('settings.home.scanDesc')));

        // Try to mount a camera scan button if BarcodeDetector is available.
        const supportsBarcode = typeof globalThis.BarcodeDetector === 'function';
        const scanBtn = el('button', {
            class: 'bw-btn bw-btn-secondary',
            attrs: { type: 'button', disabled: supportsBarcode ? null : 'disabled' },
            onClick: async () => {
                try {
                    await startCameraScan(modal, (qrText) => {
                        urlInput.value = qrText;
                        // Auto-advance to the next step on a successful scan.
                        nextBtn.click();
                    }, (a) => { scanAbort = a; });
                } catch (e) {
                    toast(e && e.message ? e.message : String(e), 'error');
                }
            },
        }, t('settings.home.scanCamera'));
        modal.appendChild(scanBtn);
        if (!supportsBarcode) {
            modal.appendChild(el('p', { class: 'settings-help settings-warn' },
                t('settings.home.scanUnsupported')));
        }

        const urlInput = el('input', {
            type: 'text',
            class: 'bw-input',
            attrs: { placeholder: t('settings.home.urlPlaceholder'), 'aria-label': 'bwhome:// URL' },
        });
        modal.appendChild(urlInput);

        const nameInput = el('input', {
            type: 'text',
            class: 'bw-input',
            attrs: { placeholder: t('settings.home.deviceNamePlaceholder'), 'aria-label': 'Device name' },
        });
        // Pre-fill with a sensible default so the form is one click away
        // from done.
        nameInput.value = (typeof navigator !== 'undefined' && navigator.userAgent)
            ? defaultDeviceName(navigator.userAgent)
            : 'Browser';
        modal.appendChild(nameInput);

        const err = el('div', { class: 'settings-err', attrs: { 'aria-live': 'polite' } });
        modal.appendChild(err);

        const nextBtn = el('button', {
            class: 'bw-btn bw-btn-primary',
            attrs: { type: 'button' },
            onClick: async () => {
                err.textContent = '';
                try {
                    claimedQr = parseQrPayload(urlInput.value);
                    deviceName = nameInput.value.trim() || 'Browser';
                    const ident = await deviceIdentity();
                    await claim({
                        tunnelUrl: claimedQr.tunnelUrl,
                        oneTimeToken: claimedQr.oneTimeToken,
                        devicePubkey: ident.publicKeyHex,
                        deviceName,
                    });
                    showStep2();
                } catch (e) {
                    err.textContent = e && e.message ? e.message : String(e);
                }
            },
        }, t('settings.home.next'));

        const cancelBtn = el('button', {
            class: 'bw-btn bw-btn-link',
            attrs: { type: 'button' },
            onClick: close,
        }, t('settings.home.cancel'));

        modal.appendChild(el('div', { class: 'settings-actions' }, nextBtn, cancelBtn));
    }

    function showStep2() {
        if (scanAbort) { try { scanAbort(); } catch (_) {} scanAbort = null; }
        clear(modal);
        modal.appendChild(el('h2', { class: 'bw-modal-title' }, t('settings.home.confirm')));
        modal.appendChild(el('p', { class: 'settings-help' }, t('settings.home.codeDesc')));
        if (claimedQr.peerFingerprint) {
            modal.appendChild(el('p', { class: 'settings-help' },
                `Home FP: ${claimedQr.peerFingerprint}`));
        }
        const codeInput = el('input', {
            type: 'text',
            class: 'bw-input',
            attrs: {
                placeholder: t('settings.home.codePlaceholder'),
                inputmode: 'numeric',
                maxlength: '6',
                pattern: '[0-9]{6}',
                autocomplete: 'one-time-code',
            },
        });
        modal.appendChild(codeInput);

        const err = el('div', { class: 'settings-err', attrs: { 'aria-live': 'polite' } });
        modal.appendChild(err);

        const confirmBtn = el('button', {
            class: 'bw-btn bw-btn-primary',
            attrs: { type: 'button' },
            onClick: async () => {
                err.textContent = '';
                try {
                    const bundle = await confirmPair({
                        tunnelUrl: claimedQr.tunnelUrl,
                        oneTimeToken: claimedQr.oneTimeToken,
                        code: codeInput.value.trim(),
                    });
                    await savePairingBundle({
                        ...bundle,
                        tunnel_url: claimedQr.tunnelUrl,
                        device_name: deviceName,
                    });
                    toast(t('settings.home.success'), 'success');
                    close();
                    await refresh(card);
                } catch (e) {
                    err.textContent = e && e.message ? e.message : String(e);
                }
            },
        }, t('settings.home.confirm'));

        const cancelBtn = el('button', {
            class: 'bw-btn bw-btn-link',
            attrs: { type: 'button' },
            onClick: close,
        }, t('settings.home.cancel'));

        modal.appendChild(el('div', { class: 'settings-actions' }, confirmBtn, cancelBtn));
        try { codeInput.focus(); } catch (_) {}
    }

    showStep1();
    document.body.appendChild(overlay);
}

/**
 * Drive a one-shot QR scan via BarcodeDetector + getUserMedia. Resolves
 * the first decoded `bwhome://pair?...` value via `onResult`. The caller
 * gets an abort callback via `onAbort` so it can stop the camera when
 * advancing away from the modal step.
 */
async function startCameraScan(host, onResult, onAbort) {
    if (typeof globalThis.BarcodeDetector !== 'function') {
        throw new Error('BarcodeDetector not available');
    }
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        throw new Error('getUserMedia not available');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
    });
    const video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    video.srcObject = stream;
    video.style.width = '100%';
    video.style.maxHeight = '320px';
    host.appendChild(video);

    let stopped = false;
    const stop = () => {
        stopped = true;
        try { stream.getTracks().forEach((tk) => tk.stop()); } catch (_) {}
        try { if (video.parentNode) video.parentNode.removeChild(video); } catch (_) {}
    };
    onAbort(stop);

    const detector = new globalThis.BarcodeDetector({ formats: ['qr_code'] });
    const tick = async () => {
        if (stopped) return;
        try {
            const codes = await detector.detect(video);
            for (const c of codes) {
                if (c.rawValue && c.rawValue.startsWith('bwhome://')) {
                    stop();
                    onResult(c.rawValue);
                    return;
                }
            }
        } catch (_) { /* keep trying — transient errors are common */ }
        if (!stopped) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

function defaultDeviceName(ua) {
    if (/iPhone/i.test(ua)) return 'iPhone';
    if (/iPad/i.test(ua)) return 'iPad';
    if (/Android/i.test(ua)) return 'Android';
    if (/Mac/i.test(ua)) return 'Mac';
    if (/Windows/i.test(ua)) return 'Windows';
    if (/Linux/i.test(ua)) return 'Linux';
    return 'Browser';
}
