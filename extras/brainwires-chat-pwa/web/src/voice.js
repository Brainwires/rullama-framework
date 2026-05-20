// brainwires-chat-pwa — voice (TTS + STT) façade
//
// Wraps the WASM `WebTts` and `WebStt` classes from `pkg/`. Lazy
// singletons so we don't allocate the SpeechRecognition object until
// the first time the user actually wants to use voice.
//
// Voice prefs live in the `voicePrefs` IDB store (see db.js).
// Recognized keys:
//   - tts.voiceUri      — string, default null (let the OS pick)
//   - tts.rate          — number, 0.1 .. 10
//   - tts.pitch         — number, 0 .. 2
//   - tts.volume        — number, 0 .. 1
//   - tts.lang          — string, BCP-47, default null
//   - stt.lang          — string, BCP-47, default 'en-US'
//   - stt.continuous    — boolean, default false
//   - stt.interim       — boolean, default true
//   - stt.maxAlternatives — number, default 1
//   - stt.enabled       — boolean, default true

import { getWasm } from './state.js';
import { getVoicePref, setVoicePref } from './sql-db.js';

let _tts = null;
let _stt = null;

class SttUnsupported extends Error {
    constructor(msg = 'SpeechRecognition unavailable in this browser') {
        super(msg);
        this.name = 'STT_UNSUPPORTED';
    }
}
export const STT_UNSUPPORTED = SttUnsupported;

// ── Pref helpers ───────────────────────────────────────────────

async function readPref(key, fallback) {
    try {
        const v = await getVoicePref(key);
        return v === undefined ? fallback : v;
    } catch (_) {
        return fallback;
    }
}

async function writePref(key, value) {
    try { await setVoicePref(key, value); } catch (_) { /* best-effort */ }
}

export const voicePrefs = {
    get: readPref,
    set: writePref,
};

// ── TTS ────────────────────────────────────────────────────────

/**
 * Lazy TTS singleton. Returns a `WebTts` instance.
 * @returns {Promise<object>}
 */
export async function getTts() {
    if (_tts) return _tts;
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        throw new SttUnsupported('speechSynthesis unavailable');
    }
    const wasm = await getWasm();
    if (typeof wasm.WebTts !== 'function') {
        throw new Error('wasm.WebTts not exported — rebuild the wasm crate');
    }
    _tts = new wasm.WebTts();
    return _tts;
}

/**
 * Speak `text` using the user's saved voice prefs.
 *
 * @param {string} text
 * @param {object} [opts]  override values (voiceUri, rate, pitch, volume, lang)
 */
export async function speak(text, opts = {}) {
    if (!text || typeof text !== 'string') return;
    const tts = await getTts();
    const voiceUri = opts.voiceUri ?? await readPref('tts.voiceUri', null);
    const rate = opts.rate ?? await readPref('tts.rate', 1.0);
    const pitch = opts.pitch ?? await readPref('tts.pitch', 1.0);
    const volume = opts.volume ?? await readPref('tts.volume', 1.0);
    const lang = opts.lang ?? await readPref('tts.lang', null);
    try { tts.cancel(); } catch (_) {}
    tts.speak(text, voiceUri || null, rate, pitch, volume, lang || null);
}

export async function cancelSpeak() {
    if (!_tts) return;
    try { _tts.cancel(); } catch (_) {}
}

/**
 * Snapshot of available TTS voices.
 * @returns {Promise<Array<{uri:string,name:string,lang:string,default:boolean}>>}
 */
export async function listVoices() {
    try {
        const tts = await getTts();
        const v = tts.voices();
        if (Array.isArray(v)) return v;
        return [];
    } catch (_) {
        return [];
    }
}

/**
 * Persist the chosen TTS voice URI.
 * @param {string | null} uri
 */
export async function setTtsVoice(uri) {
    await writePref('tts.voiceUri', uri || null);
}

// ── STT ────────────────────────────────────────────────────────

function _hasSttSupport() {
    if (typeof window === 'undefined') return false;
    return ('SpeechRecognition' in window) || ('webkitSpeechRecognition' in window);
}

/**
 * Lazy STT singleton. Throws `STT_UNSUPPORTED` when the browser lacks
 * `SpeechRecognition`.
 * @returns {Promise<object>}
 */
export async function getStt() {
    if (_stt) return _stt;
    if (!_hasSttSupport()) throw new SttUnsupported();
    const wasm = await getWasm();
    if (typeof wasm.WebStt !== 'function') {
        throw new Error('wasm.WebStt not exported — rebuild the wasm crate');
    }
    _stt = new wasm.WebStt();
    return _stt;
}

/**
 * Start listening. Returns a `stop()` function that gracefully ends
 * the session. Throws `{name: 'STT_UNSUPPORTED'}` when unavailable.
 *
 * @param {object} args
 * @param {(text: string, isFinal: boolean, confidence: number) => void} args.onResult
 * @param {() => void} [args.onEnd]
 * @param {(err: string, msg?: string) => void} [args.onError]
 * @param {object} [args.opts]   inline overrides for the saved prefs
 * @returns {Promise<() => void>} a stopper function
 */
export async function listen({ onResult, onEnd, onError, opts = {} } = {}) {
    if (typeof onResult !== 'function') {
        throw new Error('voice.listen: onResult callback required');
    }
    const enabled = await readPref('stt.enabled', true);
    if (!enabled) throw new SttUnsupported('STT disabled in settings');

    const stt = await getStt();
    const lang = opts.lang ?? await readPref('stt.lang', 'en-US');
    const continuous = opts.continuous ?? await readPref('stt.continuous', false);
    const interim = opts.interim ?? await readPref('stt.interim', true);
    const maxAlts = opts.maxAlternatives ?? await readPref('stt.maxAlternatives', 1);

    try {
        if (typeof stt.setOnEnd === 'function') {
            stt.setOnEnd(() => { try { onEnd && onEnd(); } catch (_) {} });
        }
        if (typeof stt.setOnError === 'function') {
            stt.setOnError((err, msg) => {
                try { onError && onError(String(err || ''), msg ? String(msg) : null); }
                catch (_) {}
            });
        }
        stt.start(lang, !!continuous, !!interim, maxAlts | 0, (text, isFinal, confidence) => {
            try { onResult(String(text || ''), !!isFinal, Number(confidence) || 0); }
            catch (_) {}
        });
    } catch (e) {
        // Surface as STT_UNSUPPORTED so the UI can fall back gracefully.
        if (e && e.name === 'STT_UNSUPPORTED') throw e;
        throw new SttUnsupported(e && e.message ? e.message : 'failed to start STT');
    }

    return function stop() {
        try { stt.stop(); } catch (_) {}
    };
}

/** Persist the chosen STT language. */
export async function setSttLang(code) {
    await writePref('stt.lang', code || 'en-US');
}

/** Whether STT is supported in this browser. Cheap, no async. */
export function isSttSupported() { return _hasSttSupport(); }
