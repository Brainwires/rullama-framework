// brainwires-chat-pwa — i18n
//
// Async catalog loader with English fallback. Mirrors the pattern from
// `parolnet/pwa/src/i18n.js`, scaled to 35 languages so the chat UI
// matches Gemma 4's multilingual coverage.
//
// Boot calls `initI18n(savedLang)`. When `savedLang` is null we detect
// the system locale via `navigator.languages`. The chosen catalog is
// fetched on top of an English fallback dict — `t()` falls through
// `current → en → key` so a missing key never blanks out the UI.

export const SUPPORTED_LANGS = [
    'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'uk',
    'cs', 'hu', 'ro', 'el', 'sv', 'da', 'no', 'fi', 'tr', 'ar',
    'he', 'fa', 'ur', 'hi', 'bn', 'ta', 'mr', 'zh-CN', 'zh-TW', 'ja',
    'ko', 'id', 'ms', 'vi', 'th',
];

export const RTL_LANGS = ['ar', 'he', 'fa', 'ur'];

// Native-name labels for the picker. Listed in `SUPPORTED_LANGS` order.
export const LANG_NAMES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch',
    'it': 'Italiano',
    'pt': 'Português',
    'nl': 'Nederlands',
    'pl': 'Polski',
    'ru': 'Русский',
    'uk': 'Українська',
    'cs': 'Čeština',
    'hu': 'Magyar',
    'ro': 'Română',
    'el': 'Ελληνικά',
    'sv': 'Svenska',
    'da': 'Dansk',
    'no': 'Norsk',
    'fi': 'Suomi',
    'tr': 'Türkçe',
    'ar': 'العربية',
    'he': 'עברית',
    'fa': 'فارسی',
    'ur': 'اردو',
    'hi': 'हिन्दी',
    'bn': 'বাংলা',
    'ta': 'தமிழ்',
    'mr': 'मराठी',
    'zh-CN': '简体中文',
    'zh-TW': '繁體中文',
    'ja': '日本語',
    'ko': '한국어',
    'id': 'Bahasa Indonesia',
    'ms': 'Bahasa Melayu',
    'vi': 'Tiếng Việt',
    'th': 'ไทย',
};

let _dict = {};
let _enDict = {};
let _code = 'en';

/**
 * Initialize i18n. Loads English first (so `t()` always has a fallback)
 * then loads the requested locale on top. Falls back to system-detected
 * locale when `savedLang` is null/empty, and to `'en'` if neither yields
 * a supported code.
 *
 * @param {string|null|undefined} savedLang
 * @returns {Promise<void>}
 */
export async function initI18n(savedLang) {
    let target = savedLang || detectLanguage();
    if (!SUPPORTED_LANGS.includes(target)) target = 'en';

    // English first — fallback dict survives even if the chosen
    // catalog fetch fails.
    try {
        const enResp = await fetch('./lang/en.json', { cache: 'force-cache' });
        if (enResp.ok) _enDict = await enResp.json();
    } catch (_) { /* offline / cache miss — _enDict stays empty */ }

    if (target === 'en') {
        _dict = _enDict;
        _code = 'en';
    } else {
        await loadLocale(target);
    }
    applyToDOM();
}

/**
 * Load a locale dictionary from `lang/<code>.json`. Falls back silently
 * to English when the network/file is unavailable.
 *
 * Kept as the public entry point for backwards compatibility — most
 * code should call `initI18n()` (boot) or `changeLanguage()` (settings).
 *
 * @param {string} code
 * @returns {Promise<Record<string,string>>}
 */
export async function loadLocale(code) {
    const target = code || 'en';
    try {
        const url = `./lang/${encodeURIComponent(target)}.json`;
        const resp = await fetch(url, { cache: 'force-cache' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data && typeof data === 'object') {
            _dict = data;
            _code = target;
            return data;
        }
    } catch (_) {
        _dict = _enDict;
        _code = 'en';
    }
    return _dict;
}

/**
 * Look up a key. Fallback chain: current locale → English → key string.
 *
 * @param {string} key
 * @param {Record<string, string|number>} [vars]  optional `{name}` substitutions
 * @returns {string}
 */
export function t(key, vars) {
    let raw;
    if (Object.prototype.hasOwnProperty.call(_dict, key)) raw = _dict[key];
    else if (Object.prototype.hasOwnProperty.call(_enDict, key)) raw = _enDict[key];
    else raw = key;
    if (!vars || typeof vars !== 'object') return String(raw);
    return String(raw).replace(/\{(\w+)\}/g, (_, name) => {
        return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : `{${name}}`;
    });
}

/** Current locale code (the last successfully-loaded one). */
export function currentLocale() { return _code; }

/** Alias kept for parity with the parolnet API. */
export function getCurrentLang() { return _code; }

/**
 * Persist the chosen language and reload the page so every view paints
 * with the new catalog. Settings persistence is done by the caller via
 * `setSetting('language', code)` — this helper just swaps the in-memory
 * dict + applies DOM attrs in case the caller doesn't reload.
 */
export async function changeLanguage(code) {
    if (!SUPPORTED_LANGS.includes(code)) return;
    await loadLocale(code);
    applyToDOM();
}

/**
 * Reflect the current locale on the document root: `<html lang>` and
 * `<html dir>` (rtl/ltr). The chat-pwa renders all strings via JS
 * `t()` calls (no `data-i18n` HTML attributes), so a single call here
 * is enough — full re-render happens on language change via reload.
 */
export function applyToDOM() {
    if (typeof document === 'undefined') return;
    const isRtl = RTL_LANGS.includes(_code);
    document.documentElement.lang = _code;
    document.documentElement.dir = isRtl ? 'rtl' : 'ltr';
}

/**
 * Pure, testable matcher: given the browser's ordered preference list
 * and the set of supported locale codes, return the best match or 'en'.
 *
 * Rules (lifted from `parolnet/pwa/src/i18n.js`):
 *   1. Exact match on a preferred tag (case-insensitive).
 *   2. Chinese region aliasing: zh-Hant / zh-HK / zh-TW / zh-MO → zh-TW;
 *      any other zh-* (Hans, CN, SG, …) → zh-CN.
 *   3. Base-language match (e.g. 'fr-CA' → 'fr', 'pt-BR' → 'pt').
 *   4. Fallback: 'en'.
 *
 * The first preference that yields any match wins — we don't skip ahead
 * to a later preference merely because the earlier one only matched at
 * the base level. That's what the user actually asked for.
 */
export function detectLocale(prefs, supported) {
    if (!Array.isArray(prefs) || prefs.length === 0) prefs = ['en'];
    if (!Array.isArray(supported) || supported.length === 0) return 'en';
    const supportedLower = supported.map((s) => s.toLowerCase());
    const origByLower = new Map(supported.map((s) => [s.toLowerCase(), s]));

    for (const raw of prefs) {
        if (!raw || typeof raw !== 'string') continue;
        const pref = raw.trim();
        if (!pref) continue;
        const lower = pref.toLowerCase();

        // 1. Exact match.
        if (supportedLower.includes(lower)) return origByLower.get(lower);

        const base = lower.split('-')[0];

        // 2. Chinese region aliasing.
        if (base === 'zh') {
            const isTraditional = /(^|-)hant(-|$)/.test(lower) || /-(hk|tw|mo)(-|$)/.test(lower);
            if (isTraditional && supportedLower.includes('zh-tw')) return origByLower.get('zh-tw');
            if (supportedLower.includes('zh-cn')) return origByLower.get('zh-cn');
        }

        // 3. Base-language match.
        if (supportedLower.includes(base)) return origByLower.get(base);
    }

    // 4. Fallback.
    return supportedLower.includes('en') ? origByLower.get('en') : supported[0];
}

function detectLanguage() {
    const prefs = (typeof navigator !== 'undefined' && Array.isArray(navigator.languages) && navigator.languages.length)
        ? navigator.languages
        : [(typeof navigator !== 'undefined' && (navigator.language || navigator.userLanguage)) || 'en'];
    return detectLocale(prefs, SUPPORTED_LANGS);
}

/** For tests — install a dictionary directly without fetching. */
export function _setDictForTests(dict, code = 'en') {
    _dict = dict || {};
    _enDict = dict || {};
    _code = code;
}
