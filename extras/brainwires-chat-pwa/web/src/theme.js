// brainwires-chat-pwa — theme switcher
//
// Three modes: 'light', 'dark', 'system' (follows prefers-color-scheme).
// Applied by setting `data-theme` on <html>; CSS in styles.css does the rest.
// Persisted via the existing settings store. `system` is the default and
// keeps the original behavior (dark, with prefers-color-scheme: light overrides).

import { getSetting, setSetting } from './sql-db.js';
import { appEvents } from './state.js';

const SETTING_KEY = 'ui.theme';
const VALID = ['light', 'dark', 'system'];

let _current = 'system';
let _mql = null;

function apply(mode) {
    const root = document.documentElement;
    if (!root) return;
    root.setAttribute('data-theme', mode);
    root.style.colorScheme = mode === 'system' ? 'dark light' : mode;
}

export function getTheme() {
    return _current;
}

export async function setTheme(mode) {
    if (!VALID.includes(mode)) mode = 'system';
    _current = mode;
    apply(mode);
    await setSetting(SETTING_KEY, mode);
    appEvents.dispatchEvent(new CustomEvent('theme-changed', { detail: { theme: mode } }));
}

export async function loadTheme() {
    let saved;
    try { saved = await getSetting(SETTING_KEY); } catch (_) { /* IDB may not be open yet */ }
    const mode = VALID.includes(saved) ? saved : 'system';
    _current = mode;
    apply(mode);

    // The OS-level dark/light flip only matters in 'system' mode, but we
    // listen unconditionally so consumers (e.g. a syntax-highlight theme
    // injector) can react. CSS itself handles the visual flip.
    if (typeof matchMedia === 'function' && !_mql) {
        _mql = matchMedia('(prefers-color-scheme: light)');
        const onChange = () => {
            appEvents.dispatchEvent(new CustomEvent('theme-system-changed', {
                detail: { systemPrefersLight: _mql.matches },
            }));
        };
        if (typeof _mql.addEventListener === 'function') _mql.addEventListener('change', onChange);
        else if (typeof _mql.addListener === 'function') _mql.addListener(onChange); // legacy Safari
    }
}
