// brainwires-chat-pwa — view router
//
// Vanilla, no framework. Slugs: 'chat' | 'settings' | 'unlock'.
//
// Each registered view module exports `render(root)` — it should
// (re)build its DOM under the supplied root container. The router
// owns three sibling root nodes inside `<main id="app">`, toggling
// `is-active` to switch between them. We mount lazily (build the
// inner DOM the first time a view becomes active) so cold-start
// only pays for the first view.

import { clear } from './utils.js';

const _viewRoots = new Map();   // slug → root element
const _viewMods  = new Map();   // slug → module with render(root)
const _mounted   = new Set();   // slug set, tracks whether render() ran

let _appRoot = null;
let _current = null;

const KNOWN_SLUGS = ['unlock', 'chat', 'settings'];

/**
 * Register a view module. Should be called once per slug at boot.
 *
 * @param {string} slug
 * @param {{ render: (root: HTMLElement) => void, onShow?: () => void, onHide?: () => void }} mod
 */
export function register(slug, mod) {
    if (!slug || !mod || typeof mod.render !== 'function') {
        throw new Error('views.register: slug + module with render() required');
    }
    _viewMods.set(slug, mod);
}

/**
 * Initialize the router. Creates the per-view root containers under
 * `<main id="app">` so view modules have stable mount points to render
 * into.
 *
 * @param {HTMLElement} appRoot the `<main id="app">` element
 */
export function init(appRoot) {
    _appRoot = appRoot;
    if (!appRoot) throw new Error('views.init: app root required');
    clear(appRoot);
    for (const slug of KNOWN_SLUGS) {
        const root = document.createElement('section');
        root.id = `view-${slug}`;
        root.className = `view view-${slug}`;
        root.setAttribute('role', 'region');
        root.setAttribute('aria-label', slug);
        root.hidden = true;
        appRoot.appendChild(root);
        _viewRoots.set(slug, root);
    }
}

/**
 * Switch the visible view. Re-rendering is the responsibility of the
 * view module itself if it wants to refresh; we only call render() on
 * first mount, then call onShow() on every subsequent activation.
 *
 * @param {string} slug
 */
export function mount(slug) {
    if (!_appRoot) throw new Error('views.mount: call init(appRoot) first');
    if (!_viewRoots.has(slug)) {
        // Permissive: register a stub root so unknown views don't crash boot.
        const fallback = document.createElement('section');
        fallback.id = `view-${slug}`;
        fallback.className = `view view-${slug}`;
        fallback.hidden = true;
        _appRoot.appendChild(fallback);
        _viewRoots.set(slug, fallback);
    }

    // Hide the previous view.
    if (_current && _viewRoots.has(_current)) {
        const prev = _viewRoots.get(_current);
        prev.hidden = true;
        prev.classList.remove('is-active');
        const prevMod = _viewMods.get(_current);
        if (prevMod && typeof prevMod.onHide === 'function') {
            try { prevMod.onHide(); } catch (e) { console.warn(`onHide(${_current}):`, e); }
        }
    }

    const root = _viewRoots.get(slug);
    const mod = _viewMods.get(slug);
    if (mod && !_mounted.has(slug)) {
        try { mod.render(root); _mounted.add(slug); }
        catch (e) { console.error(`render(${slug}) failed:`, e); }
    }
    root.hidden = false;
    root.classList.add('is-active');
    _appRoot.dataset.view = slug;
    _current = slug;

    // Sync URL so a page reload stays on the current view.
    const url = new URL(location.href);
    if (slug === 'chat') {
        url.searchParams.delete('page');
    } else {
        url.searchParams.set('page', slug);
    }
    history.replaceState(null, '', url);
    if (mod && typeof mod.onShow === 'function') {
        try { mod.onShow(); } catch (e) { console.warn(`onShow(${slug}):`, e); }
    }
}

/** @returns {string | null} */
export function current() { return _current; }

/** Force a fresh render() the next time `slug` is mounted. */
export function invalidate(slug) {
    _mounted.delete(slug);
    if (_viewRoots.has(slug)) clear(_viewRoots.get(slug));
}

/** Test-only: reset the router's internal state. */
export function _resetForTests() {
    _viewRoots.clear();
    _viewMods.clear();
    _mounted.clear();
    _appRoot = null;
    _current = null;
}
