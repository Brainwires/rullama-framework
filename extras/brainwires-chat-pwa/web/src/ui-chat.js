// brainwires-chat-pwa — chat view
//
// Mobile-first chat surface. Vanilla DOM, no framework. Renders:
//   - header (drawer toggle, title, overflow menu)
//   - off-canvas conversation drawer
//   - messages list (user / assistant bubbles)
//   - composer with mic + textarea + send + provider chip
//
// Streaming is fed from `state.events`:
//   - 'chat_chunk' { conversationId, messageId, delta }
//   - 'chat_done'  { conversationId, messageId, usage }
//   - 'chat_error' { conversationId, messageId, error }
// The same payloads also surface on `appEvents` as 'chat-chunk' /
// 'chat-done' / 'chat-error' (see boot.js fan-out). We listen on
// `state.events` because the underscore form is closer to source-of-truth.

import { events as stateEvents, getSessionKey, isSessionUnlocked } from './state.js';
import {
    listConversations,
    putConversation,
    deleteConversation,
    listMessages,
    putMessage,
    setSetting,
    getSetting,
    partsToText,
} from './sql-db.js';
import { listProviders, startChat } from './providers/index.js';
import * as localProvider from './providers/local.js';
import * as homeProvider from './home-provider.js';
import { isDownloaded } from './model-store.js';
import { el, clear, toast, isMobile, genId, escapeHtml } from './utils.js';
import { t } from './i18n.js';
import { renderMarkdown } from './markdown.js';
import { highlightWithin } from './code-highlight.js';
import { extractThinking, buildReasoningElement } from './reasoning-display.js';
import * as attachments from './attachments.js';
import { buildStrip as buildAttachmentStrip } from './ui-attachments.js';
import { isVisionModel, imageToBase64 } from './vision.js';
import { retrieve as ragRetrieve, formatRetrievalAsSystem } from './rag.js';
import { openPicker as openMcpPicker, resolveEnabledTools, findServerForTool } from './ui-mcp-picker.js';
import * as mcp from './mcp-client.js';
import { MAX_TOOL_ITERATIONS, extractToolUses, wrapToolResult } from './mcp-tool-loop.js';

// math.js (KaTeX + theme) is dynamically imported on first render so the
// ~80 KB gz cost is paid only by sessions that actually contain math.
let _mathMod = null;
async function maybeRenderMath(node) {
    if (!node || !node.textContent || node.textContent.indexOf('$') < 0) return;
    if (!_mathMod) {
        try { _mathMod = await import('./math.js'); }
        catch (e) { console.warn('[bw] math import failed:', e && e.message ? e.message : e); return; }
    }
    _mathMod.renderMathWithin(node);
}
import * as voice from './voice.js';
import { isDownloadActive, activeModelId } from './ui-download-banner.js';
import * as cryptoStore from '../crypto-store.js';
import { mount as mountView } from './views.js';

// ── Module state ───────────────────────────────────────────────

let _root = null;
let _conversationId = null;
let _conversation = null;          // { id, title, providerId, ... }
let _messages = [];                // array of { messageId, role, content, ... }
let _streaming = null;             // { messageId, bubble, contentNode, finalized }
let _autoScroll = true;
let _activeProviderId = null;
// Per-turn tool-call counter and abort flag — survive across the
// repeated runProvider() calls inside one user turn so the cap and
// cancellation in mcp-tool-loop.js semantics translate to the live UI.
// `_aborted` is reserved for a future Stop button; flipping it bails
// out of executeToolUsesAndContinue between iterations.
let _toolIterations = 0;
let _aborted = false;
// Cached "is the home agent paired and reachable?" flag. The bundle
// lives in IDB and may be encrypted with the session key, so we resolve
// it asynchronously and cache the answer so the synchronous picker code
// (cycleProvider / updateProviderChip) doesn't have to await on every
// click. Refreshed on view init and via refreshHomeAvailability() after
// pairing / unlock events.
let _homeAvailable = false;

async function refreshHomeAvailability() {
    try { _homeAvailable = await homeProvider.isAvailable(); }
    catch (_) { _homeAvailable = false; }
}

// Filter out the home provider when the user isn't paired (or the
// session is locked and the bundle is encrypted). All other runtimes
// pass through unchanged.
function listAvailableProviders() {
    const all = listProviders();
    return all.filter((p) => p.runtime !== 'home' || _homeAvailable);
}

// DOM cache
const _ui = {
    titleEl: null,
    listEl: null,
    drawerEl: null,
    drawerListEl: null,
    composer: null,
    textarea: null,
    sendBtn: null,
    micBtn: null,
    providerChip: null,
    homeStatusPill: null,
    listening: false,
};

// ── Public render ──────────────────────────────────────────────

export async function render(root) {
    _root = root;
    clear(root);
    root.appendChild(buildLayout());
    subscribeStreams();
    bindVisualViewport();
    await refreshConversations();
    await refreshActiveProvider();

    // Pick the most recent conversation (or create a new one).
    let id = await getSetting('chat.activeConversationId');
    if (!id) {
        const all = await listConversations();
        id = all && all.length ? all[0].id : null;
    }
    if (!id) id = await newConversation(/* silent */ true);
    await loadConversation(id);
}

export function onShow() {
    if (_ui.textarea && !isMobile()) {
        try { _ui.textarea.focus(); } catch (_) {}
    }
}

// ── Layout ─────────────────────────────────────────────────────

function buildLayout() {
    // Header
    const titleEl = el('h1', { class: 'chat-title', attrs: { 'aria-live': 'polite' } }, t('chat.title'));
    _ui.titleEl = titleEl;

    const drawerToggle = el('button', {
        class: 'icon-btn',
        attrs: { type: 'button', 'aria-label': t('nav.menu'), 'aria-controls': 'conversation-drawer' },
        onClick: () => toggleDrawer(true),
    }, glyph('menu'));

    const overflowBtn = el('button', {
        class: 'icon-btn',
        attrs: { type: 'button', 'aria-label': 'More', 'aria-haspopup': 'menu' },
        onClick: (e) => openOverflowMenu(e.currentTarget),
    }, glyph('dots'));

    const settingsBtn = el('button', {
        class: 'icon-btn',
        attrs: { type: 'button', 'aria-label': t('nav.settings') },
        onClick: () => mountView('settings'),
    }, glyph('gear'));

    const header = el('header', { class: 'chat-header' },
        drawerToggle,
        titleEl,
        settingsBtn,
        overflowBtn,
    );

    // Drawer (off-canvas)
    const drawerListEl = el('ol', { class: 'drawer-list', attrs: { 'aria-label': 'Conversations' } });
    const drawerEl = el('nav', {
        id: 'conversation-drawer',
        class: 'drawer',
        attrs: { 'aria-label': 'Conversations', 'aria-hidden': 'true' },
    },
        el('div', { class: 'drawer-header' },
            el('button', {
                class: 'icon-btn',
                attrs: { type: 'button', 'aria-label': t('nav.close') },
                onClick: () => toggleDrawer(false),
            }, glyph('x')),
            el('strong', { class: 'drawer-title' }, t('app.title')),
        ),
        el('button', {
            class: 'btn drawer-new',
            attrs: { type: 'button' },
            onClick: async () => { const id = await newConversation(); toggleDrawer(false); await loadConversation(id); },
        }, '+ ' + t('chat.newChat')),
        drawerListEl,
    );
    _ui.drawerEl = drawerEl;
    _ui.drawerListEl = drawerListEl;
    const scrim = el('div', {
        class: 'drawer-scrim',
        attrs: { 'aria-hidden': 'true' },
        onClick: () => toggleDrawer(false),
    });

    // Messages
    const listEl = el('ol', { class: 'chat-messages', attrs: { 'aria-live': 'polite' } });
    _ui.listEl = listEl;
    listEl.addEventListener('scroll', () => {
        const nearBottom = listEl.scrollTop + listEl.clientHeight >= listEl.scrollHeight - 24;
        _autoScroll = nearBottom;
    }, { passive: true });

    // Composer
    const textarea = el('textarea', {
        class: 'composer-input',
        attrs: {
            'aria-label': t('chat.placeholder'),
            placeholder: t('chat.placeholder'),
            rows: 1,
            spellcheck: 'true',
            autocomplete: 'off',
            autocapitalize: 'sentences',
        },
    });
    _ui.textarea = textarea;

    textarea.addEventListener('input', () => {
        autoSizeTextarea(textarea);
        updateSendDisabled();
    });
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isMobile()) {
            e.preventDefault();
            handleSend();
        }
    });

    const micBtn = el('button', {
        class: 'icon-btn mic-btn',
        attrs: { type: 'button', 'aria-label': t('voice.start') },
    }, glyph('mic'));
    _ui.micBtn = micBtn;
    bindMic(micBtn);

    // Attach button — hidden until a vision-capable model is selected.
    // The hidden file input is detached; the button triggers it via .click().
    const attachInput = el('input', {
        type: 'file',
        attrs: { accept: 'image/*,application/pdf', multiple: '', style: 'display:none' },
        onChange: async (e) => {
            const files = Array.from(e.currentTarget.files || []);
            if (files.length) await attachments.addFiles(files);
            e.currentTarget.value = '';
        },
    });
    const attachBtn = el('button', {
        class: 'icon-btn attach-btn',
        attrs: { type: 'button', 'aria-label': t('chat.attach'), hidden: '' },
        onClick: () => attachInput.click(),
    }, glyph('paperclip'));
    _ui.attachBtn = attachBtn;

    const sendBtn = el('button', {
        class: 'icon-btn send-btn',
        attrs: { type: 'button', 'aria-label': t('chat.send'), disabled: '' },
        onClick: handleSend,
    }, glyph('send'));
    _ui.sendBtn = sendBtn;

    const toolsBtn = el('button', {
        class: 'icon-btn tools-btn',
        attrs: { type: 'button', 'aria-label': t('mcp.picker.title') },
        onClick: () => {
            if (!_conversationId) { toast(t('mcp.picker.noConversation'), 'error'); return; }
            openMcpPicker(_conversationId);
        },
    }, glyph('tools'));
    _ui.toolsBtn = toolsBtn;

    const providerChip = el('button', {
        class: 'provider-chip',
        attrs: { type: 'button', 'aria-label': t('chat.provider') },
        onClick: cycleProvider,
    }, t('chat.provider'));
    _ui.providerChip = providerChip;

    // M12 — home-transport status pill. Hidden by default; only visible
    // when the active provider is the home agent. State text/colour is
    // refreshed by updateHomeStatusPill() on every provider change and
    // every 'home-transport-state' event.
    const homeStatusPill = el('span', {
        class: 'pill home-status-pill',
        attrs: { hidden: '', 'aria-live': 'polite', 'data-state': 'idle' },
    }, '');
    _ui.homeStatusPill = homeStatusPill;

    const syncStatusPill = el('span', {
        class: 'pill sync-status-pill',
        attrs: { hidden: '', 'aria-live': 'polite', 'data-state': 'idle' },
    }, '');
    _ui.syncStatusPill = syncStatusPill;

    const attachmentStrip = buildAttachmentStrip();

    const composer = el('form', {
        class: 'composer',
        attrs: { 'aria-label': 'Composer' },
        onSubmit: (e) => { e.preventDefault(); handleSend(); },
    },
        attachmentStrip,
        el('div', { class: 'composer-row' },
            attachBtn,
            attachInput,
            toolsBtn,
            micBtn,
            textarea,
            sendBtn,
        ),
        el('div', { class: 'composer-meta' },
            providerChip,
            homeStatusPill,
            syncStatusPill,
        ),
    );
    _ui.composer = composer;

    // Paste-from-clipboard image support on the textarea. Gated by the
    // attach button's visibility so we never silently swallow images on
    // non-vision providers.
    textarea.addEventListener('paste', async (e) => {
        if (attachBtn.hasAttribute('hidden')) return;
        const items = e.clipboardData ? Array.from(e.clipboardData.items) : [];
        const files = items
            .filter((it) => it.kind === 'file' && it.type && it.type.startsWith('image/'))
            .map((it) => it.getAsFile())
            .filter(Boolean);
        if (files.length) {
            e.preventDefault();
            await attachments.addFiles(files);
        }
    });

    // Drag-and-drop on the chat shell. Same gate as paste.
    function onDragOver(e) {
        if (attachBtn.hasAttribute('hidden')) return;
        if (!e.dataTransfer || !Array.from(e.dataTransfer.items || []).some((it) => it.kind === 'file')) return;
        e.preventDefault();
    }
    async function onDrop(e) {
        if (attachBtn.hasAttribute('hidden')) return;
        const files = Array.from(e.dataTransfer ? e.dataTransfer.files : []);
        const accepted = files.filter((f) => f.type.startsWith('image/') || f.type === 'application/pdf');
        if (accepted.length === 0) return;
        e.preventDefault();
        await attachments.addFiles(accepted);
    }
    composer.addEventListener('dragover', onDragOver);
    composer.addEventListener('drop', onDrop);

    return el('div', { class: 'chat-shell' },
        header,
        scrim,
        drawerEl,
        listEl,
        composer,
    );
}

// ── Conversations & messages ───────────────────────────────────

async function refreshConversations() {
    const conversations = await listConversations();
    const list = _ui.drawerListEl;
    if (!list) return;
    clear(list);
    if (!conversations.length) {
        list.appendChild(el('li', { class: 'drawer-empty' }, t('chat.empty')));
        return;
    }
    for (const c of conversations) {
        const item = el('li', { class: 'drawer-item' },
            el('button', {
                class: 'drawer-link' + (c.id === _conversationId ? ' is-active' : ''),
                attrs: { type: 'button' },
                onClick: async () => { toggleDrawer(false); await loadConversation(c.id); },
            }, c.title || t('chat.title')),
            el('button', {
                class: 'icon-btn drawer-del',
                attrs: { type: 'button', 'aria-label': t('chat.delete') },
                onClick: async (e) => {
                    e.stopPropagation();
                    if (!confirm(t('chat.confirmDelete'))) return;
                    await deleteConversation(c.id);
                    if (c.id === _conversationId) {
                        const all = await listConversations();
                        const next = all && all.length ? all[0].id : await newConversation(true);
                        await loadConversation(next);
                    } else {
                        await refreshConversations();
                    }
                    toast(t('chat.deleted'), 'success');
                },
            }, glyph('x')),
        );
        list.appendChild(item);
    }
}

async function newConversation(silent = false) {
    const id = genId('conv');
    const row = await putConversation({ id, title: t('chat.newChat'), providerId: _activeProviderId });
    _conversation = row;
    _conversationId = id;
    _messages = [];
    await setSetting('chat.activeConversationId', id);
    if (!silent) {
        await refreshConversations();
        renderMessages();
        setTitle(row.title);
    }
    return id;
}

async function loadConversation(id) {
    _conversationId = id;
    await setSetting('chat.activeConversationId', id);
    const all = await listConversations();
    _conversation = all.find((c) => c.id === id) || null;
    setTitle(_conversation ? (_conversation.title || t('chat.title')) : t('chat.title'));
    _messages = await listMessages(id);
    await refreshConversations();
    renderMessages();
}

function setTitle(title) {
    if (_ui.titleEl) _ui.titleEl.textContent = title || t('chat.title');
}

function renderMessages() {
    if (!_ui.listEl) return;
    clear(_ui.listEl);
    if (_messages.length === 0) {
        const emptyState = el('li', { class: 'chat-empty-state' },
            el('img', { class: 'chat-logo', attrs: { src: 'icons/icon-192.png', alt: 'Brainwires Chat' } }),
            el('h2', {}, 'Brainwires Chat'),
            el('p', { class: 'build-stamp', id: 'build-stamp' }),
        );
        _ui.listEl.appendChild(emptyState);
        import('../build-info.js').then(info => {
            const stamp = document.getElementById('build-stamp');
            if (!stamp || !info) return;
            const parts = [info.BUILD_GIT, info.BUILD_TIME].filter(Boolean);
            stamp.textContent = parts.join(' — ') || 'dev';
            wireHardRefresh(stamp);
        }).catch(() => {});
        return;
    }
    for (const m of _messages) {
        _ui.listEl.appendChild(buildBubble(m));
    }
    // Initial scroll-to-bottom on conversation load. One rAF gets us
    // past the synchronous append; the second rAF + ResizeObserver
    // covers the markdown / image / async-bubble case where bubble
    // height changes after first paint.
    requestAnimationFrame(() => {
        scrollToBottom(true);
        requestAnimationFrame(() => scrollToBottom(true));
    });
    if (typeof ResizeObserver !== 'undefined' && _ui.listEl && !_ui.listEl._initialScrollObs) {
        const obs = new ResizeObserver(() => {
            if (_autoScroll) scrollToBottom(true);
        });
        // Watch each message bubble for late layout (images, code blocks).
        for (const child of _ui.listEl.children) obs.observe(child);
        _ui.listEl._initialScrollObs = obs;
        // Stop forcing scroll once the user has had a moment to interact.
        setTimeout(() => {
            obs.disconnect();
            delete _ui.listEl._initialScrollObs;
        }, 1500);
    }
}

function buildBubble(m) {
    const isUser = m.role === 'user';
    const cls = isUser ? 'bubble bubble-user' : 'bubble bubble-assistant';
    const textContent = partsToText(m.content || '');
    const imageParts = Array.isArray(m.content)
        ? m.content.filter((p) => p && p.type === 'image' && typeof p.data === 'string')
        : [];
    const toolUseParts = Array.isArray(m.content)
        ? m.content.filter((p) => p && p.type === 'tool_use')
        : [];
    const toolResultParts = Array.isArray(m.content)
        ? m.content.filter((p) => p && p.type === 'tool_result')
        : [];
    const { thinking, body } = isUser
        ? { thinking: null, body: textContent }
        : extractThinking(textContent);
    const contentNode = el('div', { class: 'bubble-content' });
    if (thinking) contentNode.appendChild(buildReasoningElement(thinking));
    if (imageParts.length > 0) {
        const gallery = el('div', { class: 'bubble-images' });
        for (const p of imageParts) {
            const img = el('img', {
                class: 'bubble-image',
                attrs: {
                    src: `data:${p.mediaType || 'image/jpeg'};base64,${p.data}`,
                    alt: 'attached image',
                    loading: 'lazy',
                    decoding: 'async',
                },
            });
            gallery.appendChild(img);
        }
        contentNode.appendChild(gallery);
    }
    // Tool result parts (user-role bubble only emits these in practice).
    for (const r of toolResultParts) {
        contentNode.appendChild(buildToolResultEl(r));
    }
    const bodyNode = el('div', { class: 'bubble-body' });
    bodyNode.innerHTML = renderMarkdown(body);
    contentNode.appendChild(bodyNode);
    highlightWithin(bodyNode);
    maybeRenderMath(bodyNode);
    // Tool use parts (assistant-role bubble) — keep these AFTER the body
    // so the natural reading order matches the streamed sequence: model
    // talks, then calls a tool.
    for (const p of toolUseParts) {
        contentNode.appendChild(buildToolUseEl(p));
    }

    const actions = el('div', { class: 'bubble-actions' });
    actions.appendChild(el('button', {
        class: 'icon-btn',
        attrs: { type: 'button', 'aria-label': t('chat.copy') },
        onClick: () => copyText(textContent),
    }, glyph('copy')));
    if (!isUser) {
        actions.appendChild(el('button', {
            class: 'icon-btn',
            attrs: { type: 'button', 'aria-label': t('chat.speak') },
            onClick: async () => { try { await voice.speak(textContent); } catch (_) {} },
        }, glyph('speaker')));
        actions.appendChild(el('button', {
            class: 'icon-btn',
            attrs: { type: 'button', 'aria-label': t('chat.regenerate') },
            onClick: () => regenerateAt(m.messageId),
        }, glyph('refresh')));
    }

    const li = el('li', { class: cls, attrs: { 'data-msg-id': m.messageId } },
        contentNode,
        actions,
    );
    return li;
}

// Render `{type:'tool_use', id, name, input}` as a collapsible block
// styled consistently with the reasoning <details>.
function buildToolUseEl(p) {
    const details = el('details', { class: 'tool-call' });
    const summary = el('summary', { class: 'tool-call-summary' },
        el('span', { class: 'tool-call-label' }, t('mcp.tools.callLabel')),
        el('code', { class: 'tool-call-name' }, p.name || ''),
    );
    details.appendChild(summary);
    const pre = el('pre', { class: 'tool-call-input' });
    const code = el('code', { class: 'language-json' });
    code.textContent = JSON.stringify(p.input || {}, null, 2);
    pre.appendChild(code);
    details.appendChild(pre);
    highlightWithin(details);
    return details;
}

// Render `{type:'tool_result', toolUseId, content, is_error?}` similarly.
function buildToolResultEl(p) {
    const details = el('details', { class: 'tool-result' + (p.is_error ? ' tool-result-error' : '') });
    const summary = el('summary', { class: 'tool-result-summary' },
        el('span', { class: 'tool-result-label' },
            p.is_error ? t('mcp.tools.errorLabel') : t('mcp.tools.resultLabel')),
        el('code', { class: 'tool-result-id' }, p.toolUseId || ''),
    );
    details.appendChild(summary);
    const pre = el('pre', { class: 'tool-result-content' });
    const code = el('code', { class: 'language-json' });
    code.textContent = typeof p.content === 'string'
        ? p.content
        : JSON.stringify(p.content == null ? '' : p.content, null, 2);
    pre.appendChild(code);
    details.appendChild(pre);
    highlightWithin(details);
    return details;
}

// ── Send + streaming ───────────────────────────────────────────

async function handleSend() {
    const text = _ui.textarea ? _ui.textarea.value.trim() : '';
    const attached = attachments.getAll();
    if (!text && attached.length === 0) return;
    if (!_activeProviderId) { toast(t('error.noProvider'), 'error'); return; }
    if (!await canUseProvider(_activeProviderId)) return;

    if (!_conversationId) await newConversation(true);

    // If any image attachments are queued, pre-process them into parts[].
    // Resize + base64 happens before the user message is persisted so that
    // a successful save means the image is fully prepared for retries.
    let content = text;
    if (attached.length > 0) {
        const parts = [];
        for (const a of attached) {
            if (a.kind !== 'image') continue;
            try {
                const { data, mediaType } = await imageToBase64(a.file);
                parts.push({ type: 'image', mediaType, data });
            } catch (e) {
                toast(`Failed to process ${a.name}: ${e.message || e}`, 'error');
            }
        }
        if (text) parts.push({ type: 'text', text });
        content = parts.length > 0 ? parts : text;
    }

    const userMsg = {
        conversationId: _conversationId,
        messageId: genId('msg'),
        role: 'user',
        content,
        createdAt: Date.now(),
        updatedAt: Date.now(),
    };
    await putMessage(userMsg);
    _messages.push(userMsg);
    _ui.listEl.querySelector('.chat-empty-state')?.remove();
    _ui.listEl.appendChild(buildBubble(userMsg));

    _ui.textarea.value = '';
    autoSizeTextarea(_ui.textarea);
    attachments.clear();
    updateSendDisabled();

    // If this is the first user message, set the conversation title to a
    // snippet of the text (parts content with no text falls back to a
    // generic image-message label).
    if (_messages.filter((m) => m.role === 'user').length === 1) {
        const titleSrc = text || (attached.length > 0 ? `[image] ${attached[0].name}` : '');
        const snip = titleSrc.length > 48 ? titleSrc.slice(0, 45) + '…' : titleSrc;
        await putConversation({ ..._conversation, id: _conversationId, title: snip });
        setTitle(snip);
        await refreshConversations();
    }

    // Best-effort RAG retrieval. Failures (no embedding model, no docs,
    // wasm not yet rebuilt) silently fall through to a no-context send so
    // chat keeps working before the user has set up a library.
    let history = _messages.map((m) => ({ role: m.role, content: m.content }));
    try {
        const hits = await ragRetrieve(text, { conversationId: _conversationId, k: 4 });
        const sys = formatRetrievalAsSystem(hits);
        if (sys) history = [{ role: 'system', content: sys }, ...history];
    } catch (e) {
        console.warn('[bw] rag retrieve skipped:', e && e.message ? e.message : e);
    }

    // Resolve enabled MCP tools for this conversation. The picker stores
    // per-conversation enable state; the loop and provider envelope share
    // the same shape.
    let tools = [];
    try {
        tools = await resolveEnabledTools(_conversationId);
    } catch (e) {
        console.warn('[bw] mcp tool resolve skipped:', e && e.message ? e.message : e);
    }

    // Reset per-turn loop state. The cap counter and abort flag persist
    // across the (possibly multiple) runProvider invocations below.
    _toolIterations = 0;
    _aborted = false;
    await runProvider(history, tools);
}

async function runProvider(messages, tools) {
    const messageId = genId('msg');
    const placeholder = {
        conversationId: _conversationId,
        messageId,
        role: 'assistant',
        content: '',
        createdAt: Date.now(),
        updatedAt: Date.now(),
    };
    _messages.push(placeholder);
    const bubble = buildBubble(placeholder);
    _ui.listEl.appendChild(bubble);
    const contentNode = bubble.querySelector('.bubble-content');
    _streaming = {
        messageId,
        bubble,
        contentNode,
        bodyNode: contentNode.querySelector('.bubble-body'),
        reasoningNode: null,
        accum: '',
        // Live tool_use parts arrive via the chat_tool_use SW event;
        // buildBubble re-renders the parts list when this changes.
        toolUseParts: [],
        finalized: false,
        userMessages: messages,
        tools: Array.isArray(tools) ? tools : [],
    };
    scrollToBottom(false);
    await putMessage(placeholder);

    // Resolve API key & session key for cloud providers.
    let apiKeyEncrypted = null;
    let sessionKey = null;
    const providerInfo = listProviders().find((p) => p.id === _activeProviderId);
    if (providerInfo && providerInfo.runtime === 'cloud' && _activeProviderId !== 'ollama') {
        try {
            const blob = await getSetting(`provider.${_activeProviderId}.apiKey`);
            if (blob && blob.encrypted) {
                apiKeyEncrypted = blob.encrypted;
                sessionKey = getSessionKey();
                if (!sessionKey) {
                    streamingError(t('error.locked'));
                    return;
                }
            } else if (blob && blob.plaintext) {
                // We need the SW to substitute, so we have to pack this
                // as an encrypted blob. The SW always expects encrypted.
                // For unencrypted-storage users, we encrypt with an
                // ephemeral session key on the fly so the SW can decrypt.
                // Simpler path: complain and ask them to set a passphrase.
                streamingError('API key stored without encryption — set a passphrase in Settings to send via cloud providers.');
                return;
            } else {
                streamingError(t('error.noKey'));
                return;
            }
        } catch (e) {
            streamingError(e && e.message ? e.message : String(e));
            return;
        }
    }

    const params = {};
    if (providerInfo && _activeProviderId === 'ollama') {
        const baseUrl = await getSetting('provider.ollama.baseUrl');
        if (baseUrl) params.baseUrl = baseUrl;
    }
    const modelOverride = await getSetting(`provider.${_activeProviderId}.model`);
    if (modelOverride) params.model = modelOverride;
    if (Array.isArray(tools) && tools.length) {
        // Strip the picker-only `_serverId` field — providers ignore
        // unknown keys but keep the wire format minimal.
        params.tools = tools.map((tool) => ({
            name: tool.name,
            description: tool.description,
            input_schema: tool.input_schema,
        }));
    }

    const result = await startChat({
        provider: _activeProviderId,
        conversationId: _conversationId,
        messageId,
        messages,
        params,
        apiKeyEncrypted,
        sessionKey,
    });
    if (!result.ok) {
        streamingError(result.error || t('error.generic'));
    }
}

// Render `_streaming.accum` into the bubble, splitting an optional leading
// `<thinking>...</thinking>` block into a collapsible reasoning <details>
// above the body. Called from chunk / done / error handlers so all three
// paths share the dual-pane logic.
function renderStreamingFrame(streaming) {
    if (!streaming) return;
    const { thinking, body } = extractThinking(streaming.accum || '');
    if (thinking) {
        if (!streaming.reasoningNode) {
            streaming.reasoningNode = buildReasoningElement(thinking);
            streaming.contentNode.insertBefore(streaming.reasoningNode, streaming.bodyNode);
        } else {
            const bodyEl = streaming.reasoningNode.querySelector('.reasoning-body');
            if (bodyEl) bodyEl.innerHTML = renderMarkdown(thinking);
        }
    } else if (streaming.reasoningNode) {
        // Closing tag arrived after we already rendered partial reasoning;
        // extraction now succeeds and the branch above runs. The non-thinking
        // case here means the model never opened a thinking tag — leave the
        // reasoning node alone if the user opened it.
    }
    streaming.bodyNode.innerHTML = renderMarkdown(body);
}

function streamingError(msg) {
    if (_streaming) {
        const bodyNode = _streaming.bodyNode || _streaming.contentNode;
        bodyNode.innerHTML = `<em class="bubble-error">${escapeHtml(msg)}</em>`;
        _streaming.finalized = true;
        _streaming = null;
    }
    toast(msg, 'error');
}

function subscribeStreams() {
    stateEvents.addEventListener('chat_chunk', (e) => {
        const d = e.detail || {};
        if (!_streaming || _streaming.messageId !== d.messageId) return;
        if (typeof d.delta !== 'string') return;
        _streaming.accum += d.delta;
        renderStreamingFrame(_streaming);
        if (_autoScroll) scrollToBottom(false);
    });
    stateEvents.addEventListener('chat_tool_use', (e) => {
        const d = e.detail || {};
        if (!_streaming || _streaming.messageId !== d.messageId) return;
        const tu = d.tool_use;
        if (!tu || typeof tu.name !== 'string') return;
        _streaming.toolUseParts.push({
            type: 'tool_use',
            id: tu.id || '',
            name: tu.name,
            input: tu.input || {},
        });
        // Live-render: append a <details> to the bubble so the user sees
        // each tool call as it is reassembled.
        try {
            _streaming.contentNode.appendChild(buildToolUseEl(_streaming.toolUseParts[_streaming.toolUseParts.length - 1]));
        } catch (_) {}
        if (_autoScroll) scrollToBottom(false);
    });
    stateEvents.addEventListener('chat_done', (e) => {
        const d = e.detail || {};
        if (!_streaming || _streaming.messageId !== d.messageId) return;
        renderStreamingFrame(_streaming);
        const bodyNode = _streaming.bodyNode;
        // Snapshot streaming context BEFORE finalize clears _streaming.
        const ctx = {
            messageId: _streaming.messageId,
            tools: _streaming.tools,
            toolUseParts: _streaming.toolUseParts.slice(),
            text: _streaming.accum || '',
        };
        finalizeStreaming();
        highlightWithin(bodyNode);
        maybeRenderMath(bodyNode);
        // If the assistant emitted tool_use parts, drive the loop.
        // Errors are surfaced via toast; iteration cap is enforced too.
        if (ctx.toolUseParts.length > 0) {
            executeToolUsesAndContinue(ctx).catch((err) => {
                console.warn('[bw] tool loop failed:', err && err.message ? err.message : err);
                toast(err && err.message ? err.message : String(err), 'error');
            });
        }
    });
    stateEvents.addEventListener('chat_error', (e) => {
        const d = e.detail || {};
        if (!_streaming || _streaming.messageId !== d.messageId) return;
        const err = d.error || t('error.generic');
        renderStreamingFrame(_streaming);
        const bodyNode = _streaming.bodyNode;
        bodyNode.insertAdjacentHTML('beforeend', `<em class="bubble-error"> — ${escapeHtml(err)}</em>`);
        highlightWithin(bodyNode);
        maybeRenderMath(bodyNode);
        _streaming.finalized = true;
        _streaming = null;
        toast(err, 'error');
    });

    // M12 — home-transport status events: keep the pill in lockstep
    // and surface a toast when the link drops while the user is on
    // the home agent. Suppress the toast for transient blips during
    // M10's reconnect flow — only 'failed' (after retries exhausted)
    // turns into a user-visible alert.
    stateEvents.addEventListener('home-transport-state', (e) => {
        const d = e.detail || {};
        updateHomeStatusPill();
        if (d.next === 'failed' || d.next === 'closed') {
            updateSyncStatusPill('idle');
        }
        if (d.next === 'failed' && _activeProviderId === homeProvider.id) {
            toast(t('home.status.failed'), 'error');
        }
    });
    // After unpair, hide "Home agent" from the picker and reset the pill.
    stateEvents.addEventListener('home-unpaired', () => {
        refreshActiveProvider().catch(() => {});
        updateHomeStatusPill();
        updateSyncStatusPill('idle');
    });

    // Sync status events — update the sync pill and toast on pull.
    stateEvents.addEventListener('sync-update', (e) => {
        const entries = e.detail?.entries;
        const count = Array.isArray(entries) ? entries.length : 0;
        if (count > 0) {
            updateSyncStatusPill('synced');
            toast(t('sync.toast.pulled', { count }), 'info', 3000);
            refreshConversations().catch(() => {});
        }
    });
}

// Run the tool calls produced by the just-finished assistant message,
// post a synthetic user-role tool_result message, and resume the chat.
// Capped at MAX_TOOL_ITERATIONS — past that we stop and toast.
async function executeToolUsesAndContinue(ctx) {
    if (_aborted) return;
    if (_toolIterations >= MAX_TOOL_ITERATIONS) {
        toast(t('mcp.tools.loopLimit'), 'error');
        return;
    }
    _toolIterations += 1;

    // Persist the assistant message with text + tool_use parts so the
    // history we replay back to the provider matches what the user sees.
    const assistantParts = [];
    if (ctx.text) assistantParts.push({ type: 'text', text: ctx.text });
    for (const p of ctx.toolUseParts) assistantParts.push(p);
    const assistantIdx = _messages.findIndex((m) => m.messageId === ctx.messageId);
    if (assistantIdx >= 0) {
        _messages[assistantIdx].content = assistantParts;
        try {
            await putMessage({
                conversationId: _conversationId,
                messageId: ctx.messageId,
                role: 'assistant',
                content: assistantParts,
                updatedAt: Date.now(),
            });
        } catch (_) {}
    }

    // Run each tool call; collect results.
    const tuList = extractToolUses(assistantParts);
    const resultParts = [];
    for (const tu of tuList) {
        if (_aborted) return;
        try {
            const server = await findServerForTool(_conversationId, tu.name);
            if (!server) {
                resultParts.push(wrapToolResult(tu, { ok: false, error: `tool '${tu.name}' has no enabled server` }));
                continue;
            }
            const value = await mcp.callTool(server, tu.name, tu.input);
            resultParts.push(wrapToolResult(tu, { ok: true, value }));
        } catch (err) {
            const msg = err && err.message ? err.message : String(err);
            resultParts.push(wrapToolResult(tu, { ok: false, error: msg }));
        }
    }

    // Persist + render a synthetic user-role tool-result message.
    const resultMsg = {
        conversationId: _conversationId,
        messageId: genId('msg'),
        role: 'user',
        content: resultParts,
        createdAt: Date.now(),
        updatedAt: Date.now(),
    };
    await putMessage(resultMsg);
    _messages.push(resultMsg);
    _ui.listEl.appendChild(buildBubble(resultMsg));
    if (_autoScroll) scrollToBottom(false);

    if (_aborted) return;

    // Resume the chat with the extended history. Tools stay enabled so
    // the model can chain calls — capped above.
    const history = _messages.map((m) => ({ role: m.role, content: m.content }));
    await runProvider(history, ctx.tools);
}

async function finalizeStreaming() {
    if (!_streaming) return;
    const id = _streaming.messageId;
    const text = _streaming.accum || '';
    // Update local cache + db.
    const idx = _messages.findIndex((m) => m.messageId === id);
    if (idx >= 0) _messages[idx].content = text;
    try {
        await putMessage({
            conversationId: _conversationId,
            messageId: id,
            role: 'assistant',
            content: text,
            updatedAt: Date.now(),
        });
        await putConversation({ ..._conversation, id: _conversationId, updatedAt: Date.now() });
    } catch (_) { /* best-effort */ }
    _streaming.finalized = true;
    _streaming = null;
    refreshConversations();
}

async function regenerateAt(messageId) {
    // Find the user message that produced this assistant message and
    // resend up to (but not including) the assistant message.
    const idx = _messages.findIndex((m) => m.messageId === messageId);
    if (idx < 0) return;
    const slice = _messages.slice(0, idx);
    // Drop the old assistant from the array + DOM; we'll spawn a new one.
    _messages.splice(idx, 1);
    const node = _ui.listEl.querySelector(`[data-msg-id="${messageId}"]`);
    if (node) node.remove();
    let tools = [];
    try { tools = await resolveEnabledTools(_conversationId); }
    catch (_) {}
    _toolIterations = 0;
    _aborted = false;
    await runProvider(slice.map((m) => ({ role: m.role, content: m.content })), tools);
}

// ── Provider handling ──────────────────────────────────────────

async function refreshActiveProvider() {
    const stored = await getSetting('chat.activeProvider');
    await refreshHomeAvailability();
    const providers = listAvailableProviders();
    if (!providers.length) return;
    if (stored && providers.find((p) => p.id === stored)) {
        _activeProviderId = stored;
    } else {
        const local = providers.find((p) => p.id === 'local');
        _activeProviderId = (local || providers[0]).id;
    }
    updateProviderChip();
    updateSendDisabled();
    await updateAttachVisibility();
}

async function cycleProvider() {
    // Re-check pairing each cycle so a fresh pair shows up without a
    // page reload (and an unpair drops the option mid-session).
    await refreshHomeAvailability();
    const providers = listAvailableProviders();
    if (!providers.length) return;
    const idx = providers.findIndex((p) => p.id === _activeProviderId);
    const next = providers[(idx + 1) % providers.length];
    _activeProviderId = next.id;
    await setSetting('chat.activeProvider', next.id);
    updateProviderChip();
    updateSendDisabled();
    await updateAttachVisibility();
}

// Show the attach button only when the active (provider, model) pair
// supports image inputs. Reads the saved per-provider model so the
// gating reflects the user's selection, not just the provider default.
async function updateAttachVisibility() {
    if (!_ui.attachBtn) return;
    let model = '';
    try { model = await getSetting(`provider.${_activeProviderId}.model`); } catch (_) {}
    const providers = listProviders();
    const p = providers.find((x) => x.id === _activeProviderId);
    if (!model && p) model = p.defaultModel;
    if (isVisionModel(_activeProviderId, model)) {
        _ui.attachBtn.removeAttribute('hidden');
    } else {
        _ui.attachBtn.setAttribute('hidden', '');
        // Drop any pending attachments — they can't be sent through this
        // provider anyway and would silently be discarded at submit.
        attachments.clear();
    }
}

function updateProviderChip() {
    if (!_ui.providerChip) return;
    const providers = listProviders();
    const p = providers.find((x) => x.id === _activeProviderId);
    _ui.providerChip.textContent = p ? p.displayName : t('chat.provider');
    updateHomeStatusPill();
}

/**
 * M12 — sync the connection-status pill to the home-transport state.
 * Hidden when the home agent isn't the active provider, or in the
 * 'idle' / 'closed' states (nothing to surface). Exported via the
 * module-level `homeStatusPillState()` helper for the unit tests.
 */
function updateHomeStatusPill() {
    const pill = _ui.homeStatusPill;
    if (!pill) return;
    const activeIsHome = _activeProviderId === homeProvider.id;
    const state = homeProvider.getTransportState ? homeProvider.getTransportState() : 'idle';
    pill.dataset.state = state;
    if (!activeIsHome || state === 'idle' || state === 'closed' || state === 'closing') {
        pill.setAttribute('hidden', '');
        pill.textContent = '';
        pill.classList.remove('pill-ok', 'pill-warn', 'pill-err');
        return;
    }
    pill.removeAttribute('hidden');
    pill.classList.remove('pill-ok', 'pill-warn', 'pill-err');
    if (state === 'connecting') {
        pill.classList.add('pill-warn');
        pill.textContent = t('home.status.connecting');
    } else if (state === 'connected') {
        pill.classList.add('pill-ok');
        pill.textContent = t('home.status.connected');
    } else if (state === 'reconnecting') {
        pill.classList.add('pill-warn');
        pill.textContent = t('home.status.reconnecting');
    } else if (state === 'failed') {
        pill.classList.add('pill-err');
        pill.textContent = t('home.status.failed');
    } else {
        // Unknown state — show muted with the raw label so we don't lie.
        pill.classList.add('pill-warn');
        pill.textContent = state;
    }
}

let _syncAutoHideTimer = null;
function updateSyncStatusPill(syncState) {
    const pill = _ui.syncStatusPill;
    if (!pill) return;
    if (_syncAutoHideTimer) { clearTimeout(_syncAutoHideTimer); _syncAutoHideTimer = null; }
    const activeIsHome = _activeProviderId === homeProvider.id;
    const transportUp = homeProvider.getTransportState?.() === 'connected';
    if (!activeIsHome || !transportUp || syncState === 'idle') {
        pill.setAttribute('hidden', '');
        pill.textContent = '';
        pill.dataset.state = 'idle';
        pill.classList.remove('pill-ok', 'pill-warn', 'pill-err');
        return;
    }
    pill.removeAttribute('hidden');
    pill.dataset.state = syncState;
    pill.classList.remove('pill-ok', 'pill-warn', 'pill-err');
    if (syncState === 'syncing') {
        pill.classList.add('pill-warn');
        pill.textContent = t('sync.status.syncing');
    } else if (syncState === 'synced') {
        pill.classList.add('pill-ok');
        pill.textContent = t('sync.status.synced');
        _syncAutoHideTimer = setTimeout(() => updateSyncStatusPill('idle'), 4000);
    } else if (syncState === 'error') {
        pill.classList.add('pill-err');
        pill.textContent = t('sync.status.error');
        _syncAutoHideTimer = setTimeout(() => updateSyncStatusPill('idle'), 6000);
    }
}

async function canUseProvider(id) {
    const p = listProviders().find((x) => x.id === id);
    if (!p) return false;
    if (p.runtime === 'local') {
        // Need model downloaded + nothing in flight.
        if (isDownloadActive() && activeModelId() === p.defaultModel) {
            toast(t('chat.modelLoading'), 'info');
            return false;
        }
        const ok = await isDownloaded(p.defaultModel);
        if (!ok) {
            toast(t('chat.modelNotReady'), 'error');
            return false;
        }
        if (!localProvider.isLocalModelLoaded()) {
            // Lazy-load — runProvider will block on this otherwise.
            try { await localProvider.loadLocalModel(p.defaultModel); }
            catch (e) { toast(e && e.message ? e.message : String(e), 'error'); return false; }
        }
        return true;
    }
    if (p.runtime === 'home') {
        // Pairing presence was already verified via listAvailableProviders;
        // double-check here so the picker can't get stale.
        const ok = await homeProvider.isAvailable();
        if (!ok) {
            toast(t('chat.homeNotPaired'), 'error');
            return false;
        }
        return true;
    }
    if (id === 'ollama') return true;
    if (!isSessionUnlocked()) {
        toast(t('error.locked'), 'error');
        return false;
    }
    const blob = await getSetting(`provider.${id}.apiKey`);
    if (!blob || (!blob.encrypted && !blob.plaintext)) {
        toast(t('error.noKey'), 'error');
        return false;
    }
    return true;
}

function updateSendDisabled() {
    if (!_ui.sendBtn || !_ui.textarea) return;
    const hasText = _ui.textarea.value.trim().length > 0;
    let disabled = !hasText;
    // Local provider with active download → disable send.
    if (!disabled) {
        const p = listProviders().find((x) => x.id === _activeProviderId);
        if (p && p.runtime === 'local' && isDownloadActive() && activeModelId() === p.defaultModel) {
            disabled = true;
        }
    }
    if (disabled) _ui.sendBtn.setAttribute('disabled', '');
    else _ui.sendBtn.removeAttribute('disabled');
}

// ── Drawer ─────────────────────────────────────────────────────

function toggleDrawer(open) {
    if (!_ui.drawerEl) return;
    if (open) {
        _ui.drawerEl.classList.add('is-open');
        _ui.drawerEl.removeAttribute('aria-hidden');
        document.body.classList.add('drawer-open');
    } else {
        _ui.drawerEl.classList.remove('is-open');
        _ui.drawerEl.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('drawer-open');
    }
}

// ── Overflow menu ──────────────────────────────────────────────

function openOverflowMenu(anchor) {
    let host = document.getElementById('overflow-menu-host');
    if (host) { host.remove(); host = null; }
    host = el('div', { id: 'overflow-menu-host', class: 'overflow-host', attrs: { role: 'menu' } });
    const close = () => { try { host.remove(); } catch (_) {} document.removeEventListener('click', closeOnOutside, true); };
    const closeOnOutside = (e) => { if (!host.contains(e.target) && e.target !== anchor) close(); };
    setTimeout(() => document.addEventListener('click', closeOnOutside, true), 0);

    host.appendChild(el('button', {
        class: 'menu-item',
        attrs: { type: 'button', role: 'menuitem' },
        onClick: async () => {
            close();
            const next = prompt(t('chat.renamePrompt'), _conversation && _conversation.title || '');
            if (!next || !_conversationId) return;
            await putConversation({ ..._conversation, id: _conversationId, title: next });
            _conversation = { ...(_conversation || {}), id: _conversationId, title: next };
            setTitle(next);
            await refreshConversations();
        },
    }, t('chat.rename')));

    host.appendChild(el('button', {
        class: 'menu-item',
        attrs: { type: 'button', role: 'menuitem' },
        onClick: async () => {
            close();
            if (!_conversationId) return;
            const data = { conversation: _conversation, messages: _messages };
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = el('a', { attrs: { href: url, download: `${(_conversation && _conversation.title) || 'chat'}.json` } });
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            toast(t('chat.exported'), 'success');
        },
    }, t('chat.export')));

    host.appendChild(el('button', {
        class: 'menu-item menu-item-danger',
        attrs: { type: 'button', role: 'menuitem' },
        onClick: async () => {
            close();
            if (!_conversationId) return;
            if (!confirm(t('chat.confirmDelete'))) return;
            await deleteConversation(_conversationId);
            const all = await listConversations();
            const next = all && all.length ? all[0].id : await newConversation(true);
            await loadConversation(next);
            toast(t('chat.deleted'), 'success');
        },
    }, t('chat.delete')));

    // Position near the anchor.
    const rect = anchor.getBoundingClientRect();
    host.style.position = 'fixed';
    host.style.top = `${Math.round(rect.bottom + 4)}px`;
    host.style.right = `${Math.round(window.innerWidth - rect.right)}px`;
    document.body.appendChild(host);
}

// ── Mic / STT ──────────────────────────────────────────────────

function bindMic(btn) {
    let stopFn = null;
    let pressActive = false;
    let toggleMode = false;

    const begin = async () => {
        if (_ui.listening) return;
        try {
            stopFn = await voice.listen({
                onResult: (text, isFinal) => {
                    if (!isFinal) return;
                    if (!_ui.textarea) return;
                    const t0 = _ui.textarea.value;
                    _ui.textarea.value = (t0 ? t0 + ' ' : '') + text;
                    autoSizeTextarea(_ui.textarea);
                    updateSendDisabled();
                },
                onEnd: () => {
                    _ui.listening = false;
                    btn.classList.remove('is-listening');
                    btn.setAttribute('aria-label', t('voice.start'));
                },
                onError: (err) => {
                    _ui.listening = false;
                    btn.classList.remove('is-listening');
                    if (err === 'not-allowed' || err === 'service-not-allowed') {
                        toast(t('voice.permissionDenied'), 'error');
                    }
                },
            });
            _ui.listening = true;
            btn.classList.add('is-listening');
            btn.setAttribute('aria-label', t('voice.stop'));
        } catch (e) {
            if (e && e.name === 'STT_UNSUPPORTED') {
                toast(t('voice.unsupported'), 'error');
            } else {
                toast(e && e.message ? e.message : String(e), 'error');
            }
        }
    };
    const end = () => {
        if (stopFn) { try { stopFn(); } catch (_) {} stopFn = null; }
    };

    // Hold-to-talk via pointer events (covers mouse + touch + pen).
    btn.addEventListener('pointerdown', (e) => {
        if (e.button !== undefined && e.button !== 0) return;
        pressActive = true;
        toggleMode = false;
        // If the press never holds long enough, treat as toggle (tap).
        const start = Date.now();
        begin();
        const release = () => {
            if (!pressActive) return;
            pressActive = false;
            const held = Date.now() - start;
            if (held < 250) {
                // Treat as toggle: leave listening on; next tap stops.
                toggleMode = true;
                btn.removeEventListener('pointerup', release, true);
                btn.removeEventListener('pointerleave', release, true);
                btn.removeEventListener('pointercancel', release, true);
                return;
            }
            end();
            btn.removeEventListener('pointerup', release, true);
            btn.removeEventListener('pointerleave', release, true);
            btn.removeEventListener('pointercancel', release, true);
        };
        btn.addEventListener('pointerup', release, true);
        btn.addEventListener('pointerleave', release, true);
        btn.addEventListener('pointercancel', release, true);
    });

    // Click handler for the toggle-mode tap-to-stop.
    btn.addEventListener('click', () => {
        if (toggleMode && _ui.listening) {
            toggleMode = false;
            end();
        }
    });
}

// ── Code block copy button (event delegation) ─────────────────
// Wire the copy buttons in code blocks via event delegation. The button
// markup is produced by markdown.js with [data-bw-copy].
document.addEventListener('click', (e) => {
    const t0 = e.target;
    if (!(t0 instanceof Element)) return;
    if (!t0.matches('[data-bw-copy]')) return;
    const codeBlock = t0.parentElement && t0.parentElement.querySelector('pre code');
    if (!codeBlock) return;
    const text = codeBlock.textContent || '';
    copyText(text);
});

async function copyText(text) {
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
        } else {
            const ta = el('textarea', {});
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
        }
        toast(t('chat.copied'), 'success', 1200);
    } catch (_) {
        toast(t('error.generic'), 'error');
    }
}

// ── Composer / scroll plumbing ─────────────────────────────────

function autoSizeTextarea(ta) {
    if (!ta) return;
    ta.style.height = 'auto';
    const max = parseFloat(getComputedStyle(ta).lineHeight) * 6 || 144;
    ta.style.height = Math.min(ta.scrollHeight, max) + 'px';
}

function scrollToBottom(force = false) {
    if (!_ui.listEl) return;
    if (!force && !_autoScroll) return;
    _ui.listEl.scrollTop = _ui.listEl.scrollHeight;
}

function bindVisualViewport() {
    if (typeof window === 'undefined' || !window.visualViewport) return;
    const apply = () => {
        const vv = window.visualViewport;
        const offset = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
        document.documentElement.style.setProperty('--vv-bottom', `${Math.round(offset)}px`);
        if (_autoScroll) scrollToBottom(false);
    };
    window.visualViewport.addEventListener('resize', apply);
    window.visualViewport.addEventListener('scroll', apply);
    apply();
}

// ── Glyphs (inline SVG; no remote icon font) ───────────────────

function glyph(name) {
    const svgNS = 'http://www.w3.org/2000/svg';
    const paths = {
        menu: 'M3 6h18M3 12h18M3 18h18',
        x: 'M6 6l12 12M6 18L18 6',
        gear: 'M12 8a4 4 0 100 8 4 4 0 000-8z M19 12c0-.6-.06-1.2-.18-1.74l1.92-1.5-2-3.46-2.32.94c-.86-.7-1.86-1.22-2.94-1.5L13 2h-4l-.48 2.74c-1.08.28-2.08.8-2.94 1.5l-2.32-.94-2 3.46 1.92 1.5C3.06 10.8 3 11.4 3 12s.06 1.2.18 1.74l-1.92 1.5 2 3.46 2.32-.94c.86.7 1.86 1.22 2.94 1.5L9 22h4l.48-2.74c1.08-.28 2.08-.8 2.94-1.5l2.32.94 2-3.46-1.92-1.5c.12-.54.18-1.14.18-1.74z',
        send: 'M2 12l20-9-9 20-2-9-9-2z',
        mic: 'M12 14a3 3 0 003-3V6a3 3 0 10-6 0v5a3 3 0 003 3z M5 11a7 7 0 0014 0M12 18v3',
        copy: 'M9 9h11v11H9z M5 5h11v3M5 5v11h3',
        speaker: 'M5 9v6h4l5 5V4L9 9H5z M16 8a5 5 0 010 8',
        refresh: 'M4 12a8 8 0 0114-5l2-2v6h-6l3-3a6 6 0 10-1 9',
        dots: 'M5 12a1 1 0 102 0 1 1 0 10-2 0z M11 12a1 1 0 102 0 1 1 0 10-2 0z M17 12a1 1 0 102 0 1 1 0 10-2 0z',
        paperclip: 'M21 11l-9 9a5 5 0 01-7-7l9-9a3.5 3.5 0 015 5l-9 9a2 2 0 11-3-3l8-8',
        // Wrench-and-screwdriver "tools" glyph for the MCP picker.
        tools: 'M14.7 6.3a4 4 0 005.4 5.4L21 13l-7 7-1.3-.9a4 4 0 00-5.4-5.4L6 13l7-7zM3 21l6-6',
    };
    const d = paths[name] || paths.dots;
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '20');
    svg.setAttribute('height', '20');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    const p = document.createElementNS(svgNS, 'path');
    p.setAttribute('d', d);
    svg.appendChild(p);
    return svg;
}

// Mark crypto-store import as used (download banner already imports it
// directly; we re-import here only because future encryption-on-send
// support will live in this module).
void cryptoStore;

// ── Build-stamp hard refresh (dblclick + long-press) ─────────────────

function wireHardRefresh(stamp) {
    let pressTimer = null;
    let fired = false;
    stamp.addEventListener('dblclick', (e) => { e.preventDefault(); hardRefresh(); });
    stamp.addEventListener('pointerdown', () => {
        fired = false;
        pressTimer = setTimeout(() => { fired = true; hardRefresh(); }, 800);
    });
    stamp.addEventListener('pointerup', () => clearTimeout(pressTimer));
    stamp.addEventListener('pointercancel', () => clearTimeout(pressTimer));
    stamp.addEventListener('pointermove', () => clearTimeout(pressTimer));
    stamp.addEventListener('contextmenu', (e) => e.preventDefault());
    stamp.addEventListener('click', (e) => { if (fired) { e.preventDefault(); e.stopPropagation(); } });
}

async function hardRefresh() {
    if ('caches' in self) {
        const ks = await caches.keys();
        await Promise.all(ks.map(k => caches.delete(k)));
    }
    if ('serviceWorker' in navigator) {
        const rs = await navigator.serviceWorker.getRegistrations();
        await Promise.all(rs.map(r => r.unregister()));
    }
    location.reload();
}
