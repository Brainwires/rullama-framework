// brainwires-chat-pwa — per-conversation MCP tool picker
//
// Composer-attached drawer that lists every configured MCP server and
// the tools it advertises (via `tools/list`). Per-tool toggles persist
// to the `mcpToolState` IDB store keyed by the active conversation id.
//
// Tool definitions are cached module-locally (one promise per server)
// so opening the picker mid-session doesn't trigger an `initialize` +
// `tools/list` round-trip every time. Settings → MCP servers' "Test"
// button is the canonical way to refresh after a server changes.

import { el, clear, toast } from './utils.js';
import {
    listMcpServers,
    setMcpToolEnabled,
    listMcpToolStateForConversation,
} from './sql-db.js';
import * as mcp from './mcp-client.js';
import { t } from './i18n.js';

// serverId → Promise<Tool[]>
// Tool shape (from mcp-client.js): { name, description?, inputSchema? }
const _toolCache = new Map();

/**
 * Resolve (and cache) the tools advertised by a server. The first call
 * runs `initialize` + `tools/list`; subsequent calls return the same
 * promise. Failures are cached so we don't retry on every keystroke;
 * users can re-test from Settings → MCP servers to invalidate.
 *
 * @param {{ id: string, url: string, headers?: object }} server
 * @returns {Promise<Array<{ name: string, description?: string, inputSchema?: object }>>}
 */
export function getCachedTools(server) {
    if (!server || !server.id) return Promise.resolve([]);
    if (_toolCache.has(server.id)) return _toolCache.get(server.id);
    const p = (async () => {
        try {
            await mcp.initialize(server);
            const tools = await mcp.listTools(server);
            return Array.isArray(tools) ? tools : [];
        } catch (e) {
            // Surface once; cache the empty result so we don't spam the
            // network. The Settings panel "Test" button is the recovery
            // path — it doesn't go through this cache.
            console.warn('[bw] mcp listTools failed for', server.id, e && e.message ? e.message : e);
            return [];
        }
    })();
    _toolCache.set(server.id, p);
    return p;
}

/** Drop the cached tools for one server (or all). */
export function invalidateToolCache(serverId) {
    if (serverId === undefined) _toolCache.clear();
    else _toolCache.delete(serverId);
}

let _activeSheet = null;

function closeSheet() {
    if (!_activeSheet) return;
    try { _activeSheet.scrim.remove(); } catch (_) {}
    try { _activeSheet.sheet.remove(); } catch (_) {}
    document.removeEventListener('keydown', _activeSheet.onKey, true);
    _activeSheet = null;
}

async function buildServerCard(server, conversationId, stateByKey) {
    const tools = await getCachedTools(server);
    const card = el('div', { class: 'settings-card mcp-picker-card' });
    const header = el('div', { class: 'settings-card-header' },
        el('h3', { class: 'settings-card-title' }, server.displayName || server.url),
    );
    card.appendChild(header);

    if (tools.length === 0) {
        card.appendChild(el('p', { class: 'settings-help' }, t('mcp.picker.empty')));
        return card;
    }

    const checkboxes = [];

    const allOnBtn = el('button', {
        class: 'bw-btn bw-btn-secondary bw-btn-sm',
        attrs: { type: 'button' },
        onClick: async () => {
            for (const cb of checkboxes) {
                if (!cb.checked) {
                    cb.checked = true;
                    await setMcpToolEnabled(conversationId, server.id, cb.dataset.toolName, true);
                }
            }
        },
    }, t('mcp.picker.allOn'));
    const allOffBtn = el('button', {
        class: 'bw-btn bw-btn-secondary bw-btn-sm',
        attrs: { type: 'button' },
        onClick: async () => {
            for (const cb of checkboxes) {
                if (cb.checked) {
                    cb.checked = false;
                    await setMcpToolEnabled(conversationId, server.id, cb.dataset.toolName, false);
                }
            }
        },
    }, t('mcp.picker.allOff'));
    card.appendChild(el('div', { class: 'settings-actions' }, allOnBtn, allOffBtn));

    const list = el('ul', { class: 'mcp-picker-tools' });
    for (const tool of tools) {
        const stateKey = `${server.id}::${tool.name}`;
        const enabled = !!(stateByKey.get(stateKey));
        const cb = el('input', {
            type: 'checkbox',
            class: 'mcp-picker-checkbox',
            attrs: { 'data-tool-name': tool.name },
        });
        cb.checked = enabled;
        cb.dataset.toolName = tool.name;
        cb.addEventListener('change', async () => {
            try {
                await setMcpToolEnabled(conversationId, server.id, tool.name, cb.checked);
            } catch (e) {
                toast(e && e.message ? e.message : String(e), 'error');
            }
        });
        checkboxes.push(cb);
        const label = el('label', { class: 'mcp-picker-tool' },
            cb,
            el('div', { class: 'mcp-picker-tool-meta' },
                el('strong', { class: 'mcp-picker-tool-name' }, tool.name),
                tool.description
                    ? el('div', { class: 'mcp-picker-tool-desc' }, tool.description)
                    : null,
            ),
        );
        list.appendChild(el('li', {}, label));
    }
    card.appendChild(list);
    return card;
}

/**
 * Open the picker for the given conversation. Anchored as a drawer-style
 * bottom sheet; ESC / outside-click closes.
 *
 * @param {string} conversationId
 */
export async function openPicker(conversationId) {
    if (_activeSheet) closeSheet();
    if (!conversationId) {
        toast(t('mcp.picker.noConversation'), 'error');
        return;
    }

    const scrim = el('div', { class: 'mcp-picker-scrim', attrs: { 'aria-hidden': 'true' } });
    scrim.addEventListener('click', closeSheet);

    const body = el('div', { class: 'mcp-picker-body' });
    const sheet = el('div', {
        class: 'mcp-picker-sheet',
        attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-label': t('mcp.picker.title') },
    },
        el('div', { class: 'mcp-picker-header' },
            el('strong', { class: 'mcp-picker-title' }, t('mcp.picker.title')),
            el('button', {
                class: 'bw-btn bw-btn-secondary bw-btn-sm',
                attrs: { type: 'button', 'aria-label': t('nav.close') },
                onClick: closeSheet,
            }, t('nav.close')),
        ),
        body,
    );

    const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); closeSheet(); } };
    document.addEventListener('keydown', onKey, true);

    document.body.appendChild(scrim);
    document.body.appendChild(sheet);
    _activeSheet = { sheet, scrim, onKey };

    body.appendChild(el('p', { class: 'settings-help mcp-picker-loading' }, t('settings.testing')));

    const [servers, states] = await Promise.all([
        listMcpServers(),
        listMcpToolStateForConversation(conversationId),
    ]);
    const stateByKey = new Map();
    for (const s of states) {
        stateByKey.set(`${s.serverId}::${s.toolName}`, !!s.enabled);
    }

    clear(body);
    if (!servers || servers.length === 0) {
        body.appendChild(el('p', { class: 'settings-help' }, t('mcp.picker.noServers')));
        return;
    }
    for (const server of servers) {
        try {
            const card = await buildServerCard(server, conversationId, stateByKey);
            body.appendChild(card);
        } catch (e) {
            body.appendChild(el('div', { class: 'settings-card' },
                el('p', { class: 'settings-status settings-status-err' },
                    e && e.message ? e.message : String(e)),
            ));
        }
    }
}

/**
 * Resolve the per-conversation enabled tool state into the
 * `params.tools` array that providers expect:
 *   `[{name, description, input_schema, _serverId}]`
 *
 * `_serverId` is attached for convenience so callers (the tool execution
 * loop) can route invocations back to the originating server without
 * re-scanning. Pre-existing serializers (anthropic / openai) only read
 * `name`, `description`, `input_schema` — extra fields are ignored.
 *
 * @param {string} conversationId
 * @returns {Promise<Array<{name: string, description: string, input_schema: object, _serverId: string}>>}
 */
export async function resolveEnabledTools(conversationId) {
    if (!conversationId) return [];
    const [servers, states] = await Promise.all([
        listMcpServers(),
        listMcpToolStateForConversation(conversationId),
    ]);
    const enabled = states.filter((s) => s.enabled);
    if (enabled.length === 0) return [];
    const serverById = new Map(servers.map((s) => [s.id, s]));
    const out = [];
    for (const row of enabled) {
        const server = serverById.get(row.serverId);
        if (!server) continue;
        const tools = await getCachedTools(server);
        const def = tools.find((t0) => t0.name === row.toolName);
        if (!def) continue;
        out.push({
            name: def.name,
            description: typeof def.description === 'string' ? def.description : '',
            input_schema: def.inputSchema || { type: 'object', properties: {} },
            _serverId: server.id,
        });
    }
    return out;
}

/**
 * Look up the server that hosts a given tool name, restricted to tools
 * currently enabled for the conversation. Returns `null` if no server
 * advertises this tool.
 *
 * @param {string} conversationId
 * @param {string} toolName
 * @returns {Promise<{ id: string, url: string, headers?: object } | null>}
 */
export async function findServerForTool(conversationId, toolName) {
    if (!conversationId || !toolName) return null;
    const [servers, states] = await Promise.all([
        listMcpServers(),
        listMcpToolStateForConversation(conversationId),
    ]);
    const serverById = new Map(servers.map((s) => [s.id, s]));
    for (const row of states) {
        if (!row.enabled || row.toolName !== toolName) continue;
        const server = serverById.get(row.serverId);
        if (server) return server;
    }
    return null;
}
