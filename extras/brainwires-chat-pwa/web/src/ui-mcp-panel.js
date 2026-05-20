// brainwires-chat-pwa — Settings → MCP servers panel
//
// CRUD over the mcpServers IDB store plus a "Test" button that runs
// initialize + tools/list and surfaces the response inline. Per-conversation
// tool enable/disable lives in mcpToolState — the picker UI for that lands
// alongside provider tool-call serialization in a follow-up.

import { el, clear, toast, genId } from './utils.js';
import { listMcpServers, putMcpServer, deleteMcpServer } from './sql-db.js';
import * as mcp from './mcp-client.js';
import { t } from './i18n.js';

let _root = null;
let _list = null;

function buildAddForm(onAdded) {
    const urlInput = el('input', {
        type: 'url',
        class: 'bw-input',
        attrs: { placeholder: 'https://mcp.example.com/v1' },
    });
    const nameInput = el('input', {
        type: 'text',
        class: 'bw-input',
        attrs: { placeholder: 'Display name' },
    });
    const tokenInput = el('input', {
        type: 'password',
        class: 'bw-input',
        attrs: { placeholder: 'Bearer token (optional)', autocomplete: 'off' },
    });
    const err = el('div', { class: 'settings-err' });

    const addBtn = el('button', {
        class: 'bw-btn bw-btn-primary bw-btn-sm',
        attrs: { type: 'button' },
        onClick: async () => {
            err.textContent = '';
            const url = urlInput.value.trim();
            const displayName = nameInput.value.trim() || url;
            const headers = {};
            if (tokenInput.value.trim()) headers['authorization'] = `Bearer ${tokenInput.value.trim()}`;
            if (!url) { err.textContent = 'URL required'; return; }
            const row = { id: genId('mcp'), url, displayName, headers, enabledByDefault: true };
            try {
                await putMcpServer(row);
                urlInput.value = ''; nameInput.value = ''; tokenInput.value = '';
                toast(t('settings.mcp.added'), 'success');
                if (onAdded) await onAdded();
            } catch (e) {
                err.textContent = e && e.message ? e.message : String(e);
            }
        },
    }, t('settings.mcp.add'));

    return el('div', { class: 'settings-form' },
        el('label', { class: 'bw-label' }, 'URL', urlInput),
        el('label', { class: 'bw-label' }, t('settings.mcp.name'), nameInput),
        el('label', { class: 'bw-label' }, t('settings.mcp.token'), tokenInput),
        addBtn,
        err,
    );
}

async function buildServerCard(server) {
    const status = el('div', { class: 'settings-status', attrs: { 'aria-live': 'polite' } });
    const toolsList = el('div', { class: 'settings-help' });

    const testBtn = el('button', {
        class: 'bw-btn bw-btn-secondary bw-btn-sm',
        attrs: { type: 'button' },
        onClick: async () => {
            status.textContent = t('settings.testing');
            status.className = 'settings-status';
            try {
                await mcp.initialize(server);
                const tools = await mcp.listTools(server);
                status.textContent = t('settings.mcp.testOk', { count: tools.length });
                status.className = 'settings-status settings-status-ok';
                clear(toolsList);
                if (tools.length > 0) {
                    toolsList.appendChild(el('strong', {}, t('settings.mcp.tools') + ':'));
                    const ul = el('ul', { class: 'settings-help' });
                    for (const tl of tools) {
                        ul.appendChild(el('li', {}, `${tl.name}${tl.description ? ' — ' + tl.description : ''}`));
                    }
                    toolsList.appendChild(ul);
                }
            } catch (e) {
                status.textContent = e && e.message ? e.message : String(e);
                status.className = 'settings-status settings-status-err';
            }
        },
    }, t('settings.test'));

    const delBtn = el('button', {
        class: 'bw-btn bw-btn-danger bw-btn-sm',
        attrs: { type: 'button' },
        onClick: async () => {
            if (!confirm(t('settings.mcp.confirmDelete'))) return;
            try {
                await deleteMcpServer(server.id);
                mcp.dropSession(server.url);
                toast(t('settings.mcp.deleted'), 'success');
                await refreshList();
            } catch (e) {
                toast(e && e.message ? e.message : String(e), 'error');
            }
        },
    }, t('settings.mcp.delete'));

    return el('div', { class: 'settings-card' },
        el('div', { class: 'settings-card-header' },
            el('h3', { class: 'settings-card-title' }, server.displayName || server.url),
            el('span', { class: 'pill pill-muted' }, new URL(server.url).hostname),
        ),
        el('p', { class: 'settings-help' }, server.url),
        el('div', { class: 'settings-actions' }, testBtn, delBtn),
        status,
        toolsList,
    );
}

async function refreshList() {
    if (!_list) return;
    clear(_list);
    const servers = await listMcpServers();
    if (servers.length === 0) {
        _list.appendChild(el('p', { class: 'settings-help' }, t('settings.mcp.empty')));
        return;
    }
    for (const s of servers) _list.appendChild(await buildServerCard(s));
}

export async function render() {
    const body = el('div', { class: 'settings-card-list' });
    _root = body;
    body.appendChild(el('div', { class: 'settings-card' }, buildAddForm(refreshList)));
    _list = el('div', { class: 'settings-card-list' });
    body.appendChild(_list);
    await refreshList();
    return body;
}
