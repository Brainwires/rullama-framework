// brainwires-chat-pwa — Settings → Private RAG panel
//
// Lists ingested documents (global library only — per-conversation panels
// can be added later from the chat header). Provides:
//   - a file picker that ingests PDF / TXT into the global library
//   - per-row delete with index rebuild
//   - a small progress strip during ingestion

import { el, clear, toast } from './utils.js';
import { listRagDocs } from './sql-db.js';
import { ingest, deleteRagDoc } from './rag.js';
import { t } from './i18n.js';

let _root = null;
let _list = null;
let _progress = null;

function fmtBytes(n) {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
}

async function refreshList() {
    if (!_list) return;
    clear(_list);
    const docs = await listRagDocs(null);
    if (docs.length === 0) {
        _list.appendChild(el('p', { class: 'settings-help' }, t('settings.rag.empty')));
        return;
    }
    for (const d of docs) {
        const row = el('div', { class: 'settings-card-header' },
            el('span', { class: 'settings-card-title' }, d.name),
            el('span', { class: 'pill pill-muted' }, `${d.type} · ${fmtBytes(d.bytes || 0)}`),
            el('button', {
                class: 'bw-btn bw-btn-danger bw-btn-sm',
                attrs: { type: 'button' },
                onClick: async () => {
                    if (!confirm(t('settings.rag.confirmDelete'))) return;
                    try {
                        await deleteRagDoc(d.id, null);
                        toast(t('settings.rag.deleted'), 'success');
                        await refreshList();
                    } catch (e) {
                        toast(e && e.message ? e.message : String(e), 'error');
                    }
                },
            }, t('settings.rag.delete')),
        );
        _list.appendChild(row);
    }
}

function setProgress(phase, current, total) {
    if (!_progress) return;
    if (!phase || phase === 'done') {
        _progress.textContent = '';
        return;
    }
    const pct = total ? ` ${current}/${total}` : '';
    _progress.textContent = `${t(`settings.rag.phase.${phase}`) || phase}${pct}`;
}

async function handleIngest(file) {
    if (!file) return;
    try {
        await ingest(file, {
            conversationId: null,
            onProgress: ({ phase, current, total }) => setProgress(phase, current, total),
        });
        setProgress('done');
        toast(t('settings.rag.ingested'), 'success');
        await refreshList();
    } catch (e) {
        setProgress('done');
        toast(e && e.message ? e.message : String(e), 'error');
    }
}

export async function render() {
    const body = el('div', { class: 'settings-card' });
    _root = body;

    const fileInput = el('input', {
        type: 'file',
        attrs: { accept: '.pdf,application/pdf,.txt,text/plain', style: 'display:none' },
        onChange: async (e) => {
            const f = (e.currentTarget.files || [])[0];
            e.currentTarget.value = '';
            if (f) await handleIngest(f);
        },
    });
    body.appendChild(fileInput);
    body.appendChild(el('button', {
        class: 'bw-btn bw-btn-primary',
        attrs: { type: 'button' },
        onClick: () => fileInput.click(),
    }, t('settings.rag.ingest')));

    _progress = el('div', { class: 'settings-status', attrs: { 'aria-live': 'polite' } });
    body.appendChild(_progress);

    _list = el('div', { class: 'settings-card-list' });
    body.appendChild(_list);

    await refreshList();
    return body;
}
