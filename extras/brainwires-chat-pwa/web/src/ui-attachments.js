// brainwires-chat-pwa — composer attachment chip strip
//
// Renders a horizontally-scrolling row of chips for pending attachments
// (image thumbnails + name pills). Subscribes to the attachments queue's
// 'change' event so the strip auto-refreshes on add/remove/clear.

import { el, clear } from './utils.js';
import { t } from './i18n.js';
import * as attachments from './attachments.js';

let _strip = null;

function buildChip(entry) {
    const chip = el('div', {
        class: `att-chip att-chip-${entry.kind}`,
        attrs: { 'data-att-id': entry.id, role: 'group', 'aria-label': entry.name },
    });
    if (entry.kind === 'image' && entry.dataUrl) {
        const img = el('img', {
            class: 'att-thumb',
            attrs: { src: entry.dataUrl, alt: entry.name, draggable: 'false' },
        });
        chip.appendChild(img);
    } else {
        chip.appendChild(el('span', { class: 'att-name' }, entry.name));
    }
    chip.appendChild(el('button', {
        class: 'att-remove',
        attrs: { type: 'button', 'aria-label': t('chat.remove') },
        onClick: () => attachments.remove(entry.id),
    }, '×'));
    return chip;
}

function render() {
    if (!_strip) return;
    clear(_strip);
    const items = attachments.getAll();
    if (items.length === 0) {
        _strip.classList.add('is-empty');
        _strip.setAttribute('hidden', '');
        return;
    }
    _strip.classList.remove('is-empty');
    _strip.removeAttribute('hidden');
    for (const e of items) _strip.appendChild(buildChip(e));
}

/**
 * Build (and bind) the chip strip element. Caller mounts it inside the
 * composer above the textarea row. Returns the root element.
 */
export function buildStrip() {
    _strip = el('div', { class: 'att-strip is-empty', attrs: { hidden: '' } });
    attachments.on('change', render);
    render();
    return _strip;
}
