// brainwires-chat-pwa — Cross-device sync orchestrator
//
// Pushes local changelog entries to the home daemon via `sync/push`,
// pulls remote entries via `sync/pull`, applies them locally with
// trigger suppression. Listens for `sync/notify` server-push
// notifications to pull immediately when another device pushes.

import {
    initSyncState, getSyncState, setSyncState,
    getSyncLogSince, beginApplying, endApplying,
    getSnapshotForEntry, applyRemoteEntry,
} from './sql-db.js';

const PUSH_INTERVAL_MS = 5_000;
const PULL_INTERVAL_MS = 30_000;

export class SyncManager {
    constructor(transport) {
        this._transport = transport;
        this._deviceId = null;
        this._running = false;
        this._pushTimer = null;
        this._pullTimer = null;
        this._pushing = false;
        this._pulling = false;
        this._onUpdate = null;
    }

    set onUpdate(fn) { this._onUpdate = typeof fn === 'function' ? fn : null; }

    async start() {
        if (this._running) return;
        this._running = true;
        this._deviceId = await initSyncState();

        this._transport.dispatcher.onNotification('sync/notify', () => {
            this._doPull();
        });

        await this._doPush();
        await this._doPull();

        this._pushTimer = setInterval(() => this._doPush(), PUSH_INTERVAL_MS);
        this._pullTimer = setInterval(() => this._doPull(), PULL_INTERVAL_MS);
    }

    stop() {
        this._running = false;
        if (this._pushTimer) { clearInterval(this._pushTimer); this._pushTimer = null; }
        if (this._pullTimer) { clearInterval(this._pullTimer); this._pullTimer = null; }
    }

    async pushNow() { return this._doPush(); }

    async _doPush() {
        if (!this._running || this._pushing) return;
        this._pushing = true;
        try {
            const lastSeq = parseInt(await getSyncState('last_push_seq') || '0', 10);
            const entries = await getSyncLogSince(lastSeq);
            if (entries.length === 0) return;

            const enriched = [];
            const now = Date.now();
            for (const e of entries) {
                const snapshot = await getSnapshotForEntry(e);
                enriched.push({
                    table: e.tableName,
                    row_key: e.rowKey,
                    op: e.op,
                    ts: now,
                    snapshot,
                });
            }

            await this._transport.request('sync/push', {
                device_id: this._deviceId,
                entries: enriched,
            });

            const maxSeq = entries[entries.length - 1].seq;
            await setSyncState('last_push_seq', String(maxSeq));
        } catch (err) {
            console.warn('[sync] push failed:', err);
        } finally {
            this._pushing = false;
        }
    }

    async _doPull() {
        if (!this._running || this._pulling) return;
        this._pulling = true;
        try {
            const lastSeq = parseInt(await getSyncState('last_pull_seq') || '0', 10);

            const result = await this._transport.request('sync/pull', {
                device_id: this._deviceId,
                since: lastSeq,
            });

            if (!result || !Array.isArray(result.entries) || result.entries.length === 0) return;

            await beginApplying();
            try {
                for (const entry of result.entries) {
                    await applyRemoteEntry(entry);
                }
            } finally {
                await endApplying();
            }

            await setSyncState('last_pull_seq', String(result.latest_seq));

            await this._transport.request('sync/ack', {
                device_id: this._deviceId,
                seq: result.latest_seq,
            }).catch(() => {});

            if (this._onUpdate) {
                try { this._onUpdate(result.entries); } catch (_) {}
            }
        } catch (err) {
            console.warn('[sync] pull failed:', err);
        } finally {
            this._pulling = false;
        }
    }
}
