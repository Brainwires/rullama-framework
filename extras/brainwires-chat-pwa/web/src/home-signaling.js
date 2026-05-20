// brainwires-chat-pwa — home-daemon HTTP signaling client.
//
// Talks to the daemon's /signal/* endpoints. Handles long-poll cursor for
// ICE, surfaces a uniform error shape, and offers an AbortController hook
// so callers can cancel a pending long-poll on disconnect / page hide.
//
// Auth: M5 sends no headers beyond Content-Type. M8 (pairing) layers in
//   - Authorization: Bearer <device_token>
//   - CF-Access-Client-Id / CF-Access-Client-Secret
// via an `extraHeaders()` callback the caller plugs in once paired.
//
// Wire shape (all JSON unless noted):
//   POST /signal/session            -> 200 { session_id, ice_servers }
//   POST /signal/offer/{id}         -> 204 (synchronous: answer is in storage)
//   GET  /signal/answer/{id}        -> 200 { sdp, type } | 204 (long-poll miss)
//   POST /signal/ice/{id}           -> 204
//   GET  /signal/ice/{id}?since=N   -> 200 { candidates, cursor }
//   DELETE /signal/{id}             -> 204
//   GET  /.well-known/agent-card.json -> 200 AgentCard

/**
 * SignalingClient — thin wrapper over fetch() for the home-daemon's
 * signaling endpoints. All methods throw on non-2xx with a uniform message.
 */
export class SignalingClient {
    /**
     * @param {{
     *   baseUrl: string,
     *   fetchImpl?: typeof fetch,
     *   extraHeaders?: () => Record<string,string>,
     * }} opts
     */
    constructor({ baseUrl, fetchImpl, extraHeaders } = {}) {
        if (!baseUrl || typeof baseUrl !== 'string') {
            throw new Error('SignalingClient: baseUrl is required');
        }
        this._baseUrl = baseUrl.replace(/\/$/, '');
        // Capture fetch lazily so tests that swap globalThis.fetch on a per-test
        // basis still work without re-constructing the client.
        this._fetchImpl = fetchImpl || null;
        this._extraHeaders = typeof extraHeaders === 'function' ? extraHeaders : () => ({});
    }

    _fetch(...args) {
        const f = this._fetchImpl || globalThis.fetch;
        if (typeof f !== 'function') {
            throw new Error('SignalingClient: no fetch implementation available');
        }
        return f(...args);
    }

    _headers(extra) {
        return { ...(this._extraHeaders() || {}), ...(extra || {}) };
    }

    async _failed(op, resp) {
        // Don't spend a network turn re-reading the body — just surface status.
        return new Error(`signaling: ${op} failed: ${resp.status}`);
    }

    /** POST /signal/session → { session_id, ice_servers } */
    async createSession() {
        let resp;
        try {
            resp = await this._fetch(`${this._baseUrl}/signal/session`, {
                method: 'POST',
                headers: this._headers({ 'Content-Type': 'application/json' }),
                body: '{}',
            });
        } catch (e) {
            throw new Error(`signaling: createSession network error: ${e && e.message ? e.message : e}`);
        }
        if (!resp.ok) throw await this._failed('createSession', resp);
        return await resp.json();
    }

    /** POST /signal/offer/{id} body { sdp, type:"offer" }. Returns void on 204. */
    async postOffer(sessionId, sdp) {
        let resp;
        const body = JSON.stringify({ sdp, type: 'offer' });
        try {
            resp = await this._fetch(`${this._baseUrl}/signal/offer/${encodeURIComponent(sessionId)}`, {
                method: 'POST',
                headers: this._headers({ 'Content-Type': 'application/json' }),
                body,
            });
        } catch (e) {
            throw new Error(`signaling: postOffer network error: ${e && e.message ? e.message : e}`);
        }
        if (resp.status === 404) throw new Error('signaling: postOffer failed: 404 (session expired)');
        if (!resp.ok && resp.status !== 204) throw await this._failed('postOffer', resp);
    }

    /**
     * GET /signal/answer/{id} (long-poll up to ~25s server-side).
     * Returns null on 204 (no answer yet) — caller should re-poll.
     * @param {string} sessionId
     * @param {AbortSignal} [signal]
     */
    async pollAnswer(sessionId, signal) {
        let resp;
        try {
            resp = await this._fetch(`${this._baseUrl}/signal/answer/${encodeURIComponent(sessionId)}`, {
                method: 'GET',
                headers: this._headers(),
                signal,
            });
        } catch (e) {
            if (e && e.name === 'AbortError') throw e;
            throw new Error(`signaling: pollAnswer network error: ${e && e.message ? e.message : e}`);
        }
        if (resp.status === 204) return null;
        if (resp.status === 404) throw new Error('signaling: pollAnswer failed: 404 (session expired)');
        if (!resp.ok) throw await this._failed('pollAnswer', resp);
        return await resp.json();
    }

    /**
     * POST /signal/ice/{id} body { candidate, sdp_mid, sdp_m_line_index }.
     * Daemon ignores end-of-candidates (null candidate) — but we still send
     * non-null candidates faithfully. Returns void on 204.
     */
    async postIce(sessionId, candidate, sdpMid, sdpMLineIndex) {
        const body = JSON.stringify({
            candidate: candidate ?? null,
            sdp_mid: sdpMid ?? null,
            sdp_m_line_index: typeof sdpMLineIndex === 'number' ? sdpMLineIndex : null,
        });
        let resp;
        try {
            resp = await this._fetch(`${this._baseUrl}/signal/ice/${encodeURIComponent(sessionId)}`, {
                method: 'POST',
                headers: this._headers({ 'Content-Type': 'application/json' }),
                body,
            });
        } catch (e) {
            throw new Error(`signaling: postIce network error: ${e && e.message ? e.message : e}`);
        }
        if (resp.status === 404) throw new Error('signaling: postIce failed: 404 (session expired)');
        if (!resp.ok && resp.status !== 204) throw await this._failed('postIce', resp);
    }

    /**
     * GET /signal/ice/{id}?since=N. Long-polls until a new candidate is
     * available or the server times out. On timeout returns the same cursor
     * with an empty candidates list — caller is responsible for re-polling.
     * @param {string} sessionId
     * @param {number} since
     * @param {AbortSignal} [signal]
     * @returns {Promise<{candidates: Array, cursor: number}>}
     */
    async pollIce(sessionId, since, signal) {
        const url = `${this._baseUrl}/signal/ice/${encodeURIComponent(sessionId)}?since=${since | 0}`;
        let resp;
        try {
            resp = await this._fetch(url, {
                method: 'GET',
                headers: this._headers(),
                signal,
            });
        } catch (e) {
            if (e && e.name === 'AbortError') throw e;
            throw new Error(`signaling: pollIce network error: ${e && e.message ? e.message : e}`);
        }
        if (resp.status === 204) return { candidates: [], cursor: since };
        if (resp.status === 404) throw new Error('signaling: pollIce failed: 404 (session expired)');
        if (!resp.ok) throw await this._failed('pollIce', resp);
        const out = await resp.json();
        return {
            candidates: Array.isArray(out.candidates) ? out.candidates : [],
            cursor: typeof out.cursor === 'number' ? out.cursor : since,
        };
    }

    /** DELETE /signal/{id}. Best-effort — swallows network errors. */
    async closeSession(sessionId) {
        try {
            await this._fetch(`${this._baseUrl}/signal/${encodeURIComponent(sessionId)}`, {
                method: 'DELETE',
                headers: this._headers(),
            });
        } catch (_) {
            // Best-effort cleanup; the server reaps abandoned sessions on TTL.
        }
    }

    /** GET /.well-known/agent-card.json — returns the parsed AgentCard. */
    async fetchAgentCard() {
        let resp;
        try {
            resp = await this._fetch(`${this._baseUrl}/.well-known/agent-card.json`, {
                method: 'GET',
                headers: this._headers(),
            });
        } catch (e) {
            throw new Error(`signaling: fetchAgentCard network error: ${e && e.message ? e.message : e}`);
        }
        if (!resp.ok) throw await this._failed('fetchAgentCard', resp);
        return await resp.json();
    }
}
