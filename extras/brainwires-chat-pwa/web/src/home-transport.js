// brainwires-chat-pwa — WebRTC transport to the home daemon.
//
// Owns one RTCPeerConnection plus the "a2a" data channel. Drives the
// browser side of the SDP offer/answer + ICE-trickle handshake against a
// SignalingClient, then exposes a tiny request/response API on top of the
// data channel using JSON-RPC ids.
//
// State machine:
//   'idle'         — constructed, no connect() yet
//   'connecting'   — connect() in flight
//   'connected'    — data channel open
//   'reconnecting' — ICE restart in flight (M10/M12; recovers to 'connected'
//                    or escalates to 'failed' on a second restart loss)
//   'closing'      — close() in flight
//   'closed'       — close() done
//   'failed'       — ICE failed / dc closed unexpectedly
//
// Wire format on the channel: a single JSON-RPC envelope per text frame.
// Length-prefixed binary framing is reserved for M11.
//
// M10 — reconnect/resume:
//   - 15 s app-level `system/ping` heartbeat. Two consecutive timeouts
//     surface a dead link and trigger ICE restart.
//   - On `iceConnectionState === 'disconnected'` for >5 s the transport
//     calls `pc.restartIce()`, fresh-offers against the same session_id,
//     and re-trickles. Two failed restarts → tear the session down and
//     re-handshake (which counts as a "new session" event).
//   - Tracks last-seen JSON-RPC reply id; after a successful restart the
//     transport calls `system/resume { last_seen_id }` and re-feeds any
//     replayed frames into the dispatcher. `dropped: true` from the
//     daemon is surfaced via the `onSessionReset` event hook so the
//     chat UI can warn the user.

// SignalingClient is the expected shape of opts.signaling — see ./home-signaling.js.
// We don't import it here to avoid a circular-looking edge for tooling; the
// JSDoc reference is informational.

/**
 * JsonRpcDispatcher — the pure piece of HomeTransport's request bookkeeping.
 *
 * Allocates monotonic ids, parks a Promise for each outstanding request, and
 * dispatches inbound replies back to the right caller. Broken out so the
 * id-allocation + reply-routing logic can be unit-tested without an
 * RTCPeerConnection in scope (Node `--test` has no RTC).
 */
export class JsonRpcDispatcher {
    constructor() {
        this._nextId = 1;
        this._pending = new Map(); // id -> { resolve, reject, timer }
        this._notificationHandlers = new Map(); // method -> callback
        // M10 — high-water mark of inbound numeric reply ids. Used as the
        // `last_seen_id` cursor for `system/resume` after an ICE restart.
        // Notifications and string-id replies don't update this.
        this._lastSeenReplyId = 0;
    }

    onNotification(method, handler) {
        this._notificationHandlers.set(method, handler);
    }

    /** Last numeric JSON-RPC reply id observed (M10 resume cursor). */
    get lastSeenReplyId() { return this._lastSeenReplyId; }

    /** Allocate a fresh id and park a Promise; returns { id, frame, promise }. */
    request(method, params, { timeoutMs = 30000 } = {}) {
        const id = this._nextId++;
        const frame = JSON.stringify({ jsonrpc: '2.0', id, method, params });
        const promise = new Promise((resolve, reject) => {
            const timer = (timeoutMs > 0 && typeof setTimeout === 'function')
                ? setTimeout(() => {
                    if (this._pending.delete(id)) {
                        reject(new Error(`jsonrpc: request '${method}' (id=${id}) timed out after ${timeoutMs}ms`));
                    }
                }, timeoutMs)
                : null;
            this._pending.set(id, { resolve, reject, timer });
        });
        return { id, frame, promise };
    }

    /**
     * Route an inbound text frame. Resolves/rejects the matching pending
     * promise. Unknown ids (server-push, stray notifications) are dropped.
     * Returns true if the frame matched a pending request, false otherwise.
     */
    dispatch(text) {
        let msg;
        try { msg = JSON.parse(text); }
        catch (_) { return false; }
        if (!msg || typeof msg !== 'object') return false;
        // Server-push notifications: method present, no id.
        if (msg.id == null && msg.method) {
            const handler = this._notificationHandlers.get(msg.method);
            if (handler) { try { handler(msg.params); } catch (_) { /* swallow */ } }
            return !!handler;
        }
        if (msg.id == null) return false;
        // M10 — bump the resume cursor for every observed numeric reply id,
        // even one that doesn't match a pending slot (replay frames after
        // an ICE restart count). String ids are out of band for the
        // outbox protocol — leave them alone.
        if (typeof msg.id === 'number' && msg.id > this._lastSeenReplyId) {
            this._lastSeenReplyId = msg.id;
        }
        const slot = this._pending.get(msg.id);
        if (!slot) return false;
        this._pending.delete(msg.id);
        if (slot.timer) clearTimeout(slot.timer);
        if (msg.error) {
            const e = new Error(msg.error.message || 'jsonrpc error');
            e.code = msg.error.code;
            e.data = msg.error.data;
            slot.reject(e);
        } else {
            slot.resolve(msg.result);
        }
        return true;
    }

    /** Reject everything still pending — used on close()/failure. */
    rejectAll(reason) {
        const err = reason instanceof Error ? reason : new Error(String(reason));
        for (const slot of this._pending.values()) {
            if (slot.timer) clearTimeout(slot.timer);
            slot.reject(err);
        }
        this._pending.clear();
    }

    get pendingCount() { return this._pending.size; }
}

/**
 * HomeTransport — owns one RTCPeerConnection + 'a2a' data channel.
 *
 * Browser-only at runtime. The Node test suite exercises JsonRpcDispatcher
 * directly; the full handshake is covered by the M3 daemon's webrtc.rs test
 * and the eventual chat-PWA e2e suite (M9+).
 */
export class HomeTransport {
    /**
     * @param {{
     *   signaling: import('./home-signaling.js').SignalingClient,
     *   iceServers?: RTCIceServer[],
     *   rtcPeerConnection?: typeof RTCPeerConnection,
     *   heartbeatIntervalMs?: number,
     *   heartbeatTimeoutMs?: number,
     *   iceDisconnectGraceMs?: number,
     *   restartTimeoutMs?: number,
     *   onSessionReset?: (info: {dropped: boolean, newSession: boolean}) => void,
     *   onStateChange?: (info: {prev: string, next: string}) => void,
     *   _clock?: { setInterval: typeof setInterval, clearInterval: typeof clearInterval, setTimeout: typeof setTimeout, clearTimeout: typeof clearTimeout, now: () => number },
     * }} opts
     */
    constructor(opts) {
        if (!opts || !opts.signaling) {
            throw new Error('HomeTransport: signaling is required');
        }
        this._signaling = opts.signaling;
        this._iceServers = Array.isArray(opts.iceServers) ? opts.iceServers : null;
        this._RTC = opts.rtcPeerConnection || (typeof RTCPeerConnection !== 'undefined' ? RTCPeerConnection : null);

        // M10 timing knobs — defaults match the plan; tests inject smaller
        // values plus a fake clock.
        this._heartbeatIntervalMs = typeof opts.heartbeatIntervalMs === 'number' ? opts.heartbeatIntervalMs : 15000;
        this._heartbeatTimeoutMs = typeof opts.heartbeatTimeoutMs === 'number' ? opts.heartbeatTimeoutMs : 5000;
        this._iceDisconnectGraceMs = typeof opts.iceDisconnectGraceMs === 'number' ? opts.iceDisconnectGraceMs : 5000;
        this._restartTimeoutMs = typeof opts.restartTimeoutMs === 'number' ? opts.restartTimeoutMs : 30000;
        this._onSessionReset = typeof opts.onSessionReset === 'function' ? opts.onSessionReset : null;
        // M12 — state-change observer. Fired (synchronously) on every
        // transition so the chat UI can keep a status pill in lockstep
        // with reconnect/resume without polling. Errors thrown by the
        // observer are swallowed — UI bugs must not break the transport.
        this._onStateChange = typeof opts.onStateChange === 'function' ? opts.onStateChange : null;

        // Clock injection. In production we use the global timer functions;
        // in tests we hand in a fake clock so the heartbeat / disconnect
        // timing is deterministic without sleeping.
        const c = opts._clock || {};
        this._setInterval = c.setInterval || ((typeof setInterval === 'function') ? setInterval.bind(globalThis) : null);
        this._clearInterval = c.clearInterval || ((typeof clearInterval === 'function') ? clearInterval.bind(globalThis) : null);
        this._setTimeout = c.setTimeout || ((typeof setTimeout === 'function') ? setTimeout.bind(globalThis) : null);
        this._clearTimeout = c.clearTimeout || ((typeof clearTimeout === 'function') ? clearTimeout.bind(globalThis) : null);
        this._now = c.now || (() => Date.now());

        this._sessionId = null;
        this._pc = null;
        this._dc = null;
        this._state = 'idle';
        this._dispatcher = new JsonRpcDispatcher();
        this._iceAbort = null;        // AbortController for the inbound ICE long-poll
        this._iceCursor = 0;
        this._inboundIcePump = null;  // Promise for the running ICE pump task
        this._connectResolve = null;
        this._connectReject = null;

        // M10 — heartbeat / ICE-restart bookkeeping.
        this._heartbeatTimer = null;
        this._missedPongs = 0;
        this._lastPongAt = 0;
        this._iceDisconnectTimer = null;
        this._restartInFlight = null;     // Promise for an in-flight ICE restart
        this._restartFailures = 0;
    }

    get sessionId() { return this._sessionId; }
    get state() { return this._state; }
    /** Wall-clock ms timestamp of the last successful pong (M10 diagnostic). */
    get lastPongAt() { return this._lastPongAt; }

    _setState(next) {
        const prev = this._state;
        if (prev === next) return;
        this._state = next;
        if (this._onStateChange) {
            try { this._onStateChange({ prev, next }); } catch (_) { /* swallow */ }
        }
    }

    /**
     * Begin the connection. Resolves when the data channel is open.
     */
    async connect() {
        if (this._state !== 'idle') {
            throw new Error(`HomeTransport.connect: already ${this._state}`);
        }
        if (!this._RTC) {
            throw new Error('HomeTransport: RTCPeerConnection unavailable in this environment');
        }
        this._setState('connecting');

        try {
            // 1. Create signaling session — server returns an ICE-server hint list.
            const session = await this._signaling.createSession();
            this._sessionId = session.session_id;
            const iceServers = this._iceServers || (Array.isArray(session.ice_servers) ? session.ice_servers : []);

            // 2. Build the peer connection + 'a2a' data channel BEFORE createOffer.
            //    (Otherwise the channel won't appear in the SDP and the home
            //    side never sees an `ondatachannel` event.)
            const pc = new this._RTC({ iceServers });
            this._pc = pc;
            const dc = pc.createDataChannel('a2a');
            this._dc = dc;

            // 3. Arm the data channel listeners early so we don't miss the
            //    'open' event in case it fires before we await it below.
            const dcOpen = new Promise((resolve, reject) => {
                dc.onopen = () => resolve();
                dc.onerror = (e) => reject(new Error(`a2a data channel error: ${e && e.message ? e.message : 'unknown'}`));
                dc.onclose = () => {
                    if (this._state === 'connected') {
                        this._setState('failed');
                        this._dispatcher.rejectAll(new Error('a2a data channel closed unexpectedly'));
                    }
                };
                dc.onmessage = (ev) => {
                    if (typeof ev.data === 'string') {
                        const matched = this._dispatcher.dispatch(ev.data);
                        if (!matched) console.debug('home-transport: dropped unmatched frame', ev.data);
                    } else {
                        // M5 only sends/receives text frames. Binary framing is M11.
                        console.debug('home-transport: ignoring non-text frame');
                    }
                };
            });

            // 4. Outbound ICE: forward each local candidate to the daemon.
            //    Fire-and-forget; an error here logs but doesn't fail the
            //    handshake (we may still succeed with already-trickled cands).
            pc.onicecandidate = (ev) => {
                if (!ev.candidate) return;
                const c = ev.candidate;
                this._signaling.postIce(
                    this._sessionId,
                    c.candidate,
                    c.sdpMid,
                    c.sdpMLineIndex,
                ).catch((e) => console.warn('home-transport: postIce failed:', e && e.message ? e.message : e));
            };

            pc.oniceconnectionstatechange = () => this._onIceConnectionStateChange();

            // 5. createOffer → setLocalDescription → POST /signal/offer.
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            await this._signaling.postOffer(this._sessionId, offer.sdp);

            // 6. Pull the answer (the home daemon stashes it before the POST
            //    returns 204, so this should fast-path).
            let answer = null;
            for (let attempt = 0; attempt < 4 && !answer; attempt++) {
                answer = await this._signaling.pollAnswer(this._sessionId);
            }
            if (!answer) throw new Error('HomeTransport.connect: no SDP answer received');
            await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });

            // 7. Inbound ICE pump: long-poll the daemon for its trickled
            //    candidates and addIceCandidate them.
            this._iceAbort = new AbortController();
            this._inboundIcePump = this._runIcePump(this._iceAbort.signal).catch((e) => {
                if (e && e.name === 'AbortError') return;
                console.warn('home-transport: ICE pump exited:', e && e.message ? e.message : e);
            });

            // 8. Wait for the data channel to open.
            await dcOpen;
            this._setState('connected');

            // 9. M10 — start the heartbeat now that the channel is up.
            this._startHeartbeat();
        } catch (err) {
            this._setState('failed');
            // Best-effort cleanup of the signaling session and PC.
            if (this._iceAbort) try { this._iceAbort.abort(); } catch (_) {}
            if (this._pc) try { this._pc.close(); } catch (_) {}
            if (this._sessionId) try { await this._signaling.closeSession(this._sessionId); } catch (_) {}
            throw err;
        }
    }

    async _runIcePump(signal) {
        // Long-poll loop. The daemon returns 204 on timeout; pollIce maps
        // that to {candidates:[], cursor:since}. We keep going until the
        // session is closed or we hit a hard error (404 from the server,
        // which we treat as "session gone").
        while (!signal.aborted && (this._state === 'connecting' || this._state === 'connected')) {
            let next;
            try {
                next = await this._signaling.pollIce(this._sessionId, this._iceCursor, signal);
            } catch (e) {
                if (e && e.name === 'AbortError') return;
                // 404 (session gone) → bail; transient network errors → bail too,
                // M10 will add reconnect.
                throw e;
            }
            for (const c of next.candidates) {
                if (!c || c.candidate == null) continue;
                try {
                    await this._pc.addIceCandidate({
                        candidate: c.candidate,
                        sdpMid: c.sdp_mid ?? undefined,
                        sdpMLineIndex: typeof c.sdp_m_line_index === 'number' ? c.sdp_m_line_index : undefined,
                    });
                } catch (e) {
                    console.warn('home-transport: addIceCandidate failed:', e && e.message ? e.message : e);
                }
            }
            this._iceCursor = next.cursor;
        }
    }

    /**
     * Send a JSON-RPC request. Resolves with the result; rejects on error
     * reply or timeout.
     * @param {string} method
     * @param {object} [params]
     * @param {{timeoutMs?: number}} [opts]
     */
    async request(method, params, { timeoutMs = 30000 } = {}) {
        if (this._state !== 'connected') {
            throw new Error(`HomeTransport.request: not connected (state=${this._state})`);
        }
        const { frame, promise } = this._dispatcher.request(method, params, { timeoutMs });
        try {
            this._dc.send(frame);
        } catch (e) {
            // Drop the pending slot so we don't leak; timeout would also
            // catch this, but failing fast is friendlier.
            this._dispatcher.rejectAll(new Error(`jsonrpc: send failed: ${e && e.message ? e.message : e}`));
            throw e;
        }
        return await promise;
    }

    // ── M10: heartbeat ─────────────────────────────────────────

    /**
     * Start the 15 s `system/ping` heartbeat. Each tick fires a ping with
     * a bounded timeout; two consecutive timeouts (or send errors) trigger
     * an ICE restart. Idempotent — safe to call multiple times.
     */
    _startHeartbeat() {
        this._stopHeartbeat();
        if (!this._setInterval) return;
        // Return the tick promise so a clock-aware test runner can await it.
        // Production timer handles ignore the return value, so this is a no-op
        // in the browser and a useful hook in unit tests.
        this._heartbeatTimer = this._setInterval(() => this._heartbeatTick(), this._heartbeatIntervalMs);
    }

    _stopHeartbeat() {
        if (this._heartbeatTimer && this._clearInterval) {
            try { this._clearInterval(this._heartbeatTimer); } catch (_) {}
        }
        this._heartbeatTimer = null;
    }

    async _heartbeatTick() {
        if (this._state !== 'connected') return;
        try {
            await this.request('system/ping', {}, { timeoutMs: this._heartbeatTimeoutMs });
            this._missedPongs = 0;
            this._lastPongAt = this._now();
        } catch (_) {
            this._missedPongs += 1;
            if (this._missedPongs >= 2) {
                console.warn('home-transport: heartbeat missed twice; triggering ICE restart');
                this._missedPongs = 0;
                // Fire-and-forget — the restart handles its own errors.
                this._triggerIceRestart('heartbeat-timeout').catch((e) => {
                    console.warn('home-transport: heartbeat-driven restart failed:', e && e.message ? e.message : e);
                });
            }
        }
    }

    // ── M10: ICE-disconnect detection + restart ────────────────

    _onIceConnectionStateChange() {
        const pc = this._pc;
        if (!pc) return;
        const s = pc.iceConnectionState;
        if (s === 'failed') {
            // Hard failure — try a restart immediately rather than waiting
            // out the disconnect grace window.
            this._cancelIceDisconnectGrace();
            if (!this._restartInFlight) {
                this._triggerIceRestart('ice-failed').catch((e) => {
                    console.warn('home-transport: ice-failed restart errored:', e && e.message ? e.message : e);
                });
            }
            return;
        }
        if (s === 'disconnected') {
            // Grace period — transient blips often recover within seconds
            // without a full ICE restart.
            this._scheduleIceDisconnectGrace();
            return;
        }
        if (s === 'connected' || s === 'completed') {
            this._cancelIceDisconnectGrace();
        }
    }

    _scheduleIceDisconnectGrace() {
        if (this._iceDisconnectTimer || !this._setTimeout) return;
        this._iceDisconnectTimer = this._setTimeout(() => {
            this._iceDisconnectTimer = null;
            const pc = this._pc;
            if (!pc) return;
            // Only restart if we're STILL disconnected. Browsers can emit
            // a disconnected→connected→disconnected flap on a wifi roam.
            if (pc.iceConnectionState === 'disconnected') {
                this._triggerIceRestart('ice-disconnected').catch((e) => {
                    console.warn('home-transport: disconnect-driven restart errored:', e && e.message ? e.message : e);
                });
            }
        }, this._iceDisconnectGraceMs);
    }

    _cancelIceDisconnectGrace() {
        if (this._iceDisconnectTimer && this._clearTimeout) {
            try { this._clearTimeout(this._iceDisconnectTimer); } catch (_) {}
        }
        this._iceDisconnectTimer = null;
    }

    /**
     * Drive an ICE restart against the same signaling session. On success,
     * sends `system/resume` and re-feeds replayed frames into the
     * dispatcher. On second consecutive failure, tears down the session
     * and reconnects fresh — surfaces a `newSession: true` reset event
     * so the chat UI can warn the user.
     *
     * Coalesces concurrent calls — a second invocation while one is in
     * flight returns the same Promise.
     */
    _triggerIceRestart(reason) {
        if (this._restartInFlight) return this._restartInFlight;
        this._restartInFlight = (async () => {
            try {
                await this._iceRestartOnce(reason);
                this._restartFailures = 0;
                await this._sendResumeRequest();
            } catch (err) {
                this._restartFailures += 1;
                if (this._restartFailures < 2) {
                    // Second attempt on the SAME session.
                    try {
                        await this._iceRestartOnce(`${reason}-retry`);
                        this._restartFailures = 0;
                        await this._sendResumeRequest();
                        return;
                    } catch (_err2) {
                        this._restartFailures += 1;
                        // Fall through to full re-handshake.
                    }
                }
                // Two restart failures → full session tear-down + fresh handshake.
                console.warn('home-transport: two ICE-restart failures; opening a new session');
                await this._teardownAndReconnect();
                this._restartFailures = 0;
                if (this._onSessionReset) {
                    try { this._onSessionReset({ dropped: true, newSession: true }); } catch (_) {}
                }
                throw err;
            }
        })();
        const p = this._restartInFlight;
        p.finally(() => { if (this._restartInFlight === p) this._restartInFlight = null; });
        return p;
    }

    async _iceRestartOnce(reason) {
        if (this._state !== 'connected' && this._state !== 'connecting' && this._state !== 'reconnecting') {
            throw new Error(`HomeTransport._iceRestartOnce: bad state ${this._state}`);
        }
        // M12 — surface the in-flight restart so the status pill goes
        // amber. We restore 'connected' once the new ICE pair is up.
        if (this._state === 'connected') this._setState('reconnecting');
        const pc = this._pc;
        if (!pc) throw new Error('HomeTransport._iceRestartOnce: no peer connection');
        // restartIce() bumps the ICE ufrag/pwd on the next negotiation.
        if (typeof pc.restartIce === 'function') {
            try { pc.restartIce(); } catch (_) { /* fallthrough — createOffer({iceRestart:true}) is a backup */ }
        }
        const offer = await pc.createOffer({ iceRestart: true });
        await pc.setLocalDescription(offer);
        await this._signaling.postOffer(this._sessionId, offer.sdp);

        // Pull the new answer. The home daemon synchronously renegotiates
        // and stashes it before /signal/offer returns, so the first poll
        // should fast-path; we still loop a few times for paranoia.
        let answer = null;
        for (let attempt = 0; attempt < 4 && !answer; attempt++) {
            answer = await this._signaling.pollAnswer(this._sessionId);
        }
        if (!answer) throw new Error('ICE restart: no SDP answer received');
        await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });

        // The existing inbound-ICE pump is still running against the same
        // session — fresh candidates trickle through it. We don't need to
        // restart the pump or reset the cursor (the daemon's outbound
        // buffer is append-only; a stale cursor is harmless).

        // Wait for the connection to actually recover.
        await this._waitIceConnected();
        // M12 — restart finished cleanly; flip back to 'connected' so the
        // status pill returns to green before system/resume.
        if (this._state === 'reconnecting') this._setState('connected');
        console.info(`home-transport: ICE restart succeeded (reason=${reason})`);
    }

    /**
     * Block until `pc.iceConnectionState` reaches `connected` or `completed`,
     * or the restart timeout elapses.
     */
    async _waitIceConnected() {
        const pc = this._pc;
        if (!pc) throw new Error('no peer connection');
        if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') return;
        const setT = this._setTimeout;
        const clearT = this._clearTimeout;
        return await new Promise((resolve, reject) => {
            let timer = null;
            const cleanup = () => {
                if (timer && clearT) { try { clearT(timer); } catch (_) {} }
                if (pc.removeEventListener) {
                    try { pc.removeEventListener('iceconnectionstatechange', onChange); } catch (_) {}
                }
            };
            const onChange = () => {
                const s = pc.iceConnectionState;
                if (s === 'connected' || s === 'completed') { cleanup(); resolve(); }
                else if (s === 'failed' || s === 'closed') { cleanup(); reject(new Error(`ICE entered ${s} during restart`)); }
            };
            if (pc.addEventListener) {
                pc.addEventListener('iceconnectionstatechange', onChange);
            }
            if (setT) {
                timer = setT(() => { cleanup(); reject(new Error('ICE restart timed out')); }, this._restartTimeoutMs);
            }
        });
    }

    async _sendResumeRequest() {
        let result;
        try {
            result = await this.request(
                'system/resume',
                { last_seen_id: this._dispatcher.lastSeenReplyId },
                { timeoutMs: 10000 },
            );
        } catch (e) {
            // Resume failure is non-fatal — the channel is back, just no
            // backfill. Log and move on.
            console.warn('home-transport: system/resume failed:', e && e.message ? e.message : e);
            return;
        }
        const replayed = Array.isArray(result && result.replayed) ? result.replayed : [];
        const dropped = !!(result && result.dropped);
        for (const frame of replayed) {
            if (typeof frame === 'string') {
                try { this._dispatcher.dispatch(frame); } catch (_) { /* ignore */ }
            }
        }
        if (dropped && this._onSessionReset) {
            try { this._onSessionReset({ dropped: true, newSession: false }); } catch (_) {}
        }
    }

    async _teardownAndReconnect() {
        // Stop heartbeat + ICE pump while we rebuild.
        this._stopHeartbeat();
        this._cancelIceDisconnectGrace();
        if (this._iceAbort) { try { this._iceAbort.abort(); } catch (_) {} this._iceAbort = null; }
        try { if (this._dc) this._dc.close(); } catch (_) {}
        try { if (this._pc) this._pc.close(); } catch (_) {}
        if (this._sessionId) {
            try { await this._signaling.closeSession(this._sessionId); } catch (_) {}
        }
        this._dc = null;
        this._pc = null;
        this._sessionId = null;
        this._iceCursor = 0;
        // Reset the dispatcher's resume cursor — the new session has a
        // fresh outbox, so the old cursor is meaningless.
        this._dispatcher = new JsonRpcDispatcher();
        // Rerun the full handshake.
        this._setState('idle');
        await this.connect();
    }

    // ── M11: binary chunking (uplink) ──────────────────────────

    /**
     * Upload a binary payload to the home daemon, auto-chunking at 256 KB
     * raw boundaries when needed. Returns a fresh `bin_id` (UUID) the
     * caller can reference in a subsequent `message/send` file part as
     * `metadata: { bin_id }`. The daemon resolves the reference and
     * inlines the bytes before the agent sees the message.
     *
     * Wire (each `bin/*` is one JSON-RPC call over the data channel):
     *   bin/begin → bin/chunk × N → bin/end (with sha256)
     *
     * Sequential by design — chunk N+1 only after chunk N's reply lands.
     * The dispatcher would serialize them anyway via id allocation, but
     * explicit awaits make the protocol invariants obvious in tracing.
     *
     * @param {Uint8Array} bytes raw payload
     * @param {string} contentType MIME type for the daemon to record
     * @param {{
     *   chunkSize?: number,           // raw bytes per chunk, default 256 KB
     *   onProgress?: (p: {sent: number, total: number}) => void,
     *   timeoutMs?: number,           // per-call JSON-RPC timeout, default 30 s
     * }} [opts]
     * @returns {Promise<string>} the bin_id
     */
    async uploadBinary(bytes, contentType, opts = {}) {
        if (this._state !== 'connected') {
            throw new Error(`HomeTransport.uploadBinary: not connected (state=${this._state})`);
        }
        if (!(bytes instanceof Uint8Array)) {
            throw new Error('HomeTransport.uploadBinary: bytes must be a Uint8Array');
        }
        const chunkSize = typeof opts.chunkSize === 'number' && opts.chunkSize > 0
            ? opts.chunkSize
            : (256 * 1024);
        const timeoutMs = typeof opts.timeoutMs === 'number' ? opts.timeoutMs : 30000;
        const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;

        const total = bytes.byteLength;
        const totalChunks = total === 0 ? 1 : Math.ceil(total / chunkSize);
        const binId = (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function')
            ? crypto.randomUUID()
            : `bin-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;

        // Compute SHA-256 once over the whole payload — the daemon
        // verifies on bin/end. Computed up-front because Web Crypto's
        // digest API is one-shot, not incremental.
        const sha = await sha256Hex(bytes);

        await this.request('bin/begin', {
            bin_id: binId,
            content_type: contentType,
            total_size: total,
            total_chunks: totalChunks,
        }, { timeoutMs });

        let sent = 0;
        for (let seq = 0; seq < totalChunks; seq++) {
            const start = seq * chunkSize;
            const end = Math.min(start + chunkSize, total);
            const slice = bytes.subarray(start, end);
            const b64 = uint8ToBase64(slice);
            await this.request('bin/chunk', {
                bin_id: binId,
                seq,
                data: b64,
            }, { timeoutMs });
            sent += slice.byteLength;
            if (onProgress) {
                try { onProgress({ sent, total }); } catch (_) { /* swallow */ }
            }
        }

        await this.request('bin/end', { bin_id: binId, sha256: sha }, { timeoutMs });
        return binId;
    }

    /** Disconnect cleanly. Idempotent. */
    async close() {
        if (this._state === 'closed' || this._state === 'closing') return;
        this._setState('closing');
        this._stopHeartbeat();
        this._cancelIceDisconnectGrace();
        if (this._iceAbort) { try { this._iceAbort.abort(); } catch (_) {} }
        this._dispatcher.rejectAll(new Error('HomeTransport closed'));
        try { if (this._dc) this._dc.close(); } catch (_) {}
        try { if (this._pc) this._pc.close(); } catch (_) {}
        if (this._sessionId) {
            try { await this._signaling.closeSession(this._sessionId); } catch (_) {}
        }
        this._setState('closed');
    }
}

// ── M11 helpers ────────────────────────────────────────────────

/**
 * Encode a Uint8Array to base64 in a way that works in browsers
 * (where btoa exists but only takes a binary string) and in node
 * (where Buffer is available). The chunked-string approach avoids
 * blowing the call-stack on multi-MB payloads — `String.fromCharCode`
 * applied to a >100 K-element array hits engine limits.
 *
 * Exported for the unit tests.
 * @param {Uint8Array} u8
 * @returns {string}
 */
export function uint8ToBase64(u8) {
    // eslint-disable-next-line no-undef
    if (typeof Buffer !== 'undefined' && Buffer.from) {
        // node test environment
        // eslint-disable-next-line no-undef
        return Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString('base64');
    }
    if (typeof btoa !== 'function') {
        throw new Error('uint8ToBase64: no btoa or Buffer available');
    }
    let bin = '';
    const CHUNK = 0x8000; // 32 KB sub-batches keep fromCharCode happy
    for (let i = 0; i < u8.length; i += CHUNK) {
        const slice = u8.subarray(i, i + CHUNK);
        bin += String.fromCharCode.apply(null, slice);
    }
    return btoa(bin);
}

/**
 * SHA-256 the input bytes and return a lowercase hex string. Uses
 * Web Crypto's `crypto.subtle.digest` (available in every PWA-target
 * browser and in node 18+).
 * @param {Uint8Array} bytes
 * @returns {Promise<string>}
 */
export async function sha256Hex(bytes) {
    if (!globalThis.crypto || !globalThis.crypto.subtle) {
        throw new Error('sha256Hex: Web Crypto unavailable');
    }
    const buf = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    const arr = new Uint8Array(buf);
    let hex = '';
    for (let i = 0; i < arr.length; i++) {
        const h = arr[i].toString(16);
        hex += h.length === 1 ? `0${h}` : h;
    }
    return hex;
}
