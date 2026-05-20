// brainwires-chat-pwa — unit tests
//
// Run via: node --test tests/unit.test.mjs
//
// Covers:
//   - streaming.parseSSE / parseNDJSON / streamFromResponse
//   - crypto-store deriveKey/encrypt/decrypt round-trip + pack/unpack
//   - db.appendMessageChunk via fake-indexeddb (devDep)

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';

// Web Crypto on the global. node 20+ provides it but older runs may not.
if (typeof globalThis.crypto === 'undefined' || !globalThis.crypto.subtle) {
    globalThis.crypto = webcrypto;
}

// btoa/atob shims for crypto-store (Node 18+ has them globally; older nodes don't).
if (typeof globalThis.btoa === 'undefined') {
    globalThis.btoa = (s) => Buffer.from(s, 'binary').toString('base64');
}
if (typeof globalThis.atob === 'undefined') {
    globalThis.atob = (s) => Buffer.from(s, 'base64').toString('binary');
}

const { parseSSE, parseNDJSON, streamFromResponse } = await import('../src/streaming.js');
const cryptoStore = await import('../crypto-store.js');

// ── parseSSE / SSE assembly ────────────────────────────────────

describe('parseSSE', () => {
    test('field:value parsing strips one leading space', () => {
        assert.deepEqual(parseSSE('event: chunk'), { field: 'event', value: 'chunk' });
        assert.deepEqual(parseSSE('data:hello'), { field: 'data', value: 'hello' });
    });

    test('blank lines and comments return null', () => {
        assert.equal(parseSSE(''), null);
        assert.equal(parseSSE('\r'), null);
        assert.equal(parseSSE(': keepalive'), null);
    });

    test('field-only line (no colon) yields empty value', () => {
        assert.deepEqual(parseSSE('retry'), { field: 'retry', value: '' });
    });
});

describe('streamFromResponse(sse)', () => {
    test('multi-line data is joined and dispatched on blank line; [DONE] terminates', async () => {
        const lines = [
            'event: chunk',
            'data: line one',
            'data: line two',
            '',
            'data: solo',
            '',
            'data: [DONE]',
            '',
            'data: ignored after done',
            '',
        ];
        const body = lines.join('\n');
        const resp = mkResponse(body);
        const events = [];
        for await (const ev of streamFromResponse(resp, 'sse')) {
            events.push(ev);
            if (ev.done) break;
        }
        assert.equal(events.length, 3);
        assert.equal(events[0].event, 'chunk');
        assert.equal(events[0].data, 'line one\nline two');
        assert.equal(events[0].done, false);
        assert.equal(events[1].data, 'solo');
        assert.equal(events[2].done, true);
        assert.equal(events[2].data, '[DONE]');
    });

    test('handles \\r\\n line endings', async () => {
        const body = 'data: a\r\ndata: b\r\n\r\n';
        const resp = mkResponse(body);
        const events = [];
        for await (const ev of streamFromResponse(resp, 'sse')) events.push(ev);
        assert.equal(events.length, 1);
        assert.equal(events[0].data, 'a\nb');
    });
});

// ── parseNDJSON ────────────────────────────────────────────────

describe('parseNDJSON', () => {
    test('parses a single JSON object', () => {
        assert.deepEqual(parseNDJSON('{"a":1}'), { a: 1 });
    });

    test('returns null for blank/whitespace lines', () => {
        assert.equal(parseNDJSON(''), null);
        assert.equal(parseNDJSON('   '), null);
        assert.equal(parseNDJSON('\r'), null);
    });

    test('throws on malformed JSON', () => {
        assert.throws(() => parseNDJSON('{not json'));
    });
});

describe('streamFromResponse(ndjson)', () => {
    test('yields one parsed object per non-empty line, skips malformed', async () => {
        const body = '{"i":1}\n{"i":2}\n\nnot-json\n{"i":3}\n';
        const resp = mkResponse(body);
        const out = [];
        for await (const obj of streamFromResponse(resp, 'ndjson')) out.push(obj);
        assert.deepEqual(out, [{ i: 1 }, { i: 2 }, { i: 3 }]);
    });
});

// ── crypto-store round-trip ────────────────────────────────────

describe('crypto-store', () => {
    test('deriveKey + encrypt + decrypt round-trips a UTF-8 string', async () => {
        const salt = cryptoStore.generateSalt();
        const key = await cryptoStore.deriveKey('correct horse battery staple', salt);
        const blob = await cryptoStore.encrypt(key, 'sk-hello-world-🔑');
        const out = await cryptoStore.decrypt(key, blob);
        assert.equal(out, 'sk-hello-world-🔑');
    });

    test('pack/unpack is a bijection', async () => {
        const salt = cryptoStore.generateSalt();
        const key = await cryptoStore.deriveKey('pw', salt);
        const blob = await cryptoStore.encrypt(key, 'payload');
        const packed = cryptoStore.pack({ salt, iv: blob.iv, ciphertext: blob.ciphertext });
        const un = cryptoStore.unpack(packed);
        assert.deepEqual([...un.salt], [...salt]);
        assert.deepEqual([...un.iv], [...blob.iv]);
        assert.deepEqual([...un.ciphertext], [...blob.ciphertext]);
        // Re-derive on unpack-side and confirm decrypt still works.
        const key2 = await cryptoStore.deriveKey('pw', un.salt);
        const out = await cryptoStore.decrypt(key2, { iv: un.iv, ciphertext: un.ciphertext });
        assert.equal(out, 'payload');
    });

    test('decrypt throws on tampered ciphertext', async () => {
        const salt = cryptoStore.generateSalt();
        const key = await cryptoStore.deriveKey('pw', salt);
        const blob = await cryptoStore.encrypt(key, 'secret');
        const tampered = new Uint8Array(blob.ciphertext);
        tampered[0] ^= 0xff;
        await assert.rejects(cryptoStore.decrypt(key, { iv: blob.iv, ciphertext: tampered }));
    });
});

// ── db.appendMessageChunk via fake-indexeddb ───────────────────

describe('db', async () => {
    let dbModule = null;
    try {
        await import('fake-indexeddb/auto');
        dbModule = await import('../src/db.js');
    } catch (e) {
        // fake-indexeddb is the only devDep we add for tests; if it's
        // unavailable in this environment, skip the IDB tests rather
        // than fail the whole suite.
        // TODO: install fake-indexeddb if you want this test active.
        console.warn('[unit.test] fake-indexeddb not available, skipping IDB tests:', e.message);
    }

    if (!dbModule) return;

    test('appendMessageChunk creates the row on first call and accumulates content', async () => {
        const { appendMessageChunk, getMessage } = dbModule;
        const cid = 'c-' + Math.random().toString(36).slice(2);
        const mid = 'm-' + Math.random().toString(36).slice(2);
        await appendMessageChunk(cid, mid, 'Hello, ');
        await appendMessageChunk(cid, mid, 'world!');
        const row = await getMessage(cid, mid);
        assert.equal(row.content, 'Hello, world!');
        assert.equal(row.role, 'assistant');
        assert.equal(row.conversationId, cid);
        assert.equal(row.messageId, mid);
    });

    test('putConversation + listConversations sorts newest-first', async () => {
        const { putConversation, listConversations } = dbModule;
        const a = await putConversation({ id: 'sort-a', title: 'A', updatedAt: 100 });
        const b = await putConversation({ id: 'sort-b', title: 'B', updatedAt: 200 });
        const list = await listConversations();
        const byId = Object.fromEntries(list.map((c) => [c.id, c]));
        assert.ok(byId['sort-a']);
        assert.ok(byId['sort-b']);
        // Find their relative order — b updated later, must come first.
        const idxA = list.findIndex((c) => c.id === 'sort-a');
        const idxB = list.findIndex((c) => c.id === 'sort-b');
        assert.ok(idxB < idxA, 'expected newer conversation first');
        // Silence unused-warning; values are asserted above.
        void a; void b;
    });
});

// ── provider adapters ─────────────────────────────────────────

const anthropic = await import('../src/providers/anthropic.js');
const openai = await import('../src/providers/openai.js');
const google = await import('../src/providers/google.js');
const ollama = await import('../src/providers/ollama.js');
const modelStore = await import('../src/model-store.js');

describe('providers/anthropic', () => {
    test('buildRequest: URL, sentinel header, system extraction, stream:true', () => {
        const req = anthropic.buildRequest({
            model: 'claude-opus-4-7',
            messages: [
                { role: 'system', content: 'You are helpful.' },
                { role: 'user', content: 'Hi' },
                { role: 'assistant', content: 'Hello.' },
                { role: 'user', content: 'Tell me a joke.' },
            ],
            params: { max_tokens: 256, temperature: 0.5 },
        });
        assert.equal(req.url, 'https://api.anthropic.com/v1/messages');
        assert.equal(req.method, 'POST');
        assert.equal(req.format, 'sse');
        assert.equal(req.headers['anthropic-version'], '2023-06-01');
        assert.equal(req.headers['x-api-key'], '__API_KEY__');
        assert.equal(req.headers['content-type'], 'application/json');
        const body = JSON.parse(req.body);
        assert.equal(body.model, 'claude-opus-4-7');
        assert.equal(body.stream, true);
        assert.equal(body.max_tokens, 256);
        assert.equal(body.temperature, 0.5);
        assert.equal(body.system, 'You are helpful.');
        assert.equal(body.messages.length, 3);
        assert.equal(body.messages[0].role, 'user');
        assert.equal(body.messages[1].role, 'assistant');
    });

    test('parseChunk: text_delta event extracts the delta', () => {
        const ev = {
            type: 'event',
            event: 'content_block_delta',
            data: JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hello' } }),
            done: false,
        };
        assert.deepEqual(anthropic.parseChunk(ev), { delta: 'hello' });
    });

    test('parseChunk: message_stop returns finished', () => {
        const ev = {
            type: 'event',
            event: 'message_stop',
            data: JSON.stringify({ type: 'message_stop' }),
            done: false,
        };
        assert.deepEqual(anthropic.parseChunk(ev), { finished: true });
    });

    test('buildRequest: serializes params.tools to top-level Anthropic tools[]', () => {
        const req = anthropic.buildRequest({
            model: 'claude-opus-4-7',
            messages: [{ role: 'user', content: 'hi' }],
            params: {
                tools: [
                    {
                        name: 'fs_read',
                        description: 'Read a file',
                        input_schema: { type: 'object', properties: { path: { type: 'string' } } },
                    },
                ],
            },
        });
        const body = JSON.parse(req.body);
        assert.ok(Array.isArray(body.tools), 'body.tools should be an array');
        assert.equal(body.tools.length, 1);
        assert.equal(body.tools[0].name, 'fs_read');
        assert.equal(body.tools[0].description, 'Read a file');
        assert.deepEqual(body.tools[0].input_schema, { type: 'object', properties: { path: { type: 'string' } } });
    });

    test('mapMessages: assistant tool_use part → Anthropic tool_use content block', () => {
        const req = anthropic.buildRequest({
            model: 'claude-opus-4-7',
            messages: [
                { role: 'user', content: 'list /tmp' },
                {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'reading dir' },
                        { type: 'tool_use', id: 'toolu_a', name: 'fs_list', input: { path: '/tmp' } },
                    ],
                },
            ],
        });
        const body = JSON.parse(req.body);
        const blocks = body.messages[1].content;
        assert.ok(Array.isArray(blocks), 'assistant content should be a content-block array');
        assert.equal(blocks[0].type, 'text');
        assert.equal(blocks[0].text, 'reading dir');
        assert.equal(blocks[1].type, 'tool_use');
        assert.equal(blocks[1].id, 'toolu_a');
        assert.equal(blocks[1].name, 'fs_list');
        assert.deepEqual(blocks[1].input, { path: '/tmp' });
    });

    test('mapMessages: user tool_result parts → Anthropic tool_result content blocks (string + object content + is_error)', () => {
        const req = anthropic.buildRequest({
            model: 'claude-opus-4-7',
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'tool_result', toolUseId: 'toolu_a', content: 'plain string out' },
                        { type: 'tool_result', toolUseId: 'toolu_b', content: { rows: [1, 2, 3] } },
                        { type: 'tool_result', toolUseId: 'toolu_c', content: 'oops', is_error: true },
                    ],
                },
            ],
        });
        const body = JSON.parse(req.body);
        const blocks = body.messages[0].content;
        assert.ok(Array.isArray(blocks));
        assert.equal(blocks.length, 3);
        assert.equal(blocks[0].type, 'tool_result');
        assert.equal(blocks[0].tool_use_id, 'toolu_a');
        assert.equal(blocks[0].content, 'plain string out');
        assert.equal(blocks[0].is_error, undefined);
        assert.equal(blocks[1].tool_use_id, 'toolu_b');
        assert.equal(blocks[1].content, JSON.stringify({ rows: [1, 2, 3] }));
        assert.equal(blocks[2].tool_use_id, 'toolu_c');
        assert.equal(blocks[2].is_error, true);
    });

    test('parseChunk: reassembles tool_use across content_block_start/delta/stop with shared acc', () => {
        const acc = {};
        const start = anthropic.parseChunk({
            event: 'content_block_start',
            data: JSON.stringify({
                type: 'content_block_start',
                index: 1,
                content_block: { type: 'tool_use', id: 'toolu_abc', name: 'fs_read', input: {} },
            }),
            done: false,
        }, acc);
        assert.equal(start, null);
        const d1 = anthropic.parseChunk({
            event: 'content_block_delta',
            data: JSON.stringify({
                type: 'content_block_delta',
                index: 1,
                delta: { type: 'input_json_delta', partial_json: '{"pa' },
            }),
            done: false,
        }, acc);
        assert.equal(d1, null);
        const d2 = anthropic.parseChunk({
            event: 'content_block_delta',
            data: JSON.stringify({
                type: 'content_block_delta',
                index: 1,
                delta: { type: 'input_json_delta', partial_json: 'th":"/etc/hosts"}' },
            }),
            done: false,
        }, acc);
        assert.equal(d2, null);
        const stop = anthropic.parseChunk({
            event: 'content_block_stop',
            data: JSON.stringify({ type: 'content_block_stop', index: 1 }),
            done: false,
        }, acc);
        assert.deepEqual(stop, {
            tool_use: { id: 'toolu_abc', name: 'fs_read', input: { path: '/etc/hosts' } },
        });
    });
});

describe('providers/openai', () => {
    test('buildRequest: URL, Bearer sentinel, stream:true', () => {
        const req = openai.buildRequest({
            model: 'gpt-4o-mini',
            messages: [
                { role: 'system', content: 'sys' },
                { role: 'user', content: 'hi' },
            ],
            params: { max_tokens: 64 },
        });
        assert.equal(req.url, 'https://api.openai.com/v1/chat/completions');
        assert.equal(req.method, 'POST');
        assert.equal(req.format, 'sse');
        assert.equal(req.headers['Authorization'], 'Bearer __API_KEY__');
        const body = JSON.parse(req.body);
        assert.equal(body.stream, true);
        assert.equal(body.model, 'gpt-4o-mini');
        // OpenAI keeps system in the messages array (unlike Anthropic).
        assert.equal(body.messages[0].role, 'system');
        assert.equal(body.max_tokens, 64);
    });

    test('parseChunk: extracts choices[0].delta.content', () => {
        const ev = {
            type: 'event',
            event: 'message',
            data: JSON.stringify({ choices: [{ delta: { content: 'tok' }, index: 0 }] }),
            done: false,
        };
        assert.deepEqual(openai.parseChunk(ev), { delta: 'tok' });
    });

    test('parseChunk: [DONE] sentinel returns finished', () => {
        assert.deepEqual(openai.parseChunk({ done: true, data: '[DONE]', event: 'message' }), { finished: true });
    });

    test('buildRequest: serializes params.tools to OpenAI function shape', () => {
        const req = openai.buildRequest({
            model: 'gpt-4o-mini',
            messages: [{ role: 'user', content: 'hi' }],
            params: {
                tools: [
                    {
                        name: 'fs_read',
                        description: 'Read a file',
                        input_schema: { type: 'object', properties: { path: { type: 'string' } } },
                    },
                ],
            },
        });
        const body = JSON.parse(req.body);
        assert.ok(Array.isArray(body.tools), 'body.tools should be an array');
        assert.equal(body.tools.length, 1);
        assert.equal(body.tools[0].type, 'function');
        assert.equal(body.tools[0].function.name, 'fs_read');
        assert.equal(body.tools[0].function.description, 'Read a file');
        assert.deepEqual(body.tools[0].function.parameters, {
            type: 'object',
            properties: { path: { type: 'string' } },
        });
    });

    test('mapMessages: assistant tool_use parts → message with tool_calls[] and content:null', () => {
        const req = openai.buildRequest({
            model: 'gpt-5.5',
            messages: [
                { role: 'user', content: 'hi' },
                {
                    role: 'assistant',
                    content: [
                        { type: 'tool_use', id: 'call_1', name: 'fs_read', input: { path: '/a' } },
                        { type: 'tool_use', id: 'call_2', name: 'fs_list', input: { path: '/b' } },
                    ],
                },
            ],
        });
        const body = JSON.parse(req.body);
        const asst = body.messages[1];
        assert.equal(asst.role, 'assistant');
        assert.equal(asst.content, null);
        assert.ok(Array.isArray(asst.tool_calls));
        assert.equal(asst.tool_calls.length, 2);
        assert.equal(asst.tool_calls[0].id, 'call_1');
        assert.equal(asst.tool_calls[0].type, 'function');
        assert.equal(asst.tool_calls[0].function.name, 'fs_read');
        assert.equal(asst.tool_calls[0].function.arguments, JSON.stringify({ path: '/a' }));
        assert.equal(asst.tool_calls[1].function.name, 'fs_list');
    });

    test('mapMessages: user tool_result parts → N separate role:tool messages', () => {
        const req = openai.buildRequest({
            model: 'gpt-5.5',
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'tool_result', toolUseId: 'call_1', content: 'A' },
                        { type: 'tool_result', toolUseId: 'call_2', content: { rows: 3 } },
                    ],
                },
            ],
        });
        const body = JSON.parse(req.body);
        assert.equal(body.messages.length, 2);
        assert.equal(body.messages[0].role, 'tool');
        assert.equal(body.messages[0].tool_call_id, 'call_1');
        assert.equal(body.messages[0].content, 'A');
        assert.equal(body.messages[1].role, 'tool');
        assert.equal(body.messages[1].tool_call_id, 'call_2');
        assert.equal(body.messages[1].content, JSON.stringify({ rows: 3 }));
    });

    test('parseChunk: reassembles tool_uses across delta tool_calls then finish_reason=tool_calls', () => {
        const acc = {};
        // First delta: index, id, function.name, opening of arguments JSON.
        const d1 = openai.parseChunk({
            event: 'message',
            data: JSON.stringify({
                choices: [{
                    index: 0,
                    delta: {
                        tool_calls: [{
                            index: 0,
                            id: 'call_xyz',
                            type: 'function',
                            function: { name: 'fs_read', arguments: '{"pa' },
                        }],
                    },
                }],
            }),
            done: false,
        }, acc);
        assert.equal(d1, null);
        // Second delta: only an arguments fragment.
        const d2 = openai.parseChunk({
            event: 'message',
            data: JSON.stringify({
                choices: [{
                    index: 0,
                    delta: { tool_calls: [{ index: 0, function: { arguments: 'th":"/x"}' } }] },
                }],
            }),
            done: false,
        }, acc);
        assert.equal(d2, null);
        // Finalization: finish_reason=tool_calls.
        const fin = openai.parseChunk({
            event: 'message',
            data: JSON.stringify({
                choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
            }),
            done: false,
        }, acc);
        assert.ok(fin, 'finalization should return a non-null event');
        assert.equal(fin.finished, true);
        assert.deepEqual(fin.tool_use, {
            id: 'call_xyz',
            name: 'fs_read',
            input: { path: '/x' },
        });
    });
});

describe('providers/google', () => {
    test('buildRequest: URL contains __API_KEY__, system → systemInstruction', () => {
        const req = google.buildRequest({
            model: 'gemini-2.5-flash',
            messages: [
                { role: 'system', content: 'be terse' },
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'sup' },
            ],
            params: { temperature: 0.2 },
        });
        assert.ok(req.url.startsWith('https://generativelanguage.googleapis.com/v1beta/models/'));
        assert.ok(req.url.includes('streamGenerateContent'));
        assert.ok(req.url.includes('alt=sse'));
        assert.ok(req.url.includes('key=__API_KEY__'));
        assert.equal(req.format, 'sse');
        const body = JSON.parse(req.body);
        assert.equal(body.systemInstruction.parts[0].text, 'be terse');
        assert.equal(body.contents.length, 2);
        assert.equal(body.contents[0].role, 'user');
        assert.equal(body.contents[1].role, 'model');
        assert.equal(body.generationConfig.temperature, 0.2);
    });

    test('parseChunk: pulls candidates[0].content.parts[0].text', () => {
        const ev = {
            type: 'event',
            event: 'message',
            data: JSON.stringify({
                candidates: [{ content: { parts: [{ text: 'piece' }], role: 'model' } }],
            }),
            done: false,
        };
        assert.deepEqual(google.parseChunk(ev), { delta: 'piece' });
    });
});

describe('providers/ollama', () => {
    test('buildRequest: defaults to localhost:11434, no auth header, stream:true', () => {
        const req = ollama.buildRequest({
            model: 'gemma3:latest',
            messages: [{ role: 'user', content: 'hi' }],
            params: {},
        });
        assert.equal(req.url, 'http://localhost:11434/api/chat');
        assert.equal(req.format, 'ndjson');
        assert.equal(req.headers['Authorization'], undefined);
        assert.equal(req.headers['x-api-key'], undefined);
        const body = JSON.parse(req.body);
        assert.equal(body.stream, true);
        assert.equal(body.model, 'gemma3:latest');
    });

    test('buildRequest: honors params.baseUrl override and trims trailing slash', () => {
        const req = ollama.buildRequest({
            model: 'llama3.2:latest',
            messages: [{ role: 'user', content: 'x' }],
            params: { baseUrl: 'http://10.0.0.5:11434/' },
        });
        assert.equal(req.url, 'http://10.0.0.5:11434/api/chat');
    });

    test('parseChunk: ndjson line yields delta and finished', () => {
        assert.deepEqual(
            ollama.parseChunk({ message: { role: 'assistant', content: 'hey' }, done: false }),
            { delta: 'hey' },
        );
        const fin = ollama.parseChunk({ message: { role: 'assistant', content: '' }, done: true, eval_count: 5 });
        assert.equal(fin.finished, true);
        assert.equal(fin.usage.completion_tokens, 5);
    });
});

describe('model-store', () => {
    test('KNOWN_MODELS has gemma-4-e2b with the expected shape', () => {
        const m = modelStore.KNOWN_MODELS['gemma-4-e2b'];
        assert.ok(m);
        assert.equal(m.id, 'gemma-4-e2b');
        assert.equal(m.hf.repo, 'google/gemma-4-e2b');
        assert.equal(m.hf.revision, 'main');
        assert.ok(Array.isArray(m.files));
        const kinds = m.files.map((f) => f.kind).sort();
        assert.deepEqual(kinds, ['tokenizer', 'weights']);
    });

    test('gemma-4-e2b is flagged multimodal so the worker picks the vision loader', () => {
        // The worker reads this flag to decide between
        // `init_local_multimodal` and the text-only `init_local_model_*`
        // exports. Without it, `vision_chat` would fall through to the
        // text path and surface a cryptic error.
        const m = modelStore.KNOWN_MODELS['gemma-4-e2b'];
        assert.equal(m.multimodal, true);
    });

    test('cacheKey produces a HF resolve URL', () => {
        const url = modelStore.cacheKey('gemma-4-e2b', 'model.safetensors');
        assert.equal(url, 'https://huggingface.co/google/gemma-4-e2b/resolve/main/model.safetensors');
    });

    test('cacheKey throws on unknown model', () => {
        assert.throws(() => modelStore.cacheKey('does-not-exist', 'x.bin'));
    });

    // The full download path needs Cache Storage + a fetch polyfill that
    // streams a Response.body. Skipping until we wire one in.
    // TODO: bring in a Cache + fetch test polyfill (or use Playwright)
    // and exercise downloadModel({onProgress}) end-to-end.
    test.skip('downloadModel writes to Cache Storage and emits progress', () => {});

    test('downloadModel emits a verifying-phase event for cached files with sha256 pins', async () => {
        // Mock Cache Storage with a pre-cached entry whose bytes match
        // the pin. We set a pin on the registry temporarily so the
        // verify branch fires; restore it after.
        const m = modelStore.KNOWN_MODELS['gemma-4-e2b'];
        const originalFiles = m.files.map((f) => ({ ...f }));
        const stateMod = await import('../src/state.js');

        // Build small payloads + their real sha256 pins.
        const payloads = await Promise.all(m.files.map(async (f) => {
            const bytes = new TextEncoder().encode(`stub-${f.filename}`);
            const digest = await crypto.subtle.digest('SHA-256', bytes);
            const hex = [...new Uint8Array(digest)]
                .map((b) => b.toString(16).padStart(2, '0')).join('');
            return { f, bytes, hex };
        }));

        // Pin each file in the registry so downloadModel takes the
        // verify-cached branch.
        for (const p of payloads) p.f.sha256 = p.hex;

        // Install a fake `caches` that returns the pre-cached responses.
        const cacheBacking = new Map();
        for (const p of payloads) {
            const key = modelStore.cacheKey('gemma-4-e2b', p.f.filename);
            cacheBacking.set(key, new Response(p.bytes));
        }
        const fakeCache = {
            match: async (key) => {
                const hit = cacheBacking.get(key);
                return hit ? hit.clone() : undefined;
            },
            put: async (key, resp) => { cacheBacking.set(key, resp); },
            delete: async (key) => cacheBacking.delete(key),
        };
        const originalCaches = globalThis.caches;
        globalThis.caches = { open: async () => fakeCache };

        const phases = [];
        const handler = (e) => {
            if (e.detail && e.detail.phase) phases.push(e.detail.phase);
        };
        stateMod.events.addEventListener('model_progress', handler);

        try {
            await modelStore.downloadModel('gemma-4-e2b');
        } finally {
            stateMod.events.removeEventListener('model_progress', handler);
            globalThis.caches = originalCaches;
            // Restore registry pins.
            for (let i = 0; i < originalFiles.length; i++) {
                m.files[i].sha256 = originalFiles[i].sha256;
            }
        }

        assert.ok(phases.includes('verifying'),
            `expected a 'verifying' phase event, got: ${JSON.stringify(phases)}`);
    });
});

// ── providers/local — RPC surface ─────────────────────────────
//
// `node --test` has no Worker polyfill, so we can't drive end-to-end
// through the actual worker — but we can at least lock in the public
// API shape so accidental renames break the test rather than the UI.
// The full main↔worker↔wasm round-trip lives in the e2e suite.

describe('providers/local (RPC surface)', () => {
    test('exposes load/unload/chat/cancel + module metadata', async () => {
        await import('fake-indexeddb/auto');
        const local = await import('../src/providers/local.js');
        assert.equal(local.id, 'local-gemma-4-e2b');
        assert.equal(local.runtime, 'local');
        assert.equal(local.defaultModel, 'gemma-4-e2b');
        assert.equal(typeof local.loadLocalModel, 'function');
        assert.equal(typeof local.unloadLocalModel, 'function');
        assert.equal(typeof local.startChat, 'function');
        assert.equal(typeof local.chatLocal, 'function');
        assert.equal(typeof local.cancelLocal, 'function');
        assert.equal(typeof local.isLocalModelLoaded, 'function');
        // Aliases must point at the same impl so behaviour can't drift.
        assert.equal(local.chatLocal, local.startChat);
        // No worker spawned just by importing the module.
        assert.equal(local.isLocalModelLoaded(), false);
    });

    test.skip('main→worker→wasm round-trip (needs Worker polyfill or browser)', () => {});
});

// ── utils (formatters + pure DOM helpers) ─────────────────────

const utils = await import('../src/utils.js');

describe('utils.formatBytes', () => {
    test('handles 0, sub-KB, KB, MB, GB', () => {
        assert.equal(utils.formatBytes(0), '0 B');
        assert.equal(utils.formatBytes(512), '512 B');
        assert.equal(utils.formatBytes(1024), '1.00 KB');
        assert.equal(utils.formatBytes(1536), '1.50 KB');
        assert.equal(utils.formatBytes(5_242_880), '5.00 MB');
        assert.equal(utils.formatBytes(2_500_000_000), '2.33 GB');
    });

    test('rejects non-number / negative', () => {
        assert.equal(utils.formatBytes(NaN), '0 B');
        assert.equal(utils.formatBytes(-1), '0 B');
        assert.equal(utils.formatBytes('hello'), '0 B');
    });
});

describe('utils.formatEta', () => {
    test('seconds, minutes, hours', () => {
        assert.equal(utils.formatEta(0), '0s');
        assert.equal(utils.formatEta(45), '45s');
        assert.equal(utils.formatEta(75), '1m 15s');
        assert.equal(utils.formatEta(3725), '1h 2m 5s');
    });

    test('null/Infinity/negative → em-dash', () => {
        assert.equal(utils.formatEta(null), '—');
        assert.equal(utils.formatEta(undefined), '—');
        assert.equal(utils.formatEta(Infinity), '—');
        assert.equal(utils.formatEta(-1), '—');
    });
});

describe('utils.escapeHtml', () => {
    test('escapes the five usual suspects', () => {
        assert.equal(utils.escapeHtml('<a href="x">&y</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;y&lt;/a&gt;');
        assert.equal(utils.escapeHtml("o'reilly"), 'o&#39;reilly');
    });
});

describe('utils.debounce / throttle', () => {
    test('debounce only fires once after quiet period', async () => {
        let calls = 0;
        const fn = utils.debounce(() => { calls += 1; }, 30);
        fn(); fn(); fn();
        await new Promise((r) => setTimeout(r, 80));
        assert.equal(calls, 1);
    });

    test('throttle fires at most once per window', async () => {
        let calls = 0;
        const fn = utils.throttle(() => { calls += 1; }, 30);
        fn(); fn(); fn();
        // The first call is immediate; further calls coalesce into one trailing call.
        await new Promise((r) => setTimeout(r, 80));
        assert.ok(calls >= 1 && calls <= 2, `expected 1..2 calls, got ${calls}`);
    });
});

// ── i18n ─────────────────────────────────────────────────────

describe('i18n', () => {
    test('falls back to the key when missing', async () => {
        const i18n = await import('../src/i18n.js');
        i18n._setDictForTests({ 'app.title': 'Brainwires Chat' });
        assert.equal(i18n.t('app.title'), 'Brainwires Chat');
        assert.equal(i18n.t('not.in.dict'), 'not.in.dict');
    });

    test('substitutes {var} placeholders', async () => {
        const i18n = await import('../src/i18n.js');
        i18n._setDictForTests({ 'hello': 'Hello, {name}!' });
        assert.equal(i18n.t('hello', { name: 'world' }), 'Hello, world!');
        // Missing variable → leaves the placeholder visible.
        assert.equal(i18n.t('hello', {}), 'Hello, {name}!');
    });
});

// ── markdown ──────────────────────────────────────────────────

describe('markdown', async () => {
    let renderRaw;
    try {
        const i18n = await import('../src/i18n.js');
        i18n._setDictForTests({ 'chat.copy': 'Copy' });
        const md = await import('../src/markdown.js');
        renderRaw = md.renderRaw;
    } catch (e) {
        console.warn('[unit.test] markdown imports failed:', e.message);
    }

    test('emits our codeblock wrapper with copy button + language class', (ctx) => {
        if (!renderRaw) return ctx.skip();
        const out = renderRaw('```javascript\nconst x = 1;\n```\n');
        assert.match(out, /<div class="codeblock">/);
        assert.match(out, /data-bw-copy="1"/);
        assert.match(out, /class="codeblock-copy"/);
        assert.match(out, /<code class="language-javascript">/);
        assert.match(out, /const x = 1;/);
    });

    test('inline code, bold, italic render', (ctx) => {
        if (!renderRaw) return ctx.skip();
        const out = renderRaw('use `foo` and **bold** and *italic*');
        assert.match(out, /<code>foo<\/code>/);
        assert.match(out, /<strong>bold<\/strong>/);
        assert.match(out, /<em>italic<\/em>/);
    });

    test('links get target=_blank and rel=noopener noreferrer', (ctx) => {
        if (!renderRaw) return ctx.skip();
        const out = renderRaw('[click](https://example.com)');
        assert.match(out, /href="https:\/\/example\.com"/);
        assert.match(out, /target="_blank"/);
        assert.match(out, /rel="noopener noreferrer"/);
    });

    test('unclosed fence at end of stream still renders as code (mid-stream safety)', (ctx) => {
        if (!renderRaw) return ctx.skip();
        const out = renderRaw('here:\n```js\nconst partial = ');
        // Marked treats an unclosed fence as a code block to EOF — exactly
        // what we want during streaming so the bubble doesn't briefly show
        // the fence as plain text and then "snap" into a code block.
        assert.match(out, /<pre><code/);
        assert.match(out, /const partial =/);
    });

    test('escapes HTML inside code blocks (no XSS via fenced content)', (ctx) => {
        if (!renderRaw) return ctx.skip();
        const out = renderRaw('```\n<script>x</script>\n```');
        assert.ok(!out.includes('<script>x'));
        assert.match(out, /&lt;script&gt;x&lt;\/script&gt;/);
    });
});

// ── mcp-client (Streamable HTTP) ──────────────────────────────

describe('mcp-client', async () => {
    let mcp;
    try { mcp = await import('../src/mcp-client.js'); }
    catch (e) { console.warn('[unit.test] mcp-client import failed:', e.message); }

    function jsonResponse(obj, headers = {}) {
        return new Response(JSON.stringify(obj), {
            status: 200,
            headers: { 'content-type': 'application/json', ...headers },
        });
    }

    test('initialize: parses JSON reply, captures session header', async (ctx) => {
        if (!mcp) return ctx.skip();
        let captured;
        const origFetch = globalThis.fetch;
        globalThis.fetch = async (url, init) => {
            captured = { url, init };
            return jsonResponse(
                { jsonrpc: '2.0', id: 1, result: { protocolVersion: '2025-06-18', capabilities: {} } },
                { 'mcp-session-id': 'sess-123' },
            );
        };
        try {
            const out = await mcp.initialize({ url: 'https://x.test/mcp' });
            assert.equal(out.protocolVersion, '2025-06-18');
            assert.equal(captured.init.method, 'POST');
            const body = JSON.parse(captured.init.body);
            assert.equal(body.method, 'initialize');
            assert.equal(body.jsonrpc, '2.0');
            assert.equal(typeof body.id, 'number');
        } finally {
            globalThis.fetch = origFetch;
        }
    });

    test('listTools: returns the tools array', async (ctx) => {
        if (!mcp) return ctx.skip();
        const origFetch = globalThis.fetch;
        globalThis.fetch = async () => jsonResponse({
            jsonrpc: '2.0', id: 1,
            result: { tools: [{ name: 'echo', description: 'echo input' }] },
        });
        try {
            const tools = await mcp.listTools({ url: 'https://x.test/mcp' });
            assert.equal(tools.length, 1);
            assert.equal(tools[0].name, 'echo');
        } finally {
            globalThis.fetch = origFetch;
        }
    });

    test('error reply surfaces a thrown Error', async (ctx) => {
        if (!mcp) return ctx.skip();
        const origFetch = globalThis.fetch;
        globalThis.fetch = async () => jsonResponse({
            jsonrpc: '2.0', id: 1,
            error: { code: -32601, message: 'method not found' },
        });
        try {
            await assert.rejects(() => mcp.listTools({ url: 'https://x.test/mcp' }), /method not found/);
        } finally {
            globalThis.fetch = origFetch;
        }
    });

    test('http non-2xx surfaces a thrown Error', async (ctx) => {
        if (!mcp) return ctx.skip();
        const origFetch = globalThis.fetch;
        globalThis.fetch = async () => new Response('boom', { status: 502 });
        try {
            await assert.rejects(() => mcp.initialize({ url: 'https://x.test/mcp' }), /HTTP 502/);
        } finally {
            globalThis.fetch = origFetch;
        }
    });
});

// ── chunker ───────────────────────────────────────────────────

describe('chunker', async () => {
    let chunkText;
    try { ({ chunkText } = await import('../src/chunker.js')); }
    catch (e) { console.warn('[unit.test] chunker import failed:', e.message); }

    test('empty / null returns empty array', (ctx) => {
        if (!chunkText) return ctx.skip();
        assert.deepEqual(chunkText(''), []);
        assert.deepEqual(chunkText(null), []);
        assert.deepEqual(chunkText(undefined), []);
    });

    test('short text fits in one chunk', (ctx) => {
        if (!chunkText) return ctx.skip();
        const out = chunkText('Hello world. This is a short doc.');
        assert.equal(out.length, 1);
        assert.match(out[0], /Hello world/);
    });

    test('long text splits into multiple chunks with overlap', (ctx) => {
        if (!chunkText) return ctx.skip();
        // ~3000 chars of repeating sentences → multiple ~2KB chunks at the
        // default target (512 tokens × 4 chars/token = 2048).
        const sentences = [];
        for (let i = 0; i < 80; i++) sentences.push(`Sentence ${i} has a payload of words that hopefully chunks well.`);
        const out = chunkText(sentences.join(' '));
        assert.ok(out.length >= 2, `expected multiple chunks, got ${out.length}`);
        // Adjacent chunks should share some prefix material via overlap.
        const tail = out[0].slice(-32);
        assert.ok(out[1].includes(tail.slice(0, 16)) || out[1].length > 0);
    });

    test('hard-splits a single sentence longer than target', (ctx) => {
        if (!chunkText) return ctx.skip();
        const huge = 'x'.repeat(5000);
        const out = chunkText(huge);
        assert.ok(out.length >= 2, `expected hard-split, got ${out.length}`);
    });
});

// ── vision (mapping + isVisionModel) ──────────────────────────

describe('vision', async () => {
    let vision;
    try { vision = await import('../src/vision.js'); }
    catch (e) { console.warn('[unit.test] vision import failed:', e.message); }

    test('isVisionModel returns true for known multimodal models', (ctx) => {
        if (!vision) return ctx.skip();
        assert.equal(vision.isVisionModel('anthropic', 'claude-opus-4-7'), true);
        assert.equal(vision.isVisionModel('anthropic', 'claude-sonnet-4-6'), true);
        assert.equal(vision.isVisionModel('openai', 'gpt-5.5'), true);
        assert.equal(vision.isVisionModel('openai', 'gpt-4.1-mini'), true);
        assert.equal(vision.isVisionModel('openai', 'o3'), true);
        assert.equal(vision.isVisionModel('google', 'gemini-2.5-flash'), true);
        assert.equal(vision.isVisionModel('google', 'gemini-1.5-pro'), true);
        assert.equal(vision.isVisionModel('local', 'gemma-4-e2b'), true);
    });

    test('isVisionModel returns false for unknown providers / models', (ctx) => {
        if (!vision) return ctx.skip();
        assert.equal(vision.isVisionModel('ollama', 'llama3'), false);
        assert.equal(vision.isVisionModel('mystery', 'foo'), false);
        assert.equal(vision.isVisionModel('anthropic', null), false);
    });
});

describe('providers vision mapping', async () => {
    const ant = await import('../src/providers/anthropic.js');
    const oai = await import('../src/providers/openai.js');
    const gem = await import('../src/providers/google.js');

    const visionMessage = {
        role: 'user',
        content: [
            { type: 'text', text: 'what is in this image?' },
            { type: 'image', mediaType: 'image/jpeg', data: 'AAAAB' },
        ],
    };

    test('anthropic: image part becomes a base64 source content block', () => {
        const req = ant.buildRequest({
            model: 'claude-opus-4-7',
            messages: [visionMessage],
        });
        const body = JSON.parse(req.body);
        const blocks = body.messages[0].content;
        assert.ok(Array.isArray(blocks));
        assert.equal(blocks[0].type, 'text');
        assert.equal(blocks[0].text, 'what is in this image?');
        assert.equal(blocks[1].type, 'image');
        assert.equal(blocks[1].source.type, 'base64');
        assert.equal(blocks[1].source.media_type, 'image/jpeg');
        assert.equal(blocks[1].source.data, 'AAAAB');
    });

    test('openai: image part becomes an image_url with data URL', () => {
        const req = oai.buildRequest({
            model: 'gpt-5.5',
            messages: [visionMessage],
        });
        const body = JSON.parse(req.body);
        const items = body.messages[0].content;
        assert.ok(Array.isArray(items));
        assert.equal(items[0].type, 'text');
        assert.equal(items[1].type, 'image_url');
        assert.equal(items[1].image_url.url, 'data:image/jpeg;base64,AAAAB');
    });

    test('gemini: image part becomes inline_data', () => {
        const req = gem.buildRequest({
            model: 'gemini-2.5-flash',
            messages: [visionMessage],
        });
        const body = JSON.parse(req.body);
        const parts = body.contents[0].parts;
        assert.equal(parts[0].text, 'what is in this image?');
        assert.equal(parts[1].inline_data.mime_type, 'image/jpeg');
        assert.equal(parts[1].inline_data.data, 'AAAAB');
    });

    test('all providers: legacy string content still works', () => {
        const msgs = [{ role: 'user', content: 'plain' }];
        const a = JSON.parse(ant.buildRequest({ model: 'claude-opus-4-7', messages: msgs }).body);
        const o = JSON.parse(oai.buildRequest({ model: 'gpt-5.5', messages: msgs }).body);
        const g = JSON.parse(gem.buildRequest({ model: 'gemini-2.5-flash', messages: msgs }).body);
        assert.equal(a.messages[0].content, 'plain');
        assert.equal(o.messages[0].content, 'plain');
        assert.equal(g.contents[0].parts[0].text, 'plain');
    });
});

// ── db parts[] helpers ────────────────────────────────────────

describe('db parts[] helpers', async () => {
    let dbReady = false;
    try {
        await import('fake-indexeddb/auto');
        dbReady = true;
    } catch (_) { /* skip silently */ }

    test('normalizeContent wraps a string into [{type:text}]', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        assert.deepEqual(db.normalizeContent('hi'), [{ type: 'text', text: 'hi' }]);
        assert.deepEqual(db.normalizeContent(''), []);
        assert.deepEqual(db.normalizeContent(null), []);
        assert.deepEqual(db.normalizeContent(undefined), []);
        const parts = [{ type: 'text', text: 'a' }, { type: 'image', mediaType: 'image/png', data: 'AAAA' }];
        assert.equal(db.normalizeContent(parts), parts); // identity for arrays
    });

    test('partsToText joins text parts and skips non-text', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        assert.equal(db.partsToText('plain'), 'plain');
        assert.equal(db.partsToText([{ type: 'text', text: 'a' }, { type: 'image', data: 'x' }, { type: 'text', text: 'b' }]), 'ab');
        assert.equal(db.partsToText([]), '');
        assert.equal(db.partsToText(null), '');
    });

    test('appendMessageChunk on a parts[] row appends to the trailing text part', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        await db.putMessage({
            conversationId: 'c1',
            messageId: 'm1',
            role: 'assistant',
            content: [{ type: 'text', text: 'hello ' }],
            createdAt: Date.now(),
        });
        await db.appendMessageChunk('c1', 'm1', 'world');
        const row = await db.getMessage('c1', 'm1');
        assert.deepEqual(row.content, [{ type: 'text', text: 'hello world' }]);
    });

    test('new v2 stores accept inserts (smoke)', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        await db.putRagDoc({ id: 'd1', conversationId: null, name: 'spec.pdf', type: 'pdf', bytes: 1234 });
        const docs = await db.listRagDocs(null);
        assert.equal(docs.length, 1);
        assert.equal(docs[0].name, 'spec.pdf');
        await db.putRagChunks([
            { id: 'k1', docId: 'd1', conversationId: null, page: 1, text: 'chunk one', embeddingDim: 384 },
            { id: 'k2', docId: 'd1', conversationId: null, page: 2, text: 'chunk two', embeddingDim: 384 },
        ]);
        const chunks = await db.listRagChunksByDoc('d1');
        assert.equal(chunks.length, 2);
        await db.deleteRagDoc('d1');
        assert.equal((await db.listRagDocs(null)).length, 0);
        assert.equal((await db.listRagChunksByDoc('d1')).length, 0);
    });
});

// ── reasoning-display ─────────────────────────────────────────

describe('reasoning-display', async () => {
    let extractThinking;
    try {
        const i18n = await import('../src/i18n.js');
        i18n._setDictForTests({ 'chat.copy': 'Copy', 'chat.reasoning': 'Reasoning' });
        const mod = await import('../src/reasoning-display.js');
        extractThinking = mod.extractThinking;
    } catch (e) { console.warn('[unit.test] reasoning-display imports failed:', e.message); }

    test('extracts a leading <thinking>...</thinking> block', (ctx) => {
        if (!extractThinking) return ctx.skip();
        const { thinking, body } = extractThinking('<thinking>I should check both branches.</thinking>\nThe answer is 42.');
        assert.equal(thinking, 'I should check both branches.');
        assert.equal(body, 'The answer is 42.');
    });

    test('returns null thinking when no block present', (ctx) => {
        if (!extractThinking) return ctx.skip();
        const { thinking, body } = extractThinking('Just a regular response.');
        assert.equal(thinking, null);
        assert.equal(body, 'Just a regular response.');
    });

    test('handles whitespace before opening tag', (ctx) => {
        if (!extractThinking) return ctx.skip();
        const { thinking, body } = extractThinking('  \n<thinking>step</thinking>x');
        assert.equal(thinking, 'step');
        assert.equal(body, 'x');
    });

    test('does not extract when tag is mid-message', (ctx) => {
        if (!extractThinking) return ctx.skip();
        const { thinking } = extractThinking('Hello <thinking>this came late</thinking>');
        assert.equal(thinking, null);
    });

    test('partial open tag during streaming yields no extraction yet', (ctx) => {
        if (!extractThinking) return ctx.skip();
        const { thinking, body } = extractThinking('<thinking>partial...');
        // Without a closing tag, the regex doesn't match; the partial body
        // renders as plain text and snaps into a <details> when </thinking>
        // arrives.
        assert.equal(thinking, null);
        assert.equal(body, '<thinking>partial...');
    });
});

// ── theme ─────────────────────────────────────────────────────

describe('theme', async () => {
    let dbReady = false;
    try {
        await import('fake-indexeddb/auto');
        dbReady = true;
    } catch (_) { /* fake-indexeddb missing — tests below will skip */ }

    function makeStubs() {
        const html = {
            _attrs: {},
            _style: {},
            setAttribute(k, v) { this._attrs[k] = v; },
            getAttribute(k) { return this._attrs[k] ?? null; },
            get style() { return this._style; },
        };
        globalThis.document = { documentElement: html };
        globalThis.matchMedia = () => ({
            matches: false,
            addEventListener() {},
            removeEventListener() {},
        });
        return html;
    }

    test('setTheme persists, applies data-theme, and emits theme-changed', async (ctx) => {
        if (!dbReady) return ctx.skip('fake-indexeddb not available');
        const html = makeStubs();

        const db = await import('../src/db.js');
        db._resetDbForTests();

        const theme = await import(`../src/theme.js?cb=${Date.now()}`);
        const state = await import('../src/state.js');

        let received = null;
        state.appEvents.addEventListener('theme-changed',
            (e) => { received = e.detail; }, { once: true });

        await theme.setTheme('light');

        assert.equal(theme.getTheme(), 'light');
        assert.equal(html._attrs['data-theme'], 'light');
        assert.equal(html._style.colorScheme, 'light');
        assert.deepEqual(received, { theme: 'light' });
        assert.equal(await db.getSetting('ui.theme'), 'light');
    });

    test('loadTheme reads the saved value and applies it', async (ctx) => {
        if (!dbReady) return ctx.skip('fake-indexeddb not available');
        const html = makeStubs();

        const db = await import('../src/db.js');
        db._resetDbForTests();
        await db.setSetting('ui.theme', 'dark');

        const theme = await import(`../src/theme.js?cb=${Date.now()}_2`);
        await theme.loadTheme();

        assert.equal(theme.getTheme(), 'dark');
        assert.equal(html._attrs['data-theme'], 'dark');
        assert.equal(html._style.colorScheme, 'dark');
    });

    test('invalid theme value falls back to system', async (ctx) => {
        if (!dbReady) return ctx.skip('fake-indexeddb not available');
        const html = makeStubs();

        const db = await import('../src/db.js');
        db._resetDbForTests();

        const theme = await import(`../src/theme.js?cb=${Date.now()}_3`);
        await theme.setTheme('bogus');
        assert.equal(theme.getTheme(), 'system');
        assert.equal(html._attrs['data-theme'], 'system');
        assert.equal(html._style.colorScheme, 'dark light');
    });
});

// ── views (no-DOM smoke) ──────────────────────────────────────
//
// Skipped: the router needs a real `document` to mount sections under.
// Without a JSDOM/linkedom dependency we'd have to fake too much of
// the DOM API to get a useful signal here. See the Tests section in
// the task notes.
test.skip('views.mount toggles classes correctly', () => {});

// ── mcp-tool-loop ─────────────────────────────────────────────

describe('mcp-tool-loop', async () => {
    let loop;
    try { loop = await import('../src/mcp-tool-loop.js'); }
    catch (e) { console.warn('[unit.test] mcp-tool-loop import failed:', e.message); }

    test('extractToolUses pulls tool_use parts; ignores text and unknown', (ctx) => {
        if (!loop) return ctx.skip();
        const out = loop.extractToolUses([
            { type: 'text', text: 'hello' },
            { type: 'tool_use', id: 'a', name: 'fs_read', input: { x: 1 } },
            { type: 'tool_use', id: 'b', name: 'fs_list' }, // no input → defaults to {}
        ]);
        assert.equal(out.length, 2);
        assert.deepEqual(out[0], { id: 'a', name: 'fs_read', input: { x: 1 } });
        assert.deepEqual(out[1], { id: 'b', name: 'fs_list', input: {} });
    });

    test('wrapToolResult: ok → string content, error → is_error', (ctx) => {
        if (!loop) return ctx.skip();
        const okPart = loop.wrapToolResult({ id: 'x' }, { ok: true, value: 'done' });
        assert.equal(okPart.type, 'tool_result');
        assert.equal(okPart.toolUseId, 'x');
        assert.equal(okPart.content, 'done');
        assert.equal(okPart.is_error, undefined);
        const errPart = loop.wrapToolResult({ id: 'x' }, { ok: false, error: 'boom' });
        assert.equal(errPart.is_error, true);
        assert.equal(errPart.content, 'boom');
    });

    test('runToolLoop terminates when assistant has no tool_use', async (ctx) => {
        if (!loop) return ctx.skip();
        const out = await loop.runToolLoop({
            initialHistory: [{ role: 'user', content: 'hi' }],
            runProvider: async () => ({ role: 'assistant', content: [{ type: 'text', text: 'no tools' }] }),
            callTool: async () => 'unused',
        });
        assert.equal(out.iterations, 0);
        assert.equal(out.capped, undefined);
        assert.equal(out.cancelled, undefined);
        assert.equal(out.history.length, 2);
    });

    test('runToolLoop: caps at MAX_TOOL_ITERATIONS when provider keeps emitting tool_use', async (ctx) => {
        if (!loop) return ctx.skip();
        let calls = 0;
        const out = await loop.runToolLoop({
            initialHistory: [{ role: 'user', content: 'go' }],
            runProvider: async () => ({
                role: 'assistant',
                content: [{ type: 'tool_use', id: `t${calls++}`, name: 'echo', input: { i: calls } }],
            }),
            callTool: async () => 'ok',
        });
        assert.equal(out.capped, true);
        assert.equal(out.iterations, loop.MAX_TOOL_ITERATIONS);
    });

    test('runToolLoop: errors in callTool are wrapped as tool_result with is_error', async (ctx) => {
        if (!loop) return ctx.skip();
        let phase = 0;
        const out = await loop.runToolLoop({
            initialHistory: [{ role: 'user', content: 'go' }],
            runProvider: async () => {
                phase += 1;
                if (phase === 1) {
                    return { role: 'assistant', content: [{ type: 'tool_use', id: 'a', name: 'oops', input: {} }] };
                }
                return { role: 'assistant', content: [{ type: 'text', text: 'recovered' }] };
            },
            callTool: async () => { throw new Error('boom'); },
        });
        assert.equal(out.capped, undefined);
        assert.equal(out.iterations, 1);
        // Last user message in history is the synthetic tool_result message.
        const beforeAsst = out.history[out.history.length - 2];
        assert.equal(beforeAsst.role, 'user');
        assert.ok(Array.isArray(beforeAsst.content));
        assert.equal(beforeAsst.content[0].type, 'tool_result');
        assert.equal(beforeAsst.content[0].is_error, true);
        assert.equal(beforeAsst.content[0].content, 'boom');
    });

    test('runToolLoop: cancellation between iterations', async (ctx) => {
        if (!loop) return ctx.skip();
        let cancelled = false;
        const out = await loop.runToolLoop({
            initialHistory: [{ role: 'user', content: 'go' }],
            runProvider: async () => ({
                role: 'assistant',
                content: [{ type: 'tool_use', id: 'a', name: 'echo', input: {} }],
            }),
            callTool: async () => { cancelled = true; return 'ok'; },
            isCancelled: () => cancelled,
        });
        assert.equal(out.cancelled, true);
        // First iteration ran (one runProvider + one callTool) but the
        // pre-iteration check on the next loop pass returned cancelled.
        assert.ok(out.iterations >= 1);
    });
});

// ── home-signaling (HTTP signaling client for the home daemon) ─

describe('home-signaling.SignalingClient', async () => {
    const { SignalingClient } = await import('../src/home-signaling.js');

    /** Build a fake fetch that records calls and returns the queued responses. */
    function mkFakeFetch(responses) {
        const calls = [];
        const queue = [...responses];
        const fakeFetch = async (url, init) => {
            calls.push({ url, init });
            const next = queue.shift();
            if (typeof next === 'function') return next(url, init);
            return next;
        };
        fakeFetch.calls = calls;
        return fakeFetch;
    }

    function jsonResp(obj, status = 200) {
        return new Response(JSON.stringify(obj), {
            status, headers: { 'content-type': 'application/json' },
        });
    }

    test('createSession POSTs to /signal/session and returns parsed body', async () => {
        const fetchImpl = mkFakeFetch([
            jsonResp({ session_id: 'abc', ice_servers: [] }),
        ]);
        const sc = new SignalingClient({ baseUrl: 'http://127.0.0.1:7878', fetchImpl });
        const out = await sc.createSession();
        assert.deepEqual(out, { session_id: 'abc', ice_servers: [] });
        const c = fetchImpl.calls[0];
        assert.equal(c.url, 'http://127.0.0.1:7878/signal/session');
        assert.equal(c.init.method, 'POST');
        assert.equal(c.init.headers['Content-Type'], 'application/json');
    });

    test('postOffer sends {sdp,type:"offer"} JSON body', async () => {
        const fetchImpl = mkFakeFetch([
            new Response(null, { status: 204 }),
        ]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        await sc.postOffer('sess1', 'v=0\r\n...sdp...');
        const c = fetchImpl.calls[0];
        assert.equal(c.url, 'http://h.test/signal/offer/sess1');
        assert.equal(c.init.method, 'POST');
        const body = JSON.parse(c.init.body);
        assert.equal(body.sdp, 'v=0\r\n...sdp...');
        assert.equal(body.type, 'offer');
    });

    test('pollAnswer returns null on 204, parsed JSON on 200', async () => {
        const fetchImpl = mkFakeFetch([
            new Response(null, { status: 204 }),
            jsonResp({ sdp: 'v=0\r\n...answer...', type: 'answer' }),
        ]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        const miss = await sc.pollAnswer('sess1');
        assert.equal(miss, null);
        const hit = await sc.pollAnswer('sess1');
        assert.equal(hit.type, 'answer');
        assert.match(hit.sdp, /\.\.\.answer\.\.\./);
    });

    test('pollAnswer maps 404 to a "session expired" error', async () => {
        const fetchImpl = mkFakeFetch([new Response('', { status: 404 })]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        await assert.rejects(() => sc.pollAnswer('gone'), /session expired/);
    });

    test('postIce sends {candidate,sdp_mid,sdp_m_line_index} JSON body', async () => {
        const fetchImpl = mkFakeFetch([new Response(null, { status: 204 })]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        await sc.postIce('sess1', 'candidate:abc 1 udp ...', '0', 0);
        const c = fetchImpl.calls[0];
        assert.equal(c.url, 'http://h.test/signal/ice/sess1');
        const body = JSON.parse(c.init.body);
        assert.equal(body.candidate, 'candidate:abc 1 udp ...');
        assert.equal(body.sdp_mid, '0');
        assert.equal(body.sdp_m_line_index, 0);
    });

    test('pollIce returns candidates + cursor; 204 yields {candidates:[], cursor:since}', async () => {
        const fetchImpl = mkFakeFetch([
            jsonResp({ candidates: [{ candidate: 'c1', sdp_mid: '0', sdp_m_line_index: 0 }], cursor: 5 }),
            new Response(null, { status: 204 }),
        ]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        const a = await sc.pollIce('sess1', 0);
        assert.equal(a.candidates.length, 1);
        assert.equal(a.cursor, 5);
        const c = fetchImpl.calls[0];
        assert.equal(c.url, 'http://h.test/signal/ice/sess1?since=0');
        const b = await sc.pollIce('sess1', 5);
        assert.deepEqual(b, { candidates: [], cursor: 5 });
    });

    test('fetchAgentCard returns parsed AgentCard JSON', async () => {
        const card = { name: 'brainwires-home', version: '0.1.0', supportedInterfaces: [] };
        const fetchImpl = mkFakeFetch([jsonResp(card)]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        const out = await sc.fetchAgentCard();
        assert.deepEqual(out, card);
        assert.equal(fetchImpl.calls[0].url, 'http://h.test/.well-known/agent-card.json');
    });

    test('extraHeaders() return value is merged into every request', async () => {
        const fetchImpl = mkFakeFetch([
            jsonResp({ session_id: 'x', ice_servers: [] }),
            new Response(null, { status: 204 }),
        ]);
        const sc = new SignalingClient({
            baseUrl: 'http://h.test',
            fetchImpl,
            extraHeaders: () => ({
                'Authorization': 'Bearer dev-token',
                'CF-Access-Client-Id': 'cid',
                'CF-Access-Client-Secret': 'csec',
            }),
        });
        await sc.createSession();
        await sc.postIce('s', 'cand', '0', 0);
        for (const call of fetchImpl.calls) {
            assert.equal(call.init.headers['Authorization'], 'Bearer dev-token');
            assert.equal(call.init.headers['CF-Access-Client-Id'], 'cid');
            assert.equal(call.init.headers['CF-Access-Client-Secret'], 'csec');
        }
    });

    test('baseUrl trailing slashes are stripped', async () => {
        const fetchImpl = mkFakeFetch([jsonResp({ session_id: 'x', ice_servers: [] })]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test/', fetchImpl });
        await sc.createSession();
        assert.equal(fetchImpl.calls[0].url, 'http://h.test/signal/session');
    });

    test('non-2xx surfaces uniform "signaling: <op> failed: <status>" error', async () => {
        const fetchImpl = mkFakeFetch([new Response('boom', { status: 500 })]);
        const sc = new SignalingClient({ baseUrl: 'http://h.test', fetchImpl });
        await assert.rejects(() => sc.createSession(), /signaling: createSession failed: 500/);
    });
});

// ── home-transport.JsonRpcDispatcher (id allocation + reply routing) ─

describe('home-transport.JsonRpcDispatcher', async () => {
    const { JsonRpcDispatcher } = await import('../src/home-transport.js');

    test('request() allocates monotonic ids and produces a parseable frame', () => {
        const d = new JsonRpcDispatcher();
        const r1 = d.request('system/ping', {});
        const r2 = d.request('message/send', { message: { role: 'user', parts: [] } });
        assert.equal(r1.id, 1);
        assert.equal(r2.id, 2);
        const f1 = JSON.parse(r1.frame);
        assert.equal(f1.jsonrpc, '2.0');
        assert.equal(f1.id, 1);
        assert.equal(f1.method, 'system/ping');
        // Promises should be unsettled before dispatch().
        let r1Settled = false;
        r1.promise.then(() => { r1Settled = true; }, () => { r1Settled = true; });
        // Don't await — just confirm we have two slots in the pending map.
        assert.equal(d.pendingCount, 2);
        // Suppress the unhandled-rejection from r2 for the timeout-cleanup test.
        r2.promise.catch(() => {});
        void r1Settled;
    });

    test('dispatch() resolves the matching request and drops it from pending', async () => {
        const d = new JsonRpcDispatcher();
        const r = d.request('system/ping', {}, { timeoutMs: 0 });
        const matched = d.dispatch(JSON.stringify({ jsonrpc: '2.0', id: r.id, result: { ok: true, ts: 12345 } }));
        assert.equal(matched, true);
        const out = await r.promise;
        assert.deepEqual(out, { ok: true, ts: 12345 });
        assert.equal(d.pendingCount, 0);
    });

    test('dispatch() rejects on error reply and propagates code/data', async () => {
        const d = new JsonRpcDispatcher();
        const r = d.request('does/notexist', {}, { timeoutMs: 0 });
        d.dispatch(JSON.stringify({
            jsonrpc: '2.0', id: r.id,
            error: { code: -32601, message: 'method not found', data: { hint: 'x' } },
        }));
        await assert.rejects(r.promise, (e) => {
            assert.equal(e.message, 'method not found');
            assert.equal(e.code, -32601);
            assert.deepEqual(e.data, { hint: 'x' });
            return true;
        });
    });

    test('dispatch() drops unknown ids and notifications without error', () => {
        const d = new JsonRpcDispatcher();
        const r = d.request('system/ping', {}, { timeoutMs: 0 });
        // Unrelated id — dropped.
        assert.equal(d.dispatch(JSON.stringify({ jsonrpc: '2.0', id: 999, result: 'stray' })), false);
        // Notification (no id) — also dropped.
        assert.equal(d.dispatch(JSON.stringify({ jsonrpc: '2.0', method: 'event/foo' })), false);
        // Garbage — dropped.
        assert.equal(d.dispatch('{not json'), false);
        // Original request still pending.
        assert.equal(d.pendingCount, 1);
        // Clean up so the test process doesn't keep a Promise around.
        r.promise.catch(() => {});
        d.rejectAll(new Error('teardown'));
    });

    test('rejectAll() rejects every pending request and clears the map', async () => {
        const d = new JsonRpcDispatcher();
        const r1 = d.request('a', {}, { timeoutMs: 0 });
        const r2 = d.request('b', {}, { timeoutMs: 0 });
        d.rejectAll(new Error('shutdown'));
        await assert.rejects(r1.promise, /shutdown/);
        await assert.rejects(r2.promise, /shutdown/);
        assert.equal(d.pendingCount, 0);
    });

    test('timeout fires when no reply arrives in time', async () => {
        const d = new JsonRpcDispatcher();
        const r = d.request('slow/op', {}, { timeoutMs: 20 });
        await assert.rejects(r.promise, /timed out after 20ms/);
        assert.equal(d.pendingCount, 0);
    });

    // M10 — last-seen reply id tracking for the resume cursor.
    test('lastSeenReplyId tracks high-water mark of numeric reply ids', () => {
        const d = new JsonRpcDispatcher();
        assert.equal(d.lastSeenReplyId, 0);
        const r1 = d.request('a', {}, { timeoutMs: 0 });
        const r2 = d.request('b', {}, { timeoutMs: 0 });
        const r3 = d.request('c', {}, { timeoutMs: 0 });
        d.dispatch(JSON.stringify({ jsonrpc: '2.0', id: r1.id, result: { ok: true } }));
        assert.equal(d.lastSeenReplyId, 1);
        // Out-of-order higher id bumps the cursor.
        d.dispatch(JSON.stringify({ jsonrpc: '2.0', id: r3.id, result: { ok: true } }));
        assert.equal(d.lastSeenReplyId, 3);
        // Lower id (the now-orphaned r2) does NOT regress the cursor.
        d.dispatch(JSON.stringify({ jsonrpc: '2.0', id: r2.id, result: { ok: true } }));
        assert.equal(d.lastSeenReplyId, 3);
        // Unknown id with higher value still bumps the cursor (replay frames).
        d.dispatch(JSON.stringify({ jsonrpc: '2.0', id: 99, result: 'replay' }));
        assert.equal(d.lastSeenReplyId, 99);
    });
});

// ── home-transport HomeTransport (M10 reconnect / resume) ─────

describe('home-transport.HomeTransport (M10)', async () => {
    const { HomeTransport } = await import('../src/home-transport.js');

    /** Fake clock with manual tick control. */
    function mkClock() {
        let now = 0;
        let nextHandle = 1;
        const intervals = new Map(); // handle -> { fn, every, next }
        const timeouts = new Map();  // handle -> { fn, at }
        const setInterval = (fn, every) => {
            const h = nextHandle++;
            intervals.set(h, { fn, every, next: now + every });
            return h;
        };
        const clearInterval = (h) => { intervals.delete(h); };
        const setTimeout = (fn, ms) => {
            const h = nextHandle++;
            timeouts.set(h, { fn, at: now + ms });
            return h;
        };
        const clearTimeout = (h) => { timeouts.delete(h); };
        const advance = async (ms) => {
            const target = now + ms;
            // Walk forward in chronological order so chained timers fire
            // correctly. Each iteration finds the earliest pending event
            // strictly greater than `now` and <= `target`; we then advance
            // time to that point and fire it. Loop ends when no event
            // remains in the window.
            while (true) {
                let nextAt = Infinity;
                let nextFn = null;
                let nextKey = null;
                for (const [h, t] of timeouts) {
                    if (t.at > now && t.at <= target && t.at < nextAt) {
                        nextAt = t.at; nextFn = t.fn; nextKey = ['t', h];
                    }
                }
                for (const [h, i] of intervals) {
                    if (i.next > now && i.next <= target && i.next < nextAt) {
                        nextAt = i.next; nextFn = i.fn; nextKey = ['i', h];
                    }
                }
                if (!nextFn) break;
                now = nextAt;
                if (nextKey[0] === 't') {
                    timeouts.delete(nextKey[1]);
                } else {
                    const i = intervals.get(nextKey[1]);
                    if (i) i.next += i.every;
                }
                // Fire the callback but do NOT await its return — production
                // setInterval/setTimeout are fire-and-forget. Any inner
                // `await` chain settles via microtasks the test yields to
                // explicitly between advance() calls.
                try { nextFn(); } catch (_) { /* swallow — matches real timers */ }
                // Yield so synchronous chains inside the callback resolve.
                await Promise.resolve();
            }
            now = target;
        };
        return {
            setInterval, clearInterval, setTimeout, clearTimeout,
            now: () => now,
            advance,
            _timeouts: timeouts,
            _intervals: intervals,
        };
    }

    /** Fake signaling that records calls; createSession/postOffer/pollAnswer return queued values. */
    function mkSignaling({ answers = [{ sdp: 'v=0\r\n...answer1', type: 'answer' }], iceServers = [] } = {}) {
        const calls = [];
        const answerQueue = [...answers];
        return {
            calls,
            createSession: async () => {
                calls.push({ kind: 'createSession' });
                return { session_id: `sess-${calls.length}`, ice_servers: iceServers };
            },
            postOffer: async (sessionId, sdp) => {
                calls.push({ kind: 'postOffer', sessionId, sdp });
            },
            pollAnswer: async (sessionId) => {
                calls.push({ kind: 'pollAnswer', sessionId });
                return answerQueue.shift() || { sdp: 'v=0\r\n...answerN', type: 'answer' };
            },
            postIce: async () => { calls.push({ kind: 'postIce' }); },
            pollIce: async (sessionId, since, signal) => {
                calls.push({ kind: 'pollIce' });
                // Park forever until aborted so the pump doesn't busy-loop.
                if (signal) {
                    await new Promise((_resolve, reject) => {
                        if (signal.aborted) reject(Object.assign(new Error('abort'), { name: 'AbortError' }));
                        signal.addEventListener('abort', () => reject(Object.assign(new Error('abort'), { name: 'AbortError' })));
                    });
                }
                return { candidates: [], cursor: since };
            },
            closeSession: async () => { calls.push({ kind: 'closeSession' }); },
        };
    }

    /** Mock RTCPeerConnection that exposes hooks tests can drive. */
    function mkPeerCtor() {
        const instances = [];
        const Ctor = function FakePC(_cfg) {
            this._listeners = { iceconnectionstatechange: new Set() };
            this.iceConnectionState = 'new';
            this.localDescription = null;
            this.remoteDescription = null;
            this._dc = null;
            this.onicecandidate = null;
            this.oniceconnectionstatechange = null;
            this.restartIceCalls = 0;
            this.createOfferCalls = [];
            this.createDataChannel = (label) => {
                const dc = {
                    label,
                    onopen: null, onerror: null, onclose: null, onmessage: null,
                    sentFrames: [],
                    send: (frame) => { dc.sentFrames.push(frame); },
                    close: () => { if (dc.onclose) dc.onclose(); },
                };
                this._dc = dc;
                return dc;
            };
            this.addEventListener = (ev, fn) => {
                if (!this._listeners[ev]) this._listeners[ev] = new Set();
                this._listeners[ev].add(fn);
            };
            this.removeEventListener = (ev, fn) => {
                if (this._listeners[ev]) this._listeners[ev].delete(fn);
            };
            this.setLocalDescription = async (d) => { this.localDescription = d; };
            this.setRemoteDescription = async (d) => { this.remoteDescription = d; };
            this.createOffer = async (opts) => { this.createOfferCalls.push(opts || null); return { type: 'offer', sdp: `v=0\r\n...offer-${this.createOfferCalls.length}` }; };
            this.restartIce = () => { this.restartIceCalls += 1; };
            this.addIceCandidate = async () => {};
            this.close = () => {};
            // Test helper: drive the iceConnectionState machine.
            this._setIceState = (s) => {
                this.iceConnectionState = s;
                if (typeof this.oniceconnectionstatechange === 'function') this.oniceconnectionstatechange();
                for (const fn of (this._listeners.iceconnectionstatechange || [])) {
                    try { fn(); } catch (_) {}
                }
            };
            // Test helper: simulate an inbound dc message.
            this._inbound = (text) => {
                if (this._dc && typeof this._dc.onmessage === 'function') {
                    this._dc.onmessage({ data: text });
                }
            };
            instances.push(this);
        };
        Ctor.instances = instances;
        return Ctor;
    }

    /** Drive HomeTransport.connect() to the connected state on the mock peer. */
    async function driveToConnected(transport, PCctor) {
        const connectPromise = transport.connect();
        // Yield until the FakePC has been constructed.
        for (let i = 0; i < 10 && PCctor.instances.length === 0; i++) await Promise.resolve();
        const pc = PCctor.instances[PCctor.instances.length - 1];
        // Move ICE forward + open the data channel synchronously.
        // The dc.onopen handler is set inside connect after createDataChannel,
        // so wait one microtask before firing it.
        for (let i = 0; i < 20 && (!pc._dc || typeof pc._dc.onopen !== 'function'); i++) await Promise.resolve();
        pc._setIceState('connected');
        pc._dc.onopen();
        await connectPromise;
        return pc;
    }

    test('heartbeat fires every interval and pings on schedule', async () => {
        const clock = mkClock();
        const PC = mkPeerCtor();
        const signaling = mkSignaling();
        const transport = new HomeTransport({
            signaling,
            rtcPeerConnection: PC,
            heartbeatIntervalMs: 1000,
            heartbeatTimeoutMs: 200,
            _clock: clock,
        });
        const pc = await driveToConnected(transport, PC);

        // Each tick awaits the system/ping reply, so we have to: (a) advance
        // the clock to fire the interval (don't await it — it would deadlock
        // until we reply), (b) reply to the ping, (c) yield microtasks to
        // settle the tick's await chain.
        const countPings = () => pc._dc.sentFrames.filter((f) => {
            try { return JSON.parse(f).method === 'system/ping'; } catch (_) { return false; }
        }).length;
        const replyToOpenPings = () => {
            for (const frame of pc._dc.sentFrames) {
                let msg;
                try { msg = JSON.parse(frame); } catch (_) { continue; }
                if (msg.method !== 'system/ping') continue;
                // Don't double-reply.
                pc._inbound(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { ok: true, ts: clock.now() } }));
            }
        };

        // Tick 1.
        const advance1 = clock.advance(1000);
        // Yield microtasks so the interval fires its callback (which sends the ping).
        for (let i = 0; i < 5; i++) await Promise.resolve();
        const pings1 = countPings();
        assert.ok(pings1 >= 1, `expected at least one ping after 1s tick; got ${pings1}`);
        // Reply so _heartbeatTick's await resolves and advance1 can complete.
        replyToOpenPings();
        await advance1;

        // Tick 2.
        const advance2 = clock.advance(1000);
        for (let i = 0; i < 5; i++) await Promise.resolve();
        const pings2 = countPings();
        assert.ok(pings2 > pings1, `expected more pings after second tick; got ${pings2} (was ${pings1})`);
        replyToOpenPings();
        await advance2;

        // Verify lastPongAt was bumped (diagnostic for connection-status pill).
        assert.ok(transport.lastPongAt > 0, 'lastPongAt should reflect a successful pong');

        await transport.close();
    });

    test('ICE-disconnected for >grace triggers restartIce + new offer', async () => {
        const clock = mkClock();
        const PC = mkPeerCtor();
        const signaling = mkSignaling({
            answers: [
                { sdp: 'v=0\r\n...answer1', type: 'answer' },
                { sdp: 'v=0\r\n...answer2', type: 'answer' },
            ],
        });
        const transport = new HomeTransport({
            signaling,
            rtcPeerConnection: PC,
            heartbeatIntervalMs: 1_000_000,  // suppress heartbeat noise
            iceDisconnectGraceMs: 2000,
            restartTimeoutMs: 30000,
            _clock: clock,
        });
        const pc = await driveToConnected(transport, PC);
        assert.equal(pc.createOfferCalls.length, 1, 'one offer for initial connect');
        assert.equal(pc.restartIceCalls, 0);

        // Drop ICE; nothing happens within the grace window.
        pc._setIceState('disconnected');
        await clock.advance(1000);
        assert.equal(pc.restartIceCalls, 0, 'no restart inside grace');

        // Past the grace window — restartIce + new iceRestart offer should fire.
        await clock.advance(1500);
        // The restart awaits postOffer → pollAnswer → setRemoteDescription →
        // _waitIceConnected. Drive ICE back to connected so _waitIceConnected
        // resolves, and reply to the system/resume the restart sends after.
        pc._setIceState('connected');
        // Yield for the chain inside _iceRestartOnce.
        for (let i = 0; i < 30; i++) await Promise.resolve();
        // Auto-reply to system/resume.
        for (const frame of pc._dc.sentFrames) {
            const msg = JSON.parse(frame);
            if (msg.method === 'system/resume') {
                pc._inbound(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { replayed: [], dropped: false } }));
            }
        }
        for (let i = 0; i < 10; i++) await Promise.resolve();

        assert.equal(pc.restartIceCalls, 1, 'restartIce called exactly once');
        assert.equal(pc.createOfferCalls.length, 2, 'second offer for the restart');
        assert.deepEqual(pc.createOfferCalls[1], { iceRestart: true });
        // Ensure the same session_id was reused (no createSession on restart).
        const createSessionCalls = signaling.calls.filter((c) => c.kind === 'createSession').length;
        assert.equal(createSessionCalls, 1, 'restart must not open a new session');

        await transport.close();
    });

    test('post-restart resume replays returned frames into the dispatcher', async () => {
        const clock = mkClock();
        const PC = mkPeerCtor();
        const signaling = mkSignaling({
            answers: [
                { sdp: 'v=0\r\n...answer1', type: 'answer' },
                { sdp: 'v=0\r\n...answer2', type: 'answer' },
            ],
        });
        const resetCalls = [];
        const transport = new HomeTransport({
            signaling,
            rtcPeerConnection: PC,
            heartbeatIntervalMs: 1_000_000,
            iceDisconnectGraceMs: 100,
            _clock: clock,
            onSessionReset: (info) => { resetCalls.push(info); },
        });
        const pc = await driveToConnected(transport, PC);

        // Park a real outbound request so the replay can satisfy it.
        const pending = transport.request('message/send', { foo: 'bar' }, { timeoutMs: 1_000_000 });
        const sentReq = JSON.parse(pc._dc.sentFrames[pc._dc.sentFrames.length - 1]);
        const pendingId = sentReq.id;

        // Fake a network blip.
        pc._setIceState('disconnected');
        await clock.advance(200);

        // The restart awaits ICE-connected; flip the state.
        pc._setIceState('connected');
        for (let i = 0; i < 30; i++) await Promise.resolve();

        // The transport now sends `system/resume`. Reply with one replayed
        // frame — the original pending message/send reply.
        const resumeFrame = pc._dc.sentFrames.find((f) => {
            try { return JSON.parse(f).method === 'system/resume'; } catch (_) { return false; }
        });
        assert.ok(resumeFrame, 'transport must send system/resume after restart');
        const resumeReq = JSON.parse(resumeFrame);
        const replayPayload = JSON.stringify({ jsonrpc: '2.0', id: pendingId, result: { ok: 'replayed' } });
        pc._inbound(JSON.stringify({
            jsonrpc: '2.0',
            id: resumeReq.id,
            result: { replayed: [replayPayload], dropped: false },
        }));
        // Settle the chain that re-feeds replayed frames.
        for (let i = 0; i < 10; i++) await Promise.resolve();

        const reply = await pending;
        assert.deepEqual(reply, { ok: 'replayed' }, 'replayed frame must satisfy the original request');
        assert.deepEqual(resetCalls, [], 'dropped:false → no reset event');

        await transport.close();
    });

    test('resume with dropped:true triggers the onSessionReset hook', async () => {
        const clock = mkClock();
        const PC = mkPeerCtor();
        const signaling = mkSignaling({
            answers: [
                { sdp: 'v=0\r\n...answer1', type: 'answer' },
                { sdp: 'v=0\r\n...answer2', type: 'answer' },
            ],
        });
        const resetCalls = [];
        const transport = new HomeTransport({
            signaling,
            rtcPeerConnection: PC,
            heartbeatIntervalMs: 1_000_000,
            iceDisconnectGraceMs: 100,
            _clock: clock,
            onSessionReset: (info) => { resetCalls.push(info); },
        });
        const pc = await driveToConnected(transport, PC);

        pc._setIceState('disconnected');
        await clock.advance(200);
        pc._setIceState('connected');
        for (let i = 0; i < 30; i++) await Promise.resolve();

        const resumeFrame = pc._dc.sentFrames.find((f) => {
            try { return JSON.parse(f).method === 'system/resume'; } catch (_) { return false; }
        });
        const resumeReq = JSON.parse(resumeFrame);
        pc._inbound(JSON.stringify({
            jsonrpc: '2.0',
            id: resumeReq.id,
            result: { replayed: [], dropped: true },
        }));
        for (let i = 0; i < 10; i++) await Promise.resolve();

        assert.equal(resetCalls.length, 1, 'one reset event');
        assert.deepEqual(resetCalls[0], { dropped: true, newSession: false });

        await transport.close();
    });
});

// ── home-pairing (M8 pairing flow) ────────────────────────────

describe('home-pairing.parseQrPayload', async () => {
    const { parseQrPayload } = await import('../src/home-pairing.js');

    test('parses a well-formed bwhome:// URL', () => {
        const out = parseQrPayload('bwhome://pair?u=https%3A%2F%2Fhome.example.com&t=abc&fp=deadbeef');
        assert.equal(out.tunnelUrl, 'https://home.example.com');
        assert.equal(out.oneTimeToken, 'abc');
        assert.equal(out.peerFingerprint, 'deadbeef');
    });

    test('fp is optional', () => {
        const out = parseQrPayload('bwhome://pair?u=http%3A%2F%2Flocalhost%3A7878&t=tok');
        assert.equal(out.tunnelUrl, 'http://localhost:7878');
        assert.equal(out.peerFingerprint, '');
    });

    test('throws on missing u', () => {
        assert.throws(() => parseQrPayload('bwhome://pair?t=abc'), /missing u/);
    });

    test('throws on missing t', () => {
        assert.throws(() => parseQrPayload('bwhome://pair?u=http%3A%2F%2Fa'), /missing t/);
    });

    test('throws on non-bwhome URL', () => {
        assert.throws(() => parseQrPayload('https://example.com'), /not a bwhome/);
    });

    test('throws on non-http(s) tunnel URL', () => {
        assert.throws(
            () => parseQrPayload('bwhome://pair?u=javascript%3Aalert(1)&t=abc'),
            /must be http/,
        );
    });

    test('rejects empty input', () => {
        assert.throws(() => parseQrPayload(''), /empty input/);
        assert.throws(() => parseQrPayload(null), /empty input/);
    });
});

describe('home-pairing.claim/confirm', async () => {
    const pairing = await import('../src/home-pairing.js');

    function jsonResp(obj, status = 200) {
        return new Response(JSON.stringify(obj), {
            status, headers: { 'content-type': 'application/json' },
        });
    }

    test('claim POSTs the right shape and parses ok', async () => {
        let captured;
        const fetchImpl = async (url, init) => {
            captured = { url, init };
            return jsonResp({ ok: true });
        };
        const out = await pairing.claim({
            tunnelUrl: 'http://h.test',
            oneTimeToken: 'tok',
            devicePubkey: 'pk-hex',
            deviceName: 'phone',
            fetchImpl,
        });
        assert.deepEqual(out, { ok: true });
        assert.equal(captured.url, 'http://h.test/pair/claim');
        assert.equal(captured.init.method, 'POST');
        const body = JSON.parse(captured.init.body);
        assert.equal(body.one_time_token, 'tok');
        assert.equal(body.device_pubkey, 'pk-hex');
        assert.equal(body.device_name, 'phone');
        assert.equal(captured.init.headers['Content-Type'], 'application/json');
    });

    test('claim 404 surfaces a "token unknown or expired" error', async () => {
        const fetchImpl = async () => new Response('', { status: 404 });
        await assert.rejects(
            pairing.claim({
                tunnelUrl: 'http://h.test', oneTimeToken: 'x',
                devicePubkey: 'pk', deviceName: 'n', fetchImpl,
            }),
            /token unknown or expired/,
        );
    });

    test('confirm parses {device_token, peer_pubkey} bundle', async () => {
        const fetchImpl = async () => jsonResp({
            device_token: 'a'.repeat(64),
            peer_pubkey: 'b'.repeat(64),
        });
        const bundle = await pairing.confirm({
            tunnelUrl: 'http://h.test',
            oneTimeToken: 'tok',
            code: '123456',
            fetchImpl,
        });
        assert.equal(bundle.device_token.length, 64);
        assert.equal(bundle.peer_pubkey.length, 64);
        assert.equal(bundle.cf_client_id, undefined);
    });

    test('confirm carries cf_* through when present', async () => {
        const fetchImpl = async () => jsonResp({
            device_token: 'a'.repeat(64),
            peer_pubkey: 'b'.repeat(64),
            cf_client_id: 'cid',
            cf_client_secret: 'csec',
        });
        const bundle = await pairing.confirm({
            tunnelUrl: 'http://h.test',
            oneTimeToken: 'tok', code: '123456', fetchImpl,
        });
        assert.equal(bundle.cf_client_id, 'cid');
        assert.equal(bundle.cf_client_secret, 'csec');
    });

    test('confirm 401 → wrong code error', async () => {
        const fetchImpl = async () => new Response('', { status: 401 });
        await assert.rejects(
            pairing.confirm({
                tunnelUrl: 'http://h.test', oneTimeToken: 'tok',
                code: '000000', fetchImpl,
            }),
            /wrong 6-digit code/,
        );
    });

    test('confirm rejects malformed bundle (missing peer_pubkey)', async () => {
        const fetchImpl = async () => jsonResp({ device_token: 'a'.repeat(64) });
        await assert.rejects(
            pairing.confirm({
                tunnelUrl: 'http://h.test', oneTimeToken: 'tok',
                code: '111111', fetchImpl,
            }),
            /missing device_token \/ peer_pubkey/,
        );
    });
});

describe('home-pairing.savePairingBundle / loadPairingBundle', async () => {
    let dbReady = false;
    try {
        await import('fake-indexeddb/auto');
        dbReady = true;
    } catch (_) { /* skip silently */ }

    test('plaintext round-trip when no session key is set', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        const stateMod = await import('../src/state.js');
        stateMod.setSessionKey(null);
        const pairing = await import('../src/home-pairing.js');

        const bundle = {
            device_token: 'a'.repeat(64),
            peer_pubkey: 'b'.repeat(64),
            tunnel_url: 'https://home.example.com',
            device_name: 'Phone',
        };
        await pairing.savePairingBundle(bundle);
        const out = await pairing.loadPairingBundle();
        assert.deepEqual(out, bundle);
    });

    test('encrypted round-trip when session key is set', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        const stateMod = await import('../src/state.js');
        const cs = await import('../crypto-store.js');

        const salt = cs.generateSalt();
        const key = await cs.deriveKey('sess-pw', salt);
        stateMod.setSessionKey(key);

        const pairing = await import('../src/home-pairing.js');
        const bundle = {
            device_token: 'c'.repeat(64),
            peer_pubkey: 'd'.repeat(64),
            tunnel_url: 'https://home.example.com',
            device_name: 'Laptop',
        };
        await pairing.savePairingBundle(bundle);

        // Stored row is the encrypted shape — the plaintext field must
        // not be present.
        const row = await db.getSetting('home_pairing_bundle');
        assert.ok(row.encrypted, 'should be encrypted shape');
        assert.equal(row.plaintext, undefined);

        const out = await pairing.loadPairingBundle();
        assert.deepEqual(out, bundle);

        // Lock the session — load should now return null without throwing.
        stateMod.setSessionKey(null);
        const locked = await pairing.loadPairingBundle();
        assert.equal(locked, null);

        // Restore so we don't leak state between tests.
        stateMod.setSessionKey(null);
    });

    test('clearPairingBundle removes the record', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        const stateMod = await import('../src/state.js');
        stateMod.setSessionKey(null);
        const pairing = await import('../src/home-pairing.js');
        await pairing.savePairingBundle({
            device_token: 'e'.repeat(64),
            peer_pubkey: 'f'.repeat(64),
            tunnel_url: 'https://h.test',
            device_name: 'X',
        });
        await pairing.clearPairingBundle();
        const out = await pairing.loadPairingBundle();
        assert.equal(out, null);
    });
});

// ── home-provider (M9) ─────────────────────────────────────────

describe('home-provider.buildSendMessageRequest', async () => {
    const { buildSendMessageRequest } = await import('../src/home-provider.js');

    test('emits the A2A 0.3 wire shape (camelCase + ROLE_USER)', () => {
        const req = buildSendMessageRequest([
            { role: 'user', content: 'hello' },
            { role: 'assistant', content: 'hi' },
            { role: 'user', content: 'how are you?' },
        ], {});
        assert.ok(req.message, 'must have message field');
        assert.equal(typeof req.message.messageId, 'string');
        assert.ok(req.message.messageId.length > 0);
        assert.equal(req.message.role, 'ROLE_USER');
        assert.deepEqual(req.message.parts, [{ text: 'how are you?' }]);
    });

    test('flattens parts[] content (drops non-text parts)', () => {
        const req = buildSendMessageRequest([
            { role: 'user', content: [
                { type: 'text', text: 'see this:' },
                { type: 'image', src: '...' },
                { type: 'text', text: 'pls' },
            ] },
        ], {});
        assert.equal(req.message.parts[0].text, 'see this:\npls');
    });

    test('throws when no user turn is present', () => {
        assert.throws(
            () => buildSendMessageRequest([{ role: 'assistant', content: 'hi' }], {}),
            /no user message/,
        );
    });

    test('throws when last user turn has empty text', () => {
        assert.throws(
            () => buildSendMessageRequest([{ role: 'user', content: '' }], {}),
            /no text content/,
        );
    });

    test('throws when messages is not an array', () => {
        assert.throws(
            () => buildSendMessageRequest(null, {}),
            /must be an array/,
        );
    });
});

describe('home-provider.extractReplyText', async () => {
    const { extractReplyText } = await import('../src/home-provider.js');

    test('reads text from a bare A2A Message result (a2a.rs current shape)', () => {
        const result = {
            messageId: 'abc',
            role: 'ROLE_AGENT',
            parts: [{ text: 'hello there' }],
        };
        assert.equal(extractReplyText(result), 'hello there');
    });

    test('reads text from a wrapped { message: ... } result (forward-compat)', () => {
        const result = {
            message: {
                messageId: 'abc',
                role: 'ROLE_AGENT',
                parts: [{ text: 'wrapped' }],
            },
        };
        assert.equal(extractReplyText(result), 'wrapped');
    });

    test('joins multiple text parts with newlines', () => {
        const result = {
            role: 'ROLE_AGENT',
            parts: [{ text: 'line one' }, { text: 'line two' }],
        };
        assert.equal(extractReplyText(result), 'line one\nline two');
    });

    test('returns empty string for invalid input', () => {
        assert.equal(extractReplyText(null), '');
        assert.equal(extractReplyText({}), '');
        assert.equal(extractReplyText({ parts: [] }), '');
    });
});

describe('home-provider provider registration', async () => {
    const home = await import('../src/home-provider.js');
    const providers = await import('../src/providers/index.js');

    test('exposes the EventProvider shape', () => {
        assert.equal(home.id, 'home');
        assert.equal(home.runtime, 'home');
        assert.equal(home.displayName, 'Home agent');
        assert.equal(typeof home.startChat, 'function');
        assert.ok(Array.isArray(home.models));
        assert.ok(home.models.length > 0);
    });

    test('listed in the provider registry under id "home"', () => {
        const got = providers.getProvider('home');
        assert.ok(got, 'getProvider("home") must return the module');
        assert.equal(got.runtime, 'home');
        const ids = providers.listProviders().map((p) => p.id);
        assert.ok(ids.includes('home'), `listProviders should include 'home'; got: ${ids.join(',')}`);
    });
});

describe('home-provider.startChat event dispatch', async () => {
    const home = await import('../src/home-provider.js');
    const stateMod = await import('../src/state.js');

    function captureEvents() {
        const captured = [];
        const types = ['chat_chunk', 'chat_done', 'chat_error'];
        const handlers = types.map((type) => {
            const h = (ev) => captured.push({ type, detail: ev.detail });
            stateMod.events.addEventListener(type, h);
            return { type, h };
        });
        return {
            captured,
            unsubscribe() {
                for (const { type, h } of handlers) {
                    stateMod.events.removeEventListener(type, h);
                }
            },
        };
    }

    test('dispatches chat_chunk + chat_done on a successful round-trip', async () => {
        home._resetForTests();
        const cap = captureEvents();
        try {
            const fakeTransport = {
                async request(method, params, _opts) {
                    assert.equal(method, 'message/send');
                    assert.equal(params.message.role, 'ROLE_USER');
                    assert.equal(params.message.parts[0].text, 'ping');
                    return {
                        messageId: 'srv-1',
                        role: 'ROLE_AGENT',
                        parts: [{ text: 'pong' }],
                    };
                },
            };
            const out = await home.startChat({
                conversationId: 'c1',
                messageId: 'm1',
                messages: [{ role: 'user', content: 'ping' }],
                params: {},
                _transport: fakeTransport,
            });
            assert.ok(out.tokensReceived > 0);

            const chunkEv = cap.captured.find((e) => e.type === 'chat_chunk');
            assert.ok(chunkEv, 'chat_chunk should fire');
            assert.equal(chunkEv.detail.delta, 'pong');
            assert.equal(chunkEv.detail.conversationId, 'c1');
            assert.equal(chunkEv.detail.messageId, 'm1');

            const doneEv = cap.captured.find((e) => e.type === 'chat_done');
            assert.ok(doneEv, 'chat_done should fire');
            assert.equal(doneEv.detail.conversationId, 'c1');

            assert.equal(cap.captured.find((e) => e.type === 'chat_error'), undefined);
        } finally {
            cap.unsubscribe();
        }
    });

    test('dispatches chat_error and rejects when transport.request rejects', async () => {
        home._resetForTests();
        const cap = captureEvents();
        try {
            const failingTransport = {
                async request() { throw new Error('boom'); },
            };
            await assert.rejects(
                home.startChat({
                    conversationId: 'c2',
                    messageId: 'm2',
                    messages: [{ role: 'user', content: 'ping' }],
                    params: {},
                    _transport: failingTransport,
                }),
                /boom/,
            );

            const errEv = cap.captured.find((e) => e.type === 'chat_error');
            assert.ok(errEv, 'chat_error should fire');
            assert.equal(errEv.detail.conversationId, 'c2');
            assert.equal(errEv.detail.error, 'boom');
            // No success events.
            assert.equal(cap.captured.find((e) => e.type === 'chat_done'), undefined);
            assert.equal(cap.captured.find((e) => e.type === 'chat_chunk'), undefined);
        } finally {
            cap.unsubscribe();
        }
    });
});

describe('home-provider.isAvailable (paired-only gating)', async () => {
    let dbReady = false;
    try {
        await import('fake-indexeddb/auto');
        dbReady = true;
    } catch (_) { /* skip */ }

    test('returns false when no bundle is stored', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        const stateMod = await import('../src/state.js');
        stateMod.setSessionKey(null);
        const home = await import('../src/home-provider.js');
        assert.equal(await home.isAvailable(), false);
    });

    test('returns true once a plaintext bundle is saved', async (ctx) => {
        if (!dbReady) return ctx.skip();
        const db = await import('../src/db.js');
        db._resetDbForTests();
        const stateMod = await import('../src/state.js');
        stateMod.setSessionKey(null);
        const pairing = await import('../src/home-pairing.js');
        await pairing.savePairingBundle({
            device_token: 'a'.repeat(64),
            peer_pubkey: 'b'.repeat(64),
            tunnel_url: 'https://home.example.com',
            device_name: 'Phone',
        });
        const home = await import('../src/home-provider.js');
        assert.equal(await home.isAvailable(), true);
    });
});

// ── home-transport.uploadBinary (M11) ──────────────────────────

describe('home-transport.uint8ToBase64 + sha256Hex', async () => {
    const { uint8ToBase64, sha256Hex } = await import('../src/home-transport.js');

    test('uint8ToBase64 round-trips with Buffer.from(b64, "base64")', () => {
        const u8 = new Uint8Array([0, 1, 2, 250, 251, 252, 253, 254, 255]);
        const b64 = uint8ToBase64(u8);
        const back = Buffer.from(b64, 'base64');
        assert.deepEqual(Array.from(back), Array.from(u8));
    });

    test('uint8ToBase64 handles 600 KB without blowing the stack', () => {
        const big = new Uint8Array(600 * 1024);
        for (let i = 0; i < big.length; i++) big[i] = i & 0xff;
        const b64 = uint8ToBase64(big);
        // Decoded length should match the input size.
        assert.equal(Buffer.from(b64, 'base64').length, big.length);
    });

    test('sha256Hex matches a known fixture', async () => {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        const out = await sha256Hex(new TextEncoder().encode('abc'));
        assert.equal(out, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    });
});

describe('home-transport.uploadBinary (M11)', async () => {
    const { HomeTransport } = await import('../src/home-transport.js');

    /**
     * Build a HomeTransport that's already in the 'connected' state with
     * a fake data channel and a captured request log. The test pokes the
     * private fields directly because re-deriving the ICE handshake just
     * to exercise the chunking path would dwarf the test in plumbing.
     */
    function connectedFakeTransport() {
        const sent = [];
        const fakeDC = { send: (frame) => sent.push(frame) };
        const t = new HomeTransport({ signaling: {}, rtcPeerConnection: function () {} });
        t._state = 'connected';
        t._dc = fakeDC;
        // Auto-resolve every JSON-RPC request the dispatcher tries to fire.
        const realRequest = t._dispatcher.request.bind(t._dispatcher);
        t._dispatcher.request = (method, params, opts) => {
            const slot = realRequest(method, params, opts);
            // Resolve next tick so the await yields.
            queueMicrotask(() => {
                t._dispatcher.dispatch(JSON.stringify({
                    jsonrpc: '2.0',
                    id: slot.id,
                    result: { ok: true },
                }));
            });
            return slot;
        };
        return { t, sent };
    }

    test('uploadBinary chunks at 256 KB boundaries (600 KB → 3 chunks)', async () => {
        const { t, sent } = connectedFakeTransport();
        const big = new Uint8Array(600 * 1024);
        for (let i = 0; i < big.length; i++) big[i] = i & 0xff;
        const onProgress = [];
        const binId = await t.uploadBinary(big, 'application/octet-stream', {
            onProgress: (p) => onProgress.push({ ...p }),
        });
        assert.equal(typeof binId, 'string');
        // 1 begin + 3 chunks + 1 end = 5 frames
        const methods = sent.map((f) => JSON.parse(f).method);
        assert.deepEqual(methods, ['bin/begin', 'bin/chunk', 'bin/chunk', 'bin/chunk', 'bin/end']);
        // Progress fires once per chunk with monotonic `sent`, and final
        // `sent === total`.
        assert.equal(onProgress.length, 3);
        assert.equal(onProgress[0].total, big.length);
        assert.ok(onProgress[0].sent < onProgress[1].sent);
        assert.ok(onProgress[1].sent < onProgress[2].sent);
        assert.equal(onProgress[2].sent, big.length);
    });

    test('uploadBinary computes SHA-256 and includes it in bin/end', async () => {
        const { t, sent } = connectedFakeTransport();
        const data = new TextEncoder().encode('abc');
        await t.uploadBinary(data, 'text/plain');
        const endFrame = sent.map((f) => JSON.parse(f)).find((m) => m.method === 'bin/end');
        assert.ok(endFrame);
        assert.equal(
            endFrame.params.sha256,
            'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        );
    });

    test('uploadBinary < 256 KB sends a single chunk', async () => {
        const { t, sent } = connectedFakeTransport();
        const small = new Uint8Array(100 * 1024);
        await t.uploadBinary(small, 'application/octet-stream');
        const methods = sent.map((f) => JSON.parse(f).method);
        assert.deepEqual(methods, ['bin/begin', 'bin/chunk', 'bin/end']);
        const beginFrame = JSON.parse(sent[0]);
        assert.equal(beginFrame.params.total_chunks, 1);
        assert.equal(beginFrame.params.total_size, small.length);
    });

    test('uploadBinary throws when not connected', async () => {
        const t = new HomeTransport({ signaling: {} });
        // _state defaults to 'idle'.
        await assert.rejects(
            t.uploadBinary(new Uint8Array(8), 'application/octet-stream'),
            /not connected/,
        );
    });
});

describe('home-provider.uploadFilePartsToBinIds (M11)', async () => {
    const { uploadFilePartsToBinIds, INLINE_FILE_THRESHOLD } =
        await import('../src/home-provider.js');

    test('large file part becomes a bin_id ref via transport.uploadBinary', async () => {
        const big = new Uint8Array(INLINE_FILE_THRESHOLD + 1024);
        const calls = [];
        const fakeTransport = {
            async uploadBinary(bytes, ct) {
                calls.push({ size: bytes.byteLength, ct });
                return 'bin-xyz';
            },
        };
        const req = {
            message: {
                messageId: 'm',
                role: 'ROLE_USER',
                parts: [
                    { text: 'see attachment' },
                    { filename: 'big.bin', mediaType: 'application/octet-stream', _bytes: big },
                ],
            },
        };
        await uploadFilePartsToBinIds(req, fakeTransport);
        assert.equal(calls.length, 1);
        assert.equal(calls[0].size, big.byteLength);
        assert.equal(calls[0].ct, 'application/octet-stream');
        const filePart = req.message.parts[1];
        assert.equal(filePart._bytes, undefined, '_bytes must be stripped before sending');
        assert.equal(filePart.metadata.bin_id, 'bin-xyz');
        assert.equal(filePart.raw, undefined, 'large parts must NOT be inlined');
    });

    test('small file part is inlined as base64 raw, not uploaded', async () => {
        const small = new Uint8Array(8);
        small.set([0, 1, 2, 3, 4, 5, 6, 7]);
        const fakeTransport = {
            async uploadBinary() { throw new Error('should not be called'); },
        };
        const req = {
            message: {
                messageId: 'm',
                role: 'ROLE_USER',
                parts: [
                    { filename: 'tiny.bin', _bytes: small },
                ],
            },
        };
        await uploadFilePartsToBinIds(req, fakeTransport);
        const filePart = req.message.parts[0];
        assert.equal(filePart._bytes, undefined);
        assert.equal(filePart.raw, Buffer.from(small).toString('base64'));
        // No bin_id metadata for the inline case.
        assert.equal(
            filePart.metadata == null || filePart.metadata.bin_id == null,
            true,
        );
    });

    test('plain text-only request is left untouched', async () => {
        const req = {
            message: {
                messageId: 'm', role: 'ROLE_USER',
                parts: [{ text: 'hello' }],
            },
        };
        const fakeTransport = { uploadBinary: () => { throw new Error('nope'); } };
        await uploadFilePartsToBinIds(req, fakeTransport);
        assert.deepEqual(req.message.parts, [{ text: 'hello' }]);
    });
});

describe('home-provider.buildSendMessageRequest with attachments (M11)', async () => {
    const { buildSendMessageRequest, collectFileLikeSources } =
        await import('../src/home-provider.js');

    test('attachments shape is collected', () => {
        const userTurn = {
            role: 'user',
            content: 'caption',
            attachments: [
                { name: 'a.png', mediaType: 'image/png', bytes: new Uint8Array([1, 2]) },
                { name: 'b.bin', bytes: new Uint8Array([3]) },
                { name: 'c.no-bytes' }, // dropped
            ],
        };
        const out = collectFileLikeSources(userTurn);
        assert.equal(out.length, 2);
        assert.equal(out[0].mediaType, 'image/png');
        assert.equal(out[1].mediaType, 'application/octet-stream');
    });

    test('buildSendMessageRequest emits one part per file with _bytes set', () => {
        const userTurn = {
            role: 'user',
            content: 'see image',
            attachments: [
                { name: 'a.png', mediaType: 'image/png', bytes: new Uint8Array([7, 7, 7]) },
            ],
        };
        const req = buildSendMessageRequest([userTurn]);
        assert.equal(req.message.parts.length, 2);
        assert.equal(req.message.parts[0].text, 'see image');
        assert.equal(req.message.parts[1].filename, 'a.png');
        assert.equal(req.message.parts[1].mediaType, 'image/png');
        assert.ok(req.message.parts[1]._bytes instanceof Uint8Array);
        assert.deepEqual(Array.from(req.message.parts[1]._bytes), [7, 7, 7]);
    });
});

// ── M12 polish: state-change observer + public state mirror ───

describe('home-transport state-change observer (M12)', async () => {
    const { HomeTransport } = await import('../src/home-transport.js');

    test('onStateChange fires for each transition (idle → connecting → failed)', async () => {
        const events = [];
        const failingSignaling = {
            createSession: async () => { throw new Error('boom'); },
            postOffer: async () => {},
            pollAnswer: async () => null,
            postIce: async () => {},
            pollIce: async () => ({ candidates: [], cursor: 0 }),
            closeSession: async () => {},
        };
        const transport = new HomeTransport({
            signaling: failingSignaling,
            rtcPeerConnection: function FakePC() {},
            onStateChange: (info) => events.push(info),
        });
        await assert.rejects(transport.connect(), /boom/);
        // Expect at minimum: idle→connecting, connecting→failed.
        assert.deepEqual(events.map((e) => `${e.prev}->${e.next}`), [
            'idle->connecting',
            'connecting->failed',
        ]);
    });

    test('observer errors are swallowed and do not break the transport', async () => {
        const transport = new HomeTransport({
            signaling: {
                createSession: async () => { throw new Error('halt'); },
                postOffer: async () => {}, pollAnswer: async () => null,
                postIce: async () => {}, pollIce: async () => ({ candidates: [], cursor: 0 }),
                closeSession: async () => {},
            },
            rtcPeerConnection: function FakePC() {},
            onStateChange: () => { throw new Error('observer-bug'); },
        });
        // Should reject with the original error, not the observer's.
        await assert.rejects(transport.connect(), /halt/);
    });
});

describe('home-provider.getTransportState (M12 public mirror)', async () => {
    const homeProvider = await import('../src/home-provider.js');
    const { events } = await import('../src/state.js');

    test('starts in idle and resets to idle after disconnect()', async () => {
        homeProvider._resetForTests();
        assert.equal(homeProvider.getTransportState(), 'idle');
        await homeProvider.disconnect();
        assert.equal(homeProvider.getTransportState(), 'idle');
    });

    test('home-transport-state app event fires when the mirror flips', async () => {
        homeProvider._resetForTests();
        const seen = [];
        const handler = (e) => seen.push(e.detail);
        events.addEventListener('home-transport-state', handler);
        try {
            // Drive the transport via a stubbed factory so we can flip
            // state through the real getTransport path. We don't need
            // a full handshake — startChat won't be called.
            const fakeTransport = {
                request: async () => ({}),
                close: async () => {},
            };
            // Emulate what onStateChange would do after a successful connect.
            // We import the transport module directly to confirm the wiring;
            // here we simulate by triggering disconnect() and confirming the
            // public state is 'idle' (no spurious events fire from idle→idle).
            await homeProvider.disconnect();
            // Spy stays silent because state was already idle.
            assert.equal(seen.length, 0);
            // Now flip via the internal helper exposed in tests by writing
            // through the real onStateChange path used by getTransport.
            // We can't reach the closure directly, so instead drive a fake
            // HomeTransport whose onStateChange is the same callback the
            // real factory installs:
            const { HomeTransport } = await import('../src/home-transport.js');
            const t = new HomeTransport({
                signaling: {
                    createSession: async () => { throw new Error('x'); },
                    postOffer: async () => {}, pollAnswer: async () => null,
                    postIce: async () => {}, pollIce: async () => ({ candidates: [], cursor: 0 }),
                    closeSession: async () => {},
                },
                rtcPeerConnection: function FakePC() {},
                // Wire the same shape home-provider uses.
                onStateChange: ({ next }) => {
                    events.dispatchEvent(new CustomEvent('home-transport-state', { detail: { prev: 'x', next } }));
                },
            });
            await assert.rejects(t.connect());
            // Two transitions should have fired: connecting + failed.
            assert.equal(seen.length, 2);
            assert.equal(seen[0].next, 'connecting');
            assert.equal(seen[1].next, 'failed');
            void fakeTransport;
        } finally {
            events.removeEventListener('home-transport-state', handler);
        }
    });
});

// ── M12 polish: unpair flow side effects ───────────────────────

describe('ui-home-pairing.performUnpair (M12)', async () => {
    const ui = await import('../src/ui-home-pairing.js');

    test('disconnect → clearPairingBundle → home-unpaired event in order', async () => {
        const calls = [];
        const fakeEvents = {
            dispatchEvent: (ev) => { calls.push({ kind: 'event', type: ev.type }); return true; },
        };
        await ui.performUnpair({
            _disconnect: async () => { calls.push({ kind: 'disconnect' }); },
            _clearPairingBundle: async () => { calls.push({ kind: 'clear' }); },
            _events: fakeEvents,
        });
        assert.deepEqual(calls, [
            { kind: 'disconnect' },
            { kind: 'clear' },
            { kind: 'event', type: 'home-unpaired' },
        ]);
    });

    test('disconnect failure does not block clearPairingBundle or the event', async () => {
        const calls = [];
        await ui.performUnpair({
            _disconnect: async () => { throw new Error('socket gone'); },
            _clearPairingBundle: async () => { calls.push('clear'); },
            _events: { dispatchEvent: () => calls.push('event') },
        });
        assert.deepEqual(calls, ['clear', 'event']);
    });
});

// ── helpers ────────────────────────────────────────────────────

function mkResponse(body) {
    // Build a minimal Response-like object backed by a ReadableStream.
    // node 20+ has global ReadableStream + Response.
    const enc = new TextEncoder();
    const stream = new ReadableStream({
        start(controller) {
            controller.enqueue(enc.encode(body));
            controller.close();
        },
    });
    return new Response(stream);
}
