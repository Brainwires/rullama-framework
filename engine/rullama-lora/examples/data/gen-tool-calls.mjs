// Generates `tool-call-app-intents.jsonl` — the canonical function-calling
// fine-tune dataset. Each line is `{prompt, completion}` where the completion
// is a single tool call in the EXACT wire format the chat renderer parses:
//
//     <tool_call>{"name":"...","arguments":{...}}</tool_call>
//
// Same contract as web/src/lib/toolFormat.ts (and the schema in
// tool-schema.txt, which is prepended as a System turn at train + infer time).
//
// Design (Phase 2 — drive eval toward 0 fails):
//   - ~350 examples, heavy variety, ONE consistent key set per tool so the
//     model learns "extract value -> THIS key" instead of memorizing values.
//   - Multi-slot tools (send_email, add_calendar_event) are OVER-represented —
//     they were the v2 failure mode.
//   - `schedule`/`put on my calendar` phrasings all map to add_calendar_event,
//     to kill v2's wrong-tool-name (`schedule`) failure.
//   - A few values are RESERVED (held out) for eval — see HELDOUT below; the
//     generator never emits them, so eval prompts test generalization.
//   - Deterministic (no RNG): templates are cycled by index.
//
// Run:  node gen-tool-calls.mjs > tool-call-app-intents.jsonl

const OPEN = "<tool_call>";
const CLOSE = "</tool_call>";
const call = (name, args) => `${OPEN}${JSON.stringify({ name, arguments: args })}${CLOSE}`;

const rows = [];
const add = (prompt, name, args) => rows.push({ prompt, completion: call(name, args) });
// cycle a template list deterministically by index
const pick = (arr, i) => arr[i % arr.length];

// Values deliberately kept OUT of training, reserved for held-out eval:
//   timer 7 · city "Miami" · email Priya/"budget review" · "classical music"
//   reminder "call grandma"/"tonight" · calendar "root canal"/"next Friday"

// ── set_timer(duration_minutes) ────────────────────────────────────────────
{
    const mins = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 60, 75, 90];
    const tpl = [
        (n) => `Set a timer for ${n} minutes.`,
        (n) => `Start a ${n}-minute timer.`,
        (n) => `Timer for ${n} minutes, please.`,
        (n) => `Can you set a ${n} minute timer?`,
        (n) => `Give me a ${n} minute timer.`,
    ];
    // Natural-language duration (copy the user's wording — no unit conversion;
    // small models mangle "30 seconds" when forced into a minutes field).
    const mlabel = (n) => `${n} ${n === 1 ? "minute" : "minutes"}`;
    mins.forEach((n, i) => add(pick(tpl, i)(n), "set_timer", { duration: mlabel(n) }));
    add("Set a timer for 30 seconds.", "set_timer", { duration: "30 seconds" });
    add("Set a timer for 90 seconds.", "set_timer", { duration: "90 seconds" });
    add("Set a timer for half an hour.", "set_timer", { duration: "half an hour" });
    add("Set a timer for an hour.", "set_timer", { duration: "an hour" });
}

// ── get_weather(location) ──────────────────────────────────────────────────
{
    const cities = ["Tokyo", "Paris", "London", "New York", "Berlin", "Austin",
        "Seattle", "Cairo", "Sydney", "Toronto", "Boston", "Denver", "Chicago",
        "Madrid", "Rome", "Dubai", "Mumbai", "Lagos", "Oslo", "Lima",
        "Bangkok", "Vienna", "Dublin", "Houston", "Portland"];
    const tpl = [
        (c) => `What's the weather in ${c}?`,
        (c) => `What's the weather like in ${c}?`,
        (c) => `How's the weather in ${c} today?`,
        (c) => `Will it rain in ${c} today?`,
        (c) => `What's the forecast for ${c}?`,
        (c) => `Tell me the weather in ${c}.`,
    ];
    cities.forEach((c, i) => add(pick(tpl, i)(c), "get_weather", { location: c }));
}

// ── send_email(to, subject) ── OVER-REPRESENTED (2-slot) ───────────────────
{
    const to = ["Alice", "Bob", "Carlos", "Dana", "Sarah", "Tom", "mom", "my manager",
        "the team", "the client", "Dr. Lee", "the landlord", "Jordan", "Priyanka", "Wei"];
    const subj = ["lunch tomorrow", "the quarterly report", "weekend plans",
        "Friday's release", "the project update", "the interview schedule",
        "the invoice", "the meeting notes", "vacation dates", "the contract",
        "the design review", "the new hire", "next week's agenda", "the refund",
        "the schedule change"];
    const tpl = [
        (t, s) => `Email ${t} about ${s}.`,
        (t, s) => `Send an email to ${t} about ${s}.`,
        (t, s) => `Send ${t} an email regarding ${s}.`,
        (t, s) => `Write an email to ${t} about ${s}.`,
        (t, s) => `Can you email ${t} about ${s}?`,
        (t, s) => `Draft an email to ${t} about ${s}.`,
    ];
    // pair each recipient with several subjects (offset so pairs vary)
    let k = 0;
    for (let i = 0; i < to.length; i++) {
        for (let j = 0; j < 5; j++) {
            const s = subj[(i * 5 + j) % subj.length];
            add(pick(tpl, k)(to[i], s), "send_email", { to: to[i], subject: s });
            k++;
        }
    }
}

// ── add_calendar_event(title, date) ── OVER-REPRESENTED (2-slot) ───────────
{
    const titles = ["dentist appointment", "team standup", "birthday party",
        "flight to Chicago", "1:1 with Sam", "yoga class", "doctor's appointment",
        "lunch with Maria", "project deadline", "parent-teacher conference",
        "gym session", "car service", "haircut", "board meeting", "vet visit"];
    const dates = ["Monday", "tomorrow", "Saturday", "next Tuesday", "Friday",
        "this Thursday", "June 20th", "next week", "Wednesday morning",
        "Sunday afternoon", "the 15th", "tomorrow at 2pm", "Thursday", "Monday at 9am"];
    // varied verbs — ALL map to add_calendar_event (kills v2's `schedule` name bug)
    const tpl = [
        (t, d) => `Add ${t} to my calendar for ${d}.`,
        (t, d) => `Schedule ${t} on ${d}.`,
        (t, d) => `Put ${t} on my calendar for ${d}.`,
        (t, d) => `Create a calendar event for ${t} on ${d}.`,
        (t, d) => `Add an event: ${t} on ${d}.`,
        (t, d) => `Book ${t} for ${d}.`,
    ];
    let k = 0;
    for (let i = 0; i < titles.length; i++) {
        for (let j = 0; j < 5; j++) {
            const d = dates[(i * 5 + j) % dates.length];
            add(pick(tpl, k)(titles[i], d), "add_calendar_event", { title: titles[i], date: d });
            k++;
        }
    }
}

// ── play_music(query) ──────────────────────────────────────────────────────
{
    const q = ["Bohemian Rhapsody", "some jazz", "the latest Taylor Swift album",
        "lo-fi beats", "my workout playlist", "Beethoven's 9th", "something relaxing",
        "Daft Punk", "the Beatles", "90s hip hop", "rain sounds", "Miles Davis",
        "a focus playlist", "Adele", "country music", "the new Drake song",
        "some blues", "my road trip playlist", "Mozart", "synthwave", "Pink Floyd",
        "ambient music", "Kendrick Lamar", "a piano playlist", "reggae", "Fleetwood Mac"];
    const tpl = [
        (x) => `Play ${x}.`,
        (x) => `Can you play ${x}?`,
        (x) => `Put on ${x}.`,
        (x) => `I want to listen to ${x}.`,
        (x) => `Start playing ${x}.`,
    ];
    q.forEach((x, i) => add(pick(tpl, i)(x), "play_music", { query: x }));
}

// ── set_reminder(text, time) ── 2-slot ─────────────────────────────────────
{
    const text = ["call the doctor", "take out the trash", "buy milk", "pay rent",
        "water the plants", "submit the form", "pick up the kids", "send the invoice",
        "take my medication", "feed the cat", "book the flight", "renew the license",
        "back up my laptop", "stretch", "drink water"];
    const time = ["at 3pm", "tonight", "tomorrow morning", "on the 1st", "this evening",
        "by noon", "at 9am", "in an hour", "tomorrow", "at 5", "after lunch", "before bed"];
    const tpl = [
        (t, w) => `Remind me to ${t} ${w}.`,
        (t, w) => `Set a reminder to ${t} ${w}.`,
        (t, w) => `Can you remind me to ${t} ${w}?`,
        (t, w) => `Reminder: ${t} ${w}.`,
    ];
    let k = 0;
    for (let i = 0; i < text.length; i++) {
        for (let j = 0; j < 3; j++) {
            const w = time[(i * 3 + j) % time.length];
            add(pick(tpl, k)(text[i], w), "set_reminder", { text: text[i], time: w });
            k++;
        }
    }
}

for (const r of rows) process.stdout.write(JSON.stringify(r) + "\n");
const counts = {};
for (const r of rows) { const n = JSON.parse(r.completion.slice(OPEN.length, -CLOSE.length)).name; counts[n] = (counts[n] || 0) + 1; }
process.stderr.write(`generated ${rows.length} examples: ${JSON.stringify(counts)}\n`);
