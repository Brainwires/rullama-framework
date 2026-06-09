// Generates `tool-call-app-intents.jsonl` — the canonical function-calling
// fine-tune dataset. Each line is `{prompt, completion}` where the completion
// is a single tool call in the EXACT wire format the chat renderer parses:
//
//     <tool_call>{"name":"...","arguments":{...}}</tool_call>
//
// This is the SAME contract defined in web/src/lib/toolFormat.ts. Training the
// adapter to emit this shape is what makes Gemma 4 e2b a reliable function
// caller for this fixed tool set — the renderer then surfaces the call as a
// structured block. Deterministic (no RNG) so the dataset is reproducible.
//
// Run:  node gen-tool-calls.mjs > tool-call-app-intents.jsonl

const OPEN = "<tool_call>";
const CLOSE = "</tool_call>";

/** One tool call → the exact completion string the renderer expects. */
function call(name, args) {
    return `${OPEN}${JSON.stringify({ name, arguments: args })}${CLOSE}`;
}

const rows = [];
function add(prompt, name, args) {
    rows.push({ prompt, completion: call(name, args) });
}

// ---- set_timer -------------------------------------------------------------
for (const [n, unit, mins] of [
    [5, "minutes", 5], [10, "minutes", 10], [20, "minutes", 20],
    [1, "minute", 1], [2, "minutes", 2], [30, "minutes", 30],
    [45, "minutes", 45], [3, "minutes", 3], [15, "minutes", 15],
]) {
    add(`Set a timer for ${n} ${unit}.`, "set_timer", { duration_minutes: mins });
    add(`Start a ${n}-${unit.replace(/s$/, "")} timer.`, "set_timer", { duration_minutes: mins });
}
add("Can you set a timer for half an hour?", "set_timer", { duration_minutes: 30 });
add("Timer for an hour please.", "set_timer", { duration_minutes: 60 });

// ---- get_weather -----------------------------------------------------------
for (const city of ["Tokyo", "Paris", "London", "New York", "Berlin", "Austin", "Seattle", "Cairo", "Sydney", "Toronto"]) {
    add(`What's the weather in ${city}?`, "get_weather", { location: city });
}
add("Will it rain in Boston today?", "get_weather", { location: "Boston" });
add("How hot is it in Phoenix right now?", "get_weather", { location: "Phoenix" });
add("Give me the forecast for Denver.", "get_weather", { location: "Denver" });

// ---- send_email ------------------------------------------------------------
for (const [to, subj] of [
    ["alice", "lunch tomorrow"], ["bob", "the quarterly report"],
    ["mom", "weekend plans"], ["the team", "Friday's release"],
    ["carlos", "project update"], ["dana", "interview schedule"],
]) {
    add(`Email ${to} about ${subj}.`, "send_email", { to, subject: subj });
    add(`Send an email to ${to} regarding ${subj}.`, "send_email", { to, subject: subj });
}

// ---- add_calendar_event ----------------------------------------------------
for (const [title, date] of [
    ["dentist appointment", "Monday"], ["team standup", "tomorrow at 9am"],
    ["birthday party", "Saturday"], ["flight to Chicago", "next Tuesday"],
    ["1:1 with Sam", "Thursday afternoon"], ["yoga class", "Wednesday evening"],
]) {
    add(`Add ${title} to my calendar for ${date}.`, "add_calendar_event", { title, date });
    add(`Schedule ${title} ${date.startsWith("next") || date.startsWith("tomorrow") ? "" : "on "}${date}.`.replace(/\s+/g, " ").trim() + "", "add_calendar_event", { title, date });
}

// ---- play_music ------------------------------------------------------------
for (const q of [
    "Bohemian Rhapsody", "some jazz", "the latest Taylor Swift album",
    "lo-fi study beats", "my workout playlist", "Beethoven's 9th",
    "something relaxing", "Daft Punk",
]) {
    add(`Play ${q}.`, "play_music", { query: q });
    add(`Can you put on ${q}?`, "play_music", { query: q });
}

// ---- set_reminder ----------------------------------------------------------
for (const [text, time] of [
    ["call the doctor", "3pm"], ["take out the trash", "tonight"],
    ["buy milk", "tomorrow morning"], ["pay rent", "the 1st"],
    ["water the plants", "this evening"], ["submit the form", "by noon"],
]) {
    add(`Remind me to ${text} at ${time}.`, "set_reminder", { text, time });
    add(`Set a reminder to ${text} ${time}.`, "set_reminder", { text, time });
}

for (const r of rows) process.stdout.write(JSON.stringify(r) + "\n");
process.stderr.write(`generated ${rows.length} examples\n`);
