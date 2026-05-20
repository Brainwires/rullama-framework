import { assert, assertEquals } from "@std/assert";
import {
  type ExpectedBehavior,
  evaluate,
  type Fixture,
  FixtureCase,
  type FixtureRunner,
  loadFixturesFromDir,
  type RunOutcome,
} from "./fixtures.ts";

function happyOutcome(seq: string[], text: string): RunOutcome {
  return {
    output_text: text,
    tool_sequence: seq.slice(),
    finish_reason: "end_turn",
    duration_ms: 5,
  };
}

Deno.test("evaluate passes when all assertions hold", () => {
  const expected: ExpectedBehavior = {
    tool_sequence: ["read_file", "edit_file"],
    assertions: [
      { contains: "fn bar" },
      { tool_called: "edit_file" },
      { finish_reason: "end_turn" },
    ],
  };
  const outcome = happyOutcome(
    ["read_file", "edit_file"],
    "updated: fn bar() {}",
  );
  assertEquals(evaluate(expected, outcome), null);
});

Deno.test("evaluate fails on tool sequence mismatch", () => {
  const expected: ExpectedBehavior = {
    tool_sequence: ["read_file", "edit_file"],
    assertions: [],
  };
  const outcome = happyOutcome(["edit_file"], "");
  const err = evaluate(expected, outcome);
  assert(err !== null);
  assert(err.includes("tool_sequence mismatch"));
});

Deno.test("evaluate fails on missing substring", () => {
  const expected: ExpectedBehavior = {
    tool_sequence: [],
    assertions: [{ contains: "bar" }],
  };
  const outcome = happyOutcome([], "only foo here");
  assert(evaluate(expected, outcome) !== null);
});

Deno.test("evaluate regex assertion", () => {
  const expected: ExpectedBehavior = {
    tool_sequence: [],
    assertions: [{ regex: "^updated:" }],
  };
  const outcome = happyOutcome([], "updated: ok");
  assertEquals(evaluate(expected, outcome), null);
});

Deno.test("load fixtures from tmpdir in sorted order", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const a = `
name: aa
category: test
messages:
  - { role: user, content: "hi" }
expected:
  assertions:
    - contains: "hi"
`;
    const b = `
name: bb
category: test
messages:
  - { role: user, content: "go" }
expected:
  assertions:
    - finish_reason: end_turn
`;
    await Deno.writeTextFile(`${dir}/a_first.yaml`, a);
    await Deno.writeTextFile(`${dir}/b_second.yml`, b);
    await Deno.writeTextFile(`${dir}/ignore_me.txt`, "");

    const fixtures = await loadFixturesFromDir(dir);
    assertEquals(fixtures.length, 2);
    assertEquals(fixtures[0].name, "aa");
    assertEquals(fixtures[1].name, "bb");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

class StubRunner implements FixtureRunner {
  outcome: RunOutcome;
  constructor(outcome: RunOutcome) {
    this.outcome = outcome;
  }
  run(_f: Fixture, _t: number): Promise<RunOutcome> {
    return Promise.resolve(this.outcome);
  }
}

Deno.test("fixture case bridges to trial result", async () => {
  const fixture: Fixture = {
    name: "f1",
    category: "smoke",
    model: null,
    messages: [{ role: "user", content: "hi" }],
    expected: {
      tool_sequence: [],
      assertions: [{ contains: "hi" }],
    },
  };
  const good = new FixtureCase(
    fixture,
    new StubRunner(happyOutcome([], "hi there")),
  );
  const r1 = await good.run(0);
  assert(r1.success);

  const failing: Fixture = {
    ...fixture,
    expected: {
      tool_sequence: [...fixture.expected.tool_sequence],
      assertions: [{ contains: "BYE" }],
    },
  };
  const bad = new FixtureCase(
    failing,
    new StubRunner(happyOutcome([], "hi there")),
  );
  const r2 = await bad.run(0);
  assert(!r2.success);
  assert(r2.error !== null && r2.error.includes("missing expected substring"));
});
