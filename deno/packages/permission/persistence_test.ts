import { assert, assertEquals } from "@std/assert";
import { AuditLogger, createAuditEvent } from "./audit.ts";

Deno.test("AuditLogger writes an owner-only file and surfaces write failures", () => {
  const dir = Deno.makeTempDirSync();
  try {
    const logger = AuditLogger.withPath(`${dir}/nested/audit.jsonl`);
    logger.log(createAuditEvent("tool_execution"));
    logger.flush();
    const mode = Deno.statSync(`${dir}/nested/audit.jsonl`).mode! & 0o777;
    assertEquals(mode, 0o600);
    assertEquals(logger.lastError, undefined);

    // A path whose parent is a regular file cannot be created or written.
    const broken = AuditLogger.withPath(`${dir}/nested/audit.jsonl/not-a-dir`);
    broken.log(createAuditEvent("tool_execution"));
    broken.flush();
    assert(broken.lastError !== undefined);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});
