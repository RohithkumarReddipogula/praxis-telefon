import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ENV } from "./helpers.ts";

// Runs `node server.ts` as a real process (the production start command) so the
// TypeScript execution model, the imports and the port are tested, not assumed.

function run(env: Record<string, string>, ms: number): Promise<{ out: string; err: string; code: number | null; child: ReturnType<typeof spawn> }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ["server.ts"], { env: { PATH: process.env.PATH ?? "", ...env } });
    let out = "";
    let err = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    const done = () => resolve({ out, err, code: child.exitCode, child });
    child.on("exit", done);
    setTimeout(done, ms);
  });
}

test("the server starts, listens on PORT and answers /health", async () => {
  const port = String(20000 + Math.floor(Math.random() * 20000));
  const db = join(mkdtempSync(join(tmpdir(), "srv-")), "usage.db");
  const { out, child } = await run({ ...ENV, PORT: port, DB_PATH: db, PUBLIC_ORIGIN: `http://localhost:${port}` }, 2500);
  try {
    assert.match(out, /"event":"starting"/);
    assert.match(out, /"event":"db_ok"/);
    assert.match(out, new RegExp(`"event":"listening","port":${port}`));
    const res = await fetch(`http://127.0.0.1:${port}/health`);
    assert.equal(await res.text(), "ok");
    const page = await fetch(`http://127.0.0.1:${port}/`);
    assert.equal(page.status, 200);
  } finally {
    child.kill();
  }
});

test("missing variables end the process at once with a logged, value-free reason", async () => {
  const { out, err, code } = await run({}, 5000);
  assert.equal(code, 1);
  assert.match(out, /"event":"starting"/);
  assert.match(err, /"event":"fatal","where":"config"/);
  assert.match(err, /LIVEKIT_URL/);
});
