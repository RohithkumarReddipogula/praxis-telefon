import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { UsageStore, berlinDay } from "../src/store.ts";

const LIMITS = { webMinutesPerDay: 20, webMinutesPerMonth: 60, callSeconds: 300, visitorTokensPerDay: 3 };
const DAY = new Date("2026-11-09T10:00:00+01:00");
const at = (days: number, hour = 10) => new Date(DAY.getTime() + days * 86_400_000 + (hour - 10) * 3_600_000);
const store = (over: Partial<typeof LIMITS> = {}, path = ":memory:") => new UsageStore(path, { ...LIMITS, ...over });

test("four tokens of five minutes fill the twenty minutes; the fifth is busy", () => {
  const s = store();
  for (let i = 0; i < 4; i++) {
    const r = s.reserve(`v${i}`, DAY);
    assert.ok(r.ok && r.seconds === 300);
  }
  assert.deepEqual(s.reserve("v9", DAY), { ok: false, reason: "daily" });
});

test("the monthly cap of sixty minutes stops the demo on the fourth day", () => {
  const s = store();
  for (let d = 0; d < 3; d++) for (let i = 0; i < 4; i++) assert.ok(s.reserve(`v${d}-${i}`, at(d)).ok);
  assert.deepEqual(s.reserve("late", at(3)), { ok: false, reason: "monthly" });
});

test("a visitor gets three tokens a day and the limit is per visitor", () => {
  const s = store();
  for (let i = 0; i < 3; i++) assert.ok(s.reserve("alice", DAY).ok);
  assert.deepEqual(s.reserve("alice", DAY), { ok: false, reason: "visitor" });
  assert.ok(s.reserve("bob", DAY).ok); // the day still has 5 minutes
});

test("a refused visitor does not use up the day's minutes", () => {
  const s = store();
  for (let i = 0; i < 3; i++) s.reserve("alice", DAY);
  s.reserve("alice", DAY); // refused: visitor limit
  assert.ok(s.reserve("bob", DAY).ok); // 4th token of the day is still free
});

test("the last token of a day or month is shortened; under sixty seconds is refused", () => {
  const s = store({ webMinutesPerDay: 7 }); // 420 s
  const a = s.reserve("a", DAY);
  const b = s.reserve("b", DAY);
  assert.ok(a.ok && a.seconds === 300 && b.ok && b.seconds === 120);
  assert.deepEqual(s.reserve("c", DAY), { ok: false, reason: "daily" });
  const t = store({ webMinutesPerDay: 5 });
  assert.ok(t.reserve("a", DAY).ok);
  const small = store({ webMinutesPerMonth: 6 }); // month: 360 s -> 300 + 60
  small.reserve("a", DAY);
  const last = small.reserve("b", at(1));
  assert.ok(last.ok && last.seconds === 60);
});

test("a released reservation gives its minutes back", () => {
  const s = store();
  const ids = [0, 1, 2, 3].map((i) => s.reserve(`v${i}`, DAY));
  assert.ok(ids[0].ok);
  s.release((ids[0] as { id: number }).id);
  assert.ok(s.reserve("again", DAY).ok);
});

test("days and months follow Berlin time", () => {
  assert.equal(berlinDay(new Date("2026-11-09T23:30:00+01:00")), "2026-11-09");
  assert.equal(berlinDay(new Date("2026-11-09T22:30:00Z")), "2026-11-09"); // 23:30 Berlin
  assert.equal(berlinDay(new Date("2026-11-09T23:30:00Z")), "2026-11-10"); // 00:30 Berlin
  const s = store();
  for (let i = 0; i < 4; i++) s.reserve(`v${i}`, new Date("2026-11-09T23:30:00+01:00"));
  assert.ok(s.reserve("x", new Date("2026-11-10T00:30:00+01:00")).ok); // a new Berlin day
  const m = store();
  for (let d = 0; d < 3; d++) for (let i = 0; i < 4; i++) m.reserve(`v${d}${i}`, new Date(`2026-11-${26 + d}T12:00:00+01:00`));
  assert.equal(m.reserve("z", new Date("2026-11-30T23:30:00+01:00")).ok, false);
  assert.ok(m.reserve("z", new Date("2026-12-01T00:30:00+01:00")).ok); // a new month
});

test("the numbers survive a restart (a file on the volume)", () => {
  const path = join(mkdtempSync(join(tmpdir(), "usage-")), "usage.db");
  const first = store({}, path);
  for (let i = 0; i < 4; i++) first.reserve(`v${i}`, DAY);
  const second = store({}, path); // a new process opening the same file
  assert.deepEqual(second.reserve("v9", DAY), { ok: false, reason: "daily" });
});

test("two connections to one file cannot overbook the day", () => {
  const path = join(mkdtempSync(join(tmpdir(), "usage-")), "usage.db");
  const a = store({}, path);
  const b = store({}, path);
  const results = [a, b, a, b, a, b].map((s, i) => s.reserve(`v${i}`, DAY).ok);
  assert.equal(results.filter(Boolean).length, 4);
});

test("visitor keys are forgotten after the day; minutes are kept for the month", () => {
  const path = join(mkdtempSync(join(tmpdir(), "usage-")), "usage.db");
  const s = store({}, path);
  s.reserve("alice-key", DAY);
  s.reserve("alice-key", at(1)); // next day: yesterday's visitor value is erased
  const rows = new DatabaseSync(path).prepare("SELECT day, visitor, seconds FROM reservations ORDER BY id").all();
  assert.equal(rows[0].visitor, null);
  assert.equal(rows[0].seconds, 300);
  assert.equal(rows[1].visitor, "alice-key");
});

test("the usage file holds no IP address, only the keyed value it was given", () => {
  const path = join(mkdtempSync(join(tmpdir(), "usage-")), "usage.db");
  store({}, path).reserve("0123456789abcdef", DAY);
  const dump = JSON.stringify(new DatabaseSync(path).prepare("SELECT * FROM reservations").all());
  assert.ok(!dump.includes("203.0.113"));
});
