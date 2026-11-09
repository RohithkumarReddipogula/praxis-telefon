import assert from "node:assert/strict";
import { test } from "node:test";
import { FORBIDDEN_FIELDS, buildSpec, parseRequest } from "../src/token.ts";
import { config } from "./helpers.ts";

let n = 0;
const hex = (bytes: number) => String(++n).padStart(bytes * 2, "0");

test("the request may only carry the captcha token and a language", () => {
  assert.deepEqual(parseRequest('{"turnstile_token":"t","lang":"en"}'), { ok: true, turnstileToken: "t", lang: "en" });
  assert.deepEqual(parseRequest('{"turnstile_token":"t"}'), { ok: true, turnstileToken: "t", lang: "de" });
});

test("fields a client must not set are rejected", () => {
  for (const field of FORBIDDEN_FIELDS) {
    const r = parseRequest(JSON.stringify({ turnstile_token: "t", [field]: "x" }));
    assert.equal(r.ok, false, field);
  }
  assert.ok(FORBIDDEN_FIELDS.includes("room_config") && FORBIDDEN_FIELDS.includes("participant_attributes"));
});

test("bad JSON, a non-object and unknown fields are rejected", () => {
  for (const body of ["{", "[]", '"x"', "null", '{"turnstile_token":"t","extra":1}', '{"turnstile_token":5}']) {
    assert.equal(parseRequest(body).ok, false, body);
  }
  assert.equal(parseRequest('{"turnstile_token":"t","lang":"fr"}').ok, false);
});

test("the token spec is short-lived, microphone-only and dispatches the agent", () => {
  const spec = buildSpec(config(), 300, hex);
  assert.equal(spec.ttlSeconds, 120);
  assert.match(spec.room, /^web-[0-9a-f]+$/);
  assert.match(spec.identity, /^visitor-[0-9a-f]+$/);
  assert.deepEqual(spec.attributes, { channel: "web" }); // the agent's web bucket
  assert.deepEqual(spec.grant, {
    roomJoin: true,
    room: spec.room,
    canPublish: true,
    canPublishSources: ["microphone"],
    canSubscribe: true,
    canPublishData: false,
    canUpdateOwnMetadata: false,
  });
  assert.deepEqual(spec.roomConfig, {
    agents: [{ agentName: "praxis-telefon" }],
    maxParticipants: 2,
    emptyTimeout: 30,
    departureTimeout: 10,
  });
});

test("every spec gets its own room and identity", () => {
  const a = buildSpec(config(), 300, hex);
  const b = buildSpec(config(), 300, hex);
  assert.notEqual(a.room, b.room);
  assert.notEqual(a.identity, b.identity);
});

test("the room is closed shortly after the reserved time", () => {
  const spec = buildSpec(config(), 180, hex);
  assert.equal(spec.maxSeconds, 180);
});
