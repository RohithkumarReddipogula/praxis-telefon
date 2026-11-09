import assert from "node:assert/strict";
import { test } from "node:test";
import { AGENT_TIMEOUT_MS, INITIAL, messageKey, next } from "../public/state.js";

const run = (events: string[], from: string = INITIAL) => events.reduce(next, from);

test("the normal call: notice, captcha, start, token, connect, agent, live, ended", () => {
  assert.equal(INITIAL, "notice");
  assert.equal(run(["captcha_ok"]), "ready");
  assert.equal(run(["captcha_ok", "start"]), "requesting");
  assert.equal(run(["captcha_ok", "start", "token_ok"]), "connecting");
  assert.equal(run(["captcha_ok", "start", "token_ok", "connected"]), "waiting_agent");
  assert.equal(run(["captcha_ok", "start", "token_ok", "connected", "agent_audio"]), "live");
  assert.equal(run(["captcha_ok", "start", "token_ok", "connected", "agent_audio", "disconnected"]), "ended");
});

test("start only works after the captcha", () => {
  assert.equal(run(["start"]), "notice");
  assert.equal(run(["captcha_ok", "captcha_lost", "start"]), "notice");
});

test("every way the demo cannot run ends on a message, never on silence", () => {
  const upToToken = ["captcha_ok", "start"];
  assert.equal(run([...upToToken, "token_busy"]), "busy");
  assert.equal(run([...upToToken, "token_visitor_limit"]), "visitor_limit");
  assert.equal(run([...upToToken, "token_captcha_failed"]), "captcha_failed");
  assert.equal(run([...upToToken, "token_error"]), "busy"); // an endpoint error reads as busy
  assert.equal(run([...upToToken, "token_ok", "mic_denied"]), "mic_denied");
  assert.equal(run([...upToToken, "token_ok", "connected", "agent_busy"]), "busy");
  assert.equal(run([...upToToken, "token_ok", "connected", "agent_timeout"]), "busy");
  assert.equal(run([...upToToken, "token_ok", "connect_failed"]), "busy");
});

test("the agent's busy signal also works while the call is live", () => {
  assert.equal(run(["captcha_ok", "start", "token_ok", "connected", "agent_audio", "agent_busy"]), "busy");
});

test("an agent timeout in a live call changes nothing", () => {
  assert.equal(run(["captcha_ok", "start", "token_ok", "connected", "agent_audio", "agent_timeout"]), "live");
});

test("final states stay final until the page is reloaded", () => {
  for (const final of ["busy", "visitor_limit", "captcha_failed", "mic_denied", "ended"]) {
    assert.equal(run(["captcha_ok", "start", "token_ok", "connected"], final), final);
  }
});

test("the agent has fifteen seconds to answer", () => {
  assert.equal(AGENT_TIMEOUT_MS, 15_000);
});

test("every state has a text, and the final ones are messages", () => {
  for (const state of ["busy", "visitor_limit", "captcha_failed", "mic_denied", "ended", "live", "waiting_agent"]) {
    assert.ok(messageKey(state), state);
  }
  assert.equal(messageKey("busy"), "busy");
});
