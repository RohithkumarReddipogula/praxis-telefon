import assert from "node:assert/strict";
import { test } from "node:test";
import { ConfigError, loadConfig } from "../src/config.ts";
import { ENV } from "./helpers.ts";

test("a complete environment loads with the agreed defaults", () => {
  const c = loadConfig(ENV);
  assert.equal(c.agentName, "praxis-telefon");
  assert.equal(c.webMinutesPerDay, 20);
  assert.equal(c.webMinutesPerMonth, 60);
  assert.equal(c.callSeconds, 300); // 5 minutes: at most 4 calls a day
  assert.equal(c.tokenTtlSeconds, 120);
  assert.equal(c.visitorTokensPerDay, 3);
  assert.equal(c.trustedProxyHops, 1);
});

test("missing variables are named, never their values", () => {
  const { LIVEKIT_API_SECRET: _a, TURNSTILE_SECRET_KEY: _b, ...rest } = ENV;
  assert.throws(
    () => loadConfig(rest),
    (e: unknown) =>
      e instanceof ConfigError &&
      e.message.includes("LIVEKIT_API_SECRET") &&
      e.message.includes("TURNSTILE_SECRET_KEY") &&
      !e.message.includes(ENV.IP_HASH_KEY),
  );
});

test("a short IP hash key is refused without echoing it", () => {
  assert.throws(
    () => loadConfig({ ...ENV, IP_HASH_KEY: "short" }),
    (e: unknown) =>
      e instanceof ConfigError && e.message.includes("IP_HASH_KEY") && !e.message.includes("short"),
  );
});

test("the public origin must be https (or localhost for development) and have no path", () => {
  assert.doesNotThrow(() => loadConfig({ ...ENV, PUBLIC_ORIGIN: "http://localhost:8080" }));
  assert.throws(() => loadConfig({ ...ENV, PUBLIC_ORIGIN: "http://demo.example.app" }), ConfigError);
  assert.throws(() => loadConfig({ ...ENV, PUBLIC_ORIGIN: "https://demo.example.app/x" }), ConfigError);
});

test("LiveKit URL must be a wss URL", () => {
  assert.throws(() => loadConfig({ ...ENV, LIVEKIT_URL: "https://x.livekit.cloud" }), ConfigError);
});

test("numeric limits must be positive whole numbers", () => {
  assert.throws(() => loadConfig({ ...ENV, WEB_MINUTES_PER_MONTH: "0" }), /WEB_MINUTES_PER_MONTH/);
  assert.throws(() => loadConfig({ ...ENV, WEB_CALL_SECONDS: "abc" }), /WEB_CALL_SECONDS/);
  assert.equal(loadConfig({ ...ENV, WEB_MINUTES_PER_DAY: "10" }).webMinutesPerDay, 10);
});

test("the port comes from PORT (Railway assigns it), with 8080 as the fallback", () => {
  assert.equal(loadConfig({ ...ENV, PORT: "4321" }).port, 4321);
  assert.equal(loadConfig(ENV).port, 8080);
  assert.throws(() => loadConfig({ ...ENV, PORT: "abc" }), /PORT/);
});
