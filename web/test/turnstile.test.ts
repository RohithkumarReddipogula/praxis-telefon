import assert from "node:assert/strict";
import { test } from "node:test";
import { verifyTurnstile } from "../src/turnstile.ts";

function fakeFetch(reply: unknown, calls: { url: string; body: string }[] = []) {
  return (async (url: string, init: { body: string }) => {
    calls.push({ url, body: init.body });
    return { ok: true, json: async () => reply } as Response;
  }) as unknown as typeof fetch;
}

test("a successful check posts the secret, the token and the address to Cloudflare", async () => {
  const calls: { url: string; body: string }[] = [];
  const ok = await verifyTurnstile("tok", "secret", "203.0.113.7", fakeFetch({ success: true }, calls));
  assert.equal(ok, true);
  assert.equal(calls[0].url, "https://challenges.cloudflare.com/turnstile/v0/siteverify");
  const form = new URLSearchParams(calls[0].body);
  assert.deepEqual([form.get("secret"), form.get("response"), form.get("remoteip")], ["secret", "tok", "203.0.113.7"]);
});

test("a refused token is false", async () => {
  assert.equal(await verifyTurnstile("tok", "s", null, fakeFetch({ success: false, "error-codes": ["x"] })), false);
});

test("no token means no request at all", async () => {
  const calls: { url: string; body: string }[] = [];
  assert.equal(await verifyTurnstile("", "s", null, fakeFetch({ success: true }, calls)), false);
  assert.equal(await verifyTurnstile(undefined, "s", null, fakeFetch({ success: true }, calls)), false);
  assert.equal(calls.length, 0);
});

test("when Cloudflare cannot be reached the check fails closed", async () => {
  const boom = (async () => { throw new Error("network"); }) as unknown as typeof fetch;
  assert.equal(await verifyTurnstile("tok", "s", null, boom), false);
  const bad = (async () => ({ ok: false, json: async () => ({}) }) as Response) as unknown as typeof fetch;
  assert.equal(await verifyTurnstile("tok", "s", null, bad), false);
});

test("the remote address is left out when unknown", async () => {
  const calls: { url: string; body: string }[] = [];
  await verifyTurnstile("tok", "s", null, fakeFetch({ success: true }, calls));
  assert.equal(new URLSearchParams(calls[0].body).has("remoteip"), false);
});
