import assert from "node:assert/strict";
import { test } from "node:test";
import { createHandler } from "../src/handler.ts";
import { config, fakeDeps, post } from "./helpers.ts";

const get = (path: string, headers: Record<string, string> = {}) =>
  new Request(`https://demo.example.app${path}`, { headers });

async function json(res: Response): Promise<Record<string, unknown>> {
  return (await res.json()) as Record<string, unknown>;
}

// --- the happy path ----------------------------------------------------------------------

test("a human with budget left gets a token for a fresh web room", async () => {
  const { deps, signed, logs } = fakeDeps();
  const res = await createHandler(deps)(post({ turnstile_token: "good-token" }));
  assert.equal(res.status, 201);
  const body = await json(res);
  assert.equal(body.server_url, "wss://demo-abc123.livekit.cloud");
  assert.match(String(body.participant_token), /^signed-jwt-for-web-/);
  assert.equal(body.max_seconds, 300);
  assert.equal(signed.length, 1);
  assert.deepEqual(logs, [{ event: "token_issued", reserved_s: 300 }]);
});

test("the response is never cached", async () => {
  const { deps } = fakeDeps();
  const res = await createHandler(deps)(post({ turnstile_token: "good-token" }));
  assert.equal(res.headers.get("cache-control"), "no-store");
});

// --- who may ask ---------------------------------------------------------------------------

test("another origin is refused before anything else happens", async () => {
  const { deps, captchaCalls } = fakeDeps();
  const res = await createHandler(deps)(post({ turnstile_token: "good-token" }, { origin: "https://evil.example" }));
  assert.equal(res.status, 403);
  assert.equal(captchaCalls.length, 0);
});

test("a request without an origin is refused", async () => {
  const { deps } = fakeDeps();
  const req = new Request("https://demo.example.app/api/token", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ turnstile_token: "good-token" }),
  });
  assert.equal((await createHandler(deps)(req)).status, 403);
});

test("only POST with JSON of reasonable size", async () => {
  const handler = createHandler(fakeDeps().deps);
  assert.equal((await handler(get("/api/token"))).status, 405);
  assert.equal((await handler(post("x", { "content-type": "text/plain" }))).status, 415);
  assert.equal((await handler(post({ turnstile_token: "x".repeat(5000) }))).status, 413);
  assert.equal((await handler(post("{"))).status, 400);
});

test("the client cannot choose room, identity, attributes or config", async () => {
  const { deps, signed } = fakeDeps();
  for (const field of ["room_name", "participant_identity", "participant_attributes", "room_config"]) {
    const res = await createHandler(deps)(post({ turnstile_token: "good-token", [field]: "x" }));
    assert.equal(res.status, 400, field);
  }
  assert.equal(signed.length, 0);
});

// --- captcha, visitor, budget ------------------------------------------------------------

test("a failed captcha is refused and costs no budget", async () => {
  const { deps, signed, logs } = fakeDeps();
  const handler = createHandler(deps);
  for (let i = 0; i < 10; i++) assert.equal((await handler(post({ turnstile_token: "bad" }))).status, 403);
  assert.equal(signed.length, 0);
  assert.ok(logs.every((l) => l.event === "captcha_failed"));
  assert.equal((await handler(post({ turnstile_token: "good-token" }, { "x-forwarded-for": "198.51.100.1" }))).status, 201);
});

test("the captcha sees the visitor's address from the trusted proxy entry", async () => {
  const { deps, captchaCalls } = fakeDeps();
  await createHandler(deps)(post({ turnstile_token: "good-token" }, { "x-forwarded-for": "6.6.6.6, 203.0.113.9" }));
  assert.deepEqual(captchaCalls, [{ token: "good-token", ip: "203.0.113.9" }]);
});

test("a visitor's fourth token of the day is refused with its own message", async () => {
  const { deps } = fakeDeps();
  const handler = createHandler(deps);
  for (let i = 0; i < 3; i++) assert.equal((await handler(post({ turnstile_token: "good-token" }))).status, 201);
  const res = await handler(post({ turnstile_token: "good-token" }));
  assert.equal(res.status, 429);
  assert.deepEqual(await json(res), { status: "visitor_limit" });
});

test("when the day is used up the answer is busy (503), also for another visitor", async () => {
  const { deps, signed } = fakeDeps();
  const handler = createHandler(deps);
  for (let i = 0; i < 4; i++) {
    const res = await handler(post({ turnstile_token: "good-token" }, { "x-forwarded-for": `198.51.100.${i}` }));
    assert.equal(res.status, 201);
  }
  const res = await handler(post({ turnstile_token: "good-token" }, { "x-forwarded-for": "198.51.100.99" }));
  assert.equal(res.status, 503);
  assert.deepEqual(await json(res), { status: "busy" });
  assert.equal(signed.length, 4);
});

test("the monthly cap also answers busy, with the same body", async () => {
  const { deps } = fakeDeps({}, config({ WEB_MINUTES_PER_DAY: "5", WEB_MINUTES_PER_MONTH: "5" }));
  const handler = createHandler(deps);
  assert.equal((await handler(post({ turnstile_token: "good-token" }, { "x-forwarded-for": "198.51.100.1" }))).status, 201);
  const res = await handler(post({ turnstile_token: "good-token" }, { "x-forwarded-for": "198.51.100.2" }));
  assert.deepEqual([res.status, await json(res)], [503, { status: "busy" }]);
});

test("if signing fails the reserved minutes come back and the answer is generic", async () => {
  let fail = true;
  const { deps, logs } = fakeDeps({
    sign: async (spec) => {
      if (fail) throw new Error("sdk exploded with details");
      return `ok-${spec.room}`;
    },
  });
  const handler = createHandler(deps);
  const res = await handler(post({ turnstile_token: "good-token" }));
  assert.equal(res.status, 500);
  const text = JSON.stringify(await json(res));
  assert.ok(!text.includes("sdk exploded"));
  assert.ok(logs.some((l) => l.event === "error" && !JSON.stringify(l).includes("sdk exploded")));
  fail = false;
  for (let i = 0; i < 4; i++) {
    const r = await handler(post({ turnstile_token: "good-token" }, { "x-forwarded-for": `198.51.100.${i}` }));
    assert.equal(r.status, 201); // all four tokens of the day are still there
  }
});

test("logs never hold the address, the visitor key, the captcha token or the JWT", async () => {
  const { deps, logs } = fakeDeps();
  const handler = createHandler(deps);
  await handler(post({ turnstile_token: "good-token" }));
  await handler(post({ turnstile_token: "bad" }));
  const text = JSON.stringify(logs);
  for (const secret of ["203.0.113.7", "good-token", "signed-jwt", "web-", "visitor-"]) {
    assert.ok(!text.includes(secret), secret);
  }
});

// --- the page and what the browser is told -------------------------------------------------

test("the page is noindex, not cached, and locked down", async () => {
  const res = await createHandler(fakeDeps().deps)(get("/"));
  assert.equal(res.status, 200);
  assert.match(res.headers.get("x-robots-tag") ?? "", /noindex/);
  assert.equal(res.headers.get("referrer-policy"), "no-referrer");
  assert.equal(res.headers.get("x-content-type-options"), "nosniff");
  assert.equal(res.headers.get("cache-control"), "no-store");
  assert.match(res.headers.get("permissions-policy") ?? "", /microphone=\(self\)/);
  assert.match(res.headers.get("permissions-policy") ?? "", /camera=\(\)/);
  assert.equal(res.headers.get("set-cookie"), null);
});

test("the content security policy allows only ourselves, Cloudflare Turnstile and LiveKit", async () => {
  const res = await createHandler(fakeDeps().deps)(get("/"));
  const csp = res.headers.get("content-security-policy") ?? "";
  assert.match(csp, /default-src 'none'/);
  assert.match(csp, /script-src 'self' https:\/\/challenges\.cloudflare\.com/);
  assert.match(csp, /frame-src https:\/\/challenges\.cloudflare\.com/);
  assert.match(csp, /connect-src 'self' wss:\/\/demo-abc123\.livekit\.cloud https:\/\/demo-abc123\.livekit\.cloud/);
  assert.ok(!csp.includes("unsafe-inline") && !csp.includes("unsafe-eval"));
  assert.match(csp, /frame-ancestors 'none'/);
});

test("robots.txt keeps crawlers out and health is plain", async () => {
  const handler = createHandler(fakeDeps().deps);
  assert.match(await (await handler(get("/robots.txt"))).text(), /Disallow: \//);
  const health = await handler(get("/health"));
  assert.deepEqual([health.status, await health.text()], [200, "ok"]);
});

test("the page config holds only public values", async () => {
  const res = await createHandler(fakeDeps().deps)(get("/config.json"));
  const body = await json(res);
  assert.deepEqual(Object.keys(body).sort(), ["callMinutes", "contactEmail", "turnstileSiteKey"]);
  const text = JSON.stringify(body);
  for (const secret of ["turnstile-secret-for-tests", "livekit-secret-for-tests", "ip-hash-key", "APIkeyForTests"]) {
    assert.ok(!text.includes(secret), secret);
  }
});

test("only whitelisted files are served; nothing else, nothing outside the folder", async () => {
  const handler = createHandler(fakeDeps().deps);
  assert.equal((await handler(get("/app.js"))).status, 200);
  for (const path of ["/../server.ts", "/.env", "/src/config.ts", "/%2e%2e/server.ts", "/usage.db", "/test/helpers.ts"]) {
    assert.equal((await handler(get(path))).status, 404, path);
  }
  assert.equal((await handler(new Request("https://demo.example.app/", { method: "DELETE" }))).status, 405);
});
