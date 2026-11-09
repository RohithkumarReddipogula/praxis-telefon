import assert from "node:assert/strict";
import { test } from "node:test";
import { clientIp, visitorKey } from "../src/visitor.ts";

const KEY = "ip-hash-key-0123456789abcdef";

test("the visitor key is keyed, stable and shows no digits of the address", () => {
  const a = visitorKey("203.0.113.7", KEY);
  assert.equal(a, visitorKey("203.0.113.7", KEY));
  assert.notEqual(a, visitorKey("203.0.113.8", KEY));
  assert.notEqual(a, visitorKey("203.0.113.7", "another-key-0123456789abcdef"));
  assert.match(a, /^[0-9a-f]{16}$/);
  assert.ok(!a.includes("203"));
});

test("IPv6 spellings of one address give one key", () => {
  assert.equal(visitorKey("2001:DB8::1", KEY), visitorKey("2001:db8:0:0:0:0:0:1", KEY));
});

test("the client address is the entry the trusted proxy appended", () => {
  const h = (xff: string) => new Headers({ "x-forwarded-for": xff });
  assert.equal(clientIp(h("203.0.113.7"), 1), "203.0.113.7");
  // a client can prepend fake entries; the proxy's own entry is the last one
  assert.equal(clientIp(h("6.6.6.6, 203.0.113.7"), 1), "203.0.113.7");
  assert.equal(clientIp(h("6.6.6.6, 203.0.113.7, 10.0.0.1"), 2), "203.0.113.7");
});

test("a missing or unusable header gives no address", () => {
  assert.equal(clientIp(new Headers(), 1), null);
  assert.equal(clientIp(new Headers({ "x-forwarded-for": "not an ip" }), 1), null);
  assert.equal(clientIp(new Headers({ "x-forwarded-for": "1.2.3.4" }), 3), null);
});

test("no address still gets a (shared, conservative) key", () => {
  assert.equal(visitorKey(null, KEY), visitorKey(null, KEY));
  assert.notEqual(visitorKey(null, KEY), visitorKey("203.0.113.7", KEY));
});
