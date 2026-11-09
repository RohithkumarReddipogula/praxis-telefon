import assert from "node:assert/strict";
import { test } from "node:test";
import { buildSpec } from "../src/token.ts";
import { config } from "./helpers.ts";

// Needs livekit-server-sdk (npm install); skipped until it is installed.
let sdk: typeof import("../src/sign.ts") | null = null;
try {
  sdk = await import("../src/sign.ts");
} catch {
  sdk = null;
}

function claims(jwt: string): Record<string, any> {
  return JSON.parse(Buffer.from(jwt.split(".")[1], "base64url").toString());
}

test("the signed token carries exactly the spec", { skip: sdk === null && "livekit-server-sdk not installed" }, async () => {
  const cfg = config();
  const spec = buildSpec(cfg, 300, (n) => "ab".repeat(n));
  const jwt = await sdk!.signToken(cfg, spec);
  const c = claims(jwt);
  assert.equal(c.iss, cfg.apiKey);
  assert.equal(c.sub, spec.identity);
  assert.equal(c.video.room, spec.room);
  assert.equal(c.video.roomJoin, true);
  assert.deepEqual(c.video.canPublishSources, ["microphone"]);
  assert.equal(c.video.canPublishData, false);
  assert.equal(c.video.canUpdateOwnMetadata, false);
  assert.deepEqual(c.attributes, { channel: "web" });
  assert.equal(c.exp - c.nbf <= 125, true); // two minutes (plus clock slack)
  assert.equal(c.roomConfig.agents[0].agentName, "praxis-telefon");
  assert.equal(c.roomConfig.maxParticipants, 2);
});

test("the token is signed with the project secret and no other", { skip: sdk === null && "livekit-server-sdk not installed" }, async () => {
  const cfg = config();
  const jwt = await sdk!.signToken(cfg, buildSpec(cfg, 300, (n) => "cd".repeat(n)));
  const { createHmac } = await import("node:crypto");
  const [h, p, s] = jwt.split(".");
  const expected = createHmac("sha256", cfg.apiSecret).update(`${h}.${p}`).digest("base64url");
  assert.equal(s, expected);
});
