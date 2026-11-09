import { UsageStore } from "../src/store.ts";
import { loadConfig, type Config } from "../src/config.ts";
import type { Deps } from "../src/handler.ts";

export const ENV: Record<string, string> = {
  LIVEKIT_URL: "wss://demo-abc123.livekit.cloud",
  LIVEKIT_API_KEY: "APIkeyForTests",
  LIVEKIT_API_SECRET: "livekit-secret-for-tests-0123456789",
  TURNSTILE_SITE_KEY: "site-key-public",
  TURNSTILE_SECRET_KEY: "turnstile-secret-for-tests",
  IP_HASH_KEY: "ip-hash-key-0123456789abcdef",
  PUBLIC_ORIGIN: "https://demo.example.app",
  CONTACT_EMAIL: "kontakt@example.org",
  DB_PATH: ":memory:",
};

export const NOW = new Date("2026-11-09T10:00:00+01:00");

export function config(over: Record<string, string> = {}): Config {
  return loadConfig({ ...ENV, ...over });
}

export interface Fake {
  deps: Deps;
  logs: Record<string, unknown>[];
  signed: unknown[];
  captchaCalls: { token: string; ip: string | null }[];
}

export function fakeDeps(over: Partial<Deps> = {}, cfg: Config = config()): Fake {
  const logs: Record<string, unknown>[] = [];
  const signed: unknown[] = [];
  const captchaCalls: { token: string; ip: string | null }[] = [];
  let n = 0;
  const deps: Deps = {
    config: cfg,
    store: new UsageStore(cfg.dbPath, {
      webMinutesPerDay: cfg.webMinutesPerDay,
      webMinutesPerMonth: cfg.webMinutesPerMonth,
      callSeconds: cfg.callSeconds,
      visitorTokensPerDay: cfg.visitorTokensPerDay,
    }),
    verifyCaptcha: async (token, ip) => {
      captchaCalls.push({ token, ip });
      return token === "good-token";
    },
    sign: async (spec) => {
      signed.push(spec);
      return `signed-jwt-for-${spec.room}`;
    },
    log: (record) => logs.push(record),
    now: () => NOW,
    randomHex: (bytes) => String(++n).padStart(bytes * 2, "0"),
    files: new Map([
      ["/", { body: "<html><body>page</body></html>", type: "text/html; charset=utf-8" }],
      ["/app.js", { body: "// app", type: "text/javascript; charset=utf-8" }],
    ]),
    ...over,
  };
  return { deps, logs, signed, captchaCalls };
}

export function post(body: unknown, headers: Record<string, string> = {}): Request {
  return new Request("https://demo.example.app/api/token", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: "https://demo.example.app",
      "x-forwarded-for": "203.0.113.7",
      ...headers,
    },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}
