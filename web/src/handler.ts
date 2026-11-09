// The whole HTTP surface as a function Request -> Response (testable without a socket).
//
// POST /api/token is the only dynamic route. Order of checks, cheapest and most
// fundamental first: origin, content type and size, the request's fields, the captcha
// (a refused captcha costs no budget), then visitor limit and budget in one transaction.
// Every refusal is a clear JSON answer the page turns into a message; a failure inside
// is a generic 500 (the page reads it as "busy"). Logs hold event names and counts only:
// never the address, the visitor key, the captcha token or the JWT.

import type { Config } from "./config.ts";
import type { UsageStore } from "./store.ts";
import { buildSpec, parseRequest, type TokenSpec } from "./token.ts";
import { clientIp, visitorKey } from "./visitor.ts";

export interface StaticFile {
  body: string | Uint8Array;
  type: string;
}

export interface Deps {
  config: Config;
  store: UsageStore;
  verifyCaptcha: (token: string, ip: string | null) => Promise<boolean>;
  sign: (spec: TokenSpec) => Promise<string>;
  log: (record: Record<string, unknown>) => void;
  now: () => Date;
  randomHex: (bytes: number) => string;
  files: Map<string, StaticFile>;
}

const MAX_BODY = 2048;

function securityHeaders(): Headers {
  const h = new Headers();
  h.set("x-content-type-options", "nosniff");
  h.set("referrer-policy", "no-referrer");
  h.set("x-robots-tag", "noindex, nofollow, noarchive");
  h.set("permissions-policy", "microphone=(self), camera=(), geolocation=(), payment=()");
  h.set("cross-origin-opener-policy", "same-origin");
  h.set("cache-control", "no-store");
  return h;
}

function contentSecurityPolicy(config: Config): string {
  const live = config.livekitUrl;
  const liveHttps = live.replace(/^wss:/, "https:");
  return [
    "default-src 'none'",
    "script-src 'self' https://challenges.cloudflare.com",
    "style-src 'self'",
    "img-src 'self' data:",
    "media-src 'self' blob:",
    "font-src 'self'",
    `connect-src 'self' ${live} ${liveHttps}`,
    "frame-src https://challenges.cloudflare.com",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join("; ");
}

function respond(status: number, body: BodyInit | null, type: string, extra: Record<string, string> = {}): Response {
  const headers = securityHeaders();
  headers.set("content-type", type);
  for (const [k, v] of Object.entries(extra)) headers.set(k, v);
  return new Response(body, { status, headers });
}

const json = (status: number, data: unknown): Response =>
  respond(status, JSON.stringify(data), "application/json; charset=utf-8");

export function createHandler(deps: Deps): (req: Request) => Promise<Response> {
  const { config } = deps;

  async function token(req: Request): Promise<Response> {
    if (req.headers.get("origin") !== config.publicOrigin) return json(403, { error: "origin" });
    if (!(req.headers.get("content-type") ?? "").startsWith("application/json")) return json(415, { error: "content_type" });
    if (Number(req.headers.get("content-length") ?? 0) > MAX_BODY) return json(413, { error: "too_large" });
    const raw = await req.text();
    if (raw.length > MAX_BODY) return json(413, { error: "too_large" });

    const parsed = parseRequest(raw);
    if (!parsed.ok) {
      deps.log({ event: "bad_request", why: parsed.error });
      return json(400, { error: parsed.error });
    }

    const ip = clientIp(req.headers, config.trustedProxyHops);
    if (!(await deps.verifyCaptcha(parsed.turnstileToken, ip))) {
      deps.log({ event: "captcha_failed" });
      return json(403, { error: "captcha" });
    }

    const reservation = deps.store.reserve(visitorKey(ip, config.ipHashKey), deps.now());
    if (!reservation.ok) {
      deps.log({ event: "token_refused", reason: reservation.reason });
      return reservation.reason === "visitor"
        ? json(429, { status: "visitor_limit" })
        : json(503, { status: "busy" }); // daily and monthly look the same to the visitor
    }

    try {
      const spec = buildSpec(config, reservation.seconds, deps.randomHex);
      const participantToken = await deps.sign(spec);
      deps.log({ event: "token_issued", reserved_s: reservation.seconds });
      return json(201, {
        server_url: config.livekitUrl,
        participant_token: participantToken,
        max_seconds: reservation.seconds,
      });
    } catch {
      deps.store.release(reservation.id); // no token was handed out: give the minutes back
      deps.log({ event: "error", where: "sign" });
      return json(500, { error: "server" });
    }
  }

  return async function handle(req: Request): Promise<Response> {
    try {
      const path = new URL(req.url).pathname;
      const isRead = req.method === "GET" || req.method === "HEAD";

      if (path === "/api/token") {
        if (req.method !== "POST") return respond(405, "method not allowed", "text/plain", { allow: "POST" });
        return await token(req);
      }
      if (!isRead) return respond(405, "method not allowed", "text/plain", { allow: "GET, HEAD" });

      if (path === "/health") return respond(200, "ok", "text/plain");
      if (path === "/robots.txt") return respond(200, "User-agent: *\nDisallow: /\n", "text/plain");
      if (path === "/config.json") {
        return json(200, {
          turnstileSiteKey: config.turnstileSiteKey,
          callMinutes: Math.round(config.callSeconds / 60),
          contactEmail: config.contactEmail,
        });
      }
      const file = deps.files.get(path);
      if (!file) return respond(404, "not found", "text/plain");
      const extra: Record<string, string> = {};
      if (file.type.startsWith("text/html")) extra["content-security-policy"] = contentSecurityPolicy(config);
      return respond(200, req.method === "HEAD" ? null : (file.body as BodyInit), file.type, extra);
    } catch {
      deps.log({ event: "error", where: "handler" });
      return json(500, { error: "server" });
    }
  };
}
