// Configuration from environment variables, validated at start. Errors name the
// variables, never their values. Secrets (LiveKit secret, Turnstile secret, IP hash key)
// come from the host's secret store (Railway variables) and are never logged.

export class ConfigError extends Error {}

export interface Config {
  livekitUrl: string;
  apiKey: string;
  apiSecret: string;
  turnstileSiteKey: string;
  turnstileSecret: string;
  ipHashKey: string;
  publicOrigin: string;
  contactEmail: string;
  agentName: string;
  dbPath: string;
  port: number;
  webMinutesPerDay: number;
  webMinutesPerMonth: number;
  callSeconds: number;
  visitorTokensPerDay: number;
  tokenTtlSeconds: number;
  trustedProxyHops: number;
}

type Env = Record<string, string | undefined>;

export function loadConfig(env: Env): Config {
  const problems: string[] = [];
  const get = (name: string): string => (env[name] ?? "").trim();
  const required = (name: string): string => {
    const value = get(name);
    if (!value) problems.push(`${name} is missing or empty`);
    return value;
  };
  const positive = (name: string, fallback: number): number => {
    const raw = get(name);
    if (!raw) return fallback;
    if (!/^[1-9][0-9]*$/.test(raw)) {
      problems.push(`${name} must be a positive whole number`);
      return fallback;
    }
    return Number(raw);
  };

  const livekitUrl = required("LIVEKIT_URL");
  if (livekitUrl && !/^wss:\/\/[^/]+$/.test(livekitUrl)) problems.push("LIVEKIT_URL must be a wss:// URL without a path");
  const publicOrigin = required("PUBLIC_ORIGIN");
  if (publicOrigin) {
    const ok = /^https:\/\/[^/]+$/.test(publicOrigin) || /^http:\/\/localhost(:[0-9]+)?$/.test(publicOrigin);
    if (!ok) problems.push("PUBLIC_ORIGIN must be https://host (or http://localhost) without a path");
  }
  const ipHashKey = required("IP_HASH_KEY");
  if (ipHashKey && ipHashKey.length < 16) problems.push("IP_HASH_KEY must be at least 16 characters");

  const config: Config = {
    livekitUrl,
    apiKey: required("LIVEKIT_API_KEY"),
    apiSecret: required("LIVEKIT_API_SECRET"),
    turnstileSiteKey: required("TURNSTILE_SITE_KEY"),
    turnstileSecret: required("TURNSTILE_SECRET_KEY"),
    ipHashKey,
    publicOrigin,
    contactEmail: required("CONTACT_EMAIL"),
    agentName: get("AGENT_NAME") || "praxis-telefon",
    dbPath: get("DB_PATH") || "/data/usage.db",
    port: positive("PORT", 8080),
    webMinutesPerDay: positive("WEB_MINUTES_PER_DAY", 20),
    webMinutesPerMonth: positive("WEB_MINUTES_PER_MONTH", 60),
    callSeconds: positive("WEB_CALL_SECONDS", 300),
    visitorTokensPerDay: positive("VISITOR_TOKENS_PER_DAY", 3),
    tokenTtlSeconds: positive("TOKEN_TTL_SECONDS", 120),
    trustedProxyHops: positive("TRUSTED_PROXY_HOPS", 1),
  };
  if (problems.length) throw new ConfigError(`Invalid configuration:\n  - ${problems.join("\n  - ")}`);
  return config;
}
