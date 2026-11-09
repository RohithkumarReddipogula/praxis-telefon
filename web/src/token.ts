// What a visitor may ask for, and what the token we issue contains. The client sends only
// the captcha token and a language; room, identity, attributes and the agent dispatch are
// ours. The attribute channel=web is what makes the agent apply the web limits.

export const FORBIDDEN_FIELDS = [
  "room_name",
  "participant_identity",
  "participant_name",
  "participant_metadata",
  "participant_attributes",
  "room_config",
  "roomName",
  "identity",
  "attributes",
  "metadata",
  "roomConfig",
];

export type Lang = "de" | "en";
export type ParsedRequest = { ok: true; turnstileToken: string; lang: Lang } | { ok: false; error: string };

export function parseRequest(raw: string): ParsedRequest {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return { ok: false, error: "invalid_json" };
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) return { ok: false, error: "invalid_body" };
  const body = data as Record<string, unknown>;
  for (const key of Object.keys(body)) {
    if (FORBIDDEN_FIELDS.includes(key)) return { ok: false, error: "forbidden_field" };
    if (key !== "turnstile_token" && key !== "lang") return { ok: false, error: "unknown_field" };
  }
  if (typeof body.turnstile_token !== "string" || body.turnstile_token === "") return { ok: false, error: "no_captcha" };
  const lang = body.lang ?? "de";
  if (lang !== "de" && lang !== "en") return { ok: false, error: "invalid_lang" };
  return { ok: true, turnstileToken: body.turnstile_token, lang };
}

export interface TokenSpec {
  room: string;
  identity: string;
  ttlSeconds: number;
  maxSeconds: number;
  attributes: Record<string, string>;
  grant: {
    roomJoin: true;
    room: string;
    canPublish: true;
    canPublishSources: string[];
    canSubscribe: true;
    canPublishData: false;
    canUpdateOwnMetadata: false;
  };
  roomConfig: {
    agents: { agentName: string }[];
    maxParticipants: number;
    emptyTimeout: number;
    departureTimeout: number;
  };
}

interface TokenConfig {
  agentName: string;
  tokenTtlSeconds: number;
}

export function buildSpec(config: TokenConfig, maxSeconds: number, randomHex: (bytes: number) => string): TokenSpec {
  const room = `web-${randomHex(8)}`;
  return {
    room,
    identity: `visitor-${randomHex(6)}`,
    ttlSeconds: config.tokenTtlSeconds, // time to connect; a call can last longer
    maxSeconds,
    attributes: { channel: "web" },
    grant: {
      roomJoin: true,
      room,
      canPublish: true,
      canPublishSources: ["microphone"],
      canSubscribe: true,
      canPublishData: false,
      canUpdateOwnMetadata: false,
    },
    roomConfig: {
      agents: [{ agentName: config.agentName }],
      maxParticipants: 2, // the visitor and the agent
      emptyTimeout: 30,
      departureTimeout: 10,
    },
  };
}
