// Signs the participant token with the LiveKit server SDK. Needs `npm install`
// (livekit-server-sdk); everything it signs is decided in token.ts.

import { AccessToken, RoomAgentDispatch, RoomConfiguration, TrackSource } from "livekit-server-sdk";
import type { Config } from "./config.ts";
import type { TokenSpec } from "./token.ts";

export async function signToken(config: Config, spec: TokenSpec): Promise<string> {
  const token = new AccessToken(config.apiKey, config.apiSecret, {
    identity: spec.identity,
    ttl: spec.ttlSeconds,
    attributes: spec.attributes,
  });
  // The SDK wants its TrackSource enum, not the strings of the spec.
  const sources = spec.grant.canPublishSources.map((name) => {
    if (name !== "microphone") throw new Error("only the microphone may be published");
    return TrackSource.MICROPHONE;
  });
  token.addGrant({ ...spec.grant, canPublishSources: sources });
  token.roomConfig = new RoomConfiguration({
    agents: spec.roomConfig.agents.map((a) => new RoomAgentDispatch({ agentName: a.agentName })),
    maxParticipants: spec.roomConfig.maxParticipants,
    emptyTimeout: spec.roomConfig.emptyTimeout,
    departureTimeout: spec.roomConfig.departureTimeout,
  });
  return token.toJwt();
}
