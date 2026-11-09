// The page as a state machine: every event leads to a state, and every final state is a
// message; there is no state in which the visitor is left without an explanation.

export const INITIAL = "notice";
export const AGENT_TIMEOUT_MS = 15_000; // the agent must answer within this, else "busy"

const TRANSITIONS = {
  notice: { captcha_ok: "ready" },
  ready: { captcha_lost: "notice", start: "requesting" },
  requesting: {
    token_ok: "connecting",
    token_busy: "busy",
    token_visitor_limit: "visitor_limit",
    token_captcha_failed: "captcha_failed",
    token_error: "busy", // an endpoint error reads as busy
  },
  connecting: {
    connected: "waiting_agent",
    agent_audio: "live",
    agent_busy: "busy",
    mic_denied: "mic_denied",
    connect_failed: "busy",
  },
  waiting_agent: {
    agent_audio: "live",
    agent_busy: "busy",
    agent_timeout: "busy", // no answer: the agent or its voice is down
    disconnected: "busy", // the room closed before anything was heard
  },
  live: { agent_busy: "busy", disconnected: "ended" },
};

export function next(state, event) {
  return TRANSITIONS[state]?.[event] ?? state;
}

const MESSAGE_KEYS = {
  notice: "captcha_wait",
  ready: "ready",
  requesting: "connecting",
  connecting: "connecting",
  waiting_agent: "waiting_agent",
  live: "live",
  busy: "busy",
  visitor_limit: "visitor_limit",
  captcha_failed: "captcha_failed",
  mic_denied: "mic_denied",
  ended: "ended",
};

export function messageKey(state) {
  return MESSAGE_KEYS[state];
}

export const FINAL_STATES = ["busy", "visitor_limit", "captcha_failed", "mic_denied", "ended"];
