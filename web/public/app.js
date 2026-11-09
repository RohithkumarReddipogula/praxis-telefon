// The page. livekit-client is bundled to /livekit-client.js by `npm run build` (needs
// `npm install`). Flow: notice -> captcha -> token -> connect -> agent greets -> live.
import { Room, RoomEvent, Track } from "/livekit-client.js";
import { AGENT_TIMEOUT_MS, FINAL_STATES, INITIAL, messageKey, next } from "/state.js";
import { notice, text } from "/texts.js";

const $ = (id) => document.getElementById(id);
let lang = (navigator.language || "de").startsWith("en") ? "en" : "de";
let state = INITIAL;
let config;
let captchaToken = "";
let room;
let timer;

function render() {
  document.documentElement.lang = lang;
  $("title").textContent = text(lang, "title");
  $("intro").textContent = text(lang, "intro");
  $("notice-title").textContent = text(lang, "notice_title");
  $("notice-list").replaceChildren(...notice(lang, config).map((line) => Object.assign(document.createElement("li"), { textContent: line })));
  $("lang").textContent = text(lang, "switch_language");
  $("start").textContent = text(lang, "consent_button");
  $("status").textContent = text(lang, messageKey(state));
  $("start").disabled = state !== "ready";
  $("start").hidden = state !== "notice" && state !== "ready";
}

function dispatch(event) {
  state = next(state, event);
  clearTimeout(timer);
  if (state === "waiting_agent") timer = setTimeout(() => dispatch("agent_timeout"), AGENT_TIMEOUT_MS);
  if (FINAL_STATES.includes(state) && room) room.disconnect();
  render();
}

async function start() {
  dispatch("start");
  let res;
  try {
    res = await fetch("/api/token", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ turnstile_token: captchaToken, lang }),
    });
  } catch { return dispatch("token_error"); }
  if (res.status === 503) return dispatch("token_busy");
  if (res.status === 429) return dispatch("token_visitor_limit");
  if (res.status === 403) return dispatch("token_captcha_failed");
  if (res.status !== 201) return dispatch("token_error");
  const { server_url, participant_token } = await res.json();
  dispatch("token_ok");
  room = new Room();
  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) { $("status").after(track.attach()); dispatch("agent_audio"); }
  });
  room.on(RoomEvent.ParticipantAttributesChanged, (attrs) => {
    if (attrs.demo_status === "busy") dispatch("agent_busy"); // set by the agent before it speaks
  });
  room.on(RoomEvent.Disconnected, () => dispatch("disconnected"));
  try {
    await room.connect(server_url, participant_token);
    await room.localParticipant.setMicrophoneEnabled(true);
  } catch (error) {
    return dispatch(error?.name === "NotAllowedError" ? "mic_denied" : "connect_failed");
  }
  dispatch("connected");
}

config = await (await fetch("/config.json")).json();
$("lang").addEventListener("click", () => { lang = lang === "de" ? "en" : "de"; render(); });
$("start").addEventListener("click", start);
window.onTurnstile = () => window.turnstile.render("#captcha", {
  sitekey: config.turnstileSiteKey,
  callback: (token) => { captchaToken = token; dispatch("captcha_ok"); },
  "expired-callback": () => { captchaToken = ""; dispatch("captcha_lost"); },
});
const wait = setInterval(() => { if (window.turnstile) { clearInterval(wait); window.onTurnstile(); } }, 200);
render();
