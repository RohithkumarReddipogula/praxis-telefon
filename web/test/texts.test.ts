import assert from "node:assert/strict";
import { test } from "node:test";
import { TEXTS, notice, text } from "../public/texts.js";

test("German and English have exactly the same keys", () => {
  assert.deepEqual(Object.keys(TEXTS.de).sort(), Object.keys(TEXTS.en).sort());
  for (const lang of ["de", "en"] as const) {
    for (const [key, value] of Object.entries(TEXTS[lang])) {
      assert.ok(String(value).trim().length > 0, `${lang}.${key}`);
    }
  }
});

test("the busy message is the one the user asked for", () => {
  assert.equal(text("de", "busy"), "Die Demo ist heute ausgelastet. Bitte versuchen Sie es morgen wieder.");
  assert.equal(text("en", "busy"), "The demo is busy today, please try again tomorrow.");
});

test("the privacy notice says what matters, in both languages", () => {
  const ctx = { callMinutes: 5, contactEmail: "kontakt@example.org" };
  const de = notice("de", ctx).join(" ");
  const en = notice("en", ctx).join(" ");
  for (const [lang, s] of [["de", de], ["en", en]] as const) {
    for (const word of ["Deepgram", "ElevenLabs", "OpenRouter", "LiveKit", "Cloudflare", "Railway"]) {
      assert.ok(s.includes(word), `${lang}: ${word}`);
    }
    assert.ok(s.includes("5"), `${lang}: call length`);
    assert.ok(s.includes("kontakt@example.org"), `${lang}: contact`);
  }
  assert.match(de, /erfunden/);
  assert.match(de, /keine echten/);
  assert.match(de, /keine Aufnahme/);
  assert.match(en, /fictional/);
  assert.match(en, /no real/i);
  assert.match(en, /not recorded|no recording/i);
});

test("the notice makes no claim that everything stays in the EU", () => {
  const ctx = { callMinutes: 5, contactEmail: "kontakt@example.org" };
  for (const lang of ["de", "en"] as const) {
    const s = notice(lang, ctx).join(" ");
    assert.ok(!/alles in der EU|everything (stays )?in the EU|nur in der EU|only in the EU/i.test(s));
  }
});

test("no postal address and no placeholders are left in the texts", () => {
  const ctx = { callMinutes: 5, contactEmail: "kontakt@example.org" };
  for (const lang of ["de", "en"] as const) {
    const s = notice(lang, ctx).join(" ") + Object.values(TEXTS[lang]).join(" ");
    assert.ok(!/\{[a-zA-Z]+\}/.test(s), "unfilled placeholder");
    assert.ok(!/\b\d{5}\b/.test(s), "something that looks like a postcode");
  }
});
