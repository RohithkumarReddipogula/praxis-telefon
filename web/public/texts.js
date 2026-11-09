// Page texts, German and English. Same keys in both (test/texts.test.ts). The privacy
// notice has placeholders {callMinutes} and {contactEmail} that notice() fills.

export const TEXTS = {
  de: {
    title: "Demo: KI-Telefonassistent einer Arztpraxis",
    intro:
      "Sprechen Sie mit einem KI-Assistenten, der Termine für die erfundene Hausarztpraxis Dr. Weber vergibt. Das Gespräch läuft über Ihr Mikrofon.",
    notice_title: "Wichtiger Hinweis",
    consent_button: "Hinweis gelesen – Demo starten",
    switch_language: "English",
    captcha_wait: "Bitte warten Sie kurz, die Sicherheitsprüfung läuft.",
    ready: "Bereit. Mit dem Knopf starten Sie das Gespräch und erlauben das Mikrofon.",
    connecting: "Verbindung wird aufgebaut …",
    waiting_agent: "Der Assistent meldet sich gleich …",
    live: "Das Gespräch läuft. Sprechen Sie einfach los.",
    busy: "Die Demo ist heute ausgelastet. Bitte versuchen Sie es morgen wieder.",
    visitor_limit: "Sie haben die Demo heute schon genutzt. Bitte versuchen Sie es morgen wieder.",
    captcha_failed: "Die Sicherheitsprüfung ist fehlgeschlagen. Bitte laden Sie die Seite neu.",
    mic_denied: "Ohne Zugriff auf das Mikrofon kann die Demo nicht starten. Bitte erlauben Sie es und laden Sie die Seite neu.",
    ended: "Das Gespräch ist beendet. Danke für Ihren Test!",
    remaining: "Verbleibende Zeit",
    contact: "Kontakt",
  },
  en: {
    title: "Demo: AI phone assistant for a doctor's office",
    intro:
      "Talk to an AI assistant that books appointments for the fictional practice Hausarztpraxis Dr. Weber. The conversation uses your microphone.",
    notice_title: "Important notice",
    consent_button: "I have read the notice – start the demo",
    switch_language: "Deutsch",
    captcha_wait: "Please wait a moment, the security check is running.",
    ready: "Ready. The button starts the conversation and asks for microphone access.",
    connecting: "Connecting …",
    waiting_agent: "The assistant will answer in a moment …",
    live: "The conversation is running. Just start talking.",
    busy: "The demo is busy today, please try again tomorrow.",
    visitor_limit: "You have already used the demo today. Please try again tomorrow.",
    captcha_failed: "The security check failed. Please reload the page.",
    mic_denied: "The demo cannot start without microphone access. Please allow it and reload the page.",
    ended: "The conversation has ended. Thank you for trying it!",
    remaining: "Time left",
    contact: "Contact",
  },
};

const NOTICE = {
  de: [
    "Dies ist eine Demo mit einer erfundenen Praxis (Hausarztpraxis Dr. Weber). Alle Personen und Termine sind erfunden.",
    "Bitte nennen Sie keine echten Gesundheits- oder persönlichen Daten. Erfinden Sie Namen und Geburtsdaten.",
    "Ihre Stimme wird zur Verarbeitung an diese Dienste übertragen: Deepgram (Spracherkennung), OpenRouter mit dem Sprachmodell Claude von Anthropic (Antworten), ElevenLabs (Sprachausgabe) und LiveKit Cloud (Audioübertragung, EU-Projekt, Assistent in Frankfurt). Der Webserver läuft bei Railway (Amsterdam); dessen technische Protokolle können außerhalb der EU liegen. Cloudflare Turnstile prüft, dass Sie ein Mensch sind, und kann dazu Ihre IP-Adresse verarbeiten.",
    "Es gibt keine Aufnahme und keine Speicherung des Gesprächs durch uns. Unsere Protokolle enthalten nur technische Daten wie Zähler und Zeiten.",
    "Ihre IP-Adresse wird nur als verschlüsselter Schlüsselwert gespeichert, um Missbrauch zu begrenzen, und nach dem Tag gelöscht.",
    "Ein Gespräch dauert höchstens {callMinutes} Minuten. Die Zahl der Gespräche pro Tag und Monat ist begrenzt.",
    "Kontakt: {contactEmail}",
  ],
  en: [
    "This is a demo with a fictional practice (Hausarztpraxis Dr. Weber). All people and appointments are made up.",
    "Please give no real health or personal data. Make up names and dates of birth.",
    "Your voice is sent for processing to these services: Deepgram (speech recognition), OpenRouter with Anthropic's Claude language model (answers), ElevenLabs (speech output) and LiveKit Cloud (audio transport, EU project, assistant in Frankfurt). The web server runs at Railway (Amsterdam); its technical logs may be stored outside the EU. Cloudflare Turnstile checks that you are human and may process your IP address for that.",
    "The conversation is not recorded and not stored by us. Our logs hold technical data only, such as counters and times.",
    "Your IP address is stored only as a keyed hash, to limit abuse, and erased after the day.",
    "A conversation lasts at most {callMinutes} minutes. The number of conversations per day and month is limited.",
    "Contact: {contactEmail}",
  ],
};

export function text(lang, key) {
  return TEXTS[lang][key];
}

export function notice(lang, ctx) {
  return NOTICE[lang].map((line) =>
    line.replace("{callMinutes}", String(ctx.callMinutes)).replace("{contactEmail}", ctx.contactEmail),
  );
}
