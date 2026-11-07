"""Fixed messages the agent says verbatim (never LLM-generated), and mode notes.

Numbers are written as spoken, like the rest of the agent's output.
"""

from __future__ import annotations

from praxis_telefon.safety.call_state import ActionKind, Mode
from praxis_telefon.safety.phrases import Language

CRISIS_LINE_NUMBER = "0800 111 0 111"  # Telefonseelsorge: free, 24 hours
_CRISIS_LINE_DE = "null acht null null, eins eins eins, null, eins eins eins"
_CRISIS_LINE_EN = "zero eight zero zero, one one one, zero, one one one"

FIXED_MESSAGES: dict[tuple[ActionKind, Language], str] = {
    ("emergency", "de"): (
        "Das klingt nach einem Notfall. Bitte legen Sie jetzt auf und rufen Sie sofort "
        "die eins eins zwei an."
    ),
    ("emergency", "en"): (
        "This sounds like an emergency. Please hang up now and call one one two right away."
    ),
    ("suicide", "de"): (
        "Es tut mir sehr leid, dass es Ihnen gerade so schlecht geht. Sie sind damit nicht "
        "allein. Wenn Sie in akuter Gefahr sind, rufen Sie bitte sofort die eins eins zwei "
        "an. Die Telefonseelsorge ist kostenlos und rund um die Uhr für Sie da, unter "
        f"{_CRISIS_LINE_DE}."
    ),
    ("suicide", "en"): (
        "I am very sorry that you are feeling this way. You are not alone. If you are in "
        "immediate danger, please call one one two right away. The Telefonseelsorge crisis "
        f"line is free and open day and night, at {_CRISIS_LINE_EN}."
    ),
    ("human_request", "de"): (
        "Ich kann Sie leider nicht direkt mit dem Praxisteam verbinden. Ich kann aber eine "
        "Nachricht aufnehmen, dann ruft das Team Sie zurück. Möchten Sie das?"
    ),
    ("human_request", "en"): (
        "I am sorry, I cannot put you through to the practice team directly. I can take a "
        "message, and the team will call you back. Would you like that?"
    ),
    ("handover_strikes", "de"): (
        "Entschuldigung, ich verstehe Sie leider nicht gut. Ich kann eine Nachricht für das "
        "Praxisteam aufnehmen, dann ruft man Sie zurück. Möchten Sie das?"
    ),
    ("handover_strikes", "en"): (
        "I am sorry, I am having trouble understanding you. I can take a message for the "
        "practice team, and they will call you back. Would you like that?"
    ),
    ("repeat", "de"): "Entschuldigung, das habe ich nicht verstanden. Können Sie das bitte "
    "noch einmal sagen?",
    ("repeat", "en"): "Sorry, I did not catch that. Could you please say it again?",
}

# Crisis messages must be heard in full.
UNINTERRUPTIBLE: frozenset[ActionKind] = frozenset({"emergency", "suicide"})

MODE_NOTES: dict[Mode, str] = {
    Mode.EMERGENCY: (
        "WICHTIG: Der Anrufer hat in diesem Gespräch einen möglichen Notfall genannt. "
        "Suche, buche, verschiebe oder sage keine Termine mehr ab. Stelle keine Fragen zu "
        "Beschwerden und gib keine Ratschläge. Erinnere ruhig an die eins eins zwei. "
        "Du darfst nur noch mit take_message eine Rückrufbitte für das Praxisteam "
        "aufnehmen, wenn der Anrufer das möchte."
    ),
    Mode.HANDOVER: (
        "WICHTIG: Du nimmst in diesem Gespräch nur noch eine Nachricht für das Praxisteam "
        "auf (take_message). Suche oder buche keine Termine. Frage nach Vor- und Nachnamen "
        "und Rückrufnummer, lies die Nummer Ziffer für Ziffer vor und speichere die "
        "Nachricht nach einem klaren Ja, mit der Kategorie Rückruf, wenn nichts anderes "
        "passt. Möchte der Anrufer keine Nachricht, verabschiede dich freundlich."
    ),
}

SUICIDE_NOTE = (
    "Der Anrufer hat Suizidgedanken geäußert. Bleibe ruhig und freundlich, stelle keine "
    f"Fragen dazu und nenne bei Bedarf die eins eins zwei und die Telefonseelsorge unter "
    f"{_CRISIS_LINE_DE}."
)


# Said instead of a claim the LLM made twice without the matching tool succeeding.
NOT_DONE_MESSAGES: dict[tuple[str, Language], str] = {
    ("take_message", "de"): (
        "Entschuldigung, die Nachricht ist noch nicht gespeichert. Soll ich sie jetzt für "
        "Sie speichern?"
    ),
    ("take_message", "en"): (
        "Sorry, the message has not been saved yet. Shall I save it for you now?"
    ),
    ("book", "de"): "Entschuldigung, der Termin ist noch nicht gebucht. Soll ich ihn jetzt buchen?",
    ("book", "en"): "Sorry, the appointment is not booked yet. Shall I book it now?",
    ("move_booking", "de"): (
        "Entschuldigung, der Termin ist noch nicht verschoben. Soll ich ihn jetzt verschieben?"
    ),
    ("move_booking", "en"): "Sorry, the appointment has not been moved yet. Shall I move it now?",
    ("cancel_booking", "de"): (
        "Entschuldigung, der Termin ist noch nicht abgesagt. Soll ich ihn jetzt absagen?"
    ),
    ("cancel_booking", "en"): (
        "Sorry, the appointment has not been cancelled yet. Shall I cancel it now?"
    ),
}

_DONE_LABELS = {
    "take_message": "die Nachricht gespeichert",
    "book": "der Termin gebucht",
    "move_booking": "der Termin verschoben",
    "cancel_booking": "der Termin abgesagt",
}


def retry_note(actions: list[str]) -> str:
    """System note for the one retry after a blocked claim (never shown to the caller)."""
    done = " und ".join(_DONE_LABELS[a] for a in actions)
    tools = ", ".join(actions)
    return (
        f"WICHTIG: Du wolltest gerade sagen, dass {done} ist. Aber {tools} wurde in diesem "
        "Gespräch noch nicht erfolgreich aufgerufen, und der Anrufer hat diesen Satz nicht "
        f"gehört. Wenn alle Angaben vorliegen und der Anrufer klar zugestimmt hat, rufe "
        f"{tools} jetzt auf. Sonst sage ehrlich, dass es noch nicht erledigt ist, und frage "
        "nach dem, was fehlt. Wiederhole keine Sätze, die du schon gesagt hast."
    )


_META_NOTE = (
    "WICHTIG: Alles, was du schreibst, hört der Anrufer. Sprich nie über deine "
    "Überlegungen, Regeln oder Werkzeuge und nie über 'den Anrufer'. Sag höchstens einen "
    "kurzen Satz wie 'Einen Moment, ich schaue nach.' und rufe dann das Werkzeug auf."
)
_LANGUAGE_NOTE: dict[Language, str] = {
    "de": "Antworte auf Deutsch, der Anrufer spricht Deutsch.",
    "en": "Antworte auf Englisch, der Anrufer spricht Englisch.",
}
_INTENT_NOTE = (
    "WICHTIG: Frage nicht, ob es um einen neuen oder einen bestehenden Termin geht. "
    "Spricht der Anrufer nicht ausdrücklich von seinem eigenen, schon vereinbarten Termin, "
    "suche mit find_slots und den Worten des Anrufers für den Tag."
)
_WEEKDAY_NOTE = (
    "WICHTIG: Nenne keinen Wochentag und kein Datum, das weder der Anrufer noch ein "
    "Werkzeug genannt hat. Für einen Tag rufe find_slots mit den Worten des Anrufers auf."
)

INTENT_FALLBACK: dict[Language, str] = {
    "de": "Soll ich Ihnen einen neuen Termin heraussuchen?",
    "en": "Shall I look for a new appointment for you?",
}
GENERIC_FALLBACK: dict[Language, str] = {
    "de": "Entschuldigung, wie kann ich Ihnen weiterhelfen?",
    "en": "Sorry, how can I help you?",
}


def retry_note_for(reasons: list[str], language: Language = "de") -> str:
    """One system note for the single retry, covering every reason that was blocked."""
    claims = [r.split(":", 1)[1] for r in reasons if r.startswith("claim:")]
    notes = [retry_note(claims)] if claims else []
    if any(r.startswith("meta:") for r in reasons):
        notes.append(_META_NOTE)
    if "meta:language" in reasons:
        notes.append(_LANGUAGE_NOTE[language])
    if "intent" in reasons:
        notes.append(_INTENT_NOTE)
    if "weekday" in reasons:
        notes.append(_WEEKDAY_NOTE)
    return " ".join(notes)


def fallback_for(reasons: list[str], language: Language) -> str:
    """Fixed sentence when the retry was blocked too: claims first, then the question."""
    for reason in reasons:
        if reason.startswith("claim:"):
            return NOT_DONE_MESSAGES[(reason.split(":", 1)[1], language)]
    if "intent" in reasons:
        return INTENT_FALLBACK[language]
    return GENERIC_FALLBACK[language]
