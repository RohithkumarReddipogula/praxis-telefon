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
