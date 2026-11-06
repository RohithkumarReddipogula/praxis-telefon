"""Phrase groups: what must match, what must not, and known over-triggers."""

import pytest

from praxis_telefon.safety.phrases import asks_for_human, detect_crisis, is_short_answer, normalize

EMERGENCY_DE = [
    "Ich habe starke Brustschmerzen.",
    "Ich habe seit einer Stunde Schmerzen in der Brust.",
    "Mein Mann hat Schmerzen in der linken Brust und schwitzt.",
    "Ich habe so einen Druck auf der Brust.",
    "Ich glaube, ich habe einen Herzinfarkt.",
    "Ich bekomme keine Luft.",
    "Meine Mutter kriegt kaum Luft.",
    "Ich habe Atemnot.",
    "Ich kann nicht mehr richtig atmen.",
    "Mein Kind atmet nicht.",
    "Ich glaube, mein Vater hat einen Schlaganfall.",
    "Ihr Gesicht hängt auf einer Seite.",
    "Mein Arm ist plötzlich gelähmt.",
    "Er kann auf einmal nicht mehr sprechen.",
    "Sie spricht ganz verwaschen.",
    "Mein Opa ist bewusstlos.",
    "Meine Frau ist ohnmächtig geworden.",
    "Er ist nicht mehr ansprechbar.",
    "Ich habe eine starke Blutung am Bein.",
    "Die Wunde blutet sehr stark.",
    "Es hört einfach nicht auf zu bluten.",
    "Mein Sohn hat einen allergischen Schock.",
]

EMERGENCY_EN = [
    "I have chest pain.",
    "There is a pain in my chest.",
    "I feel pressure on my chest.",
    "I think I'm having a heart attack.",
    "I can't breathe.",
    "My father has trouble breathing.",
    "I'm short of breath.",
    "I think she had a stroke.",
    "His face is drooping.",
    "My husband passed out.",
    "She is unconscious.",
    "The cut is bleeding heavily.",
    "It won't stop bleeding.",
]

SUICIDE_DE = [
    "Ich will nicht mehr leben.",
    "Ich möchte einfach nur sterben.",
    "Ich denke an Selbstmord.",
    "Ich habe Suizidgedanken.",
    "Ich will mich umbringen.",
    "Ich will mir das Leben nehmen.",
    "Ich habe Angst, dass ich mir etwas antue.",
    "Ich bin lebensmüde.",
]

SUICIDE_EN = [
    "I want to kill myself.",
    "I don't want to live anymore.",
    "I want to die.",
    "I'm thinking about suicide.",
    "I want to end my life.",
    "I'm afraid I might hurt myself.",
]

NO_CRISIS = [
    "Ich hätte gern einen Termin wegen Rückenschmerzen.",
    "Mein Kind hatte gestern Fieber.",
    "Ich habe leichte Halsschmerzen.",
    "Ich habe Schmerzen im Bein.",
    "Ich habe seit gestern Kopfschmerzen.",
    "Ich rufe wegen meiner Blutwerte an.",
    "Ich brauche einen Termin zur Blutabnahme.",
    "Ich brauche eine Überweisung zum Kardiologen.",
    "Mein Asthmaspray ist leer, ich brauche ein Rezept.",
    "Ich kann gerade nicht sprechen, rufen Sie mich bitte zurück.",
    "Ich möchte meinen Termin absagen.",
    "Ich bin heute etwas schwindelig.",
    "Der Film war zum Sterben langweilig.",
    "Ich habe keine Lust mehr zu warten.",
    "Brustkrebsvorsorge, haben Sie da einen Termin?",
    "Ich möchte zur Vorsorge kommen.",
    "My back hurts.",
    "I need a prescription refill.",
    "I'd like an appointment for a check-up.",
    "Is the practice open tomorrow?",
    "I can't talk right now, please call me back.",
    "My child has a fever.",
]

# Over-triggers by design: negation and history are not understood. Safer than a miss.
KNOWN_OVER_TRIGGERS = [
    "Ich habe keine Brustschmerzen mehr.",
    "Mein Vater hatte vor Jahren einen Schlaganfall und braucht ein Rezept.",
]


@pytest.mark.parametrize("text", EMERGENCY_DE)
def test_german_emergency_matches(text: str) -> None:
    crisis = detect_crisis(text)
    assert crisis is not None and (crisis.kind, crisis.language) == ("emergency", "de")


@pytest.mark.parametrize("text", EMERGENCY_EN)
def test_english_emergency_matches(text: str) -> None:
    crisis = detect_crisis(text)
    assert crisis is not None and (crisis.kind, crisis.language) == ("emergency", "en")


@pytest.mark.parametrize("text", SUICIDE_DE)
def test_german_suicide_matches(text: str) -> None:
    crisis = detect_crisis(text)
    assert crisis is not None and (crisis.kind, crisis.language) == ("suicide", "de")


@pytest.mark.parametrize("text", SUICIDE_EN)
def test_english_suicide_matches(text: str) -> None:
    crisis = detect_crisis(text)
    assert crisis is not None and (crisis.kind, crisis.language) == ("suicide", "en")


def test_suicide_wins_over_emergency() -> None:
    crisis = detect_crisis("Ich habe Brustschmerzen und will nicht mehr leben.")
    assert crisis is not None and crisis.kind == "suicide"


@pytest.mark.parametrize("text", NO_CRISIS)
def test_everyday_requests_do_not_match(text: str) -> None:
    assert detect_crisis(text) is None


@pytest.mark.parametrize("text", KNOWN_OVER_TRIGGERS)
def test_known_over_triggers_still_match(text: str) -> None:
    assert detect_crisis(text) is not None


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Kann ich bitte mit einem Menschen sprechen?", "de"),
        ("Ich möchte mit einer echten Person reden.", "de"),
        ("Verbinden Sie mich bitte mit der Rezeption.", "de"),
        ("Ich will mit einem Mitarbeiter sprechen.", "de"),
        ("Ich möchte mit der Sprechstundenhilfe sprechen.", "de"),
        ("Ich will nicht mit einer KI reden.", "de"),
        ("Können Sie mich durchstellen?", "de"),
        ("Can I speak to a real person?", "en"),
        ("I want to talk to a human.", "en"),
        ("Please transfer me to the receptionist.", "en"),
        ("I want to speak to someone.", "en"),
    ],
)
def test_human_request_matches(text: str, language: str) -> None:
    assert asks_for_human(text) == language


@pytest.mark.parametrize(
    "text",
    [
        "Ich hätte gern einen Termin bei Dr. Weber.",
        "Mein Name ist Erika Mustermann.",
        "Bitte richten Sie eine Nachricht aus.",
        "Wann haben Sie geöffnet?",
        "I would like an appointment.",
    ],
)
def test_ordinary_requests_are_not_human_requests(text: str) -> None:
    assert asks_for_human(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Ja",
        "Ja.",
        "Nein",
        "Genau.",
        "Okay",
        "Richtig",
        "Ja, genau.",
        "Nein, danke.",
        "Mhm.",
        "Yes",
        "Yeah, right.",
        "No",
        "Correct.",
        "12",
        "fünfzehn",
        "neunzehnhundertfünfundachtzig",
        "zwei drei eins zwei fünf",
        "the first one",
        "Mustermann",
        "Erika Mustermann.",
        "Hans-Peter Müller",
    ],
)
def test_short_answers(text: str) -> None:
    assert is_short_answer(text)


@pytest.mark.parametrize(
    "text",
    ["", "Äh.", "Hm hm.", "ähm, ja, also der", "ich weiß nicht genau was", "der die das wo"],
)
def test_not_short_answers(text: str) -> None:
    assert not is_short_answer(text)


def test_normalize_folds_umlauts_and_punctuation() -> None:
    assert normalize("Lähmung, Größe! Can't") == " laehmung groesse cant "
