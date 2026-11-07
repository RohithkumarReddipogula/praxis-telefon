"""Tool failures with stable codes.

Messages are fixed German sentences for the LLM to paraphrase. They must never contain
argument values (names, dates of birth, phone numbers); tests enforce this.
"""

from __future__ import annotations


class ToolFailure(Exception):
    code: str = "tool_failure"
    message: str = "Das hat leider nicht geklappt."

    def __init__(self) -> None:
        super().__init__(self.message)


class InvalidDateRange(ToolFailure):
    code = "invalid_date_range"
    message = "Der Zeitraum ist ungültig. Bitte höchstens 28 Tage ab heute angeben."


class InvalidPartOfDay(ToolFailure):
    code = "invalid_part_of_day"
    message = "Die Tageszeit muss morning, afternoon oder any sein."


class DoctorNotFound(ToolFailure):
    code = "doctor_not_found"
    message = "Diese Ärztin bzw. diesen Arzt gibt es in der Praxis nicht."


class SlotNotFound(ToolFailure):
    code = "slot_not_found"
    message = "Diesen Termin gibt es nicht. Bitte zuerst freie Termine suchen."


class SlotNotOffered(ToolFailure):
    code = "slot_not_offered"
    message = "Dieser Termin wurde in diesem Gespräch nicht angeboten. Bitte zuerst suchen."


class SlotInPast(ToolFailure):
    code = "slot_in_past"
    message = "Dieser Termin liegt in der Vergangenheit."


class SlotTaken(ToolFailure):
    code = "slot_taken"
    message = "Dieser Termin ist leider schon vergeben. Bitte einen anderen Termin suchen."


class InvalidName(ToolFailure):
    code = "invalid_name"
    message = "Bitte den vollständigen Vor- und Nachnamen angeben."


class InvalidDateOfBirth(ToolFailure):
    code = "invalid_date_of_birth"
    message = "Das Geburtsdatum ist ungültig. Bitte im Format Jahr-Monat-Tag angeben."


class BookingNotFound(ToolFailure):
    code = "booking_not_found"
    message = "Dazu habe ich keinen Termin gefunden."


class BookingNotVerified(ToolFailure):
    code = "booking_not_verified"
    message = "Bitte den Termin zuerst mit Name und Geburtsdatum suchen."


class BookingNotActive(ToolFailure):
    code = "booking_not_active"
    message = "Dieser Termin ist bereits abgesagt oder schon vorbei."


class InvalidCallbackNumber(ToolFailure):
    code = "invalid_callback_number"
    message = (
        "Die Rückrufnummer ist ungültig. callback_number nur als Ziffern übergeben, also "
        "030 statt null drei null, und eine deutsche Telefonnummer verwenden."
    )


class InvalidReasonCategory(ToolFailure):
    code = "invalid_reason_category"
    message = "Bitte eine der vorgegebenen Kategorien für das Anliegen wählen."


class CallbackNumberMismatch(ToolFailure):
    # M4 harness rerun 2026-11-07: the LLM turned the caller's spoken digits into a wrong
    # number; the caller heard the right one read back, the team got the wrong one.
    code = "callback_number_mismatch"
    message = (
        "Diese Nummer hat der Anrufer nicht genannt. Übergib die Rückrufnummer genau in den "
        "Worten des Anrufers, zum Beispiel null drei null, zwei drei eins zwei fünf; sie "
        "wird im Code umgewandelt."
    )


class BookingNotForCaller(ToolFailure):
    # Privacy (M4 baseline 2026-11-07): only the caller's own appointments are looked up.
    code = "booking_not_for_caller"
    message = (
        "Termine kann ich nur für die anrufende Person selbst nachsehen. Nenne keine Termine "
        "anderer Personen und frage nicht nach deren Geburtsdatum. Für Änderungen am Termin "
        "einer anderen Person biete an, eine Nachricht für das Praxisteam aufzunehmen; einen "
        "neuen Termin für eine andere Person kannst du buchen."
    )


class InvalidTopic(ToolFailure):
    # FAQ measurement 2026-11-07: the LLM sent "weekend"; it must pick a listed topic.
    code = "invalid_topic"
    message = (
        "Unbekanntes Thema. Wähle eines von: opening_hours, address, parking, "
        "repeat_prescription, referral, sick_note, holidays, insurance_card, "
        "new_patients, other."
    )


class BookingDisabled(ToolFailure):
    code = "booking_disabled"
    message = (
        "In diesem Gespräch sind keine Terminbuchungen mehr möglich. Nur noch eine "
        "Nachricht für das Praxisteam aufnehmen, wenn der Anrufer das möchte."
    )


class DateNotUnderstood(ToolFailure):
    code = "date_not_understood"
    message = (
        "Diesen Tag oder Zeitraum habe ich nicht verstanden. Den Anrufer nach einem "
        "konkreten Tag fragen, zum Beispiel morgen, nächsten Dienstag oder 12. November."
    )


class DateInPast(ToolFailure):
    code = "date_in_past"
    message = "Dieser Tag liegt in der Vergangenheit. Nach einem späteren Tag fragen."


class DateTooFar(ToolFailure):
    code = "date_too_far"
    message = "Termine gibt es nur für die nächsten vier Wochen. Nach einem früheren Tag fragen."
