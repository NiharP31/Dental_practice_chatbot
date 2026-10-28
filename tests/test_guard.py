from datetime import date

import pytest

from app import guard


@pytest.mark.parametrize("text", [
    "I'm having trouble breathing since my tooth was pulled", "it's hard to swallow",
    "the swelling is spreading to my eye", "my face is really swollen and swelling into my neck",
    "the bleeding won't stop", "I think I broke my jaw", "got hit in the face playing hockey",
])
def test_red_flags(text):
    assert guard.red_flag(text)


@pytest.mark.parametrize("text", [
    "I have a toothache", "my gums bleed a little when I brush", "I need a cleaning", "chipped my tooth on popcorn",
])
def test_not_red_flags(text):
    assert not guard.red_flag(text)


@pytest.mark.parametrize("reply", [
    "You're all set for Thursday at 9!", "I've booked you in for Tuesday.", "Your appointment has been cancelled.",
    "Great news, your cleaning is confirmed.", "I've rescheduled it to Friday.",
])
def test_unbacked_claims_are_replaced(reply):
    assert guard.check_reply(reply, action_succeeded=False, session_id="s") == guard.UNBACKED_REPLY
    assert guard.check_reply(reply, action_succeeded=True, session_id="s") == reply


@pytest.mark.parametrize("reply", [
    "You have a cleaning on Thursday, October 29 at 9:00 AM.", "Would you like me to book that?",
    "I've let our dental team know about the emergency.", "Once you confirm, I'll make the change.",
])
def test_honest_replies_pass(reply):
    assert guard.check_reply(reply, action_succeeded=False, session_id="s") == reply


def test_medical_advice_is_blocked():
    assert guard.check_reply("Try 400mg of ibuprofen and a salt water rinse.", False, "s") == guard.MEDICAL_REPLY


def test_redact_masks_phone_and_dob():
    out = guard.redact("Maria, (555) 010-1002, born 07/25/1986 or 1986-07-25 or July 25, 1986, slot 42")
    assert "010-1002" not in out and "1986" not in out and "slot 42" in out


THU_NOV_5, FRI_NOV_6 = date(2026, 11, 5), date(2026, 11, 6)


def test_claim_backed_by_existing_appointment_dates():
    reply = "You're already all set! Your cleaning is confirmed for Thursday, November 5 at 8:00 AM."
    assert guard.check_reply(reply, False, "s", booked={THU_NOV_5}) == reply
    assert guard.check_reply(reply, False, "s", booked={FRI_NOV_6}) == guard.UNBACKED_REPLY
    both = "You're all set for Nov 5 and Nov 9."  # one real date doesn't vouch for an invented one
    assert guard.check_reply(both, False, "s", booked={THU_NOV_5}) == guard.UNBACKED_REPLY
    # Gemini eval regression: "You're all set, Sam!" a turn after a real booking is a true restatement...
    assert guard.check_reply("You're all set, Sam!", False, "s", booked={THU_NOV_5}) == "You're all set, Sam!"
    # ...but not right after a booking failed (e.g. slot_taken), and not when nothing is booked.
    assert guard.check_reply("You're all set!", False, "s", booked={THU_NOV_5}, action_failed=True) == guard.UNBACKED_REPLY
    assert guard.check_reply("You're all set!", False, "s") == guard.UNBACKED_REPLY


def test_state_claims_about_existing_appointments_without_dates():
    """Eval regression: right after verifying, 'the kids are booked Thursday at 3 and 4' is true."""
    kids = "You're verified, Priya. Arjun and Anika are booked for cleanings Thursday at 3:00 and 4:00 PM."
    assert guard.check_reply(kids, False, "s", booked={THU_NOV_5}) == kids
    assert guard.check_reply(kids, False, "s", booked={FRI_NOV_6}) == guard.UNBACKED_REPLY  # wrong weekday
    assert guard.check_reply(kids, False, "s") == guard.UNBACKED_REPLY  # nothing booked at all
    assert guard.check_reply("I've booked you in!", False, "s", booked={THU_NOV_5}) == guard.UNBACKED_REPLY


def test_registration_turn_allows_all_set_but_not_booking_claims():
    ok = "You're all set in our system, Sam. What day works best for you?"
    assert guard.check_reply(ok, False, "s", registered=True) == ok
    assert guard.check_reply(ok, False, "s") == guard.UNBACKED_REPLY
    booked = "You're all set! Your cleaning is booked."
    assert guard.check_reply(booked, False, "s", registered=True) == guard.UNBACKED_REPLY
    # Gemini eval regression: registration + options in one reply. "All set up" is never a booking claim.
    setup = "Thanks, Sam! You're all set up in our system. Here are cleanings next week: Monday, November 9 at 8:00 AM"
    assert guard.check_reply(setup, False, "s", registered=True) == setup
