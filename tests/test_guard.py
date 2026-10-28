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
