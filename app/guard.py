"""Deterministic safety checks around the LLM: red-flag symptoms, unbacked booking claims,
medical advice, and PHI redaction for logs."""
import logging
import re
from datetime import date

from app import config

log = logging.getLogger("guard")

RED_FLAGS = re.compile(r"""
    (trouble|difficulty|hard|can'?t|cannot|unable)\s+(to\s+)?(breath|swallow)
  | short(ness)?\s+of\s+breath
  | swelling.{0,40}(spread|eye|neck|throat|getting\ (worse|bigger))
  | (face|cheek|jaw|neck)\s+(is\s+)?(really\s+|very\s+)?swell
  | (spread|spreading)\s+swelling
  | (won'?t|can'?t|cannot|doesn'?t|will\ not)\s+stop\s+(the\s+)?bleed | bleed\w*\s+(won'?t|doesn'?t|will\ not|isn'?t)\s+stop
  | (uncontrolled|heavy|heavily|lots\ of|a\ lot\ of)\s+bleed | bleeding\s+(heavily|a\ lot|nonstop|non-stop)
  | (broke|broken|dislocated|fractured|injured|hurt)\s+(my\s+)?jaw | jaw\s+(is\s+)?(broken|dislocated|injur)
  | (hit|punched|kicked|struck)\s+(me\s+)?(in|on)\s+the\s+(face|jaw|mouth) | (car|bike)\s+accident
""", re.I | re.X)

# Written by code, not the model, so it's always there and always says the same safe thing.
EMERGENCY_911 = ("I'm so sorry, that sounds really frightening. Symptoms like these can get serious quickly, so "
                 "please call 911 or go to the nearest emergency room now, and please don't wait for a dental "
                 "appointment. I've also let our dental team know it's urgent, so they can follow up with you.\n\n")

# Claims that an action happened. Reading back an existing appointment ("you have a cleaning on...")
# doesn't match; the system prompt steers the model to that phrasing.
CLAIM = re.compile(r"""
    \ball\ set\b(?!\s+up)
  | \byou'?re\s+(all\s+)?(booked|confirmed|scheduled|rescheduled)\b
  | \b(i'?ve|i\ have|i\ just|we'?ve|we\ have)\s+(gone\ ahead\ and\s+)?(booked|scheduled|confirmed|reserved|cancel+ed|rescheduled|moved|made)\b
  | \b(has|have)\s+been\s+(successfully\s+)?(booked|scheduled|confirmed|reserved|cancel+ed|rescheduled|moved)\b
  | \b(is|are)\s+(now\s+)?(booked|confirmed|cancel+ed|rescheduled)\b
  | \bsuccessfully\s+(booked|scheduled|cancel+ed|rescheduled)\b
  | \b(booking|appointment)\s+(is\s+)?confirmed\b
""", re.I | re.X)

# Must stay true whatever the real DB state is: we only know the claim wasn't verified.
UNBACKED_REPLY = ("I'm sorry, I couldn't confirm that in our system just now, so please don't count on it yet. "
                  "Would you like me to check your appointments or try again?")

# The subset of claims that say something was just done. These need an action this turn, or explicit
# dates matching appointments that exist; a bare "I've booked you!" is never backed by old state.
# ("You're all set!" is a state: models say it to restate a booking made a turn earlier.)
ACTION_CLAIM = re.compile(r"""
    \b(i'?ve|i\ have|i\ just|we'?ve|we\ have)\s+(gone\ ahead\ and\s+)?(booked|scheduled|confirmed|reserved|cancel+ed|rescheduled|moved|made)\b
  | \b(has|have)\s+been\s+(successfully\s+)?(booked|scheduled|confirmed|reserved|cancel+ed|rescheduled|moved)\b
  | \b(is|are)\s+now\s+(booked|confirmed|cancel+ed|rescheduled)\b
  | \bsuccessfully\s+(booked|scheduled|cancel+ed|rescheduled)\b
""", re.I | re.X)

APPOINTMENT_WORDS = re.compile(r"book|appointment|schedul|cancel|visit|cleaning|checkup|\b(at|on)\s+\d", re.I)
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH_DAY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)

MEDICAL = re.compile(r"""
    \b\d+\s?mg\b | \b(ibuprofen|advil|motrin|tylenol|acetaminophen|aspirin|naproxen|aleve|antibiotics?|amoxicillin|penicillin|orajel|benzocaine)\b
  | \b(salt\s?water|warm\s+water)\s+rinse | \brinse\s+with\b | \b(ice\s+pack|cold\s+compress|clove\s+oil)\b
  | \byou\s+(probably|likely|might|may)\s+have\s+(an?\s+)?(abscess|infection|cavity|cavities|gum\s+disease)\b
""", re.I | re.X)

MEDICAL_REPLY = ("I'm not able to give medical advice, but one of our dental team can call you back to help. "
                 "Would you like me to request a callback?")


def red_flag(text: str) -> bool:
    return bool(RED_FLAGS.search(text))


def month_days(text: str) -> set[tuple[int, int]]:
    return {(_MONTHS.index(m[1].lower()[:3]) + 1, int(m[2])) for m in _MONTH_DAY.finditer(text)}


def claim_backed(reply: str, booked: set[date]) -> bool:
    """Does a claim restate appointments that exist? `booked` holds the dates of upcoming appointments
    of patients this session may talk about. Every month-day and weekday the reply names must be one of
    them. "You're all set for Nov 5" passes on its date; "the kids are booked Thursday" (a state, no
    date) passes when it's true; "I've booked you!" with no date never passes on old state."""
    if not booked:
        return False
    named = month_days(reply)
    if not named <= {(d.month, d.day) for d in booked}:
        return False
    if not {w.lower() for w in _WEEKDAY.findall(reply)} <= {d.strftime("%A").lower() for d in booked}:
        return False
    return bool(named) or not ACTION_CLAIM.search(reply)


def check_reply(reply: str, action_succeeded: bool, session_id: str,
                booked: set[date] = frozenset(), registered: bool = False, action_failed: bool = False) -> str:
    """Last line of defence on the model's final reply. A claim is backed by a successful book/modify
    this turn, or by restating appointments that exist (claim_backed). After a successful
    registration, "you're all set in our system" is true, as long as it doesn't also talk about an
    appointment. If a book/modify failed this turn, only a success counts: "all set!" right after a
    slot_taken error is exactly the lie this guard exists for."""
    if CLAIM.search(reply) and not action_succeeded:
        registration_only = registered and not APPOINTMENT_WORDS.search(reply)
        restates = not action_failed and claim_backed(reply, booked)
        if not restates and not registration_only:
            log.warning("unbacked_claim session=%s reply=%r", session_id, redact(reply[:300]))
            return UNBACKED_REPLY
    if MEDICAL.search(reply):
        log.warning("medical_advice_blocked session=%s reply=%r", session_id, redact(reply[:300]))
        return MEDICAL_REPLY
    return reply


def canned_reply() -> str:
    return (f"I'm sorry, I'm having trouble on my end right now. Please call us at {config.OFFICE_PHONE} "
            "(Mon-Sat, 8am-6pm) and our team will help you right away.")


# --- PHI redaction for logs ----------------------------------------------------------------

_PHONE = re.compile(r"(?<!\d)(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_DATE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
                   r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(19|20)\d{2}\b", re.I)


def redact(text) -> str:
    """Mask phone numbers and full dates (DOBs). Applied to every log line by RedactFilter."""
    return _DATE.sub("[date]", _PHONE.sub("[phone]", str(text)))


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True
