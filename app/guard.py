"""Deterministic safety checks around the LLM: red-flag symptoms, unbacked booking claims,
medical advice, and PHI redaction for logs."""
import logging
import re

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

EMERGENCY_911 = ("If you're having trouble breathing or swallowing, swelling that's spreading, bleeding that "
                 "won't stop, or a jaw injury, please call 911 or go to the nearest emergency room right now. "
                 "I've alerted our team as urgent too.\n\n")

# Claims that an action happened. Reading back an existing appointment ("you have a cleaning on...")
# doesn't match; the system prompt steers the model to that phrasing.
CLAIM = re.compile(r"""
    \ball\ set\b
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

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH_DAY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)

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


def check_reply(reply: str, action_succeeded: bool, session_id: str,
                booked_days: set[tuple[int, int]] = frozenset()) -> str:
    """Last line of defence on the model's final reply. A claim is backed by a successful book/modify
    this turn, or (for "you're already all set for Nov 5") by every date it names matching an upcoming
    appointment of a patient verified in this session."""
    if CLAIM.search(reply) and not action_succeeded:
        named = month_days(reply)
        if not (named and named <= booked_days):
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
