"""Turn a patient's phrase ("later next week", "tomorrow afternoon", "Nov 3") into a search window.

The LLM passes the patient's words through untouched; all calendar math happens here against
the office clock. Anything we can't confidently interpret raises UnclearDate so the agent asks.
"""
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app import config


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class UnclearDate(ToolError):
    def __init__(self, phrase: str):
        super().__init__("unclear_date", f"Couldn't tell which dates {phrase!r} means. Ask the patient for a day or range.")


@dataclass(frozen=True)
class Window:
    first: date  # inclusive
    last: date  # inclusive
    start_min: int = config.OPEN_HOUR * 60  # minutes after midnight, inclusive
    end_min: int = config.CLOSE_HOUR * 60  # exclusive (a slot must start before this)
    weekdays: frozenset[int] | None = None

    def contains(self, dt: datetime) -> bool:
        m = dt.hour * 60 + dt.minute
        return (self.first <= dt.date() <= self.last and self.start_min <= m < self.end_min
                and (self.weekdays is None or dt.weekday() in self.weekdays))

    def describe(self) -> str:
        span = _fmt_day(self.first) if self.first == self.last else f"{_fmt_day(self.first)} - {_fmt_day(self.last)}"
        if (self.start_min, self.end_min) != (config.OPEN_HOUR * 60, config.CLOSE_HOUR * 60):
            span += f", {fmt_time(self.start_min)}-{fmt_time(self.end_min)}"
        if self.weekdays is not None:
            span += " (" + "/".join(WEEKDAYS[d][0].title()[:3] for d in sorted(self.weekdays)) + " only)"
        return span


def fmt_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{(h - 1) % 12 + 1}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _fmt_day(d: date) -> str:
    return f"{d:%a %b} {d.day}"


def fmt_slot(dt: datetime) -> str:
    return f"{dt:%A, %B} {dt.day} at {fmt_time(dt.hour * 60 + dt.minute)}"


WEEKDAYS = [("monday", "mon"), ("tuesday", "tue", "tues"), ("wednesday", "wed"),
            ("thursday", "thu", "thur", "thurs"), ("friday", "fri"), ("saturday", "sat"), ("sunday", "sun")]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "couple of": 2, "couple": 2, "few": 3}
ASAP = r" (asap|as soon as possible|soon|soonest|earliest|first available|next available|any ?time|whenever|any day|anything) "
DAYS_AHEAD = 27  # how far "asap" / bare time-of-day phrases look (seed covers 4 weeks)

# Normalise synonyms so the patterns below stay small.
SYNONYMS = [
    (r"\blater\b", "late"), (r"\bend of\b", "late"),
    (r"\b(beginning|start) of\b", "early"), (r"\b(middle of|mid-)", "mid "),
    (r"\bthe (week|month)\b", r"this \1"), (r"\b(up)?coming\b", "next"),
    (r"\btmrw?\b|\btomorow\b|\btommorow\b", "tomorrow"), (r"\bweekend\b", "saturday"),
]


def resolve_when(phrase: str, now: datetime) -> Window:
    p = phrase.lower()
    for pat, rep in SYNONYMS:
        p = re.sub(pat, rep, p)
    p = " " + " ".join(re.sub(r"[^a-z0-9/:\- ]", " ", p).split()) + " "
    today = now.date()

    weekdays = frozenset(i for i, names in enumerate(WEEKDAYS)
                         if re.search(r" (" + "|".join(names) + r")s? ", p))
    days = _day_range(p, today, weekdays)
    start_min, end_min = _minutes(p)
    if days is None:
        if not (re.search(ASAP, p) or (start_min, end_min) != (config.OPEN_HOUR * 60, config.CLOSE_HOUR * 60)):
            raise UnclearDate(phrase)
        days = (today, today + timedelta(days=DAYS_AHEAD))
    first, last = max(days[0], today), days[1]
    return Window(first, last, start_min, end_min, weekdays or None)


def _week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _day_range(p: str, today: date, weekdays: frozenset[int]) -> tuple[date, date] | None:
    if dates := _explicit_dates(p, today):
        if " week of " in p and len(dates) == 1:
            mon = _week_monday(dates[0])
            return mon, mon + timedelta(days=5)
        return min(dates), max(dates)
    if " day after tomorrow " in p:
        return (today + timedelta(days=2),) * 2
    if " tomorrow " in p:
        return (today + timedelta(days=1),) * 2
    if re.search(r" (today|tonight|this (morning|afternoon|evening)|same day) ", p):
        return today, today

    if m := re.search(r" in (\d+|a|one|two|three|four|a couple of|a few) (day|week)s? ", p):
        n = int(m[1]) if m[1].isdigit() else NUMBERS[m[1].removeprefix("a ") if m[1] != "a" else "a"]
        if m[2] == "day":
            return (today + timedelta(days=n),) * 2
        mon = _week_monday(today + timedelta(weeks=n))
        return mon, mon + timedelta(days=5)
    if m := re.search(r" next (\d+|two|three|four|couple of|couple|few) weeks ", p):
        n = int(m[1]) if m[1].isdigit() else NUMBERS[m[1]]
        return today, today + timedelta(days=7 * n - 1)

    if m := re.search(r" (?:(early|mid|late|rest of) )?(this|next) week ", p):
        mod, which = m[1], m[2]
        mon = _week_monday(today)
        # On Sunday (office closed) "this week" means the week starting tomorrow.
        if which == "next" or today.weekday() == 6:
            mon += timedelta(weeks=1)
        sat = mon + timedelta(days=5)
        if which == "this" and mod in ("late", "rest of") and today.weekday() != 6:
            if mod == "rest of" or today.weekday() >= 3:
                return today + timedelta(days=1), sat
        return {None: (mon, sat), "early": (mon, mon + timedelta(days=1)),
                "mid": (mon + timedelta(days=2), mon + timedelta(days=3)),
                "late": (mon + timedelta(days=3), sat), "rest of": (mon, sat)}[mod]

    if m := re.search(r" (?:(early|mid|late) )?(this|next) month ", p):
        mod, which = m[1], m[2]
        y, mo = today.year, today.month
        if which == "next":
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        end = monthrange(y, mo)[1]
        lo, hi = {None: (1, end), "early": (1, 10), "mid": (11, 20), "late": (21, end)}[mod]
        return date(y, mo, lo), date(y, mo, hi)

    if weekdays:
        if re.search(r" next (" + "|".join(n for i in weekdays for n in WEEKDAYS[i]) + ") ", p):
            mon = _week_monday(today) + timedelta(weeks=1)
            return mon, mon + timedelta(days=6)
        return today, today + timedelta(days=6)  # next occurrence of each named day, today included
    return None


def _explicit_dates(p: str, today: date) -> list[date]:
    found = []  # (year or None, month, day)
    for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", p):
        found.append((int(m[1]), int(m[2]), int(m[3])))
    for m in re.finditer(r"(?<![\d-])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", p):
        y = int(m[3]) if m[3] else None
        found.append((y + 2000 if y is not None and y < 100 else y, int(m[1]), int(m[2])))
    for m in re.finditer(r"\b" + MONTH_RE + r" (\d{1,2})(?:st|nd|rd|th)?\b", p):
        found.append((None, MONTHS.index(m[1][:3]) + 1, int(m[2])))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)? (?:of )?" + MONTH_RE + r"\b", p):
        found.append((None, MONTHS.index(m[2][:3]) + 1, int(m[1])))
    for m in re.finditer(r" the (\d{1,2})(?:st|nd|rd|th) ", p):  # "the 15th": this month, or next if passed
        y, mo = today.year, today.month
        if int(m[1]) < today.day:
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        found.append((y, mo, int(m[1])))

    out = []
    for y, mo, d in found:
        try:
            dt = date(y or today.year, mo, d)
            if y is None and dt < today:
                dt = date(today.year + 1, mo, d)
        except ValueError:
            raise UnclearDate(p.strip()) from None
        out.append(dt)
    return out


def _minutes(p: str) -> tuple[int, int]:
    lo, hi = config.OPEN_HOUR * 60, config.CLOSE_HOUR * 60
    morning, afternoon = " morning" in p, " afternoon" in p
    if morning and not afternoon:
        hi = 12 * 60
    elif afternoon and not morning:
        lo, hi = 12 * 60, 17 * 60
    if re.search(r" (evening|tonight|after work|after school|late in the day) ", p):
        lo = max(lo, 16 * 60)
    if " lunch" in p:
        lo, hi = 11 * 60 + 30, 13 * 60 + 30
    for m in re.finditer(r" (after|before|at|around|by) (noon|\d{1,2})(?::(\d\d))? ?(am|pm)? ", p):
        h = 12 if m[2] == "noon" else int(m[2])
        if m[2] != "noon" and h > 12:
            continue
        if m[4] == "pm" and h < 12 or (m[4] is None and m[2] != "noon" and h <= 7):
            h += 12  # "after 3" at a dental office means 3 PM
        t = h * 60 + int(m[3] or 0)
        if m[1] == "after":
            lo = max(lo, t)
        elif m[1] in ("before", "by"):
            hi = min(hi, t)
        else:
            lo, hi = max(lo, t - 60), min(hi, t + 60)
    return lo, hi
