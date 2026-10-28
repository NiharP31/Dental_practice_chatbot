from datetime import date, datetime

import pytest

from app.dates import UnclearDate, resolve_when

WED = datetime(2026, 10, 28, 9, 0)  # mid-week
SAT_MONTH_END = datetime(2026, 10, 31, 15, 0)  # Saturday AND last day of the month
SUN = datetime(2026, 11, 1, 10, 0)  # office closed
DEC_31 = datetime(2026, 12, 31, 9, 0)  # year rollover (Thursday)
JAN_30 = datetime(2027, 1, 30, 9, 0)  # Saturday before a short month


def days(phrase, now):
    w = resolve_when(phrase, now)
    return w.first, w.last


@pytest.mark.parametrize("now, phrase, expected", [
    # "later next week" = Thu-Sat of the following calendar week
    (WED, "later next week", (date(2026, 11, 5), date(2026, 11, 7))),
    (SAT_MONTH_END, "later next week", (date(2026, 11, 5), date(2026, 11, 7))),
    (SUN, "later next week", (date(2026, 11, 5), date(2026, 11, 7))),  # Sunday: next week starts tomorrow
    (WED, "early next week", (date(2026, 11, 2), date(2026, 11, 3))),
    (WED, "next week", (date(2026, 11, 2), date(2026, 11, 7))),
    (SAT_MONTH_END, "next week", (date(2026, 11, 2), date(2026, 11, 7))),
    # "early next month" = 1st-10th of next month
    (WED, "early next month", (date(2026, 11, 1), date(2026, 11, 10))),
    (SAT_MONTH_END, "early next month", (date(2026, 11, 1), date(2026, 11, 10))),
    (DEC_31, "early next month", (date(2027, 1, 1), date(2027, 1, 10))),
    (JAN_30, "end of next month", (date(2027, 2, 21), date(2027, 2, 28))),
    (SAT_MONTH_END, "later this month", (date(2026, 10, 31), date(2026, 10, 31))),  # clipped to today
    # relative days
    (SAT_MONTH_END, "tomorrow", (date(2026, 11, 1), date(2026, 11, 1))),
    (DEC_31, "tomorrow morning", (date(2027, 1, 1), date(2027, 1, 1))),
    (WED, "day after tomorrow", (date(2026, 10, 30), date(2026, 10, 30))),
    (WED, "in 2 weeks", (date(2026, 11, 9), date(2026, 11, 14))),
    (WED, "next couple of weeks", (date(2026, 10, 28), date(2026, 11, 10))),
    # this week / end of week
    (WED, "this week", (date(2026, 10, 28), date(2026, 10, 31))),
    (WED, "end of the week", (date(2026, 10, 29), date(2026, 10, 31))),
    (SUN, "this week", (date(2026, 11, 2), date(2026, 11, 7))),
    # explicit dates; past month/day rolls to next year
    (WED, "Nov 3rd", (date(2026, 11, 3), date(2026, 11, 3))),
    (WED, "11/3", (date(2026, 11, 3), date(2026, 11, 3))),
    (WED, "2026-11-05", (date(2026, 11, 5), date(2026, 11, 5))),
    (WED, "3rd of November", (date(2026, 11, 3), date(2026, 11, 3))),
    (DEC_31, "jan 4", (date(2027, 1, 4), date(2027, 1, 4))),
    (WED, "between nov 3 and nov 6", (date(2026, 11, 3), date(2026, 11, 6))),
    (WED, "the 2nd", (date(2026, 11, 2), date(2026, 11, 2))),
    (WED, "the week of nov 11", (date(2026, 11, 9), date(2026, 11, 14))),
])
def test_day_ranges(now, phrase, expected):
    assert days(phrase, now) == expected


def test_weekday_names():
    w = resolve_when("friday", WED)
    assert (w.first, w.last, w.weekdays) == (date(2026, 10, 28), date(2026, 11, 3), {4})
    w = resolve_when("next tuesday", WED)  # = Tuesday of next calendar week
    assert w.contains(datetime(2026, 11, 3, 9)) and not w.contains(datetime(2026, 11, 4, 9))
    w = resolve_when("monday or wednesday next week", WED)
    assert w.weekdays == {0, 2} and w.first == date(2026, 11, 2)
    w = resolve_when("this weekend", WED)
    assert w.contains(datetime(2026, 10, 31, 10)) and not w.contains(datetime(2026, 10, 30, 10))


@pytest.mark.parametrize("phrase, inside, outside", [
    ("tomorrow afternoon", "13:00", "11:30"),
    ("tomorrow morning", "11:30", "12:00"),
    ("tomorrow after 3", "15:00", "14:30"),
    ("tomorrow before 10am", "9:30", "10:00"),
    ("tomorrow around 2pm", "13:30", "15:00"),
    ("tomorrow evening", "17:30", "15:30"),
])
def test_time_of_day(phrase, inside, outside):
    w = resolve_when(phrase, WED)
    at = lambda hm: datetime(2026, 10, 29, *map(int, hm.split(":")))
    assert w.contains(at(inside)) and not w.contains(at(outside))


def test_asap_and_bare_time_look_ahead_from_now():
    w = resolve_when("as soon as possible", SAT_MONTH_END)
    assert w.first == date(2026, 10, 31) and (w.last - w.first).days == 27
    assert resolve_when("afternoons are best", WED).start_min == 12 * 60


@pytest.mark.parametrize("phrase", ["when the moon is full", "blue", "", "feb 30", "sometime"])
def test_unclear_phrases_raise(phrase):
    with pytest.raises(UnclearDate):
        resolve_when(phrase, WED)


def test_today_window_is_clipped_to_now():
    w = resolve_when("today", SAT_MONTH_END)
    assert not w.contains(datetime(2026, 10, 30, 16)) and w.contains(datetime(2026, 10, 31, 16))
