import threading
from datetime import date

import pytest

from app.db import Appointment, IdempotencyKey, Slot, StaffAlert
from app.tools import (ToolError, UnclearDate, book, find_slots, modify, notify_staff, register_patient,
                       verify_patient)
from tests.conftest import NOW, add_day, add_patient, patient_says, verify

THU, FRI = date(2026, 10, 29), date(2026, 10, 30)


def verified(db, sid="s1", **kw) -> int:
    pid = add_patient(db, **kw)
    assert verify(db, sid, kw.get("name", "Daniel Reyes"), kw["dob"].isoformat() if "dob" in kw else "1984-03-12")["verified"]
    return pid


# --- verify / register -------------------------------------------------------------------

def test_dob_must_come_from_the_patient(db, caplog):
    add_patient(db)
    patient_says(db, "s1", "Hi, I'm Daniel Reyes and I need to check my appointment.")
    with caplog.at_level("WARNING", logger="tools"):
        r = verify_patient(db, "s1", "Daniel Reyes", "1984-03-12", now=NOW)  # model supplied the right DOB itself
    assert r == {"verified": False, "message": "No patient matches that name and date of birth."}
    assert any("dob_not_from_patient" in m for m in caplog.messages)
    for _ in range(5):  # doesn't count toward the lockout: it's the model's mistake, not a guess
        verify_patient(db, "s1", "Daniel Reyes", "1984-03-12", now=NOW)
    patient_says(db, "s1", "Oh, my DOB is March 12th, 1984")
    assert verify_patient(db, "s1", "Daniel Reyes", "03/12/1984", now=NOW)["verified"]  # normalised match


def test_dob_provenance_ignores_system_notes_and_other_sessions(db):
    from app.prompt import NOTE_MARKER
    add_patient(db, dob=date(2026, 10, 28))  # same date as the clock note below
    patient_says(db, "s1", "Daniel Reyes here" + NOTE_MARKER + " office time is Wednesday, October 28, 2026]")
    patient_says(db, "other", "Daniel Reyes, born 2026-10-28")
    assert not verify_patient(db, "s1", "Daniel Reyes", "2026-10-28", now=NOW)["verified"]


@pytest.mark.parametrize("typed", ["03/12/1984", "3-12-84", "1984-03-12", "March 12, 1984", "12th of March 1984",
                                   "mar. 12 1984"])
def test_dob_formats_normalise(db, typed):
    add_patient(db)
    patient_says(db, "s1", f"Daniel Reyes, DOB {typed}")
    assert verify_patient(db, "s1", "Daniel Reyes", "1984-03-12", now=NOW)["verified"]

def test_verify_needs_name_and_dob_and_hides_which_was_wrong(db):
    add_patient(db)
    wrong_dob = verify(db, "s1", "Daniel Reyes", "1984-03-13")
    wrong_name = verify(db, "s1", "Danny Reyes", "1984-03-12")
    assert wrong_dob == wrong_name == {"verified": False, "message": wrong_dob["message"]}
    ok = verify(db, "s1", "  daniel   REYES ", "03/12/1984")
    assert ok["verified"] and ok["full_name"] == "Daniel Reyes"


def test_verify_locks_after_three_failures(db):
    add_patient(db)
    for _ in range(3):
        verify(db, "s1", "Daniel Reyes", "2000-01-01")
    with pytest.raises(ToolError) as e:
        verify(db, "s1", "Daniel Reyes", "1984-03-12")
    assert e.value.code == "verify_locked"


def test_register_validates_and_normalises(db):
    r = register_patient(db, "s1", "maria reyes", "+1 (555) 010-1002", "July 25, 1986", "No insurance")
    assert r["registered"] and r["full_name"] == "Maria Reyes" and r["insurance"] == "none"
    for args, code in [(("Maria", "5550101002", "1986-07-25", "Aetna"), "bad_name"),
                       (("Maria Reyes", "555-0101", "1986-07-25", "Aetna"), "bad_phone"),
                       (("Maria Reyes", "5550101002", "2099-01-01", "Aetna"), "bad_dob"),
                       (("Maria Reyes", "5550101002", "1986-07-25", " "), "bad_insurance")]:
        with pytest.raises(ToolError) as e:
            register_patient(db, "s1", *args)
        assert e.value.code == code


def test_register_existing_person_returns_same_record(db):
    pid = add_patient(db)
    r = register_patient(db, "s2", "Daniel Reyes", "5550101001", "1984-03-12", "Delta Dental")
    assert r == {"patient_id": pid, "full_name": "Daniel Reyes", "already_registered": True}


# --- find_slots ---------------------------------------------------------------------------

def test_find_slots_spreads_options_and_caps_at_three(db):
    add_day(db, THU)
    add_day(db, FRI)
    opts = find_slots(db, "cleaning", "thursday or friday", now=NOW)["options"]
    assert len(opts) == 3
    assert opts[0]["when"] == "Thursday, October 29 at 8:00 AM"
    assert opts[1]["when"] == "Thursday, October 29 at 10:00 AM"  # 2h apart, not 8:30
    assert opts[2]["when"] == "Friday, October 30 at 8:00 AM"


def test_find_slots_never_offers_past_or_booked(db):
    add_day(db, date(2026, 10, 28), booked={"09:30", "10:00"})
    opts = find_slots(db, "checkup", "today", now=NOW)["options"]
    assert opts[0]["when"].endswith("10:30 AM")  # 9:00 is now, 9:30/10:00 booked


def test_back_to_back_runs_for_families(db):
    # Only 10:00-11:30 has three consecutive open slots on Thursday.
    booked = {f"{h:02d}:{m}" for h in range(8, 18) for m in ("00", "30")} - {"08:00", "08:30", "10:00", "10:30", "11:00", "14:00"}
    ids = add_day(db, THU, booked=booked)
    r = find_slots(db, "cleaning", "thursday", count=3, now=NOW)
    assert [o["slot_ids"] for o in r["options"]] == [[ids["10:00"], ids["10:30"], ids["11:00"]]]
    assert r["options"][0]["times"] == ["10:00 AM", "10:30 AM", "11:00 AM"]


def test_no_options_suggests_next_available(db):
    add_day(db, THU, booked={f"{h:02d}:{m}" for h in range(8, 18) for m in ("00", "30")})
    add_day(db, FRI)
    r = find_slots(db, "cleaning", "tomorrow", now=NOW)
    assert r["options"] == [] and r["next_available"]["when"] == "Friday, October 30 at 8:00 AM"


def test_exclude_rejected_options(db):
    add_day(db, THU)
    first = find_slots(db, "cleaning", "thursday morning", now=NOW)["options"]
    rejected = [i for o in first for i in o["slot_ids"]]
    again = find_slots(db, "cleaning", "thursday morning", exclude_slot_ids=rejected, now=NOW)["options"]
    assert again and not {i for o in again for i in o["slot_ids"]} & set(rejected)


def test_find_slots_bad_inputs(db):
    with pytest.raises(UnclearDate):
        find_slots(db, "cleaning", "when pigs fly", now=NOW)
    with pytest.raises(ToolError):
        find_slots(db, "whitening", "tomorrow", now=NOW)


# --- book ---------------------------------------------------------------------------------

def test_book_requires_verified_patient(db):
    ids = add_day(db, THU)
    pid = add_patient(db)
    with pytest.raises(ToolError) as e:
        book(db, "stranger", 1, [pid], [ids["09:00"]], "cleaning", now=NOW)
    assert e.value.code == "not_verified"
    assert db.get(Slot, ids["09:00"]).status == "open"


def test_book_and_idempotent_retry(db):
    ids = add_day(db, THU)
    pid = verified(db)
    first = book(db, "s1", 3, [pid], [ids["09:00"]], "cleaning", now=NOW)
    retry = book(db, "s1", 3, [pid], [ids["09:00"]], "cleaning", now=NOW)
    assert first["booked"] and retry == {**first, "replayed": True}
    assert db.query(Appointment).count() == 1


def test_double_booking_under_contention(session_factory):
    with session_factory() as db:
        slot = add_day(db, THU)["09:00"]
        a = verified(db, "a")
        b = verified(db, "b", name="Maria Reyes", dob=date(1986, 7, 25))
    results, barrier = [], threading.Barrier(2, timeout=10)

    def attempt(sid, pid):
        with session_factory() as db:
            barrier.wait()
            try:
                results.append(book(db, sid, 1, [pid], [slot], "cleaning", now=NOW)["booked"])
            except ToolError as e:
                results.append(e.code)

    threads = [threading.Thread(target=attempt, args=args) for args in (("a", a), ("b", b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(results, key=str) == [True, "slot_taken"]
    with session_factory() as db:
        assert db.query(Appointment).count() == 1


def test_family_booking_is_all_or_nothing(db):
    ids = add_day(db, THU)
    parent = verified(db, name="Priya Shah", dob=date(1980, 11, 2))
    kid1 = register_patient(db, "s1", "Arjun Shah", "5550101003", "2014-05-19", "Cigna")["patient_id"]
    kid2 = register_patient(db, "s1", "Anika Shah", "5550101003", "2017-09-08", "Cigna")["patient_id"]
    db.get(Slot, ids["10:00"]).status = "booked"  # someone grabs the 3rd slot first
    db.commit()
    with pytest.raises(ToolError) as e:
        book(db, "s1", 2, [parent, kid1, kid2], [ids["09:00"], ids["09:30"], ids["10:00"]], "cleaning", now=NOW)
    assert e.value.code == "slot_taken"
    assert db.get(Slot, ids["09:00"]).status == db.get(Slot, ids["09:30"]).status == "open"
    assert db.query(Appointment).count() == 0 and db.query(IdempotencyKey).count() == 0
    # Same turn, fresh slots: failure wasn't cached, so the retry books all three.
    r = book(db, "s1", 2, [parent, kid1, kid2], [ids["14:00"], ids["14:30"], ids["15:00"]], "cleaning", now=NOW)
    assert [a["patient"] for a in r["appointments"]] == ["Priya Shah", "Arjun Shah", "Anika Shah"]


def test_book_rejects_past_slot(db):
    ids = add_day(db, date(2026, 10, 28))
    pid = verified(db)
    with pytest.raises(ToolError) as e:
        book(db, "s1", 1, [pid], [ids["08:30"]], "cleaning", now=NOW)
    assert e.value.code == "bad_slot"


# --- modify -------------------------------------------------------------------------------

def _booked(db, sid="s1"):
    ids = add_day(db, THU)
    pid = verified(db, sid)
    appt = book(db, sid, 1, [pid], [ids["09:00"]], "cleaning", now=NOW)["appointments"][0]["appointment_id"]
    return ids, appt


def test_modify_requires_verification_of_that_patient(db):
    ids, appt = _booked(db)
    add_patient(db, name="Eve Mallory", dob=date(1990, 1, 1))
    verify(db, "intruder", "Eve Mallory", "1990-01-01")  # verified as someone else
    for sid in ("intruder", "anonymous"):
        with pytest.raises(ToolError) as e:
            modify(db, sid, 5, appt, "cancel", confirm=True, now=NOW)
        assert e.value.code == "not_verified"
    assert db.get(Appointment, appt).status == "booked"


def test_cancel_needs_readback_then_yes_on_a_later_turn(db):
    ids, appt = _booked(db)
    first = modify(db, "s1", 2, appt, "cancel", now=NOW)
    assert first["needs_confirmation"] and "Thursday, October 29 at 9:00 AM" in first["details"]
    # Model tries to self-confirm in the same turn: refused.
    assert modify(db, "s1", 2, appt, "cancel", confirm=True, now=NOW)["needs_confirmation"]
    assert db.get(Appointment, appt).status == "booked"
    done = modify(db, "s1", 3, appt, "cancel", confirm=True, now=NOW)
    assert done["done"] and db.get(Appointment, appt).status == "cancelled"
    assert db.get(Slot, ids["09:00"]).status == "open"
    assert modify(db, "s1", 3, appt, "cancel", confirm=True, now=NOW)["already_applied"]


def test_confirm_must_match_what_was_read_back(db):
    ids, appt = _booked(db)
    modify(db, "s1", 2, appt, "reschedule", new_slot_id=ids["14:00"], now=NOW)
    swapped = modify(db, "s1", 3, appt, "reschedule", new_slot_id=ids["15:00"], confirm=True, now=NOW)
    assert swapped["needs_confirmation"]  # different slot than the patient agreed to


def test_reschedule_moves_slot_atomically(db):
    ids, appt = _booked(db)
    modify(db, "s1", 2, appt, "reschedule", new_slot_id=ids["14:00"], now=NOW)
    modify(db, "s1", 3, appt, "reschedule", new_slot_id=ids["14:00"], now=NOW)  # re-asked: keeps turn 2
    done = modify(db, "s1", 3, appt, "reschedule", new_slot_id=ids["14:00"], confirm=True, now=NOW)
    assert done["done"]
    assert db.get(Appointment, appt).slot_id == ids["14:00"]
    assert (db.get(Slot, ids["09:00"]).status, db.get(Slot, ids["14:00"]).status) == ("open", "booked")


def test_reschedule_to_taken_slot_changes_nothing(db):
    ids, appt = _booked(db)
    modify(db, "s1", 2, appt, "reschedule", new_slot_id=ids["14:00"], now=NOW)
    db.get(Slot, ids["14:00"]).status = "booked"
    db.commit()
    with pytest.raises(ToolError) as e:
        modify(db, "s1", 3, appt, "reschedule", new_slot_id=ids["14:00"], confirm=True, now=NOW)
    assert e.value.code == "slot_taken" and db.get(Appointment, appt).slot_id == ids["09:00"]


# --- notify_staff -------------------------------------------------------------------------

def test_notify_staff_writes_alert(db):
    r = notify_staff(db, "s1", "urgent", "Cracked molar, severe pain since last night")
    assert r["alerted"] and db.get(StaffAlert, r["alert_id"]).priority == "urgent"
    with pytest.raises(ToolError):
        notify_staff(db, "s1", "whenever", "x")


# --- idempotency fingerprint --------------------------------------------------------------

def test_two_different_bookings_in_one_turn_both_persist_and_retries_replay(db):
    ids = add_day(db, THU)
    pid = verified(db)
    a = book(db, "s1", 4, [pid], [ids["09:00"]], "cleaning", now=NOW)
    b = book(db, "s1", 4, [pid], [ids["14:00"]], "checkup", now=NOW)
    assert a["appointments"][0]["appointment_id"] != b["appointments"][0]["appointment_id"]
    assert "replayed" not in a and "replayed" not in b
    assert book(db, "s1", 4, [pid], [ids["14:00"]], "checkup", now=NOW) == {**b, "replayed": True}
    assert book(db, "s1", 4, [pid], [ids["09:00"]], "cleaning", now=NOW) == {**a, "replayed": True}
    assert db.query(Appointment).count() == 2 and db.query(IdempotencyKey).count() == 2


# --- households ---------------------------------------------------------------------------

def _household(db):
    parent = verified(db, name="Priya Shah", dob=date(1980, 11, 2))
    kids = [add_patient(db, name=n, dob=d, guardian_id=parent)
            for n, d in (("Arjun Shah", date(2014, 5, 19)), ("Anika Shah", date(2017, 9, 8)))]
    return parent, kids


def test_guardian_books_for_dependents_without_verifying_them(db):
    ids = add_day(db, THU)
    parent, kids = _household(db)
    v = verify(db, "s1", "Priya Shah", "1980-11-02")
    assert [d["full_name"] for d in v["dependents"]] == ["Arjun Shah", "Anika Shah"]
    r = book(db, "s1", 2, [parent, *kids], [ids["10:00"], ids["10:30"], ids["11:00"]], "cleaning", now=NOW)
    assert len(r["appointments"]) == 3
    # ...and can cancel a dependent's appointment too
    kid_appt = r["appointments"][1]["appointment_id"]
    modify(db, "s1", 3, kid_appt, "cancel", now=NOW)
    assert modify(db, "s1", 4, kid_appt, "cancel", confirm=True, now=NOW)["done"]


def test_dependent_link_does_not_work_in_reverse_or_for_strangers(db):
    ids = add_day(db, THU)
    parent, kids = _household(db)
    verify(db, "kid", "Arjun Shah", "2014-05-19")
    with pytest.raises(ToolError):
        book(db, "kid", 1, [parent], [ids["09:00"]], "cleaning", now=NOW)
    with pytest.raises(ToolError):
        book(db, "stranger", 1, [kids[0]], [ids["09:00"]], "cleaning", now=NOW)


def test_spouses_verify_separately(db):
    ids = add_day(db, THU)
    daniel = verified(db)
    maria = add_patient(db, name="Maria Reyes", dob=date(1986, 7, 25))
    with pytest.raises(ToolError) as e:
        book(db, "s1", 1, [daniel, maria], [ids["09:00"], ids["09:30"]], "cleaning", now=NOW)
    assert e.value.code == "not_verified"
    assert verify(db, "s1", "Maria Reyes", "1986-07-25")["verified"]
    assert book(db, "s1", 2, [daniel, maria], [ids["09:00"], ids["09:30"]], "cleaning", now=NOW)["booked"]


def test_register_child_under_verified_guardian(db):
    parent, _ = _household(db)
    with pytest.raises(ToolError):
        register_patient(db, "other", "Kiran Shah", "5550101003", "2020-02-02", "Cigna", guardian_id=parent)
    r = register_patient(db, "s1", "Kiran Shah", "5550101003", "2020-02-02", "Cigna", guardian_id=parent)
    v = verify(db, "s2", "Priya Shah", "1980-11-02")
    assert r["patient_id"] in [d["patient_id"] for d in v["dependents"]]


# --- staff alerts before identification ----------------------------------------------------

def test_alert_before_registration_gets_patient_attached_later(db):
    a = notify_staff(db, "s1", "urgent", "Cracked molar, pain 7/10")
    assert db.get(StaffAlert, a["alert_id"]).patient_id is None
    pid = register_patient(db, "s1", "Jordan Lee", "5553012244", "1995-02-14", "Cigna")["patient_id"]
    assert db.get(StaffAlert, a["alert_id"]).patient_id == pid
    assert notify_staff(db, "other", "normal", "Billing question")  # other sessions untouched
    assert db.query(StaffAlert).filter_by(session_id="other").one().patient_id is None


def test_alert_after_verification_carries_patient(db):
    pid = verified(db)
    a = notify_staff(db, "s1", "normal", "Wants a callback")
    assert db.get(StaffAlert, a["alert_id"]).patient_id == pid


def test_emergency_booking_creates_urgent_alert_if_session_has_none(db):
    ids = add_day(db, THU)
    pid = verified(db)
    book(db, "s1", 1, [pid], [ids["09:00"]], "emergency", notes="Knocked out front tooth at soccer", now=NOW)
    [alert] = db.query(StaffAlert).all()
    assert alert.priority == "urgent" and "Knocked out front tooth" in alert.summary and alert.patient_id == pid
    book(db, "s1", 1, [pid], [ids["09:00"]], "emergency", notes="x", now=NOW)  # idempotent replay
    book(db, "s1", 2, [pid], [ids["10:00"]], "emergency", now=NOW)  # session already has an urgent alert
    assert db.query(StaffAlert).count() == 1


def test_non_emergency_booking_creates_no_alert(db):
    ids = add_day(db, THU)
    book(db, "s1", 1, [verified(db)], [ids["09:00"]], "cleaning", now=NOW)
    assert db.query(StaffAlert).count() == 0
