"""Seed demo data. Startup seeding only fills empty tables and missing future days; it never wipes
anything. reset_demo() is the one destructive path: staff-triggered, and limited to the demo."""
import random
from datetime import date, datetime, time, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app import config
from app.db import (Appointment, Base, Conversation, Feedback, Patient, SessionLocal, Slot, StaffAlert, engine,
                    migrate)

WEEKS = 4
BOOKED_FRACTION = 0.3

# (full name, phone, dob, insurance, appt type, days from today for their upcoming visit, hour)
PATIENTS = [
    # spouses
    ("Daniel Reyes", "5550101001", date(1984, 3, 12), "Delta Dental", "cleaning", 3, 9),
    ("Maria Reyes", "5550101002", date(1986, 7, 25), "Delta Dental", "checkup", 5, 14),
    # parent with two kids
    ("Priya Shah", "5550101003", date(1980, 11, 2), "Cigna", "checkup", 6, 10),
    ("Arjun Shah", "5550101003", date(2014, 5, 19), "Cigna", "cleaning", 8, 15),
    ("Anika Shah", "5550101003", date(2017, 9, 8), "Cigna", "cleaning", 8, 16),
    # uninsured
    ("Tom Becker", "5550101004", date(1992, 1, 30), "none", "cleaning", 4, 11),
]
GUARDIANS = {"Arjun Shah": "Priya Shah", "Anika Shah": "Priya Shah"}  # dependent -> guardian


def ensure_slots(db: Session, today: date, rng: random.Random) -> int:
    """Create 30-min slots through today + 4 weeks for days that don't have any yet."""
    last = db.scalar(select(func.max(Slot.start)))
    day = max(today, last.date() + timedelta(days=1)) if last else today
    added = 0
    while day < today + timedelta(weeks=WEEKS):
        if day.weekday() in config.OPEN_WEEKDAYS:
            t = datetime.combine(day, time(config.OPEN_HOUR))
            while t < datetime.combine(day, time(config.CLOSE_HOUR)):
                # ponytail: "booked" slots without an appointment row stand in for other patients
                db.add(Slot(start=t, status="booked" if rng.random() < BOOKED_FRACTION else "open"))
                t += config.SLOT_LEN
                added += 1
        day += timedelta(days=1)
    return added


def _next_open_day(today: date, offset: int) -> date:
    d = today + timedelta(days=offset)
    while d.weekday() not in config.OPEN_WEEKDAYS:
        d += timedelta(days=1)
    return d


def ensure_patients(db: Session, today: date) -> int:
    if db.scalar(select(func.count(Patient.id))):
        return 0
    ids = {}
    for name, phone, dob, insurance, appt_type, offset, hour in PATIENTS:
        p = Patient(full_name=name, phone=phone, dob=dob, insurance=insurance, guardian_id=ids.get(GUARDIANS.get(name)))
        db.add(p)
        db.flush()
        ids[name] = p.id
        slot = db.scalar(select(Slot).where(Slot.start == datetime.combine(_next_open_day(today, offset), time(hour))))
        slot.status = "booked"
        db.add(Appointment(patient_id=p.id, slot_id=slot.id, appt_type=appt_type))
    return len(PATIENTS)


def seed(db: Session, today: date | None = None, rng_seed: int = 42) -> dict:
    today = today or config.office_now().date()
    slots = ensure_slots(db, today, random.Random(rng_seed))
    db.flush()
    patients = ensure_patients(db, today)
    db.commit()
    return {"slots_added": slots, "patients_added": patients}


RESET_WINDOW_DAYS = (2, 10)  # reset re-creates each seeded appointment this many days out


def _open_slot_for_reset(db: Session, today: date, offset: int, hour: int) -> Slot:
    """The seeded day/hour if it's open, else the next open slot in the 2-10 day window."""
    lo = datetime.combine(today + timedelta(days=RESET_WINDOW_DAYS[0]), time())
    hi = datetime.combine(today + timedelta(days=RESET_WINDOW_DAYS[1] + 1), time())
    preferred = max(datetime.combine(_next_open_day(today, offset), time(hour)), lo)
    q = select(Slot).where(Slot.status == "open", Slot.start < hi).order_by(Slot.start).limit(1)
    slot = db.scalar(q.where(Slot.start >= preferred)) or db.scalar(q.where(Slot.start >= lo))
    if slot is None:
        raise RuntimeError("no open slot 2-10 days out to reset a demo appointment into")
    return slot


def reset_demo(db: Session, today: date | None = None) -> dict:
    """Make the shared live demo usable again, in one transaction.

    For the seeded patients only: delete their appointments and give each one upcoming appointment
    2-10 days out. Top up open slots to 4 weeks ahead. Clear conversations, feedback and staff alerts
    that aren't linked to a patient a tester registered. Tester-registered patients and their
    appointments stay; patients and llm_usage (the spend budget) are never deleted."""
    today = today or config.office_now().date()
    try:
        slots_added = ensure_slots(db, today, random.Random())
        db.flush()
        seed_ids: dict[str, int] = {}
        for name, phone, dob, insurance, *_ in PATIENTS:
            p = db.scalar(select(Patient).where(Patient.full_name == name, Patient.dob == dob))
            if p is None:  # only if someone removed a seeded patient
                p = Patient(full_name=name, phone=phone, dob=dob, insurance=insurance)
                db.add(p)
                db.flush()
            seed_ids[name] = p.id
        for dependent, guardian in GUARDIANS.items():
            db.get(Patient, seed_ids[dependent]).guardian_id = seed_ids[guardian]
        ids = set(seed_ids.values())

        old = db.scalars(select(Appointment).where(Appointment.patient_id.in_(ids))).all()
        for a in old:
            if a.status == "booked":
                db.get(Slot, a.slot_id).status = "open"
            db.delete(a)
        db.flush()
        for name, _, _, _, appt_type, offset, hour in PATIENTS:
            slot = _open_slot_for_reset(db, today, offset, hour)
            slot.status = "booked"
            db.add(Appointment(patient_id=seed_ids[name], slot_id=slot.id, appt_type=appt_type))
            db.flush()

        # Sessions stay only if they verified/registered a tester's own (non-seed) patient.
        convs = db.scalars(select(Conversation)).all()
        keep = {c.session_id for c in convs if set((c.state or {}).get("verified", [])) - ids}
        dropped = [c for c in convs if c.session_id not in keep]
        for c in dropped:
            db.delete(c)
        feedback = db.execute(delete(Feedback).where(Feedback.session_id.not_in(keep))).rowcount
        alerts = db.execute(delete(StaffAlert).where(
            (StaffAlert.patient_id.is_(None)) | (StaffAlert.patient_id.in_(ids)))).rowcount
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"appointments_reset": len(PATIENTS), "old_appointments_removed": len(old), "slots_added": slots_added,
            "conversations_removed": len(dropped), "feedback_removed": feedback, "alerts_removed": alerts}


def init_db() -> dict:
    Base.metadata.create_all(engine)
    added = migrate(engine)
    with SessionLocal() as db:
        return {**seed(db), "columns_added": added}


if __name__ == "__main__":
    print(init_db())
