"""Seed demo data. Only fills empty tables / missing future days; never wipes anything."""
import random
from datetime import date, datetime, time, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import config
from app.db import Appointment, Base, Patient, SessionLocal, Slot, engine, migrate

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


def init_db() -> dict:
    Base.metadata.create_all(engine)
    added = migrate(engine)
    with SessionLocal() as db:
        return {**seed(db), "columns_added": added}


if __name__ == "__main__":
    print(init_db())
