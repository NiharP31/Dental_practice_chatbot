import os
from datetime import date, datetime, time, timedelta

import pytest
from sqlalchemy.orm import sessionmaker

from app import config
from app.db import LOCAL_HOSTS, Base, Patient, Slot, make_engine

NOW = datetime(2026, 10, 28, 9, 0)  # Wednesday


@pytest.fixture
def session_factory(tmp_path):
    # TEST_DATABASE_URL runs the suite against Postgres (see docker-compose.yml); default is a
    # file-backed SQLite so separate connections really contend (in-memory DBs aren't shared).
    engine = make_engine(os.getenv("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'test.db'}")
    if engine.url.host not in LOCAL_HOSTS:
        # This fixture drops every table. Use scripts/contention_check.py for a remote DB (e.g. Supabase).
        pytest.exit(f"refusing to drop tables on non-local TEST_DATABASE_URL host {engine.url.host!r}")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def db(session_factory):
    with session_factory() as s:
        yield s


def add_day(db, day: date, booked: set[str] = frozenset()) -> dict[str, int]:
    """Create a full day of slots; `booked` holds "HH:MM" strings. Returns {"HH:MM": slot_id}."""
    ids, t = {}, datetime.combine(day, time(config.OPEN_HOUR))
    while t.hour < config.CLOSE_HOUR:
        s = Slot(start=t, status="booked" if f"{t:%H:%M}" in booked else "open")
        db.add(s)
        db.flush()
        ids[f"{t:%H:%M}"] = s.id
        t += timedelta(minutes=30)
    db.commit()
    return ids


def add_patient(db, name="Daniel Reyes", dob=date(1984, 3, 12), insurance="Delta Dental", guardian_id=None) -> int:
    p = Patient(full_name=name, phone="5550101001", dob=dob, insurance=insurance, guardian_id=guardian_id)
    db.add(p)
    db.commit()
    return p.id


@pytest.fixture(autouse=True)
def _no_external_side_effects(monkeypatch):
    # A developer's .env must not make tests post to Slack or lock the staff API.
    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "")
    monkeypatch.setattr(config, "STAFF_TOKEN", "")
