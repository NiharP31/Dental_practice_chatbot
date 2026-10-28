"""Live smoke test: 3 short conversations against the real LLM, each on a fresh seeded SQLite DB.
Prints transcript, tool calls, DB changes, and any reply the guardrail replaced.

    uv run python -u scripts/smoke.py [1 2 3]
"""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import agent, config, llm  # noqa: E402
from app.db import Appointment, Base, Conversation, LlmUsage, Patient, Slot, StaffAlert, make_engine  # noqa: E402
from app.seed import seed  # noqa: E402




class GuardCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        self.events.append(record.getMessage())


def snapshot(db):
    return {
        "patients": {p.id: (p.full_name, p.phone, str(p.dob), p.insurance, p.guardian_id) for p in db.scalars(select(Patient))},
        "appointments": {a.id: (a.patient_id, str(db.get(Slot, a.slot_id).start), a.appt_type, a.status)
                         for a in db.scalars(select(Appointment))},
        "slots": {s.id: s.status for s in db.scalars(select(Slot))},
        "alerts": {a.id: (a.priority, a.summary) for a in db.scalars(select(StaffAlert))},
    }


def diff(before, after):
    out = []
    for table in before:
        for k, v in after[table].items():
            if k not in before[table]:
                out.append(f"  + {table}[{k}] {v}")
            elif before[table][k] != v:
                out.append(f"  ~ {table}[{k}] {before[table][k]} -> {v}")
    return out or ["  (no changes)"]


def pick_option(i=0, fallback="The first time works for me."):
    return lambda payload: payload["options"][i] if payload["options"] else fallback


CONVERSATIONS = {
    "1. New patient books a cleaning 'later next week'": [
        "Hi! I'm a new patient and I'd like to book a cleaning later next week.",
        "My name is Sam Rivera, phone 555-201-3344, born June 4 1990, and I have Aetna insurance.",
        pick_option(0),
        "Yes, please book it.",
    ],
    "2. Daniel Reyes reschedules": [
        "Hi, this is Daniel Reyes, date of birth 03/12/1984. I need to reschedule my appointment.",
        "Could we do early next week, mornings if possible?",
        pick_option(0),
        "Yes, that's right, please go ahead.",
    ],
    "3. 'When's my appointment?' after verifying": [
        "Hi, I'm Daniel Reyes, date of birth March 12, 1984.",
        "When's my appointment?",
    ],
}


def run(title, steps):
    print("=" * 100 + f"\n{title}\n" + "=" * 100)
    engine = make_engine(f"sqlite:///{tempfile.mkdtemp()}/smoke.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine, expire_on_commit=False)
    with Session() as db:
        seed(db)
        before = snapshot(db)
    session_id, payload, cap = f"smoke-{abs(hash(title)) % 10**8}", {"options": []}, GuardCapture()
    logging.getLogger("guard").addHandler(cap)
    for step in steps:
        text = step(payload) if callable(step) else step
        seen = len(cap.events)
        with Session() as db:
            n_before = len(db.get(Conversation, session_id).messages) if db.get(Conversation, session_id) else 0
            payload = agent.run_turn(db, session_id, text)
            new = db.get(Conversation, session_id).messages[n_before:]
        print(f"\nPATIENT: {text}")
        for m in new:
            if m["role"] == "assistant":
                if m.get("provider"):
                    print(f"   [{m['provider']}]")
                for c in m["tool_calls"]:
                    print(f"   -> tool {c['name']}({c['args']})")
            elif m["role"] == "tool":
                print(f"   <- {m['result']}")
        print(f"RILEY:   {payload['reply']}")
        if payload["options"]:
            print(f"   [quick replies: {payload['options']}]")
        for e in cap.events[seen:]:
            print(f"   !!! GUARDRAIL REPLACED REPLY: {e}")
    logging.getLogger("guard").removeHandler(cap)
    with Session() as db:
        print("\nDB changes:\n" + "\n".join(diff(before, snapshot(db))))
        calls = db.scalars(select(LlmUsage)).all()
        print(f"\nLLM: {len(calls)} calls, est. ${sum(c.cost_usd for c in calls):.4f}, "
              f"cache read/write tokens {sum(c.cache_read_tokens for c in calls)}/{sum(c.cache_write_tokens for c in calls)}")
    engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="   [%(name)s] %(message)s")
    chain = llm.providers()
    if not any(p.billed for p in chain):
        config.LLM_MAX_RPM = config.LLM_MAX_RPM or 8  # free-tier Gemini only: stay under its per-minute cap
    print(f"chain: {' -> '.join(p.name for p in chain) or '(none)'} -> canned; throttle {config.LLM_MAX_RPM or 'off'}/min")
    only = sys.argv[1:]  # e.g. `smoke.py 1 3` runs just those
    for title, steps in CONVERSATIONS.items():
        if not only or title.split(".")[0] in only:
            run(title, steps)
