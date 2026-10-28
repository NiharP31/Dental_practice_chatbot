"""Double-booking check against a real Postgres (e.g. Supabase via the transaction pooler), safe to run
against production: it works in a throwaway schema and drops only that schema afterwards.

Each round, two threads on separate connections race tools.book() for the same slot. Exactly one
must win and the other must get slot_taken, leaving one appointment.

    uv run python scripts/contention_check.py "postgresql://...:6543/postgres" [rounds]
    uv run python scripts/contention_check.py 10         # uses DATABASE_URL from .env
"""
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import config, tools  # noqa: E402
from app.db import Appointment, Base, Conversation, Patient, Slot, make_engine  # noqa: E402

FAR_FUTURE = datetime(2030, 1, 7, 8, 0)  # slots must be in the future for book() to accept them


def main() -> int:
    args = sys.argv[1:]
    rounds = int(args.pop()) if args and args[-1].isdigit() else 10
    url = args[0] if args else config.DATABASE_URL  # config loads .env, so the password never hits the shell
    if not url.startswith(("postgres://", "postgresql")):
        print("Pass a Postgres URL (or set DATABASE_URL).")
        return 2
    base = make_engine(url)
    schema = f"contention_check_{uuid.uuid4().hex[:8]}"
    with base.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = base.execution_options(schema_translate_map={None: schema})
    print(f"host {base.url.host}:{base.url.port} | schema {schema} | {rounds} rounds")
    outcomes = []
    try:
        Base.metadata.create_all(engine)
        Session = sessionmaker(engine, expire_on_commit=False)
        for n in range(rounds):
            with Session() as db:
                slot = Slot(start=FAR_FUTURE + timedelta(minutes=30 * n))
                db.add(slot)
                pids = []
                for who in "ab":
                    p = Patient(full_name=f"Racer {who.upper()}{n}", phone="5550000000",
                                dob=datetime(1990, 1, 1).date(), insurance="none")
                    db.add(p)
                    db.flush()
                    pids.append(p.id)
                    db.add(Conversation(session_id=f"cc-{n}-{who}", messages=[], state={"verified": [p.id]}))
                db.commit()
                slot_id = slot.id
            results, barrier = [], threading.Barrier(2, timeout=30)

            def attempt(sid, pid):
                with Session() as db:
                    barrier.wait()
                    try:
                        results.append(tools.book(db, sid, 1, [pid], [slot_id], "cleaning",
                                                  now=FAR_FUTURE - timedelta(days=1))["booked"])
                    except tools.ToolError as e:
                        results.append(e.code)
                    except Exception as e:  # surface driver/pooler errors instead of hiding them
                        results.append(f"{type(e).__name__}: {e}")

            threads = [threading.Thread(target=attempt, args=(f"cc-{n}-{w}", pid)) for w, pid in zip("ab", pids)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            with Session() as db:
                count = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.slot_id == slot_id))
            ok = sorted(map(str, results)) == ["True", "slot_taken"] and count == 1
            outcomes.append(ok)
            print(f"round {n + 1}: {'OK ' if ok else 'BAD'} results={results} appointments_for_slot={count}")
    finally:
        with base.begin() as c:
            c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        base.dispose()
        print(f"dropped schema {schema}")
    print(f"\n{sum(outcomes)}/{len(outcomes)} rounds: exactly one booking won, the other got slot_taken")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
