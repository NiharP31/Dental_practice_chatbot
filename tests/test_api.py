import pytest
from fastapi.testclient import TestClient

from app import config, main, tools
from app.db import Conversation, StaffAlert


@pytest.fixture
def client(session_factory, monkeypatch):
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    main._hits.clear()
    return TestClient(main.app)  # not used as a context manager: skips startup seeding


def test_per_ip_rate_limit(client, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 3)
    body = {"session_id": "sess-rl01", "rating": "up"}
    assert [client.post("/feedback", json=body).status_code for _ in range(3)] == [200, 200, 200]
    r = client.post("/feedback", json=body)
    assert r.status_code == 429 and "(555) 010-2200" in r.json()["reply"]
    assert client.get("/health").status_code == 200  # GETs aren't limited


# Header shapes recorded live on Render (Cloudflare in front) on 2026-10-28, with a spoofed X-Forwarded-For:
REAL, CF_EDGE, RENDER_HOP = "73.241.213.202", "162.158.167.153", "10.31.179.45"
SPOOFED_XFF = f"1.2.3.4,{REAL}, {CF_EDGE}, {RENDER_HOP}"


def test_client_ip_ignores_spoofed_x_forwarded_for_and_uses_the_edge_header(client, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 2)
    body = {"session_id": "sess-xff1", "rating": "up"}

    def post(real_ip, xff):
        return client.post("/feedback", json=body, headers={"True-Client-IP": real_ip, "X-Forwarded-For": xff}).status_code

    # The attacker rotates the leftmost X-Forwarded-For entry; the limiter still sees one client.
    assert [post(REAL, f"1.2.3.{i},{REAL}, {CF_EDGE}, {RENDER_HOP}") for i in range(3)] == [200, 200, 429]
    assert post("198.51.100.9", f"198.51.100.9, {CF_EDGE}, {RENDER_HOP}") == 200  # another patient, same edge
    assert set(main._hits) == {REAL, "198.51.100.9"}  # not 1.2.3.x, and not Cloudflare's edge IP


def test_client_ip_header_preference_and_fallbacks():
    from starlette.requests import Request

    def ip(headers, peer="10.31.179.45"):
        scope = {"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                 "client": (peer, 1234)}
        return main.client_ip(Request(scope))

    assert ip({"True-Client-IP": REAL, "CF-Connecting-IP": "9.9.9.9", "X-Forwarded-For": SPOOFED_XFF}) == REAL
    assert ip({"CF-Connecting-IP": REAL, "X-Forwarded-For": SPOOFED_XFF}) == REAL
    assert ip({"True-Client-IP": "not-an-ip", "CF-Connecting-IP": REAL}) == REAL  # garbage is skipped
    assert ip({"X-Forwarded-For": SPOOFED_XFF}, peer="127.0.0.1") == "127.0.0.1"  # local dev: XFF never trusted


def test_health_reports_deployed_commit(client, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "0123456789abcdef")
    assert client.get("/health").json() == {"ok": True, "version": "0123456"}


def test_feedback_thumbs_down_flags_conversation(client, session_factory):
    assert client.post("/feedback", json={"session_id": "sess-up01", "rating": "up"}).json() == {"ok": True}
    client.post("/feedback", json={"session_id": "sess-dn01", "rating": "down", "comment": "slow"})
    with session_factory() as db:
        assert db.get(Conversation, "sess-dn01").flagged
        assert not db.get(Conversation, "sess-up01")
    assert client.post("/feedback", json={"session_id": "sess-dn01", "rating": "meh"}).status_code == 422


def test_staff_api_sorts_urgent_first_and_resolves(client, session_factory):
    with session_factory() as db:
        tools.notify_staff(db, "sess-a001", "normal", "Callback about billing")
        tools.notify_staff(db, "sess-b001", "urgent", "Knocked-out tooth")
    alerts = client.get("/api/staff").json()["alerts"]
    assert [a["priority"] for a in alerts] == ["urgent", "normal"]
    client.post(f"/api/alerts/{alerts[0]['id']}/resolve")
    assert [a["summary"] for a in client.get("/api/staff").json()["alerts"]] == ["Callback about billing"]


def test_flagged_transcript_hides_internal_notes(client, session_factory):
    with session_factory() as db:
        db.add(Conversation(session_id="sess-f001", flagged=True, state={}, messages=[
            {"role": "user", "text": "my jaw is broken" + main.RED_FLAG_NOTE_MARKER + " secret"},
            {"role": "assistant", "text": "", "tool_calls": [{"id": "1", "name": "notify_staff", "args": {}}]},
            {"role": "tool", "id": "1", "name": "notify_staff", "result": {}},
            {"role": "assistant", "text": "Help is on the way.", "tool_calls": []}]))
        db.commit()
    [conv] = client.get("/api/staff").json()["flagged"]
    assert conv["transcript"] == [{"role": "user", "text": "my jaw is broken"},
                                  {"role": "assistant", "text": "Help is on the way."}]


def test_staff_api_requires_token_when_set(client, monkeypatch):
    monkeypatch.setattr(config, "STAFF_TOKEN", "s3cret")
    assert client.get("/api/staff").status_code == 401
    assert client.get("/api/staff", headers={"X-Staff-Token": "wrong"}).status_code == 401
    assert client.get("/api/staff", headers={"X-Staff-Token": "s3cret"}).status_code == 200
    assert client.post("/api/alerts/1/resolve").status_code == 401


def test_handoff_creates_normal_alert_with_context(client, session_factory):
    with session_factory() as db:
        db.add(Conversation(session_id="sess-h001", state={}, messages=[{"role": "user", "text": "billing question"}]))
        db.commit()
    r = client.post("/handoff", json={"session_id": "sess-h001"}).json()
    assert "team" in r["reply"]
    with session_factory() as db:
        [alert] = db.query(StaffAlert).all()
        assert alert.priority == "normal" and "billing question" in alert.summary


def test_health(client, monkeypatch):
    assert client.get("/health").json()["ok"] is True

    def down():
        raise RuntimeError("db down")
    monkeypatch.setattr(main, "SessionLocal", down)
    assert client.get("/health").status_code == 503


def test_slack_failure_never_breaks_the_alert(db, monkeypatch):
    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.invalid/x")

    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(tools.httpx, "post", boom)
    assert tools.post_slack("urgent", "x") is False
    assert tools.notify_staff(db, "sess-s001", "urgent", "Swelling")["alerted"]
    assert db.query(StaffAlert).count() == 1


def test_slack_message_masks_phone_and_dob(monkeypatch):
    sent = {}
    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.invalid/x")

    class Ok:
        def raise_for_status(self):
            pass
    monkeypatch.setattr(tools.httpx, "post", lambda url, json, timeout: sent.update(json) or Ok())
    assert tools.post_slack("urgent", "Call 555-010-1002, DOB 03/12/1984")
    assert "1002" not in sent["text"] and "1984" not in sent["text"] and "URGENT" in sent["text"]


def test_postgres_engine_is_set_up_for_supabase_transaction_pooler(monkeypatch):
    from sqlalchemy.pool import NullPool

    from app import db as dbmod
    seen = {}
    monkeypatch.setattr(dbmod, "create_engine", lambda url, **kw: seen.update(url=url, **kw))
    dbmod.make_engine("postgresql://postgres.ref:pw@aws-0-us-west-1.pooler.supabase.com:6543/postgres")
    assert seen["url"].startswith("postgresql+psycopg://") and seen["poolclass"] is NullPool
    assert seen["connect_args"] == {"prepare_threshold": None, "sslmode": "require"}
    dbmod.make_engine("postgresql://postgres:postgres@localhost:55432/test")  # local docker: no TLS
    assert seen["connect_args"] == {"prepare_threshold": None}


def test_supabase_project_url_is_rejected_with_a_clear_message():
    from app.db import make_engine
    with pytest.raises(ValueError, match="Transaction pooler"):
        make_engine("https://abcdefgh.supabase.co")


def test_migrate_adds_staff_alerts_patient_id_to_an_existing_table(tmp_path):
    """Supabase already has staff_alerts without patient_id; create_all won't add it."""
    from sqlalchemy import inspect, text

    from app.db import Base, make_engine, migrate
    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as c:  # simulate the pre-change schema
        c.execute(text("DROP TABLE staff_alerts"))
        c.execute(text("CREATE TABLE staff_alerts (id INTEGER PRIMARY KEY, session_id VARCHAR(64), priority VARCHAR(10), "
                       "summary TEXT, status VARCHAR(10), created_at DATETIME)"))
    assert migrate(engine) == ["staff_alerts.patient_id"]
    assert "patient_id" in {c["name"] for c in inspect(engine).get_columns("staff_alerts")}
    assert migrate(engine) == []  # idempotent


# --- shared-demo reset ------------------------------------------------------------------------

def _demo_state(db, today):
    from datetime import datetime, time, timedelta

    from app.db import Appointment, Patient, Slot
    from app.seed import PATIENTS
    seed_names = {p[0] for p in PATIENTS}
    rows = (db.query(Patient.full_name, Appointment.appt_type, Appointment.status, Slot.start)
            .join(Appointment, Appointment.patient_id == Patient.id).join(Slot, Slot.id == Appointment.slot_id).all())
    seeded = sorted((n, t, s.isoformat()) for n, t, st, s in rows if n in seed_names and st == "booked")
    lo, hi = datetime.combine(today + timedelta(days=2), time()), datetime.combine(today + timedelta(days=11), time())
    assert all(lo <= datetime.fromisoformat(s) < hi for _, _, s in seeded), seeded
    assert all(8 <= datetime.fromisoformat(s).hour < 18 for _, _, s in seeded)
    return seeded, sorted((n, t, st) for n, t, st, _ in rows if n not in seed_names)


def test_reset_demo_restores_seed_and_keeps_testers(session_factory):
    from datetime import date

    from app import tools
    from app.db import Appointment, Feedback, Patient
    from app.seed import reset_demo, seed
    today = date(2026, 11, 2)  # a Monday
    with session_factory() as db:
        seed(db, today=date(2026, 10, 28))  # seeded a week ago: Daniel's cleaning is already in the past
        daniel = db.query(Patient).filter_by(full_name="Daniel Reyes").one()
        db.query(Appointment).filter_by(patient_id=daniel.id).update({"status": "cancelled"})  # a tester cancelled it
        tester = tools.register_patient(db, "tester-1", "Pat Tester", "5559998888", "1990-01-01", "none")["patient_id"]
        tools.notify_staff(db, "tester-1", "normal", "Pat wants a callback")       # linked to a real patient: kept
        tools.notify_staff(db, "anon-1", "urgent", "Broken tooth, never identified")  # cleared
        db.add_all([Feedback(session_id="tester-1", rating="up"), Feedback(session_id="anon-1", rating="down")])
        tools.get_conversation(db, "anon-1")
        tools.get_conversation(db, "seed-only").state = {"verified": [daniel.id]}  # only touched a seeded patient
        db.commit()

        r = reset_demo(db, today=today)
        assert r["appointments_reset"] == 6 and r["conversations_removed"] == 2
        assert (r["feedback_removed"], r["alerts_removed"]) == (1, 1)
        seeded, others = _demo_state(db, today)
        assert len(seeded) == 6  # one upcoming appointment per seeded patient, 2-10 days out, office hours
        assert db.get(Patient, tester) is not None
        assert [c.session_id for c in db.query(main.Conversation)] == ["tester-1"]
        assert [f.session_id for f in db.query(Feedback)] == ["tester-1"]
        assert [a.summary for a in db.query(main.StaffAlert)] == ["Pat wants a callback"]

        again = reset_demo(db, today=today)  # idempotent
        assert _demo_state(db, today) == (seeded, others)
        assert (again["conversations_removed"], again["feedback_removed"], again["alerts_removed"]) == (0, 0, 0)


def test_reset_demo_endpoint_requires_staff_token(client, session_factory, monkeypatch):
    from app import config
    from app.seed import seed
    with session_factory() as db:
        seed(db)
    monkeypatch.setattr(config, "STAFF_TOKEN", "s3cret")
    assert client.post("/api/staff/reset-demo").status_code == 401
    r = client.post("/api/staff/reset-demo", headers={"X-Staff-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["appointments_reset"] == 6
