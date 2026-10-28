import pytest
from fastapi.testclient import TestClient

from app import config, main, tools
from app.db import Conversation, StaffAlert


@pytest.fixture
def client(session_factory, monkeypatch):
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    return TestClient(main.app)  # not used as a context manager: skips startup seeding


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
    assert client.get("/health").json() == {"ok": True}

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
