"""Agent loop with a scripted LLM: checks that code, not the model, decides what happened."""
import pytest

from app import agent, guard, llm
from app.db import Appointment, Conversation, StaffAlert
from tests.conftest import NOW, add_day, add_patient

THU = NOW.date().replace(day=29)


def say(text):
    return {"role": "assistant", "text": text, "tool_calls": []}


def call(name, **args):
    return {"role": "assistant", "text": "", "tool_calls": [{"id": f"c_{name}", "name": name, "args": args}]}


@pytest.fixture
def script(monkeypatch):
    """Queue of LLM responses; each may be a dict or a callable(history) -> dict / exception."""
    queue, seen = [], []

    def fake_complete(system, history, tools):
        seen.append(list(history))
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return {**(step(history) if callable(step) else step), "provider": "fake"}

    monkeypatch.setattr(llm, "complete", fake_complete)
    return queue, seen


def last_tool_result(history):
    return next(m["result"] for m in reversed(history) if m["role"] == "tool")


def test_full_booking_turn(db, script):
    queue, _ = script
    ids = add_day(db, THU)
    add_patient(db)
    queue += [call("verify_patient", full_name="Daniel Reyes", dob="March 12 1984"),
              call("find_slots", appt_type="cleaning", when="thursday morning")]
    queue.append(say("I looked at Thursday morning. Which works?"))
    r = agent.run_turn(db, "sess-0001", "I'm Daniel Reyes, born March 12 1984, need a cleaning thursday morning", now=NOW)
    assert r["options"][0] == "Thursday, October 29 at 8:00 AM" and r["turn"] == 1

    queue += [call("book", patient_ids=[1], slot_ids=[ids["08:00"]], appt_type="cleaning"),
              say("You're all set for Thursday at 8:00 AM!")]
    r = agent.run_turn(db, "sess-0001", "Thursday, October 29 at 8:00 AM", now=NOW)
    assert r["reply"] == "You're all set for Thursday at 8:00 AM!" and r["options"] == []
    assert db.query(Appointment).count() == 1


def test_unbacked_claim_is_replaced_and_nothing_booked(db, script):
    queue, _ = script
    queue.append(say("Great, you're booked for Friday at 2pm!"))
    r = agent.run_turn(db, "sess-0002", "book me friday 2pm", now=NOW)
    assert r["reply"] == guard.UNBACKED_REPLY and db.query(Appointment).count() == 0


def test_failed_booking_does_not_count_as_backing(db, script):
    queue, _ = script
    ids = add_day(db, THU)
    pid = add_patient(db)  # never verified in this session
    queue += [call("book", patient_ids=[pid], slot_ids=[ids["09:00"]], appt_type="cleaning"),
              say("All set, see you Thursday!")]
    r = agent.run_turn(db, "sess-0003", "book it", now=NOW)
    assert r["reply"] == guard.UNBACKED_REPLY


def test_tool_errors_and_bad_args_go_back_to_the_model(db, script):
    queue, seen = script
    queue += [call("find_slots", appt_type="cleaning", when="when pigs fly"),
              call("find_slots", appt_type="whitening", when="tomorrow"),
              call("no_such_tool"),
              say("Which day works for you?")]
    agent.run_turn(db, "sess-0004", "cleaning please", now=NOW)
    errors = [m["result"]["error"] for m in seen[-1] if m["role"] == "tool"]
    assert errors == ["unclear_date", "bad_args", "unknown_tool"]


def test_all_llms_down_gives_canned_reply_with_phone(db, script):
    queue, _ = script
    queue.append(llm.LLMUnavailable())
    r = agent.run_turn(db, "sess-0005", "hi", now=NOW)
    assert "(555) 010-2200" in r["reply"]


def test_llm_dies_after_booking_reports_what_tool_confirmed(db, script):
    queue, _ = script
    ids = add_day(db, THU)
    add_patient(db)
    queue += [call("verify_patient", full_name="Daniel Reyes", dob="1984-03-12"),
              call("book", patient_ids=[1], slot_ids=[ids["09:00"]], appt_type="checkup"),
              llm.LLMUnavailable()]
    r = agent.run_turn(db, "sess-0006", "Daniel Reyes 3/12/1984 checkup thursday 9am", now=NOW)
    assert r["reply"].startswith("Done: Daniel Reyes: checkup on Thursday, October 29 at 9:00 AM")


def test_max_rounds_stops_the_loop(db, script):
    queue, _ = script
    add_day(db, THU)
    queue += [call("find_slots", appt_type="cleaning", when="thursday")] * agent.MAX_ROUNDS
    r = agent.run_turn(db, "sess-0007", "cleaning thursday", now=NOW)
    assert "(555) 010-2200" in r["reply"] and not queue


def test_red_flag_alerts_staff_once_and_leads_with_911(db, script):
    queue, seen = script
    queue += [say("I'm so sorry. Can you tell me when this started?"), say("Thank you, hang tight.")]
    r = agent.run_turn(db, "sess-0008", "my face is swelling and it's hard to swallow", now=NOW)
    assert r["reply"].startswith("If you're having trouble breathing") and "911" in r["reply"]
    assert "[System note: red-flag" in seen[0][-1]["text"]
    agent.run_turn(db, "sess-0008", "the swelling is spreading to my eye", now=NOW)
    alerts = db.query(StaffAlert).all()
    assert len(alerts) == 1 and alerts[0].priority == "urgent"


def test_retried_turn_returns_original_reply_without_rerunning(db, script):
    queue, seen = script
    queue.append(say("Hi! How can I help?"))
    first = agent.run_turn(db, "sess-0009", "hello", turn=1, now=NOW)
    again = agent.run_turn(db, "sess-0009", "hello", turn=1, now=NOW)
    assert first == again and len(seen) == 1


def test_history_persists_across_requests(session_factory, script):
    queue, seen = script
    queue += [say("Hi! What's your name?"), say("Nice to meet you, Sam.")]
    with session_factory() as db:
        agent.run_turn(db, "sess-0010", "hello", now=NOW)
    with session_factory() as db:  # new request, new DB session
        agent.run_turn(db, "sess-0010", "I'm Sam", now=NOW)
        assert [m["role"] for m in seen[1]] == ["user", "assistant", "user"]
        assert len(db.get(Conversation, "sess-0010").messages) == 4


def test_provider_fallback_chain(monkeypatch):
    class Boom:
        name = "gemini"

        def complete(self, *a):
            raise RuntimeError("429 quota")

    class Ok:
        name = "deepseek"

        def complete(self, *a):
            return say("hello from fallback")

    monkeypatch.setattr(llm, "_PROVIDERS", [Boom(), Ok()])
    assert llm.complete("sys", [], [])["provider"] == "deepseek"
    monkeypatch.setattr(llm, "_PROVIDERS", [Boom()])
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("sys", [], [])


def test_gemini_history_round_trips_signatures_and_groups_tool_results():
    from google.genai import types
    g = object.__new__(llm.Gemini)
    g.types = types
    history = [{"role": "user", "text": "hi"},
               {"role": "assistant", "text": "", "tool_calls": [
                   {"id": "a", "name": "find_slots", "args": {"when": "tomorrow"}, "sig": "c2lnbmF0dXJl"},
                   {"id": "b", "name": "verify_patient", "args": {}}]},
               {"role": "tool", "id": "a", "name": "find_slots", "result": {"options": []}},
               {"role": "tool", "id": "b", "name": "verify_patient", "result": {"verified": False}}]
    contents = g._contents(history)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].thought_signature == b"signature"
    assert len(contents[2].parts) == 2


def test_chat_endpoint_never_returns_a_stack_trace(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    def broken(*a, **k):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(main, "run_turn", broken)
    r = TestClient(main.app).post("/chat", json={"session_id": "sess-0011", "message": "hi"})
    assert r.status_code == 200 and "(555) 010-2200" in r.json()["reply"]
    assert TestClient(main.app).post("/chat", json={"session_id": "x", "message": ""}).status_code == 422


def _gemini_with(monkeypatch, responses):
    """A Gemini provider whose SDK call pops from `responses` (exceptions are raised)."""
    from google.genai import errors, types
    g = object.__new__(llm.Gemini)
    g.types, g.errors, g.model, g.name, g.skip_until = types, errors, "m", "gemini:m", 0.0
    calls = []

    def generate_content(model, **kw):
        calls.append(model)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    g.client = type("C", (), {"models": type("M", (), {"generate_content": staticmethod(generate_content)})})()
    monkeypatch.setattr(llm.time, "sleep", lambda s: calls.append(f"sleep {s}"))
    return g, calls


def _429(delay="3s"):
    from google.genai import errors
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                                              "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                           "retryDelay": delay}]}})


def test_429_retries_once_after_suggested_delay(monkeypatch):
    g, calls = _gemini_with(monkeypatch, [_429("3s"), "ok"])
    assert g._generate() == "ok" and calls == ["m", "sleep 3.0", "m"]


def test_second_429_or_long_delay_moves_on(monkeypatch):
    from google.genai import errors
    g, calls = _gemini_with(monkeypatch, [_429("3s"), _429("3s")])
    with pytest.raises(errors.ClientError):
        g._generate()
    g, calls = _gemini_with(monkeypatch, [_429("41820s"), "never reached"])
    with pytest.raises(errors.ClientError):
        g._generate()
    assert calls == ["m"]  # too long to make a patient wait: no sleep, next model
    with pytest.raises(llm.LLMUnavailable):  # and the model is skipped (no call) until the delay passes
        g._generate()
    assert calls == ["m"]


def test_throttle_spaces_calls(monkeypatch):
    from app import config
    clock, slept = [100.0], []
    monkeypatch.setattr(config, "LLM_MAX_RPM", 8)
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm.time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(llm, "_last_call", [0.0])
    llm.throttle()
    llm.throttle()
    assert slept == [7.5]

