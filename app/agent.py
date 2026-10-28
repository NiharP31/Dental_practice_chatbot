"""One patient turn: LLM picks tools (max 6 rounds), code runs them, guard checks the reply."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import config, guard, llm, tools
from app.prompt import build_system_prompt

log = logging.getLogger("agent")
MAX_ROUNDS = 6
HISTORY_TURNS = 20  # user turns of history sent to the model
RED_FLAG_NOTE_MARKER = "\n\n[System note:"
ApptType = Literal["cleaning", "checkup", "emergency"]


# --- tool argument schemas (validated before any tool runs; also sent to the LLM) ----------

class VerifyArgs(BaseModel):
    full_name: str = Field(description="Patient's full name as they said it")
    dob: str = Field(description="Date of birth as the patient said it, e.g. 'March 12 1984' or '03/12/1984'")


class RegisterArgs(BaseModel):
    full_name: str
    phone: str
    dob: str = Field(description="Date of birth as the patient said it")
    insurance: str = Field(description='Insurance plan name, or "none" if uninsured')
    guardian_id: int | None = Field(None, description="patient_id of the verified parent, when registering a child")


class FindSlotsArgs(BaseModel):
    appt_type: ApptType
    when: str = Field(description="The patient's own words for when, e.g. 'later next week', 'tomorrow afternoon', "
                                  "'as soon as possible'. Do not convert to dates.")
    count: int = Field(1, ge=1, le=6, description="Number of back-to-back slots, one per family member")
    exclude_slot_ids: list[int] = Field(default_factory=list, description="slot_ids the patient already turned down")


class BookArgs(BaseModel):
    patient_ids: list[int] = Field(description="One per person, in the same order as slot_ids")
    slot_ids: list[int] = Field(description="From find_slots options; one per patient")
    appt_type: ApptType
    notes: str = Field("", description="Short reason for visit / emergency summary")


class ModifyArgs(BaseModel):
    appointment_id: int
    action: Literal["cancel", "reschedule"]
    new_slot_id: int | None = Field(None, description="Required for reschedule; from find_slots")
    confirm: bool = Field(False, description="true only after the patient said yes to the read-back details")


class NotifyArgs(BaseModel):
    priority: Literal["urgent", "normal"]
    summary: str = Field(description="What the staff need to know: who, what happened, what they want")


@dataclass
class Turn:
    db: Session
    session_id: str
    turn: int
    now: datetime
    options: list[dict] = field(default_factory=list)  # latest find_slots options -> UI quick replies
    action_succeeded: bool = False  # a book/modify actually changed (or replayed) something this turn
    confirmations: list[str] = field(default_factory=list)


@dataclass
class ToolSpec:
    args: type[BaseModel]
    run: Callable[[Turn, dict], dict]
    description: str


TOOLS: dict[str, ToolSpec] = {
    "verify_patient": ToolSpec(VerifyArgs, lambda t, a: tools.verify_patient(t.db, t.session_id, now=t.now, **a),
                               "Verify an existing patient by full name + date of birth. Returns their upcoming "
                               "appointments and any dependents they can manage."),
    "register_patient": ToolSpec(RegisterArgs, lambda t, a: tools.register_patient(t.db, t.session_id, **a),
                                 "Register a new patient once you have name, phone, DOB and insurance."),
    "find_slots": ToolSpec(FindSlotsArgs, lambda t, a: tools.find_slots(t.db, now=t.now, **a),
                           "Find up to 3 open appointment options in the window the patient described."),
    "book": ToolSpec(BookArgs, lambda t, a: tools.book(t.db, t.session_id, t.turn, now=t.now, **a),
                     "Book appointments for verified/registered patients. Book a whole family in one call."),
    "modify": ToolSpec(ModifyArgs, lambda t, a: tools.modify(t.db, t.session_id, t.turn, now=t.now, **a),
                       "Cancel or reschedule an appointment of a verified patient. Two-step: without confirm "
                       "it returns details to read back; with confirm=true (after the patient's yes) it applies."),
    "notify_staff": ToolSpec(NotifyArgs, lambda t, a: tools.notify_staff(t.db, t.session_id, **a),
                             "Alert the office staff (urgent for emergencies, normal for callbacks/handoffs)."),
}


def tool_schemas() -> list[dict]:
    return [{"name": n, "description": s.description, "parameters": s.args.model_json_schema()}
            for n, s in TOOLS.items()]


def dispatch(t: Turn, name: str, raw_args: dict) -> dict:
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": "unknown_tool", "message": f"No tool named {name}."}
    try:
        args = spec.args.model_validate(raw_args).model_dump()
    except ValidationError as e:
        return {"error": "bad_args", "message": str(e)[:600]}
    try:
        result = spec.run(t, args)
    except tools.ToolError as e:
        result = {"error": e.code, "message": e.message}
    except SQLAlchemyError:
        t.db.rollback()
        log.exception("db_error tool=%s session=%s", name, t.session_id)
        result = {"error": "system_error", "message": f"Temporary system problem. Apologize and offer {config.OFFICE_PHONE}."}
    log.info("tool=%s session=%s args=%s error=%s", name, t.session_id, args, result.get("error"))

    if name == "find_slots" and "options" in result:
        t.options = result["options"]
    if name == "book" and result.get("booked"):
        t.action_succeeded = True
        t.confirmations += [f"{a['patient']}: {a['type']} on {a['when']}" for a in result["appointments"]]
    if name == "modify" and result.get("done"):
        t.action_succeeded = True
        if result.get("details"):
            t.confirmations.append(result["details"])
    return result


def _recent(history: list[dict]) -> list[dict]:
    user_idx = [i for i, m in enumerate(history) if m["role"] == "user"]
    return history[user_idx[-HISTORY_TURNS]:] if len(user_idx) > HISTORY_TURNS else history


def run_turn(db: Session, session_id: str, text: str, turn: int | None = None, now: datetime | None = None) -> dict:
    now = now or config.office_now()
    conv = tools.get_conversation(db, session_id)
    state = dict(conv.state or {})
    last = state.get("turn", 0)
    if turn is not None and turn <= last and state.get("last_reply"):
        return state["last_reply"]  # client retried a turn we already answered
    turn = max(turn or 0, last + 1)
    t = Turn(db, session_id, turn, now)
    history = list(conv.messages or [])

    prefix = ""
    if guard.red_flag(text):
        prefix = guard.EMERGENCY_911
        if not state.get("red_flag_alerted"):
            tools.notify_staff(db, session_id, "urgent", f"RED FLAG symptoms reported in chat: {text[:500]}")
            tools.set_state(conv, red_flag_alerted=True)
        text += (RED_FLAG_NOTE_MARKER + " red-flag symptoms detected. The patient has already been told to call 911 / go "
                 "to the ER and staff were alerted as urgent. Don't repeat the 911 advice; be calm and brief, ask "
                 "what happened if unclear, and offer the soonest emergency slot.]")
    history.append({"role": "user", "text": text})

    system, specs, reply = build_system_prompt(now), tool_schemas(), None
    try:
        for _ in range(MAX_ROUNDS):
            msg = llm.complete(system, _recent(history), specs)
            history.append({k: msg[k] for k in ("role", "text", "tool_calls")})
            if not msg["tool_calls"]:
                reply = msg["text"].strip()
                break
            for call in msg["tool_calls"]:
                result = dispatch(t, call["name"], call["args"])
                history.append({"role": "tool", "id": call["id"], "name": call["name"], "result": result})
        else:
            log.warning("max_rounds session=%s", session_id)
    except llm.LLMUnavailable:
        log.error("llm_unavailable session=%s", session_id)

    if not reply:
        reply = guard.canned_reply()
        if t.confirmations:  # a change went through before the LLM failed: say so from the tool result
            reply = "Done: " + "; ".join(t.confirmations) + f". If you need anything else, call us at {config.OFFICE_PHONE}."
    reply = prefix + guard.check_reply(reply, t.action_succeeded, session_id,
                                       tools.session_booked_days(db, session_id, now))

    if history[-1]["role"] == "assistant" and not history[-1]["tool_calls"]:
        history[-1] = {**history[-1], "text": reply}  # keep what the patient actually saw
    else:
        history.append({"role": "assistant", "text": reply, "tool_calls": []})
    # Quick-reply buttons only while the patient is still choosing.
    options = [] if t.action_succeeded else [o["when"] for o in t.options]
    payload = {"reply": reply, "options": options, "turn": turn}
    conv.messages = history
    tools.set_state(conv, turn=turn, last_reply=payload)
    db.commit()
    return payload
