"""One patient turn: LLM picks tools (max 6 rounds), code runs them, guard checks the reply."""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import config, guard, llm, tools
from app.db import Slot
from app.prompt import NOTE_MARKER, SYSTEM, clock_note

log = logging.getLogger("agent")
MAX_ROUNDS = 6
# Cost cap per session. Also why history is never trimmed: 30 turns fits easily in context, and an
# append-only history keeps the prompt cache (and any provider's history checks) valid.
MAX_TURNS = 30
RED_FLAG_NOTE_MARKER = NOTE_MARKER
PATIENT_TAGS = re.compile(r"</?patient_message>", re.I)
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
    patient_ids: list[int] = Field(default_factory=list, description="Family bookings: who the times are for, in "
                                                                     "order (one per slot); empty otherwise")


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


_UNANSWERED = "null if the patient didn't answer or declined"


class TriageArgs(BaseModel):
    """Fill from anything the patient said in the whole conversation, not just the last message."""
    what_happened: str = Field(description="What happened, in the patient's words")
    when: str = Field("", description="When it happened, as the patient said it; empty if not answered")
    area: str = Field("", description="Which tooth or area of the mouth; empty if not answered")
    pain: int | None = Field(None, ge=0, le=10, description=f"Pain 0-10; {_UNANSWERED}")
    facial_swelling: bool | None = Field(description=_UNANSWERED)
    uncontrolled_bleeding: bool | None = Field(description=f"Bleeding that won't stop; {_UNANSWERED}")
    trouble_breathing_or_swallowing: bool | None = Field(description=_UNANSWERED)
    patient_declined: bool = Field(False, description="true only if the patient said they won't answer the questions")


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
    registered: bool = False  # register_patient succeeded this turn
    action_failed: bool = False  # a book/modify returned an error this turn
    triage_red: bool = False  # the emergency screening found a red flag this turn
    negative_backed: bool = False  # a tool said a time can't be had (book/modify error, empty search)
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
    "find_slots": ToolSpec(FindSlotsArgs, lambda t, a: tools.find_slots(t.db, now=t.now, session_id=t.session_id, **a),
                           "Find up to 3 open appointment options in the window the patient described. "
                           "Emergency times need triage_emergency first."),
    "triage_emergency": ToolSpec(TriageArgs, lambda t, a: tools.triage_emergency(t.db, t.session_id, **a),
                                 "Record the emergency screening answers (area, pain, red flags). Creates or "
                                 "updates the urgent staff alert; any red flag routes to the 911 path."),
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

    if name == "register_patient" and result.get("patient_id"):
        t.registered = True
    if name in ("book", "modify") and "error" in result:
        t.action_failed = True
    if name in ("book", "modify", "find_slots") and ("error" in result or result.get("options") == []):
        t.negative_backed = True
    if name == "triage_emergency" and result.get("red_flag"):
        t.triage_red = True
    if name == "triage_emergency":  # remember an incomplete screening so the next turn can't loop on it
        pending = {"args": args, "turn": t.turn} if result.get("error") == "screening_incomplete" else None
        tools.set_state(tools.get_conversation(t.db, t.session_id), pending_triage=pending)
        t.db.commit()
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


def selected_option(offered: list[dict], text: str, slot_ids: list[int] | None,
                    patient_ids: list[int] | None = None) -> int | None:
    """Index of the offered option the patient tapped (its slot_ids) or typed (its exact label)."""
    for i, o in enumerate(offered):
        if slot_ids:
            if o["slot_ids"] == slot_ids and patient_ids in (None, [], o.get("patient_ids", [])):
                return i
        elif text.strip().lower() == o["label"].lower():
            return i
    return None


def reoffer(db: Session, offered: list[dict], now: datetime) -> tuple[str, list[dict]]:
    """The guard's replacement for an unbacked "that time is gone": the offered times still open."""
    still = [o for o in offered if all((s := db.get(Slot, i)) and s.status == "open" and s.start > now
                                       for i in o["slot_ids"])]
    if not still:
        return "I'm sorry, I got that wrong. Which day would suit you? I'll check what's open.", []
    return ("I'm sorry, I got that wrong: these times are still open. Which one would you like?\n"
            + "\n".join(o["label"] for o in still)), still


def run_turn(db: Session, session_id: str, text: str, turn: int | None = None, now: datetime | None = None,
             slot_ids: list[int] | None = None, patient_ids: list[int] | None = None) -> dict:
    now = now or config.office_now()
    conv = tools.get_conversation(db, session_id)
    state = dict(conv.state or {})
    last = state.get("turn", 0)
    if turn is not None and turn <= last and state.get("last_reply"):
        return state["last_reply"]  # client retried a turn we already answered
    turn = max(turn or 0, last + 1)
    if turn > MAX_TURNS:
        payload = {"reply": "We've covered a lot in this chat! To keep things easy to follow, please start a new chat "
                            f"or call us at {config.OFFICE_PHONE} and our team will help.", "options": [], "turn": turn}
        tools.set_state(conv, turn=turn, last_reply=payload)
        db.commit()
        return payload
    t = Turn(db, session_id, turn, now)
    history = list(conv.messages or [])

    red = guard.red_flag(text)
    # The first red flag of a session is answered by code alone (no LLM call, so the 911 advice is
    # said exactly once); later turns go to the model with the note below.
    first_red = red and not state.get("red_flag_alerted")
    if first_red:
        tools.notify_staff(db, session_id, "urgent", f"RED FLAG symptoms reported in chat: {text[:500]}")
    offered = state.get("offered") or []
    picked = selected_option(offered, text, slot_ids, patient_ids)
    # Delimit the patient's words so instructions inside them read as patient text; strip any tags
    # they typed so they can't close the block early.
    text = f"<patient_message>\n{PATIENT_TAGS.sub('', text)}\n</patient_message>"
    # The clock rides on the user turn, not the system prompt, so the system prompt stays cacheable.
    text += clock_note(now)
    pending = state.get("pending_triage")
    if pending and pending["turn"] < turn and not red:
        text += (NOTE_MARKER + " the red-flag question was already asked again once and is still unanswered. If "
                 "this message doesn't answer it, don't ask again: call triage_emergency now with what you have, "
                 "null for the rest and patient_declined=true, then carry on with booking.]")
    if red:
        text += (NOTE_MARKER + " red-flag symptoms detected. Our system has told the patient in this chat to "
                 "call 911 or go to the ER, that staff were alerted as urgent, and asked whether someone is with "
                 "them. Don't repeat the alert, and mention 911 at most once, briefly. Be calm, kind and brief: "
                 "their safety comes first, so don't offer to book unless they say they're OK or ask for an "
                 "appointment.]")
    if picked is not None and not red:
        o = offered[picked]
        log.info("selection session=%s option=%d slot_ids=%s", session_id, picked + 1, o["slot_ids"])
        text += (NOTE_MARKER + f" the patient selected option {picked + 1} ({o['label']}): slot_ids "
                 f"{o['slot_ids']}" + (f" for patient_ids {o['patient_ids']}" if o.get("patient_ids") else "")
                 + ". Call book now with exactly these ids (for a reschedule: modify with this new_slot_id). Don't judge yourself whether this time is valid, "
                 "open or passed: book decides, and if it returns an error, tell the patient its reason.]")
    history.append({"role": "user", "text": text})
    conv.messages = list(history)  # tools (verify_patient's DOB provenance) read this turn's patient text
    if first_red:
        return _finish(t, conv, history, guard.RED_FLAG_REPLY, [], red_flag=True)

    specs, reply, answered_by = tool_schemas(), None, []
    try:
        for _ in range(MAX_ROUNDS):
            msg = llm.complete(SYSTEM, history, specs, db=db, session_id=session_id)
            answered_by.append(msg.get("provider", "?"))
            history.append({k: msg[k] for k in ("role", "text", "tool_calls", "provider") if k in msg})
            if not msg["tool_calls"]:
                reply = msg["text"].strip()
                break
            for call in msg["tool_calls"]:
                result = dispatch(t, call["name"], call["args"])
                history.append({"role": "tool", "id": call["id"], "name": call["name"], "result": result})
            if t.triage_red:  # screening found a red flag: code writes the reply, no further LLM call
                break
        else:
            log.warning("max_rounds session=%s", session_id)
    except llm.LLMUnavailable:
        log.error("llm_unavailable session=%s", session_id)
    log.info("turn session=%s turn=%d providers=%s", session_id, turn, ",".join(answered_by) or "none")
    if pending and pending["turn"] < turn and (conv.state or {}).get("pending_triage") == pending:
        # Backstop: the model ignored the note and re-asked again. Record the gap ourselves so an
        # emergency patient can always get booked; staff see "not answered" in the alert.
        log.warning("screening_auto_declined session=%s", session_id)
        tools.triage_emergency(db, session_id, **{**pending["args"], "patient_declined": True})
        tools.set_state(conv, pending_triage=None)
        db.commit()

    if t.triage_red:
        # A red flag that only came out in screening ("yes, my cheek is swelling") gets the same
        # code-written reply as one typed up front; triage_emergency already alerted staff.
        return _finish(t, conv, history, guard.RED_FLAG_REPLY, [], red_flag=True)
    shown = t.options
    if reply and offered and not t.negative_backed and guard.negative_claim(reply):
        # "That time has passed / is taken" with no tool saying so: the model judged validity itself.
        log.warning("unbacked_negative_claim session=%s reply=%r", session_id, guard.redact(reply[:300]))
        reply, kept = reoffer(db, offered, now)
        shown = [_as_found(o) for o in kept]
    if not reply:
        reply = guard.canned_reply()
        if t.confirmations:  # a change went through before the LLM failed: say so from the tool result
            reply = "Done: " + "; ".join(t.confirmations) + f". If you need anything else, call us at {config.OFFICE_PHONE}."
        elif tools.session_has_alert(db, session_id, "urgent"):  # true by DB, so it's safe to say
            reply = ("I'm sorry, I'm having trouble on my end, but our dental team has been alerted that this is "
                     f"urgent. Please call us right away at {config.OFFICE_PHONE}, and if you have trouble "
                     "breathing or swallowing, or swelling that's spreading, call 911.")
    reply = guard.check_reply(reply, t.action_succeeded, session_id,
                              tools.session_booked_days(db, session_id, now), registered=t.registered,
                              action_failed=t.action_failed, alerted=tools.session_has_alert(db, session_id))
    if not shown:  # earlier offers the reply names again ("which of these: Monday, Nov 9 at 8:00 AM...")
        shown = [_as_found(o) for o in offered if o["label"].lower() in reply.lower()]
    # Quick-reply buttons only while the patient is still choosing.
    return _finish(t, conv, history, reply, [] if t.action_succeeded else shown)


def _as_found(option: dict) -> dict:
    """A quick-reply option back in find_slots' shape."""
    return {"when": option["label"], **{k: option[k] for k in ("slot_ids", "patient_ids") if k in option}}


def _finish(t: Turn, conv, history: list[dict], reply: str, found: list[dict], red_flag: bool = False) -> dict:
    """Save what the patient saw and return it. `found` = find_slots options to show as quick replies."""
    if history[-1]["role"] == "assistant" and not history[-1]["tool_calls"]:
        history[-1] = {**history[-1], "text": reply}  # keep what the patient actually saw
    else:
        history.append({"role": "assistant", "text": reply, "tool_calls": []})
    # Each quick reply carries its slot ids, so a tap books exactly what was offered.
    options = [{"label": o["when"], "slot_ids": o["slot_ids"],
                **({"patient_ids": o["patient_ids"]} if o.get("patient_ids") else {})} for o in found]
    payload = {"reply": reply, "options": options, "turn": t.turn}
    conv.messages = history
    changes = {"turn": t.turn, "last_reply": payload}
    if options or t.action_succeeded or red_flag:  # what a tap or a typed label selects next turn
        changes["offered"] = options
    if red_flag:
        changes["red_flag_alerted"] = True
    tools.set_state(conv, **changes)
    t.db.commit()
    return payload
