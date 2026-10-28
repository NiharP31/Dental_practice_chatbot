"""Everything the agent can do to patient data. Plain typed functions; each one validates its own
inputs, enforces identity checks, and owns its transaction. Errors raise ToolError(code, message)
for the agent loop to hand back to the model."""
import hashlib
import json
import logging
import re
import threading
from datetime import date, datetime, time, timedelta
from itertools import pairwise

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app import config, guard
from app.dates import ToolError, UnclearDate, Window, fmt_slot, full_dates_in, resolve_when
from app.db import Appointment, Conversation, IdempotencyKey, Patient, Slot, StaffAlert
from app.prompt import NOTE_MARKER

__all__ = ["ToolError", "UnclearDate", "verify_patient", "register_patient", "find_slots", "book",
           "modify", "notify_staff", "get_conversation"]

log = logging.getLogger("tools")

MAX_OPTIONS = 3
MAX_VERIFY_FAILURES = 3
DOB_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d %Y", "%b %d %Y", "%d %B %Y")
NO_INSURANCE = {"none", "no", "no insurance", "uninsured", "self pay", "self-pay", "n/a", "na", "nope"}


# --- session state ------------------------------------------------------------------------

def get_conversation(db: Session, session_id: str) -> Conversation:
    conv = db.get(Conversation, session_id)
    if conv is None:
        conv = Conversation(session_id=session_id, messages=[], state={})
        db.add(conv)
        db.flush()
    return conv


def set_state(conv: Conversation, **changes) -> None:
    conv.state = {**(conv.state or {}), **changes}  # reassign so SQLAlchemy sees the JSON change


def _verified_ids(conv: Conversation) -> set[int]:
    return set((conv.state or {}).get("verified", []))


def _mark_verified(db: Session, conv: Conversation, patient_id: int) -> None:
    set_state(conv, verified=sorted(_verified_ids(conv) | {patient_id}))
    # Alerts raised before we knew who this was (e.g. an emergency) now get their patient.
    db.execute(update(StaffAlert).where(StaffAlert.session_id == conv.session_id, StaffAlert.patient_id.is_(None))
               .values(patient_id=patient_id))


def session_has_alert(db: Session, session_id: str, priority: str | None = None) -> bool:
    q = select(StaffAlert.id).where(StaffAlert.session_id == session_id)
    if priority:
        q = q.where(StaffAlert.priority == priority)
    return db.scalar(q.limit(1)) is not None


def _authorized_ids(db: Session, conv: Conversation) -> set[int]:
    """Patients this session may act for: those verified/registered here plus their dependents."""
    verified = _verified_ids(conv)
    if not verified:
        return set()
    return verified | set(db.scalars(select(Patient.id).where(Patient.guardian_id.in_(verified))))


# --- parsing helpers ----------------------------------------------------------------------

def _clean_name(name: str) -> str:
    return " ".join(name.split()).title()


def _parse_dob(raw: str) -> date:
    s = " ".join(raw.replace(",", " ").split())
    for fmt in DOB_FORMATS:
        try:
            dob = datetime.strptime(s, fmt).date()
            break
        except ValueError:
            continue
    else:
        raise ToolError("bad_dob", "Date of birth not understood. Ask for it as month, day and year.")
    today = config.office_now().date()
    if dob > today or dob.year < today.year - 120:
        raise ToolError("bad_dob", "That date of birth isn't plausible. Ask the patient to repeat it.")
    return dob


def session_booked_days(db: Session, session_id: str, now: datetime) -> set[date]:
    """Dates of every upcoming booked appointment this session may talk about."""
    ids = _authorized_ids(db, get_conversation(db, session_id))
    starts = db.scalars(select(Slot.start).join(Appointment, Appointment.slot_id == Slot.id).where(
        Appointment.patient_id.in_(ids), Appointment.status == "booked", Slot.start >= now)) if ids else []
    return {s.date() for s in starts}


def _upcoming(db: Session, patient_id: int, now: datetime) -> list[dict]:
    rows = db.execute(
        select(Appointment, Slot).join(Slot, Slot.id == Appointment.slot_id)
        .where(Appointment.patient_id == patient_id, Appointment.status == "booked", Slot.start >= now)
        .order_by(Slot.start)).all()
    return [{"appointment_id": a.id, "type": a.appt_type, "when": fmt_slot(s.start)} for a, s in rows]


# --- tools --------------------------------------------------------------------------------

def verify_patient(db: Session, session_id: str, full_name: str, dob: str, now: datetime | None = None) -> dict:
    """Existing patient: name + date of birth must both match before anything is revealed or changed."""
    now = now or config.office_now()
    conv = get_conversation(db, session_id)
    failures = (conv.state or {}).get("verify_failures", 0)
    if failures >= MAX_VERIFY_FAILURES:
        raise ToolError("verify_locked", f"Too many failed attempts. Offer a staff callback or {config.OFFICE_PHONE}.")
    d = _parse_dob(dob)
    no_match = {"verified": False, "message": "No patient matches that name and date of birth."}
    # Provenance: the DOB must be one the patient typed in this session. A model talked into
    # "verifying" someone (or guessing) can't supply it. Not counted against the lockout: it's the
    # model's mistake, not a guess by the person chatting.
    said = set().union(*(full_dates_in(m["text"].split(NOTE_MARKER)[0])
                         for m in conv.messages or [] if m["role"] == "user"))
    if d not in said:
        log.warning("dob_not_from_patient session=%s", session_id)
        return no_match
    patient = db.scalar(select(Patient).where(
        func.lower(Patient.full_name) == _clean_name(full_name).lower(), Patient.dob == d))
    if patient is None:
        set_state(conv, verify_failures=failures + 1)
        db.commit()
        # Same answer whether the name exists or not: don't leak who is a patient.
        return no_match
    _mark_verified(db, conv, patient.id)
    db.commit()
    dependents = db.scalars(select(Patient).where(Patient.guardian_id == patient.id).order_by(Patient.id))
    result = {"verified": True, "patient_id": patient.id, "full_name": patient.full_name,
              "insurance": patient.insurance, "upcoming": _upcoming(db, patient.id, now),
              "dependents": [{"patient_id": d.id, "full_name": d.full_name, "upcoming": _upcoming(db, d.id, now)}
                             for d in dependents]}
    if result["upcoming"] or any(d["upcoming"] for d in result["dependents"]):
        # Where the model is looking right now; a prompt rule alone didn't stop it asking "cancel it?" first.
        result["next_step"] = ("If the patient already said they want to cancel one of these (and which one is "
                               "clear), call modify without confirm now, before asking them anything; it returns "
                               "the details to read back. Don't ask 'shall I cancel it?' first.")
    return result


def register_patient(db: Session, session_id: str, full_name: str, phone: str, dob: str, insurance: str,
                     guardian_id: int | None = None) -> dict:
    """New patient. insurance is the plan name, or "none" for self-pay. guardian_id links a child
    to a parent verified in this session so the parent can manage their appointments later."""
    name = _clean_name(full_name)
    if len(name.split()) < 2:
        raise ToolError("bad_name", "Need the patient's full first and last name.")
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise ToolError("bad_phone", "Phone number should be 10 digits. Ask the patient to repeat it.")
    d = _parse_dob(dob)
    ins = " ".join(insurance.split())
    if not ins:
        raise ToolError("bad_insurance", 'Ask for the insurance name, or "none" if uninsured.')
    if ins.lower() in NO_INSURANCE:
        ins = "none"

    conv = get_conversation(db, session_id)
    if guardian_id is not None and guardian_id not in _verified_ids(conv):
        raise ToolError("not_verified", "The guardian must be verified (name + DOB) in this conversation first.")
    existing = db.scalar(select(Patient).where(func.lower(Patient.full_name) == name.lower(), Patient.dob == d))
    if existing:
        # Name + DOB is our identity check, so this person is effectively verified.
        _mark_verified(db, conv, existing.id)
        db.commit()
        return {"patient_id": existing.id, "full_name": existing.full_name, "already_registered": True}
    patient = Patient(full_name=name, phone=digits, dob=d, insurance=ins, guardian_id=guardian_id)
    db.add(patient)
    db.flush()
    _mark_verified(db, conv, patient.id)
    db.commit()
    return {"patient_id": patient.id, "full_name": name, "insurance": ins, "registered": True}


SCREENING_ASK = ("In ONE short message ask: which tooth or area, pain from 0 to 10, when it happened (if not "
                 "said), and whether they have any facial swelling, bleeding that won't stop, or trouble "
                 "breathing or swallowing. Then call triage_emergency with the answers.")


def find_slots(db: Session, appt_type: str, when: str, count: int = 1,
               exclude_slot_ids: list[int] | None = None, now: datetime | None = None,
               session_id: str | None = None) -> dict:
    """Up to 3 open options in the window the patient described. count>1 = back-to-back run
    (family). Options are spread across different days/times rather than 3 adjacent slots.
    Emergency slots are only offered after the red-flag screening (triage_emergency) came back clear."""
    if appt_type not in config.APPT_TYPES:
        raise ToolError("bad_type", f"appt_type must be one of {', '.join(config.APPT_TYPES)}.")
    if appt_type == "emergency" and session_id is not None:
        triage = (get_conversation(db, session_id).state or {}).get("triage")
        if triage is None:
            raise ToolError("screening_required", "Screen before offering emergency times. " + SCREENING_ASK)
        if triage["red_flag"]:
            raise ToolError("red_flag", "The patient has been told to call 911 or go to the ER. Don't offer "
                                        "appointments; stay with their safety.")
    if not 1 <= count <= 6:
        raise ToolError("bad_count", "count must be between 1 and 6.")
    now = now or config.office_now()
    w = resolve_when(when, now)
    exclude = set(exclude_slot_ids or [])

    runs = _open_runs(db, w, count, now, exclude)
    picked: list[list[Slot]] = []
    for ok in (lambda r: r[0].start.date() not in {p[0].start.date() for p in picked},  # new day
               lambda r: all(abs(r[0].start - p[0].start) >= timedelta(hours=2) for p in picked),
               lambda r: all(not set(r) & set(p) for p in picked)):  # at least non-overlapping
        for r in runs:
            if len(picked) < MAX_OPTIONS and r not in picked and ok(r):
                picked.append(r)
    picked.sort(key=lambda r: r[0].start)

    result = {"searched": w.describe(), "options": [
        {"slot_ids": [s.id for s in r], "when": fmt_slot(r[0].start),
         **({"times": [fmt_slot(s.start).split(" at ")[1] for s in r]} if count > 1 else {})}
        for r in picked]}
    if not picked:
        later = _open_runs(db, Window(w.last + timedelta(days=1), w.last + timedelta(days=60)), count, now, exclude)
        result["message"] = "No openings in that window."
        if later:
            result["next_available"] = {"slot_ids": [s.id for s in later[0]], "when": fmt_slot(later[0][0].start)}
    return result


def _open_runs(db: Session, w: Window, count: int, now: datetime, exclude: set[int]) -> list[list[Slot]]:
    slots = db.scalars(select(Slot).where(
        Slot.status == "open", Slot.start > now,
        Slot.start >= datetime.combine(w.first, time()),
        Slot.start < datetime.combine(w.last + timedelta(days=1), time()),
    ).order_by(Slot.start)).all()
    slots = [s for s in slots if w.contains(s.start) and s.id not in exclude]
    runs = [slots[i:i + count] for i in range(len(slots) - count + 1)]
    return [r for r in runs if all(b.start - a.start == config.SLOT_LEN for a, b in pairwise(r))]


def book(db: Session, session_id: str, turn: int, patient_ids: list[int], slot_ids: list[int],
         appt_type: str, notes: str = "", now: datetime | None = None) -> dict:
    """Book one or more patients (patient_ids[i] -> slot_ids[i]) in one all-or-nothing transaction.
    Idempotent per (session, turn, request): a retried call replays; a different booking in the
    same turn is a new booking."""
    fingerprint = hashlib.sha256(json.dumps([sorted(slot_ids), sorted(patient_ids), appt_type]).encode()).hexdigest()
    key = f"{session_id}:{turn}:{fingerprint[:16]}"
    if prior := db.get(IdempotencyKey, key):
        return {**prior.result, "replayed": True}
    now = now or config.office_now()
    if appt_type not in config.APPT_TYPES:
        raise ToolError("bad_type", f"appt_type must be one of {', '.join(config.APPT_TYPES)}.")
    if not patient_ids or len(patient_ids) != len(slot_ids):
        raise ToolError("bad_args", "Give exactly one slot_id per patient_id.")
    if len(set(slot_ids)) != len(slot_ids) or len(set(patient_ids)) != len(patient_ids):
        raise ToolError("bad_args", "Each patient and each slot can appear only once.")
    conv = get_conversation(db, session_id)
    if not set(patient_ids) <= _authorized_ids(db, conv):
        raise ToolError("not_verified", "Every patient must be verified (name + DOB), registered, "
                                        "or a dependent of a verified guardian first.")

    try:
        booked = []
        for pid, sid in zip(patient_ids, slot_ids):
            slot = db.get(Slot, sid)
            if slot is None or slot.start <= now:
                raise ToolError("bad_slot", f"Slot {sid} isn't a bookable future time. Search again.")
            claimed = db.execute(update(Slot).where(Slot.id == sid, Slot.status == "open")
                                 .values(status="booked")).rowcount
            if claimed == 0:
                raise ToolError("slot_taken", "That time was just taken. Nothing was booked; offer other times.")
            appt = Appointment(patient_id=pid, slot_id=sid, appt_type=appt_type, notes=notes[:1000])
            db.add(appt)
            db.flush()
            booked.append({"appointment_id": appt.id, "patient": db.get(Patient, pid).full_name,
                           "type": appt_type, "when": fmt_slot(slot.start)})
        result = {"booked": True, "appointments": booked}
        db.add(IdempotencyKey(key=key, result=result))
        db.commit()
    except Exception:
        db.rollback()
        raise
    if appt_type == "emergency" and not session_has_alert(db, session_id, "urgent"):
        # Backstop: an emergency must reach staff even if the model never called notify_staff.
        visits = "; ".join(f"{a['patient']} on {a['when']}" for a in booked)
        notify_staff(db, session_id, "urgent", f"Emergency appointment booked in chat: {visits}. "
                                               f"Patient's description: {notes.strip() or 'not given'}")
    return result


def modify(db: Session, session_id: str, turn: int, appointment_id: int, action: str,
           new_slot_id: int | None = None, confirm: bool = False, now: datetime | None = None) -> dict:
    """Reschedule or cancel. Two-phase: the first call returns details to read back; it only
    executes when called again with confirm=True on a later turn (i.e. after the patient said yes)."""
    now = now or config.office_now()
    if action not in ("cancel", "reschedule"):
        raise ToolError("bad_action", 'action must be "cancel" or "reschedule".')
    conv = get_conversation(db, session_id)
    appt = db.get(Appointment, appointment_id)
    if appt is None or appt.patient_id not in _authorized_ids(db, conv):
        # Same error for "doesn't exist" and "not yours": don't leak appointment ids.
        raise ToolError("not_verified", "Verify the patient's name and date of birth first.")

    # Retries of an already-applied change succeed quietly.
    if action == "cancel" and appt.status == "cancelled" or action == "reschedule" and appt.slot_id == new_slot_id:
        return {"done": True, "action": action, "already_applied": True}

    old = db.get(Slot, appt.slot_id)
    if appt.status != "booked" or old.start <= now:
        raise ToolError("not_modifiable", "That appointment is not upcoming, so it can't be changed here.")
    who = db.get(Patient, appt.patient_id).full_name
    if action == "cancel":
        details = f"Cancel {who}'s {appt.appt_type} on {fmt_slot(old.start)}"
    else:
        if new_slot_id is None:
            raise ToolError("bad_args", "Reschedule needs new_slot_id from find_slots.")
        new = db.get(Slot, new_slot_id)
        if new is None or new.start <= now or new.status != "open":
            raise ToolError("slot_taken", "That new time isn't available. Search again.")
        details = f"Move {who}'s {appt.appt_type} from {fmt_slot(old.start)} to {fmt_slot(new.start)}"

    pending = {"appointment_id": appointment_id, "action": action, "new_slot_id": new_slot_id}
    stored = (conv.state or {}).get("pending") or {}
    same = {k: stored.get(k) for k in pending} == pending
    asked_turn = stored["turn"] if same else turn  # re-asking the same change keeps the original turn
    if not (confirm and same and asked_turn < turn):
        set_state(conv, pending={**pending, "turn": asked_turn})
        db.commit()
        return {"needs_confirmation": True, "details": details,
                "instruction": "Read these details to the patient and ask for a clear yes. "
                               "Call modify again with confirm=true only after they agree."}

    try:
        if action == "reschedule":
            claimed = db.execute(update(Slot).where(Slot.id == new_slot_id, Slot.status == "open")
                                 .values(status="booked")).rowcount
            if claimed == 0:
                raise ToolError("slot_taken", "That new time was just taken. Nothing changed; offer other times.")
            appt.slot_id = new_slot_id
        else:
            appt.status = "cancelled"
        old.status = "open"
        set_state(conv, pending=None)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"done": True, "action": action, "details": details}


def triage_emergency(db: Session, session_id: str, what_happened: str, when: str, area: str,
                     pain: int | None, facial_swelling: bool, uncontrolled_bleeding: bool,
                     trouble_breathing_or_swallowing: bool) -> dict:
    """Record the emergency screening. Code, not the model, decides what a yes means: any red flag
    routes to the 911 path. The summary (written by code, always complete) goes on the session's
    urgent alert: the existing one is updated rather than a second alert created."""
    red = facial_swelling or uncontrolled_bleeding or trouble_breathing_or_swallowing
    yn = lambda b: "YES" if b else "no"  # noqa: E731
    summary = (("RED FLAG, told to call 911 / go to the ER. " if red else "")
               + f"Dental emergency: {what_happened.strip() or 'not described'}. When: {when.strip() or 'not given'}. "
               f"Area: {area.strip() or 'not given'}. Pain: {'not given' if pain is None else f'{pain}/10'}. "
               f"Screening: facial swelling {yn(facial_swelling)}; bleeding that won't stop {yn(uncontrolled_bleeding)}; "
               f"trouble breathing/swallowing {yn(trouble_breathing_or_swallowing)}.")
    conv = get_conversation(db, session_id)
    alert = db.scalar(select(StaffAlert).where(StaffAlert.session_id == session_id, StaffAlert.priority == "urgent",
                                               StaffAlert.status == "open").order_by(StaffAlert.id.desc()).limit(1))
    if alert:
        alert.summary = summary[:2000]
        set_state(conv, triage={"red_flag": red}, **({"red_flag_alerted": True} if red else {}))
        db.commit()
        alert_id = alert.id
    else:
        set_state(conv, triage={"red_flag": red}, **({"red_flag_alerted": True} if red else {}))
        alert_id = notify_staff(db, session_id, "urgent", summary)["alert_id"]
    return {"triaged": True, "red_flag": red, "alert_id": alert_id, "summary": summary,
            "next": ("RED FLAG: the patient is shown 911/ER advice by our system. Don't repeat it, don't offer or "
                     "book appointments; be calm and ask if someone is with them.") if red else
                    ("Screening clear. Get their details (verify or register) and offer the soonest emergency "
                     "slot; put the summary in the booking notes.")}


def notify_staff(db: Session, session_id: str, priority: str, summary: str) -> dict:
    """Write a staff alert. The DB row is the source of truth; Slack is best-effort on top. Works
    before anyone is identified; the alert is linked to the patient once they verify or register."""
    if priority not in ("urgent", "normal"):
        raise ToolError("bad_priority", 'priority must be "urgent" or "normal".')
    if not summary.strip():
        raise ToolError("bad_summary", "Summarize what the staff need to know.")
    known = sorted(_verified_ids(get_conversation(db, session_id)))
    alert = StaffAlert(session_id=session_id, priority=priority, summary=summary.strip()[:2000],
                       patient_id=known[0] if known else None)
    db.add(alert)
    db.commit()
    if config.SLACK_WEBHOOK_URL:
        # Off the request path: a slow or broken webhook must never delay or fail the patient's turn.
        threading.Thread(target=post_slack, args=(priority, alert.summary), daemon=True).start()
    return {"alerted": True, "alert_id": alert.id, "priority": priority}


def post_slack(priority: str, summary: str) -> bool:
    # ponytail: phone/DOB are masked, but names/symptoms still go to Slack; production needs a BAA'd
    # channel or just a link to /staff.
    text = f"{':rotating_light: URGENT' if priority == 'urgent' else ':bell:'} {config.OFFICE_NAME}: {guard.redact(summary)}"
    try:
        httpx.post(config.SLACK_WEBHOOK_URL, json={"text": text}, timeout=5).raise_for_status()
        return True
    except Exception as e:
        log.warning("slack_failed error=%s", type(e).__name__)
        return False
