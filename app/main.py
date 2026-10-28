import hmac
import ipaddress
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import case, select, text

from app import config, guard, tools
from app.agent import PATIENT_TAGS, RED_FLAG_NOTE_MARKER, run_turn
from app.db import Conversation, Feedback, Patient, SessionLocal, StaffAlert
from app.seed import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(guard.RedactFilter())
log = logging.getLogger("app")

SESSION_ID = r"^[A-Za-z0-9_-]{8,64}$"
STATIC = Path(__file__).resolve().parent.parent / "static"


def visible(text: str) -> str:
    """Message text as the patient wrote/saw it (drops the agent's internal system notes)."""
    return PATIENT_TAGS.sub("", text.split(RED_FLAG_NOTE_MARKER)[0]).strip()


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("RENDER") and config.DATABASE_URL.startswith("sqlite"):
        # Render's disk is wiped on every restart/spin-down: patients and bookings would vanish.
        raise RuntimeError("DATABASE_URL must point at Postgres (Supabase) on Render, not SQLite")
    # Creates missing tables and seeds empty ones; never drops or overwrites anything.
    log.info("startup seed=%s", init_db())
    if not config.STAFF_TOKEN:
        log.warning("STAFF_TOKEN is not set: the staff API is open. Set it in any shared deployment.")
    yield


app = FastAPI(title=config.OFFICE_NAME, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ponytail: per-process sliding window; with more than one instance, move it to Redis or the edge.
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


# Render serves through Cloudflare. Checked live: Cloudflare overwrites a client-sent True-Client-IP
# with the real address and rejects a client-sent CF-Connecting-IP outright, while X-Forwarded-For
# keeps whatever the client put first (so its leftmost entry is spoofable, and its rightmost
# non-10.x entry is a Cloudflare edge, not the patient).
EDGE_IP_HEADERS = ("true-client-ip", "cf-connecting-ip")


def client_ip(request: Request) -> str:
    for h in EDGE_IP_HEADERS:
        try:
            return str(ipaddress.ip_address(request.headers.get(h, "").strip()))
        except ValueError:
            continue
    return request.client.host if request.client else "unknown"  # local dev: no edge in front


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Per-IP cap on POSTs: a public URL spends real LLM credits."""
    if request.method == "POST":
        ip, now = client_ip(request), time.monotonic()
        with _hits_lock:
            q = _hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= config.RATE_LIMIT_PER_MIN:
                log.warning("rate_limited ip=%s", ip)
                return JSONResponse({"reply": "You're sending messages a little fast. Please wait a moment and try "
                                              f"again, or call us at {config.OFFICE_PHONE}.", "options": [], "error": True},
                                    status_code=429)
            q.append(now)
    return await call_next(request)


class ChatIn(BaseModel):
    session_id: str = Field(pattern=SESSION_ID)
    message: str = Field(min_length=1, max_length=2000)
    turn: int | None = Field(None, ge=1, description="Client turn counter; a retried turn returns the original reply")


class SessionIn(BaseModel):
    session_id: str = Field(pattern=SESSION_ID)


class FeedbackIn(SessionIn):
    rating: Literal["up", "down"]
    comment: str = Field("", max_length=1000)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/staff")
def staff_page():
    return FileResponse(STATIC / "staff.html")


@app.get("/health")
def health():
    # version: the deployed commit (Render sets RENDER_GIT_COMMIT), so a deploy can be confirmed
    # with GET requests instead of anything that writes.
    version = (os.getenv("RENDER_GIT_COMMIT") or "dev")[:7]
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {"ok": True, "version": version}
    except Exception:
        log.exception("health_db_failed")
        return JSONResponse({"ok": False, "version": version}, status_code=503)


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    try:
        with SessionLocal() as db:
            return run_turn(db, body.session_id, body.message.strip(), body.turn)
    except Exception:
        # Never a stack trace or blank screen for the patient.
        log.exception("chat_failed session=%s", body.session_id)
        return {"reply": guard.canned_reply(), "options": [], "turn": body.turn, "error": True}


@app.post("/handoff")
def handoff(body: SessionIn) -> dict:
    """The always-visible "Talk to a person" button. Works even when the LLM is down."""
    try:
        with SessionLocal() as db:
            conv = db.get(Conversation, body.session_id)
            last = [visible(m["text"]) for m in (conv.messages if conv else []) if m["role"] == "user"][-3:]
            summary = "Patient pressed 'Talk to a person'." + (" Recent messages: " + " | ".join(last) if last else "")
            tools.notify_staff(db, body.session_id, "normal", summary[:2000])
        return {"reply": "I've asked a member of our team to reach out to you shortly. If it's urgent, "
                         f"please call us at {config.OFFICE_PHONE}."}
    except Exception:
        log.exception("handoff_failed session=%s", body.session_id)
        return {"reply": f"Please call us at {config.OFFICE_PHONE} and our team will help you right away.", "error": True}


@app.post("/feedback")
def feedback(body: FeedbackIn) -> dict:
    with SessionLocal() as db:
        db.add(Feedback(session_id=body.session_id, rating=body.rating, comment=body.comment.strip()))
        if body.rating == "down":  # thumbs-down puts the conversation in the staff review queue
            tools.get_conversation(db, body.session_id).flagged = True
        db.commit()
    return {"ok": True}


# --- staff API ------------------------------------------------------------------------------

def require_staff(x_staff_token: str = Header("")):
    # ponytail: one shared token; per-user staff accounts + audit log when there's more than one office
    if config.STAFF_TOKEN and not hmac.compare_digest(x_staff_token, config.STAFF_TOKEN):
        raise HTTPException(401, "Staff token required")


@app.get("/api/staff", dependencies=[Depends(require_staff)])
def staff_data() -> dict:
    with SessionLocal() as db:
        alerts = db.scalars(select(StaffAlert).where(StaffAlert.status == "open").order_by(
            case((StaffAlert.priority == "urgent", 0), else_=1), StaffAlert.created_at)).all()
        flagged = db.scalars(select(Conversation).where(Conversation.flagged)
                             .order_by(Conversation.updated_at.desc()).limit(50)).all()
        comments = {f.session_id: f.comment for f in db.scalars(
            select(Feedback).where(Feedback.rating == "down", Feedback.session_id.in_([c.session_id for c in flagged])))}
        names = {p.id: p.full_name for p in db.scalars(
            select(Patient).where(Patient.id.in_({a.patient_id for a in alerts if a.patient_id})))}
        return {
            "alerts": [{"id": a.id, "priority": a.priority, "summary": a.summary, "session_id": a.session_id,
                        "patient": names.get(a.patient_id), "created_at": a.created_at.isoformat() + "Z"}
                       for a in alerts],
            "flagged": [{"session_id": c.session_id, "updated_at": c.updated_at.isoformat() + "Z",
                         "comment": comments.get(c.session_id, ""),
                         "transcript": [{"role": m["role"], "text": visible(m["text"])} for m in c.messages
                                        if m["role"] in ("user", "assistant") and m.get("text")]}
                        for c in flagged],
        }


@app.post("/api/alerts/{alert_id}/resolve", dependencies=[Depends(require_staff)])
def resolve_alert(alert_id: int) -> dict:
    with SessionLocal() as db:
        alert = db.get(StaffAlert, alert_id)
        if alert is None:
            raise HTTPException(404, "No such alert")
        alert.status = "resolved"
        db.commit()
    return {"ok": True}


@app.post("/api/conversations/{session_id}/unflag", dependencies=[Depends(require_staff)])
def unflag(session_id: str) -> dict:
    with SessionLocal() as db:
        conv = db.get(Conversation, session_id)
        if conv is None:
            raise HTTPException(404, "No such conversation")
        conv.flagged = False
        db.commit()
    return {"ok": True}
