import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import config, guard
from app.agent import run_turn
from app.db import SessionLocal
from app.seed import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(guard.RedactFilter())
log = logging.getLogger("app")

SESSION_ID = r"^[A-Za-z0-9_-]{8,64}$"


@asynccontextmanager
async def lifespan(_app):
    log.info("startup seed=%s", init_db())
    yield


app = FastAPI(title=config.OFFICE_NAME, lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str = Field(pattern=SESSION_ID)
    message: str = Field(min_length=1, max_length=2000)
    turn: int | None = Field(None, ge=1, description="Client turn counter; a retried turn returns the original reply")


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    try:
        with SessionLocal() as db:
            return run_turn(db, body.session_id, body.message.strip(), body.turn)
    except Exception:
        # Never a stack trace or blank screen for the patient.
        log.exception("chat_failed session=%s", body.session_id)
        return {"reply": guard.canned_reply(), "options": [], "turn": body.turn, "error": True}
