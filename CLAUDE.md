# Bright Smile Dental chatbot

Take-home assessment (brief: `assessment.pdf`). Front-desk chatbot for one dental office.

## Core principle
The LLM decides *which* tool to call. Code decides everything that must be correct:
date math, availability, booking, identity checks. The model never computes dates,
invents slots, or claims a booking that a tool didn't confirm (`app/guard.py` enforces it).

## Stack (decided, don't relitigate)
- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, pytest, uv.
- LLM: Gemini Flash-class via `google-genai` (native tool calling) -> DeepSeek (OpenAI-compatible)
  -> canned reply with the office phone. Model names come from env vars only.
- DB: SQLite locally/tests/evals, Supabase Postgres in prod. Switch only on `DATABASE_URL`.
- Deploy: Render free web service. Local disk is wiped on restart: nothing durable on disk in prod.
- Frontend: one static HTML/JS page served by FastAPI, no framework.

## Office
Bright Smile Dental, open 8am-6pm Mon-Sat, America/Los_Angeles, (555) 010-2200.
Accepts all major dental insurance; self-pay, membership and financing for uninsured patients.

## Layout
- `app/config.py` office facts, env, office clock
- `app/db.py` SQLAlchemy models + engine
- `app/dates.py` patient phrase ("later next week") -> date/time window
- `app/tools.py` the only code that reads/writes patient data on the agent's behalf
- `app/seed.py` seeds only empty tables, extends slots to 4 weeks out; never wipes
- `tests/` unit tests (`uv run pytest`)

## Rules
- Session state (history, verified patients, pending confirmations) lives in the `conversations`
  table, never in process memory.
- Name + DOB verification before revealing or changing anything.
- Slot claims: `UPDATE slots SET status='booked' WHERE id=:id AND status='open'`; rowcount 0 = slot_taken.
- Booking idempotency key: `f"{session_id}:{turn}"`.
- Tools raise `ToolError(code, message)`; the agent loop turns it into a tool result.
- Never log phone/DOB unmasked. Never let the LLM write SQL.
- Never commit secrets or the SQLite file.
